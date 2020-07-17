import torch
import torch.nn as nn
from .odeint import odeint
from .misc import _check_inputs


class _AugmentedDynamics(nn.Module):
    def __init__(self, base_func, n_tensors, params):
        super(_AugmentedDynamics, self).__init__()
        self.base_func = base_func
        self.n_tensors = n_tensors
        self.params = params

    def forward(self, t, y_aug):
        # Dynamics of the original system augmented with
        # the adjoint wrt y, and an integrator wrt t and args.
        y, adj_y = y_aug[:self.n_tensors], y_aug[self.n_tensors:2 * self.n_tensors]  # Ignore adj_time and adj_params.

        with torch.set_grad_enabled(True):
            t = t.to(y[0].device).detach().requires_grad_(True)
            y = tuple(y_.detach().requires_grad_(True) for y_ in y)
            func_eval = self.base_func(t, y)

            # Workaround for PyTorch bug #39784
            _t = torch.as_strided(t, (), ())
            _y = tuple(torch.as_strided(y_, (), ()) for y_ in y)
            _f_params = tuple(torch.as_strided(param, (), ()) for param in self.params)

            vjp_t, *vjp_y_and_params = torch.autograd.grad(
                func_eval, (t,) + y + self.params,
                tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
            )
        vjp_y = vjp_y_and_params[:self.n_tensors]
        vjp_params = vjp_y_and_params[self.n_tensors:]

        # autograd.grad returns None if no gradient, set to zero.
        vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
        vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
        vjp_params = tuple(torch.zeros_like(param) if vjp_param is None else vjp_param
                           for vjp_param, param in zip(vjp_params, self.params))

        return (*func_eval, *vjp_y, vjp_t, *vjp_params)


class _OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, n_tensors, *args):

        params = args[:-n_tensors]
        y0 = args[-n_tensors:]

        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.n_tensors = n_tensors

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, *params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        t, *args = ctx.saved_tensors
        params = tuple(args[:-ctx.n_tensors])
        ans = tuple(args[-ctx.n_tensors:])

        # TODO: call odeint_adjoint to implement higher order derivatives.

        with torch.no_grad():
            T = ans[0].shape[0]

            assert T == len(t), "internal error"
            adjoint_funcs = []
            i = T - 1
            t0_ = t[-1]
            func_iter = reversed(ctx.func)
            t0, t1, func_ = next(func_iter)
            while True:
                assert t1 == t0_, "internal error"
                t0_ = max(t0, t[i - 1])

                if t1 == t[i]:
                    adjoint_funcs.append([])

                adjoint_funcs[-1].append([t1, t0_, _AugmentedDynamics(func_, ctx.n_tensors, params)])

                if t0 >= t[i - 1]:
                    try:
                        t0, t1, func_ = next(func_iter)
                    except StopIteration:
                        assert t0 == t[0], "internal error"
                        break
                else:
                    t1 = t[i - 1]
                if t0 <= t[i - 1]:
                    i -= 1
            assert i == 1, "internal error"
            assert len(adjoint_funcs) == T - 1, "internal error"
            adjoint_funcs = list(reversed(adjoint_funcs))

            # TODO: split up params by region

            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = tuple(torch.zeros_like(param) for param in params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                # index by i - 1 to select the appropriate func for this time interval
                # index by 0 to get the first region of integration for this time interval, which aligns with t[i].
                # index by 2 to get the _AugmentedDynamics out of the tuple (t0, t1, func_).
                # base_func attribute to get the underlying callable
                func_i = adjoint_funcs[i - 1][0][2].base_func(t[i], ans_i)
                assert adjoint_funcs[i - 1][0][0] == t[i], "internal error"

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                aug_y0 = (*ans_i, *adj_y, adj_time, *adj_params)
                aug_ans = odeint(
                    adjoint_funcs[i - 1], aug_y0,
                    torch.tensor([t[i], t[i - 1]]),
                    rtol=ctx.adjoint_rtol, atol=ctx.adjoint_atol, method=ctx.adjoint_method, options=ctx.adjoint_options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[ctx.n_tensors:2 * ctx.n_tensors]
                adj_time = aug_ans[2 * ctx.n_tensors]
                adj_params = aug_ans[2 * ctx.n_tensors + 1:]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                adj_params = tuple(adj_param_[1] if len(adj_param_) > 0 else adj_param_ for adj_param_ in adj_params)

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (None, time_vjps, None, None, None, None, None, None, None, None, None, *adj_params, *adj_y)


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None):
    tensor_input, func, y0, t, solution = _check_inputs(func, y0, t, adjoint=True)
    if solution is not None:
        return solution

    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method
    if adjoint_options is None:
        adjoint_options = options

    # Using PyTorch's built-in mechanism for finding parameters ensures we don't get duplicates, which would result in
    # double-counting gradients. Include gradient-requiring buffers as well to allow for the possibility of
    # "computed parameters".
    all_funcs = nn.ModuleList(func_ for _, _, func_ in func)
    params = tuple(all_funcs.parameters()) + tuple(buffer for buffer in all_funcs.buffers() if buffer.requires_grad)
    n_tensors = len(y0)

    ys = _OdeintAdjointMethod.apply(func, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, n_tensors, *params, *y0)

    if tensor_input:
        ys = ys[0]
    return ys

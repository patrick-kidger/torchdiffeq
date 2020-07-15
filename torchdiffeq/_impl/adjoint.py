import itertools
import torch
import torch.nn as nn
from .odeint import odeint
from .misc import _flatten, _flatten_convert_none_to_zeros, _check_inputs


class _AugmentedDynamics(nn.Module):
    def __init__(self, func, n_tensors):
        self.func = func
        self.n_tensors = n_tensors
        self.f_params = tuple(func.parameters())

    def forward(self, t, y_aug):
        # Dynamics of the original system augmented with
        # the adjoint wrt y, and an integrator wrt t and args.
        y, adj_y = y_aug[:self.n_tensors], y_aug[self.n_tensors:2 * self.n_tensors]  # Ignore adj_time and adj_params.

        with torch.set_grad_enabled(True):
            t = t.to(y[0].device).detach().requires_grad_(True)
            y = tuple(y_.detach().requires_grad_(True) for y_ in y)
            func_eval = self.func(t, y)
            vjp_t, *vjp_y_and_params = torch.autograd.grad(
                func_eval, (t,) + y + self.f_params,
                tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
            )
        vjp_y = vjp_y_and_params[:self.n_tensors]
        vjp_params = vjp_y_and_params[self.n_tensors:]

        # autograd.grad returns None if no gradient, set to zero.
        vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
        vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, self.f_params)

        if len(self.f_params) == 0:
            vjp_params = torch.tensor(0.).to(vjp_y[0])
        return (*func_eval, *vjp_y, vjp_t, vjp_params)


class _OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, t, flat_params, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, *y0):
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        t, flat_params, *ans = ctx.saved_tensors
        n_tensors = len(ans)

        # TODO: call odeint_adjoint to implement higher order derivatives.

        augmented_dynamics = []
        for t0, t1, func_ in ctx.func:
            augmented_dynamics.append([t1, t0, _AugmentedDynamics(func_, n_tensors)])

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)  # TODO: figure out
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = ctx.func(t[i], ans_i)

                # TODO: figure out
                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]),
                    rtol=ctx.adjoint_rtol, atol=ctx.adjoint_atol, method=ctx.adjoint_method,
                    options=ctx.adjoint_options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0:
                    adj_time = adj_time[1]
                if len(adj_params) > 0:
                    adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (None, time_vjps, adj_params, None, None, None, None, None, None, None, None, *adj_y)


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

    flat_params = _flatten(itertools.chain(func_.parameters() for _, _, func_ in func))
    ys = _OdeintAdjointMethod.apply(func, t, flat_params, rtol, atol, method, options, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, *y0)

    if tensor_input:
        ys = ys[0]
    return ys

import random
import unittest
import torch
import torchdiffeq

from problems import construct_problem

error_tol = 1e-3
error_atol = 1e-3
error_rtol = 1e-3

torch.set_default_dtype(torch.float64)
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda:0')


def rand_interval(low, high):
    return torch.rand((), dtype=low.dtype, device=low.device) * (high - low) + low


def make_stacked(f, t_points):
    breakpoints = []
    if random.choice([True, False, False]):
        breakpoints.append(rand_interval(t_points[0], t_points[1]))
    for start, end in zip(t_points[1:-1], t_points[2:]):
        if random.choice([True, False, False]):
            # include the exact time points as possibilities
            breakpoints.append(start)
        if random.choice([True, False, False]):
            # include not-the-exact-time-points as possibilities
            breakpoints.append(rand_interval(start, end))
    stacked_f = []
    breakpoints = [t_points[0]] + breakpoints + [t_points[-1]]
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        stacked_f.append([start, end, f])
    return stacked_f


def rel_error(true, estimate):
    return true.sub(estimate).div(true + 1e-6).abs().max()


class TestStacked(unittest.TestCase):

    def allclose(self, tensor1, tensor2, rtol, atol):
        # Not using torch.allclose so that we get explicit printout of what the error and tol are.
        error = tensor1.sub(tensor2).abs().max()
        tol = atol + rtol * tensor2.abs().max()
        self.assertLess(error, tol)

    def test_forward(self):
        for device in devices:
            for ode in ('constant', 'linear', 'sine'):
                f, y0, t_points, sol = construct_problem(device, ode=ode)

                for _ in range(20):
                    stacked_f = make_stacked(f, t_points)  # different stacked_f each time, as it's random
                    for odeint_ in (torchdiffeq.odeint, torchdiffeq.odeint_adjoint):
                        y = odeint_(stacked_f, y0, t_points)
                        self.assertLess(rel_error(sol, y), error_tol)

    def test_backward(self):
        for device in devices:
            for ode in ('constant', 'linear', 'sine'):
                f, y0, t_points, sol = construct_problem(device, ode=ode)
                y0.requires_grad_(True)
                t_points.requires_grad_(True)

                for _ in range(20):
                    y = torchdiffeq.odeint(f, y0, t_points, rtol=1e-7, atol=1e-12)
                    k = torch.rand_like(y)
                    y.backward(k)

                    y0_grad = y0.grad.clone()
                    t_points_grad = t_points.grad.clone()
                    param_grads = tuple(param.grad.clone() for param in f.parameters())

                    y0.grad.zero_()
                    t_points.grad.zero_()
                    for param in f.parameters():
                        param.grad.zero_()

                    stacked_f = make_stacked(f, t_points)  # different stacked_f each time, as it's random

                    for odeint_ in (torchdiffeq.odeint, torchdiffeq.odeint_adjoint):
                        y = odeint_(stacked_f, y0, t_points, rtol=1e-7, atol=1e-12)
                        y.backward(k)

                        y0_grad_ = y0.grad
                        t_points_grad_ = t_points.grad
                        self.allclose(y0_grad_, y0_grad, error_rtol, error_atol)
                        self.allclose(t_points_grad_, t_points_grad, error_rtol, error_atol)
                        for param, param_grad in zip(f.parameters(), param_grads):
                            param_grad_ = param.grad
                            self.allclose(param_grad_, param_grad, error_rtol, error_atol)

                        y0.grad.zero_()
                        t_points.grad.zero_()
                        for param in f.parameters():
                            param.grad.zero_()


if __name__ == '__main__':
    unittest.main()

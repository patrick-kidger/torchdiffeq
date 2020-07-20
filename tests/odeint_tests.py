import unittest
import torch
import torchdiffeq

import problems

error_tol = 1e-4

torch.set_default_dtype(torch.float64)
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda:0')


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


def rel_error(true, estimate):
    return max_abs((true - estimate) / true)


class TestSolverError(unittest.TestCase):

    def test_euler(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device)

            y = torchdiffeq.odeint(f, y0, t_points, method='euler')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_midpoint(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device)

            y = torchdiffeq.odeint(f, y0, t_points, method='midpoint')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_rk4(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device)

            y = torchdiffeq.odeint(f, y0, t_points, method='rk4')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_explicit_adams(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device)

            y = torchdiffeq.odeint(f, y0, t_points, method='explicit_adams')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_adams(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, ode=ode)
                y = torchdiffeq.odeint(f, y0, t_points, method='adams')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_dopri5(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, ode=ode)
                y = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_adaptive_heun(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, ode=ode)
                y = torchdiffeq.odeint(f, y0, t_points, method='adaptive_heun')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_dopri8(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, ode=ode)
                y = torchdiffeq.odeint(f, y0, t_points, method='dopri8', rtol=1e-12, atol=1e-14)
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)
                
    def test_adjoint(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)


class TestSolverBackwardsInTimeError(unittest.TestCase):

    def test_euler(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points, method='euler')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_midpoint(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points, method='midpoint')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_rk4(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points, method='rk4')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_explicit_adams(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points, method='explicit_adams')
            self.assertLess(rel_error(sol, y), error_tol)

    def test_adams(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint(f, y0, t_points, method='adams')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_dopri5(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_adaptive_heun(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint(f, y0, t_points, method='adaptive_heun')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)
                
    def test_dopri8(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint(f, y0, t_points, method='dopri8')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)

    def test_adjoint(self):
        for device in devices:
            for ode in problems.PROBLEMS.keys():
                f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

                y = torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
                with self.subTest(ode=ode):
                    self.assertLess(rel_error(sol, y), error_tol)


class TestNoIntegration(unittest.TestCase):

    def test_midpoint(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='midpoint')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_rk4(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='rk4')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_explicit_adams(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='explicit_adams')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_adams(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='adams')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_dopri5(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='dopri5')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_adaptive_heun(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='adaptive_heun')
            self.assertLess(max_abs(sol[0] - y), error_tol)

    def test_dopri8(self):
        for device in devices:
            f, y0, t_points, sol = problems.construct_problem(device, reverse=True)

            y = torchdiffeq.odeint(f, y0, t_points[0:1], method='dopri8')
            self.assertLess(max_abs(sol[0] - y), error_tol)


if __name__ == '__main__':
    unittest.main()

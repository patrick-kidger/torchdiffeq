import torch
from .tsit5 import Tsit5Solver
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .adams import VariableCoefficientAdamsBashforth
from .dopri8 import Dopri8Solver
from .misc import _check_inputs

SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'adaptive_heun': AdaptiveHeunSolver,
    'dopri8': Dopri8Solver,
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. 
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    tensor_input, func, y0, t, solution = _check_inputs(func, y0, t, adjoint=False)
    if solution is not None:
        return solution

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('cannot supply `options` without specifying `method`')
        
    if method is None:
        method = 'dopri5'
        
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(
                         method, '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    solution = [y0]
    t_index = 0
    for t0, t1, func_ in func:
        prev_t_index = t_index
        while t[t_index] < t1:
            t_index += 1
        # loop invariant: t[prev_t_index - 1] < t0 <= t[prev_t_index]
        # loop invariant: t[t_index - 1] < t1 <= t[t_index]
        # loop invariant: prev_t_index <= t_index

        y0 = solution[-1]

        t_ = []
        if t0 != t[prev_t_index]:
            solution.pop()
            t_.append(t0.unsqueeze(0))
        t_.append(t[prev_t_index:t_index])
        t_.append(t1.unsqueeze(0))
        t_ = torch.cat(t_)

        solver = SOLVERS[method](func_, y0, rtol=rtol, atol=atol, **options)
        solution_ = solver.integrate(t_)
        solution.extend(solution_)

    solution = tuple(map(torch.stack, tuple(zip(*solution))))
    if tensor_input:
        solution = solution[0]
    return solution

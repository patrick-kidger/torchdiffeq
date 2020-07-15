import warnings
import torch


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])


def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([x * y for x, y in zip(xs, ys)])


def _has_converged(y0, y1, rtol, atol):
    """Checks that each element is within the error tolerance."""
    error_tol = tuple(atol + rtol * torch.max(torch.abs(y0_), torch.abs(y1_)) for y0_, y1_ in zip(y0, y1))
    error = tuple(torch.abs(y0_ - y1_) for y0_, y1_ in zip(y0, y1))
    return all((error_ < error_tol_).all() for error_, error_tol_ in zip(error, error_tol))


def _convert_to_tensor(a, dtype=None, device=None):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if dtype is not None:
        a = a.type(dtype)
    if device is not None:
        a = a.to(device)
    return a


def _is_finite(tensor):
    _check = (tensor == float('inf')) + (tensor == float('-inf')) + torch.isnan(tensor)
    return not _check.any()


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _assert_monotone(t):
    direction = t[1:] > t[:-1]
    assert direction.all() or (~direction).all(), '`t` must be strictly increasing or decreasing'


def _assert_one_dimensional(tensor_name, tensor):
    assert len(tensor.shape) == 1, "`{}` must be one dimensional but has shape {}".format(tensor_name, tensor.shape)


def _assert_zero_dimensional(tensor_name, tensor):
    assert len(tensor.shape) == 0, "`{}` must be zero dimensional but has shape {}".format(tensor_name, tensor.shape)


def _check_floating_point(tensor_name, tensor):
    if not torch.is_floating_point(tensor):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(tensor_name, tensor.type()))


def _is_iterable(inputs):
    try:
        iter(inputs)
        return True
    except TypeError:
        return False


def _norm(x):
    """Compute RMS norm."""
    if torch.is_tensor(x):
        return x.norm() / (x.numel()**0.5)
    else:
        return torch.sqrt(sum(x_.norm()**2 for x_ in x) / sum(x_.numel() for x_ in x))


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _select_initial_step(fun, t0, y0, order, rtol, atol, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    t0 = t0.to(y0[0])
    if f0 is None:
        f0 = fun(t0, y0)

    rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
    atol = atol if _is_iterable(atol) else [atol] * len(y0)

    scale = tuple(atol_ + torch.abs(y0_) * rtol_ for y0_, atol_, rtol_ in zip(y0, atol, rtol))

    d0 = tuple(_norm(y0_ / scale_) for y0_, scale_ in zip(y0, scale))
    d1 = tuple(_norm(f0_ / scale_) for f0_, scale_ in zip(f0, scale))

    if max(d0).item() < 1e-5 or max(d1).item() < 1e-5:
        h0 = torch.tensor(1e-6).to(t0)
    else:
        h0 = 0.01 * max(d0_ / d1_ for d0_, d1_ in zip(d0, d1))

    y1 = tuple(y0_ + h0 * f0_ for y0_, f0_ in zip(y0, f0))
    f1 = fun(t0 + h0, y1)

    d2 = tuple(_norm((f1_ - f0_) / scale_) / h0 for f1_, f0_, scale_ in zip(f1, f0, scale))

    if max(d1).item() <= 1e-15 and max(d2).item() <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6).to(h0), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1 + d2))**(1. / float(order + 1))

    return torch.min(100 * h0, h1)


def _compute_error_ratio(error_estimate, error_tol=None, rtol=None, atol=None, y0=None, y1=None):
    if error_tol is None:
        assert rtol is not None and atol is not None and y0 is not None and y1 is not None
        rtol if _is_iterable(rtol) else [rtol] * len(y0)
        atol if _is_iterable(atol) else [atol] * len(y0)
        error_tol = tuple(
            atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(y1_))
            for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1)
        )
    error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_, error_tol_ in zip(error_estimate, error_tol))
    mean_sq_error_ratio = tuple(torch.mean(error_ratio_ * error_ratio_) for error_ratio_ in error_ratio)
    return mean_sq_error_ratio


def _optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next step."""
    mean_error_ratio = max(mean_error_ratio)  # Compute step size based on highest ratio.
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=torch.float64, device=mean_error_ratio.device)
    error_ratio = torch.sqrt(mean_error_ratio).to(last_step)
    exponent = torch.tensor(1 / order).to(last_step)
    factor = torch.max(1 / ifactor, torch.min(error_ratio**exponent / safety, 1 / dfactor))
    return last_step / factor


def _check_t(t):
    _assert_monotone(t)
    _assert_one_dimensional('t', t)
    assert len(t) >= 1, "`t` must be of length at least one."


def _check_inputs(func, y0, t):
    # Normalise func to being a list
    if callable(func):
        func = [[t[0], t[-1], func]]
    assert isinstance(func, (tuple, list)), ("func must either be (A) a callable, or (B) a tuple or list of 3-tuples, "
                                             "with the first two elements specifying a region of integration, and the "
                                             "third element being a callable to use on that region.")
    func = list(func)  # make a copy if it's a list; convert to list if it's a tuple

    # Check that func is a list of 3-tuples or 3-element lists and normalise the region of integration to be tensors
    for i, func_ in enumerate(func):
        assert len(func_) == 3, "each element of func should be a 3-tuple of start point, end point, and callable"
        t0, t1, func_ = func_
        t0 = _convert_to_tensor(t0)
        t1 = _convert_to_tensor(t1)
        _assert_zero_dimensional('the first element of a tuple in func, representing the start point', t0)
        _assert_zero_dimensional('the second element of a tuple in func, representing the end point', t1)
        _check_floating_point('t0', t0)
        _check_floating_point('t1', t1)
        assert callable(func_), "the third element of func must be a callable"
        func[i] = [t0, t1, func_]  # convert to list; t0 and t1 are converted to tensors

    # Normalise to tupled inputs
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)
        for func_list in func:
            _, _, func_ = func_list
            func_ = lambda t, y, _base_func=func_: (_base_func(t, y[0]),)
            func_list[2] = func_
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    # Normalise to increasing t
    if _decreasing(t):
        t = -t
        for func_list in func:
            t0, t1, func_ = func_list
            func_ = lambda t, y, _base_func=func_: tuple(-f_ for f_ in _base_func(-t, y))
            func_list[0] = -t0
            func_list[1] = -t1
            func_list[2] = func_
        func = list(reversed(func))

    # Check regions of integration
    assert func[0][0] == t[0], "the first region of integration must begin at t[0]"
    assert func[-1][1] == t[-1], "the last region of integration must end at t[-1]"
    prev_t1 = t[0]
    for t0, t1, func_ in func:
        assert prev_t1 == t0, "the regions of integration must cover the whole interval"
        assert t0 < t1, "regions of integration must be monotone and of positive size"
        prev_t1 = t1

    # Check dtypes
    for y0_ in y0:
        _check_floating_point('y0', y0_)
    _check_floating_point('t', t)

    return tensor_input, func, y0, t

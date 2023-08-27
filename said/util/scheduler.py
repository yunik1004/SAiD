"""Declare schedulers"""
import numpy as np


def frange_cycle_linear(
    n_iter: int,
    start: float = 0.0,
    stop: float = 1.0,
    n_cycle: int = 10,
    ratio: float = 0.5,
) -> np.ndarray:
    """Linear cyclical schedule.
    Refer to https://github.com/haofuml/cyclical_annealing

    Parameters
    ----------
    n_iter : int
        The number of iterations
    start : float
        Starting value, by default 0.0
    stop : float
        Ending value, by default 1.0
    n_cycle : int
        The number of cycles, by default 10
    ratio : float
        Ratio of the linear increasing part, by default 0.5

    Returns
    -------
    np.ndarray
        Scheduled values
    """
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L

import numpy as np


def bootstrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data : array-like
        original sample (for regression: 2D array with columns [x, y])

    compute_stat : callable
        function that computes a univariate statistic from data
    
    n_bootstrap : int, default 1000
        number of bootstrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable
    

    Example
    -------
    TBA

    """
    # Check the inputs.
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")

    if isinstance(n_bootstrap, (bool, np.bool_)) or not isinstance(
        n_bootstrap, (int, np.integer)
    ):
        raise TypeError("n_bootstrap must be an integer")

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")

    data = np.asarray(data)

    if data.ndim not in (1, 2):
        raise ValueError("data must be a 1D or 2D array")

    if data.size == 0:
        raise ValueError("data must not be empty")

    # Resample observations and calculate one statistic per replicate.
    n = data.shape[0]
    bootstrap_stats = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample = data[indices]
        bootstrap_stats[i] = compute_stat(sample)

    return bootstrap_stats


def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate a CI from bootstrap distribution

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        bootstrap statistics from bootstrap_sample(...)

    alpha : float, default 0.05
        significance level

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI

    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty
    
    Example
    -------
    TBA
    
    """
    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError("alpha must be a number between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    stats = np.asarray(bootstrap_stats)
    if stats.size == 0:
        raise ValueError("bootstrap_stats cannot be empty")

    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def r_squared(data):
    """
    Calculate R^2 from a linear regression

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y]

    Returns
    -------
    float
        R-squared value between 0 and 1

    Raises
    ------
    ValueError
        If data doesn't have exactly 2 columns or < 2 rows
    """
    values = np.asarray(data)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("data must be a two-dimensional array with exactly 2 columns")
    if values.shape[0] < 2:
        raise ValueError("data must contain at least 2 rows")

    try:
        values = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("data must contain numeric values") from error

    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")

    x = values[:, 0]
    y = values[:, 1]

    design_matrix = np.column_stack((np.ones(len(x)), x))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    fitted = design_matrix @ coefficients

    residual_sum_squares = np.sum((y - fitted) ** 2)
    total_sum_squares = np.sum((y - np.mean(y)) ** 2)

    if np.isclose(total_sum_squares, 0):
        return 1.0 if np.isclose(residual_sum_squares, 0) else 0.0

    result = 1 - residual_sum_squares / total_sum_squares
    return float(np.clip(result, 0.0, 1.0))

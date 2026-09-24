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
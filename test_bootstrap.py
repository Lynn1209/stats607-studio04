import numpy as np
import pytest

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


# Integration test draft: review with partner.
def test_bootstrap_pipeline():
    """Check that all three functions work together."""
    x = np.arange(100, dtype=float)
    y = 2 * x + 1
    data = np.column_stack((x, y))

    stats = bootstrap_sample(data, r_squared, n_bootstrap=200)

    assert isinstance(stats, np.ndarray)
    assert stats.shape == (200,)
    assert np.all(np.isfinite(stats))
    np.testing.assert_allclose(stats, 1.0, atol=1e-10)

    ci = bootstrap_ci(stats, alpha=0.05)

    assert isinstance(ci, tuple)
    assert len(ci) == 2
    np.testing.assert_allclose(ci, (1.0, 1.0), atol=1e-10)


# Student A: tests for bootstrap_ci.
def test_ci_percentiles():
    """Assume the percentile method with linear interpolation."""
    stats = np.arange(101, dtype=float)
    ci = bootstrap_ci(stats, alpha=0.10)

    assert isinstance(ci, tuple)
    np.testing.assert_allclose(ci, (5.0, 95.0))


def test_ci_default_alpha():
    stats = np.arange(101, dtype=float)
    ci = bootstrap_ci(stats)

    np.testing.assert_allclose(ci, (2.5, 97.5))


def test_ci_constant_values():
    ci = bootstrap_ci(np.array([5.0, 5.0, 5.0]))

    np.testing.assert_allclose(ci, (5.0, 5.0))


def test_ci_single_value():
    ci = bootstrap_ci(np.array([3.0]))

    np.testing.assert_allclose(ci, (3.0, 3.0))


def test_ci_empty():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


@pytest.mark.parametrize("alpha", [-0.1, 0, 1, 1.1])
def test_ci_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1.0, 2.0, 3.0]), alpha=alpha)


# Student A: tests for r_squared.
@pytest.mark.parametrize("slope", [2.0, -2.0])
def test_r_squared_perfect_line(slope):
    x = np.arange(5, dtype=float)
    data = np.column_stack((x, slope * x + 1))

    result = r_squared(data)

    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_r_squared_zero():
    # These nonconstant x and y values have zero covariance.
    data = np.array([
        [-1.0, 1.0],
        [0.0, -2.0],
        [1.0, 1.0],
    ])

    assert r_squared(data) == pytest.approx(0.0, abs=1e-10)


def test_r_squared_known_value():
    # For regression with an intercept, R² = 0.75 here.
    data = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ])

    assert r_squared(data) == pytest.approx(0.75)


def test_r_squared_two_rows():
    data = np.array([[0.0, 1.0], [1.0, 3.0]])

    assert r_squared(data) == pytest.approx(1.0)


@pytest.mark.parametrize("data", [
    np.empty((0, 2)),
    np.array([[1.0, 2.0]]),
    np.ones((3, 1)),
    np.ones((3, 3)),
    np.array([1.0, 2.0, 3.0]),
])
def test_r_squared_invalid_shape(data):
    with pytest.raises(ValueError):
        r_squared(data)

# Student B: tests for bootstrap_sample.

def test_bootstrap_sample_output():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    stats = bootstrap_sample(data, np.mean, n_bootstrap=100)

    assert isinstance(stats, np.ndarray)
    assert stats.shape == (100,)
    assert np.all(np.isfinite(stats))


def test_bootstrap_sample_one_replicate():
    data = np.array([1.0, 2.0, 3.0])

    stats = bootstrap_sample(data, np.mean, n_bootstrap=1)

    assert stats.shape == (1,)


def test_bootstrap_sample_empty_data():
    with pytest.raises(ValueError):
        bootstrap_sample(np.array([]), np.mean)


@pytest.mark.parametrize("n_bootstrap", [0, -1, -10])
def test_bootstrap_sample_invalid_n_bootstrap(n_bootstrap):
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        bootstrap_sample(data, np.mean, n_bootstrap=n_bootstrap)


def test_bootstrap_sample_wrong_shape():
    data = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        bootstrap_sample(data, np.mean)


def test_bootstrap_sample_noncallable_stat():
    data = np.array([1.0, 2.0, 3.0])

    with pytest.raises(TypeError):
        bootstrap_sample(data, "mean")

# Bonus: statistical validation against the null distribution of R-squared.
def test_bootstrap_r_squared_null_distribution():
    """
    Compare a null-centered pairs bootstrap with the theoretical
    Beta(1/2, (n-2)/2) distribution.

    This is a large-sample approximation, not an exact finite-sample
    distributional identity for the pairs bootstrap.
    """
    from scipy.stats import beta

    rng = np.random.default_rng(607)
    n = 5000
    n_bootstrap = 4000

    # Generate independent normal variables.
    x = rng.normal(size=n)
    y = rng.normal(size=n)

    # Remove the observed linear association to impose the null.
    x = x - x.mean()
    y = y - y.mean()
    y = y - x * np.dot(x, y) / np.dot(x, x)
    data = np.column_stack((x, y))

    # Make this test reproducible without changing other tests' RNG state.
    random_state = np.random.get_state()
    try:
        np.random.seed(607)
        stats = bootstrap_sample(
            data, r_squared, n_bootstrap=n_bootstrap
        )
    finally:
        np.random.set_state(random_state)

    assert stats.shape == (n_bootstrap,)
    assert np.all(np.isfinite(stats))
    assert np.all((stats >= 0) & (stats <= 1))

    # Compare the empirical CDF with the theoretical null CDF.
    ordered = np.sort(stats)
    theoretical_cdf = beta.cdf(ordered, 0.5, (n - 2) / 2)

    empirical_upper = np.arange(1, n_bootstrap + 1) / n_bootstrap
    empirical_lower = np.arange(n_bootstrap) / n_bootstrap

    distance = max(
        np.max(empirical_upper - theoretical_cdf),
        np.max(theoretical_cdf - empirical_lower),
    )

    # Allow Monte Carlo error and finite-sample bootstrap approximation.
    assert distance < 0.05, f"CDF distance is too large: {distance:.4f}"
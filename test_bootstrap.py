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


# Student B: tests for bootstrap_sample.
def test_bootstrap_sample_happy_path():
    """Return one finite statistic for each bootstrap replicate."""
    np.random.seed(607)
    stats = bootstrap_sample([1, 2, 3, 4], np.mean, n_bootstrap=50)

    assert isinstance(stats, np.ndarray)
    assert stats.shape == (50,)
    assert np.all(np.isfinite(stats))
    assert np.all((stats >= 1) & (stats <= 4))


def test_bootstrap_sample_keeps_rows_together():
    """Keep each x-y pair together when resampling two-dimensional data."""
    data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

    def check_pairs(sample):
        assert np.all(sample[:, 1] == 10 * sample[:, 0])
        return np.mean(sample[:, 0])

    np.random.seed(607)
    stats = bootstrap_sample(data, check_pairs, n_bootstrap=20)

    assert stats.shape == (20,)


def test_bootstrap_sample_rejects_empty_data():
    with pytest.raises(ValueError):
        bootstrap_sample([], np.mean)


@pytest.mark.parametrize("n_bootstrap", [0, -1])
def test_bootstrap_sample_rejects_invalid_replicates(n_bootstrap):
    with pytest.raises(ValueError):
        bootstrap_sample([1, 2, 3], np.mean, n_bootstrap=n_bootstrap)


@pytest.mark.parametrize("n_bootstrap", [1.5, "10", True])
def test_bootstrap_sample_rejects_noninteger_replicates(n_bootstrap):
    with pytest.raises(TypeError):
        bootstrap_sample([1, 2, 3], np.mean, n_bootstrap=n_bootstrap)


def test_bootstrap_sample_rejects_noncallable_statistic():
    with pytest.raises(TypeError):
        bootstrap_sample([1, 2, 3], "mean")


@pytest.mark.parametrize("bad_data", [5, np.zeros((2, 2, 2))])
def test_bootstrap_sample_rejects_wrong_shape(bad_data):
    with pytest.raises(ValueError):
        bootstrap_sample(bad_data, np.mean)

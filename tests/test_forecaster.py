import numpy as np
import pytest
import warnings
from src.forecaster import compute_metrics


@pytest.mark.parametrize("actual, predicted, expected", [
    # 1. Perfect forecast: MAE = RMSE = MAPE = 0
    (
            np.array([10.0, 20.0, 30.0]),
            np.array([10.0, 20.0, 30.0]),
            {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}
    ),
    # 2. Constant offset: Offset of 5.0
    (
            np.array([10.0, 20.0]),
            np.array([15.0, 25.0]),
            {"MAE": 5.0, "RMSE": 5.0, "MAPE": 37.5}
    ),
    # 3. Single-element arrays
    (
            np.array([100.0]),
            np.array([110.0]),
            {"MAE": 10.0, "RMSE": 10.0, "MAPE": 10.0}
    ),
    # 4. Negative values: MAE=2.0, RMSE=2.0, MAPE=(2/10 + 2/20)/2 * 100 = 15.0
    (
            np.array([-10.0, -20.0]),
            np.array([-12.0, -18.0]),
            {"MAE": 2.0, "RMSE": 2.0, "MAPE": 15.0}
    ),
])
def test_compute_metrics_parametrized(actual, predicted, expected):
    """Covers Perfect Forecast, Constant Offset, Single-Element, and Negative Values."""
    metrics = compute_metrics(actual, predicted)
    assert metrics["MAE"] == expected["MAE"]
    assert metrics["RMSE"] == expected["RMSE"]
    assert metrics["MAPE"] == expected["MAPE"]


def test_all_zeros_actual():
    """
    5. All-zeros actual: Verify MAPE handles division by zero (returns NaN)
    without crashing or raising ZeroDivisionError.
    """
    actual = np.zeros(3)
    predicted = np.array([1.0, 2.0, 3.0])

    # Suppress the RuntimeWarning from np.mean([]) for a clean pytest output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics = compute_metrics(actual, predicted)

    assert np.isnan(metrics["MAPE"])
    assert metrics["MAE"] == 2.0
    assert metrics["RMSE"] == 2.1602


def test_multidimensional_flattening():
    """Verify the function correctly flattens arrays to compare them."""
    actual = np.array([[10, 20], [30, 40]])
    predicted = np.array([[11, 21], [31, 41]])

    metrics = compute_metrics(actual, predicted)
    assert metrics["MAE"] == 1.0
    assert metrics["RMSE"] == 1.0


def test_mismatched_shapes():
    """Ensure NumPy raises an error if arrays cannot be broadcast/subtracted."""
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 2])
    with pytest.raises(ValueError):
        compute_metrics(actual, predicted)


def test_partial_zeros_mape():
    """
    Ensure MAPE only calculates based on non-zero actuals
    """
    actual = np.array([0.0, 10.0])
    predicted = np.array([5.0, 12.0])
    metrics = compute_metrics(actual, predicted)

    assert metrics["MAPE"] == 20.0
    assert metrics["MAE"] == 3.5


def test_contains_nan():
    """
    Verify that NaNs in the input propagate to all returned metrics.
    """
    actual = np.array([10.0, np.nan])
    predicted = np.array([11.0, 11.0])

    metrics = compute_metrics(actual, predicted)

    assert np.isnan(metrics["MAE"])
    assert np.isnan(metrics["RMSE"])
    assert np.isnan(metrics["MAPE"])


def test_empty_inputs():
    """Verify behavior on empty arrays (returns NaN for all)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics = compute_metrics(np.array([]), np.array([]))

    assert np.isnan(metrics["MAE"])
    assert np.isnan(metrics["RMSE"])
    assert np.isnan(metrics["MAPE"])
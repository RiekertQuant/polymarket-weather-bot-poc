"""Tests for src/ml/calibrator.py — IsotonicCalibrator and LogisticCalibrator."""

import numpy as np
import pytest

from src.ml.calibrator import IsotonicCalibrator, LogisticCalibrator


def _synthetic_data(n: int = 50):
    rng = np.random.RandomState(42)
    p_raw = np.linspace(0.0, 1.0, n)
    noise = rng.normal(0, 0.05, n)
    y_true = ((p_raw + noise) > 0.5).astype(int)
    return p_raw, y_true


# ── TestIsotonicCalibrator ───────────────────────────────────────────


class TestIsotonicCalibrator:
    def test_unfitted_float_passthrough(self):
        cal = IsotonicCalibrator()
        assert cal.predict(0.7) == pytest.approx(0.7)

    def test_unfitted_dict_passthrough(self):
        cal = IsotonicCalibrator()
        assert cal.predict({"p_raw": 0.7}) == pytest.approx(0.7)

    def test_fit_too_few_samples_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="at least 10"):
            cal.fit(np.array([0.1, 0.2]), np.array([0, 1]))

    def test_fit_and_predict_in_range(self):
        cal = IsotonicCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        pred = cal.predict(0.7)
        assert 0.0 <= pred <= 1.0

    def test_save_and_load_roundtrip(self, tmp_path):
        cal = IsotonicCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        path = tmp_path / "iso.pkl"
        cal.save(path)

        loaded = IsotonicCalibrator.load(path)
        assert cal.predict(0.6) == pytest.approx(loaded.predict(0.6))

    def test_predict_clips_to_01(self):
        cal = IsotonicCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        # Extreme inputs should still be clipped
        assert 0.0 <= cal.predict(0.0) <= 1.0
        assert 0.0 <= cal.predict(1.0) <= 1.0


# ── TestLogisticCalibrator ───────────────────────────────────────────


class TestLogisticCalibrator:
    def test_unfitted_passthrough(self):
        cal = LogisticCalibrator()
        assert cal.predict(0.7) == pytest.approx(0.7)

    def test_fit_too_few_samples_raises(self):
        cal = LogisticCalibrator()
        with pytest.raises(ValueError, match="at least 10"):
            cal.fit(np.array([0.1, 0.2]), np.array([0, 1]))

    def test_fit_and_predict_in_range(self):
        cal = LogisticCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        pred = cal.predict(0.7)
        assert 0.0 <= pred <= 1.0

    def test_save_and_load_roundtrip(self, tmp_path):
        cal = LogisticCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        path = tmp_path / "logistic.pkl"
        cal.save(path)

        loaded = LogisticCalibrator.load(path)
        assert cal.predict(0.6) == pytest.approx(loaded.predict(0.6))

    def test_monotonicity(self):
        cal = LogisticCalibrator()
        p_raw, y_true = _synthetic_data()
        cal.fit(p_raw, y_true)

        preds = [cal.predict(x) for x in np.linspace(0.05, 0.95, 20)]
        # Higher raw prob should give higher (or equal) calibrated prob
        for i in range(len(preds) - 1):
            assert preds[i] <= preds[i + 1] + 1e-9

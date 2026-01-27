"""Tests for src/ml/features.py — FeatureBuilder."""

from datetime import date

import numpy as np
import pytest

from src.ml.features import FeatureBuilder, PredictionFeatures


def _pred(city: str = "New York City", p_raw: float = 0.75) -> PredictionFeatures:
    return PredictionFeatures(
        p_raw=p_raw,
        forecast_temp=32.0,
        threshold=30.0,
        temp_diff=2.0,
        city=city,
        target_date=date(2026, 2, 1),
        days_ahead=3,
    )


# ── TestBuildFeatures ────────────────────────────────────────────────


class TestBuildFeatures:
    def test_array_length(self):
        fb = FeatureBuilder()
        arr = fb.build_features(_pred())
        assert len(arr) == 8

    def test_nyc_onehot(self):
        fb = FeatureBuilder()
        arr = fb.build_features(_pred(city="New York City"))
        # city one-hot at positions 5,6,7
        assert arr[5] == 1.0  # NYC
        assert arr[6] == 0.0  # London
        assert arr[7] == 0.0  # Seoul

    def test_london_onehot(self):
        fb = FeatureBuilder()
        arr = fb.build_features(_pred(city="London"))
        assert arr[5] == 0.0
        assert arr[6] == 1.0
        assert arr[7] == 0.0

    def test_seoul_onehot(self):
        fb = FeatureBuilder()
        arr = fb.build_features(_pred(city="Seoul"))
        assert arr[5] == 0.0
        assert arr[6] == 0.0
        assert arr[7] == 1.0

    def test_unknown_city_zeros(self):
        fb = FeatureBuilder()
        arr = fb.build_features(_pred(city="Tokyo"))
        assert arr[5] == 0.0
        assert arr[6] == 0.0
        assert arr[7] == 0.0

    def test_numeric_positions(self):
        fb = FeatureBuilder()
        pred = _pred()
        arr = fb.build_features(pred)

        assert arr[0] == pytest.approx(pred.p_raw)
        assert arr[1] == pytest.approx(pred.forecast_temp)
        assert arr[2] == pytest.approx(pred.threshold)
        assert arr[3] == pytest.approx(pred.temp_diff)
        assert arr[4] == pytest.approx(pred.days_ahead)


# ── TestBuildDataframe ───────────────────────────────────────────────


class TestBuildDataframe:
    def test_single_prediction(self):
        fb = FeatureBuilder()
        df = fb.build_dataframe([_pred()])
        assert len(df) == 1
        assert "p_raw" in df.columns
        assert "city" in df.columns

    def test_multiple_predictions(self):
        fb = FeatureBuilder()
        preds = [_pred("New York City"), _pred("London"), _pred("Seoul")]
        df = fb.build_dataframe(preds)
        assert len(df) == 3


# ── TestGetFeatureNames ──────────────────────────────────────────────


class TestGetFeatureNames:
    def test_returns_8_names(self):
        fb = FeatureBuilder()
        names = fb.get_feature_names()
        assert len(names) == 8
        assert names[0] == "p_raw"
        assert "city_New_York_City" in names

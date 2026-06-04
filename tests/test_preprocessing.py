from __future__ import annotations

import pandas as pd

from hd_serving.preprocessing import prepare_classification_training_df, prepare_regression_training_df


def test_classification_preprocessing_label_and_feature_drop():
    df = pd.DataFrame(
        {
            "Result": [1.0, 0.8, 1.0],
            "TRVmax[kV]": [10.0, 20.0, 30.0],
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    X, y, meta = prepare_classification_training_df(df)
    assert list(y) == [1, 0, 1]
    assert "Result" not in X.columns
    assert "TRVmax[kV]" not in X.columns
    assert list(X.columns) == ["A", "B"]
    assert meta["dropped_rows"] == 0


def test_regression_preprocessing_target_and_feature_drop():
    df = pd.DataFrame(
        {
            "Time": ["t1", "t2", "t3"],
            "Result": [0.8, 1.0, 0.6],
            "CZM": ["a", "b", "c"],
            "Test": [1, 2, 3],
            "TRVmax[kV]": [10.0, 20.0, 30.0],
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )
    X, y, meta = prepare_regression_training_df(df)
    assert list(y) == [10.0, 20.0, 30.0]
    for col in ["Time", "Result", "CZM", "Test", "TRVmax[kV]"]:
        assert col not in X.columns
    assert list(X.columns) == ["A", "B"]
    assert meta["dropped_rows"] == 0

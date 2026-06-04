from __future__ import annotations

import pandas as pd

from hd_serving.training import train_classification_model, train_regression_model


def test_training_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("HD_SERVING_TRAIN_ENGINE", "sklearn")
    class_df = pd.DataFrame(
        {
            "Result": [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
            "TRVmax[kV]": [10, 20, 12, 22, 13, 23, 14, 24],
            "A": [1, 2, 1.1, 2.2, 1.2, 2.4, 1.3, 2.6],
            "B": [5, 4, 5.2, 4.2, 5.4, 4.4, 5.6, 4.6],
        }
    )
    result = train_classification_model(class_df, model_root=tmp_path)
    assert result["ok"] is True
    assert (tmp_path / "classification" / "latest" / "model.joblib").exists()

    reg_df = pd.DataFrame(
        {
            "Time": [f"t{i}" for i in range(8)],
            "CZM": [f"c{i}" for i in range(8)],
            "Test": list(range(8)),
            "Result": [0.8] * 8,
            "TRVmax[kV]": [10, 11, 12, 13, 14, 15, 16, 17],
            "A": list(range(8)),
            "B": [v * 2 for v in range(8)],
        }
    )
    result = train_regression_model(reg_df, model_root=tmp_path)
    assert result["ok"] is True
    assert (tmp_path / "regression" / "latest" / "model.joblib").exists()

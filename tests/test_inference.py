from __future__ import annotations

import pandas as pd
import pytest

from hd_serving.inference import predict_batch
from hd_serving.training import train_classification_model


def test_inference_batch_and_schema_mismatch(tmp_path):
    df = pd.DataFrame(
        {
            "Result": [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
            "TRVmax[kV]": [10, 20, 12, 22, 13, 23, 14, 24],
            "A": [1, 2, 1.1, 2.2, 1.2, 2.4, 1.3, 2.6],
            "B": [5, 4, 5.2, 4.2, 5.4, 4.4, 5.6, 4.6],
        }
    )
    train_classification_model(df, model_root=tmp_path)
    result, summary = predict_batch(df, task="classification", model_root=tmp_path)
    assert summary["ok"] is True
    assert "prediction" in result.columns
    assert "probability_success" in result.columns
    with pytest.raises(ValueError):
        predict_batch(df.drop(columns=["A"]), task="classification", model_root=tmp_path)

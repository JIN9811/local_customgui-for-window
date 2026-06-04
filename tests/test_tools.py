from __future__ import annotations

import pandas as pd

from hd_serving.data_loader import make_dataset_record
from hd_serving.schema import infer_dataset_type
from hd_serving.tools import ToolContext, get_uploaded_data_summary, predict_tool, train_tool


def test_tools_do_not_return_full_dataframe_and_block_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HD_SERVING_TRAIN_ENGINE", "sklearn")
    df = pd.DataFrame(
        {
            "Result": [1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
            "TRVmax[kV]": [10, 20, 12, 22, 13, 23],
            "A": [1, 2, 1.1, 2.2, 1.2, 2.4],
            "B": [5, 4, 5.2, 4.2, 5.4, 4.4],
        }
    )
    record = make_dataset_record("class.xlsx", df, infer_dataset_type(list(df.columns)))
    ctx = ToolContext(datasets={record.dataset_id: record}, active_dataset_id=record.dataset_id, model_root=tmp_path)
    summary = get_uploaded_data_summary(ctx)
    assert summary["ok"] is True
    assert len(summary["summary"]["preview"]) <= 5
    train = train_tool(ctx, task="classification")
    assert train["ok"] is True

    bad_record = make_dataset_record("bad.xlsx", df.drop(columns=["A"]), "classification_prediction")
    bad_ctx = ToolContext(datasets={bad_record.dataset_id: bad_record}, active_dataset_id=bad_record.dataset_id, model_root=tmp_path)
    pred = predict_tool(bad_ctx, task="classification")
    assert pred["ok"] is False
    assert pred["failure_code"] == "PREDICTION_FAILED"

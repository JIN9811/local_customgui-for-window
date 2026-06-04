from __future__ import annotations

from hd_serving.schema import infer_dataset_type, validate_columns_against_schema


def test_infer_dataset_type():
    assert infer_dataset_type(["Result", "TRVmax[kV]", "A"]) == "classification_training"
    assert infer_dataset_type(["Time", "CZM", "Test", "Result", "TRVmax[kV]", "A"]) == "regression_training"
    assert infer_dataset_type(["A", "B"]) == "prediction_input_or_unknown"


def test_validate_columns_against_schema_missing_extra_order():
    schema = {"features": ["B", "A"], "ignored_columns": ["Result"]}
    result = validate_columns_against_schema(["A", "Result", "C"], schema)
    assert result["ok"] is False
    assert result["missing_columns"] == ["B"]
    assert result["extra_columns"] == ["C"]
    assert result["ignored_present"] == ["Result"]
    assert result["ordered_feature_columns"] == ["A"]

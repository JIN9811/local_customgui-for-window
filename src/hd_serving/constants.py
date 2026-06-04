"""Shared constants for HD serving."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_ROOT = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

CLASSIFICATION_TASK = "classification"
REGRESSION_TASK = "regression"

CLASSIFICATION_TARGET = "label"
CLASSIFICATION_SOURCE_TARGET = "Result"
REGRESSION_TARGET = "TRVmax[kV]"
MOTOR_NOISE_TARGET = "sound_mean"
MOTOR_NOISE_DIRECTION_COLUMNS = ["sound_front", "sound_rear", "sound_right", "sound_left"]

CLASSIFICATION_IGNORED_COLUMNS = ["Result", "TRVmax[kV]"]
REGRESSION_IGNORED_COLUMNS = ["Time", "Result", "CZM", "Test", "TRVmax[kV]"]
MOTOR_NOISE_IGNORED_COLUMNS = [*MOTOR_NOISE_DIRECTION_COLUMNS, MOTOR_NOISE_TARGET]

DATASET_TYPES = (
    "classification_training",
    "regression_training",
    "motor_noise_regression_training",
    "classification_prediction",
    "regression_prediction",
    "unknown",
)

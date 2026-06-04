"""Excel loading and safe dataframe summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


@dataclass(slots=True)
class DatasetRecord:
    dataset_id: str
    filename: str
    dataframe: pd.DataFrame
    uploaded_at: str
    inferred_type: str
    manual_type: str | None = None

    @property
    def dataset_type(self) -> str:
        return self.manual_type or self.inferred_type


def read_excel_file(file_or_path: Any) -> pd.DataFrame:
    """Read an uploaded Excel file/path into a DataFrame."""
    return pd.read_excel(file_or_path)


def make_dataset_record(filename: str, dataframe: pd.DataFrame, inferred_type: str) -> DatasetRecord:
    safe_name = Path(filename).name or "uploaded.xlsx"
    return DatasetRecord(
        dataset_id=f"ds-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}",
        filename=safe_name,
        dataframe=dataframe,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        inferred_type=inferred_type,
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and ratios."""
    total = max(len(df), 1)
    summary = pd.DataFrame(
        {
            "column": list(df.columns),
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_ratio": [float(df[col].isna().sum() / total) for col in df.columns],
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def dataframe_summary(df: pd.DataFrame, *, max_preview_rows: int = 5) -> dict[str, Any]:
    """Return JSON-serializable dataframe summary without exposing all rows."""
    preview = df.head(max_preview_rows).replace({pd.NA: None}).where(pd.notna(df.head(max_preview_rows)), None)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_list": [str(col) for col in df.columns],
        "missing_by_column": {
            str(col): int(count)
            for col, count in df.isna().sum().sort_values(ascending=False).items()
            if int(count) > 0
        },
        "preview": preview.to_dict(orient="records"),
    }

"""CLI for classification training."""

from __future__ import annotations

import argparse

import pandas as pd

from .training import train_classification_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HD classification model.")
    parser.add_argument("--input", required=True, help="Training Excel file")
    parser.add_argument("--train-size", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    df = pd.read_excel(args.input)
    result = train_classification_model(
        df,
        source_filename=args.input,
        train_size=args.train_size,
        random_state=args.random_state,
    )
    print({key: result[key] for key in ("ok", "task", "model_id", "artifact_dir", "latest_dir")})


if __name__ == "__main__":
    main()

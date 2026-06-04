# Project Summary

This local serving program supports HD Hyundai Electric high-voltage circuit breaker numeric-data workflows.

The goal is to use geometry/test numeric variables to train:

- a classification model for interruption success/failure, and
- a regression model for `TRVmax[kV]`.

Classification converts `Result` into a binary label:

- `Result == 1.0` -> Class 1, interruption success.
- all other values -> Class 0, interruption failure.

Regression predicts `TRVmax[kV]`.

The local LLM is not a predictor. It routes user requests to deterministic tools for data summary, schema validation, training, prediction, metrics, and explanation, then explains those tool outputs in Korean.

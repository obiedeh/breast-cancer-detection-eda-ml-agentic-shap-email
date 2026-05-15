"""Schema validation for tabular medical AI demo datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TabularSchema:
    """Expected feature and target structure."""

    feature_names: list[str]
    target_name: str = "target"
    allowed_targets: tuple[int, ...] = (0, 1)


def validate_schema(
    features: pd.DataFrame,
    target: pd.Series,
    schema: TabularSchema,
) -> None:
    """Validate columns, missing values, numeric features, and binary targets."""

    missing = [name for name in schema.feature_names if name not in features.columns]
    extra = [name for name in features.columns if name not in schema.feature_names]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected feature columns: {extra}")
    if list(features.columns) != schema.feature_names:
        raise ValueError("Feature columns are present but not in the expected order")
    if features.isna().any().any():
        raise ValueError("Feature matrix contains missing values")
    non_numeric = [
        column for column in features.columns if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns: {non_numeric}")
    invalid_targets = sorted(set(target.dropna().astype(int)) - set(schema.allowed_targets))
    if invalid_targets:
        raise ValueError(f"Unexpected target values: {invalid_targets}")
    if len(features) != len(target):
        raise ValueError("Feature and target row counts differ")

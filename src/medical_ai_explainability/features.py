"""Feature preparation utilities."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def numeric_scaler(feature_names: list[str]) -> ColumnTransformer:
    """Build a numeric preprocessing step for linear margin-based models."""

    return ColumnTransformer(
        transformers=[("numeric", StandardScaler(), feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

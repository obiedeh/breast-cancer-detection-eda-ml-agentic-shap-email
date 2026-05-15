"""Dataset loading and splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetBundle:
    """Container for a tabular binary-classification dataset."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: list[str]
    target_names: list[str]


@dataclass(frozen=True)
class SplitBundle:
    """Container for a stratified train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_breast_cancer_dataset() -> DatasetBundle:
    """Load the sklearn breast cancer dataset as pandas objects."""

    raw = load_breast_cancer(as_frame=True)
    features = raw.frame.drop(columns=["target"])
    target = raw.frame["target"].astype(int)
    return DatasetBundle(
        features=features,
        target=target,
        feature_names=list(features.columns),
        target_names=[str(name) for name in raw.target_names],
    )


def split_dataset(
    dataset: DatasetBundle,
    *,
    test_size: float,
    random_state: int,
) -> SplitBundle:
    """Create a reproducible stratified split."""

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )
    return SplitBundle(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

"""Global and local explanation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureContribution:
    """Feature contribution for a global or local explanation."""

    feature: str
    value: float
    contribution: float
    method: str


def global_feature_importance(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    *,
    random_state: int,
    top_k: int,
) -> list[FeatureContribution]:
    """Calculate global feature importance using native importances or permutation fallback."""

    native = _native_feature_importance(estimator, feature_names)
    if native is not None:
        return [
            FeatureContribution(
                feature=name,
                value=float("nan"),
                contribution=float(score),
                method="native",
            )
            for name, score in native.head(top_k).items()
        ]

    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=8,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )
    importances = pd.Series(result.importances_mean, index=feature_names).sort_values(
        ascending=False
    )
    return [
        FeatureContribution(
            feature=name,
            value=float("nan"),
            contribution=float(score),
            method="permutation",
        )
        for name, score in importances.head(top_k).items()
    ]


def local_explanation(
    estimator: object,
    sample: pd.Series,
    background: pd.DataFrame,
    feature_names: list[str],
    *,
    top_k: int,
) -> list[FeatureContribution]:
    """Generate a local explanation for one sample, preferring SHAP when installed."""

    shap_values = _try_shap_values(estimator, sample, background)
    if shap_values is not None:
        contributions = pd.Series(shap_values, index=feature_names)
        method = "shap"
    else:
        importances = _native_feature_importance(estimator, feature_names)
        if importances is None:
            importances = pd.Series(1.0, index=feature_names)
        centered = sample.astype(float) - background.mean(axis=0)
        contributions = centered * importances.reindex(feature_names).fillna(0.0)
        method = "centered_native_importance"

    ranked = contributions.reindex(feature_names).abs().sort_values(ascending=False).head(top_k)
    return [
        FeatureContribution(
            feature=name,
            value=float(sample[name]),
            contribution=float(contributions[name]),
            method=method,
        )
        for name in ranked.index
    ]


def _native_feature_importance(estimator: object, feature_names: list[str]) -> pd.Series | None:
    model = _final_estimator(estimator)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
    else:
        return None
    return pd.Series(values, index=feature_names).abs().sort_values(ascending=False)


def _final_estimator(estimator: object) -> Any:
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def _try_shap_values(
    estimator: object,
    sample: pd.Series,
    background: pd.DataFrame,
) -> np.ndarray | None:
    try:
        import shap

        estimator_any: Any = estimator
        explainer = shap.Explainer(estimator_any.predict_proba, background)
        values = explainer(sample.to_frame().T)
        class_values = values.values
        if class_values.ndim == 3:
            return np.asarray(class_values[0, :, 1], dtype=float)
        return np.asarray(class_values[0], dtype=float)
    except Exception:
        return None

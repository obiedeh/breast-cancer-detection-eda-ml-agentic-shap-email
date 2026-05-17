import numpy as np
import pandas as pd
import pytest

from medical_ai_explainability.data import load_breast_cancer_dataset, split_dataset
from medical_ai_explainability.explainability import (
    FeatureContribution,
    global_feature_importance,
    local_explanation,
)
from medical_ai_explainability.models import select_champion, train_baselines


@pytest.fixture(scope="module")
def trained_champion():
    dataset = load_breast_cancer_dataset()
    split = split_dataset(dataset, test_size=0.2, random_state=42)
    results = train_baselines(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        feature_names=dataset.feature_names,
        random_state=42,
    )
    champion = select_champion(results)
    return champion, dataset, split


def test_global_feature_importance_returns_top_k(trained_champion):
    champion, dataset, split = trained_champion
    top_k = 5

    result = global_feature_importance(
        champion.estimator,
        split.X_test,
        split.y_test,
        dataset.feature_names,
        random_state=42,
        top_k=top_k,
    )

    assert len(result) == top_k
    assert all(isinstance(c, FeatureContribution) for c in result)
    assert all(c.feature in dataset.feature_names for c in result)
    assert all(c.method in ("native", "permutation") for c in result)
    assert all(c.contribution >= 0 for c in result)


def test_global_feature_importance_descending_order(trained_champion):
    champion, dataset, split = trained_champion

    result = global_feature_importance(
        champion.estimator,
        split.X_test,
        split.y_test,
        dataset.feature_names,
        random_state=42,
        top_k=10,
    )

    contributions = [c.contribution for c in result]
    assert contributions == sorted(contributions, reverse=True)


def test_local_explanation_returns_top_k(trained_champion):
    champion, dataset, split = trained_champion
    sample = split.X_test.iloc[0]
    top_k = 6

    result = local_explanation(
        champion.estimator,
        sample,
        split.X_train,
        dataset.feature_names,
        top_k=top_k,
    )

    assert len(result) == top_k
    assert all(isinstance(c, FeatureContribution) for c in result)
    assert all(c.feature in dataset.feature_names for c in result)
    assert all(isinstance(c.value, float) for c in result)


def test_local_explanation_sample_values_match_input(trained_champion):
    champion, dataset, split = trained_champion
    sample = split.X_test.iloc[0]

    result = local_explanation(
        champion.estimator,
        sample,
        split.X_train,
        dataset.feature_names,
        top_k=5,
    )

    for contrib in result:
        assert np.isclose(contrib.value, float(sample[contrib.feature]))


def test_select_champion_raises_on_empty():
    with pytest.raises(ValueError, match="No model results"):
        select_champion([])

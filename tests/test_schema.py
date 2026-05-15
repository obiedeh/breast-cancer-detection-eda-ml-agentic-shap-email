import pytest

from medical_ai_explainability.data import load_breast_cancer_dataset
from medical_ai_explainability.schema import TabularSchema, validate_schema


def test_schema_validation_accepts_dataset():
    dataset = load_breast_cancer_dataset()

    validate_schema(dataset.features, dataset.target, TabularSchema(dataset.feature_names))


def test_schema_validation_rejects_missing_column():
    dataset = load_breast_cancer_dataset()
    features = dataset.features.drop(columns=[dataset.feature_names[0]])

    with pytest.raises(ValueError, match="Missing feature columns"):
        validate_schema(features, dataset.target, TabularSchema(dataset.feature_names))

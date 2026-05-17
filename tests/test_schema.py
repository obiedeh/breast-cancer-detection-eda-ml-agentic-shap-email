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


def test_schema_validation_rejects_extra_column():
    dataset = load_breast_cancer_dataset()
    features = dataset.features.copy()
    features["unexpected_col"] = 0.0

    with pytest.raises(ValueError, match="Unexpected feature columns"):
        validate_schema(features, dataset.target, TabularSchema(dataset.feature_names))


def test_schema_validation_rejects_nan_values():
    dataset = load_breast_cancer_dataset()
    features = dataset.features.copy()
    features.iloc[0, 0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_schema(features, dataset.target, TabularSchema(dataset.feature_names))


def test_schema_validation_rejects_non_numeric_column():
    dataset = load_breast_cancer_dataset()
    features = dataset.features.copy()
    col = dataset.feature_names[0]
    features[col] = features[col].astype(str)

    with pytest.raises(ValueError, match="Non-numeric"):
        validate_schema(features, dataset.target, TabularSchema(dataset.feature_names))


def test_schema_validation_rejects_invalid_target_value():
    dataset = load_breast_cancer_dataset()
    target = dataset.target.copy()
    target.iloc[0] = 99

    with pytest.raises(ValueError, match="Unexpected target values"):
        validate_schema(dataset.features, target, TabularSchema(dataset.feature_names))


def test_schema_validation_rejects_mismatched_row_count():
    dataset = load_breast_cancer_dataset()

    with pytest.raises(ValueError, match="row counts differ"):
        validate_schema(
            dataset.features,
            dataset.target.iloc[:10],
            TabularSchema(dataset.feature_names),
        )

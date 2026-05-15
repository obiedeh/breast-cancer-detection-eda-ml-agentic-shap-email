from medical_ai_explainability.data import load_breast_cancer_dataset, split_dataset


def test_load_breast_cancer_dataset():
    dataset = load_breast_cancer_dataset()

    assert dataset.features.shape == (569, 30)
    assert len(dataset.target) == 569
    assert dataset.feature_names == list(dataset.features.columns)


def test_split_dataset_is_stratified_shape():
    dataset = load_breast_cancer_dataset()
    split = split_dataset(dataset, test_size=0.2, random_state=42)

    assert len(split.X_train) == len(split.y_train)
    assert len(split.X_test) == len(split.y_test)
    assert split.X_train.shape[1] == 30
    assert split.X_test.shape[0] == 114

from medical_ai_explainability.data import load_breast_cancer_dataset, split_dataset
from medical_ai_explainability.evaluation import calculate_metrics
from medical_ai_explainability.models import select_champion, train_baselines


def test_model_training_smoke_test():
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

    assert {result.name for result in results} == {
        "logistic_regression",
        "random_forest",
        "svm_rbf",
    }
    champion = select_champion(results)
    assert champion.metrics["roc_auc"] > 0.95


def test_metrics_calculation():
    metrics = calculate_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        y_score=[0.1, 0.4, 0.8, 0.9],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 1.0

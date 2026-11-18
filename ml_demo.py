from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASETS = {
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast cancer": load_breast_cancer,
}

MODELS = ("Logistic regression", "K-nearest neighbors", "Support vector machine", "Random forest")


def train_and_evaluate(
    dataset_name="Iris",
    model_name="Logistic regression",
    test_size=0.25,
    random_state=42,
    neighbors=5,
):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    dataset = DATASETS[dataset_name]()
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    classifiers = {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "K-nearest neighbors": KNeighborsClassifier(n_neighbors=neighbors),
        "Support vector machine": SVC(),
        "Random forest": RandomForestClassifier(n_estimators=150, random_state=random_state),
    }
    model = make_pipeline(StandardScaler(), classifiers[model_name])
    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)
    class_labels = list(range(len(dataset.target_names)))

    return {
        "model": model,
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "actual": labels_test,
        "predictions": predictions,
        "accuracy": accuracy_score(labels_test, predictions),
        "precision": precision_score(labels_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels_test, predictions, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(labels_test, predictions, labels=class_labels),
        "classification_report": classification_report(
            labels_test,
            predictions,
            labels=class_labels,
            target_names=dataset.target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

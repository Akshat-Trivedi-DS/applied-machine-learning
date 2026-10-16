"""Train and compare Logistic Regression, Decision Tree, and Random Forest.

Standard scikit-learn workflow:
  1. train_test_split (80/20, stratified so both splits keep the same
     ~22% default rate)
  2. a Pipeline per model: scale the numeric columns, one-hot encode the
     categorical ones, then the classifier -- this keeps preprocessing
     inside the model, fit fresh on each training fold instead of on the
     whole dataset up front
  3. GridSearchCV (cv=5) over a small, readable hyperparameter grid per
     model
  4. class_weight="balanced" on every model, since only ~22% of
     customers default -- without it, a model can get "good" accuracy by
     almost always predicting "no default"
  5. evaluate the winning model of each type on the held-out test set
     with accuracy, precision, recall, F1, and a confusion matrix

Run with: python -m src.train
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of opening a GUI window
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_prep import load_clean
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, add_engineered_features

RANDOM_STATE = 42
MODEL_DIR = "models"
FIGURES_DIR = "reports/figures"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_model_configs() -> dict:
    """Each entry: (pipeline, small hyperparameter grid for GridSearchCV)."""
    return {
        "logistic_regression": (
            Pipeline([
                ("preprocess", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "decision_tree": (
            Pipeline([
                ("preprocess", build_preprocessor()),
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            {"clf__max_depth": [4, 6, 8, 10], "clf__min_samples_leaf": [10, 20, 50]},
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", build_preprocessor()),
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None]},
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def plot_confusion_matrix(cm, model_name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["no default", "default"], yticklabels=["no default", "default"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix_{model_name}.png")
    plt.close()


def main():
    df = add_engineered_features(load_clean())
    X = df[FEATURE_COLUMNS]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}
    best_models = {}

    for name, (pipeline, param_grid) in get_model_configs().items():
        print(f"\n>>> Running GridSearchCV for {name} ...")
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_

        metrics = evaluate(search.best_estimator_, X_test, y_test)
        metrics["best_params"] = search.best_params_
        results[name] = metrics
        plot_confusion_matrix(metrics["confusion_matrix"], name)

        print(f"{name}: best_params={search.best_params_}")
        print(f"  accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
              f"recall={metrics['recall']:.4f}  f1={metrics['f1_score']:.4f}")
        print(f"  confusion matrix: {metrics['confusion_matrix']}")

    best_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\n>>> Best model by F1-score on the test set: {best_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_models[best_name], f"{MODEL_DIR}/best_model.joblib")

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2, default=str)

    print(f"Saved best model ({best_name}) and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

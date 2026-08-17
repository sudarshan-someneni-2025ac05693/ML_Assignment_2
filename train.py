import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_models():
    # Six models are included because the assignment text says "6 ML models"
    # while the numbered list names five. The five named models are all present,
    # and SVM is included as the sixth model.
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )),
        ]),
    }


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    # Keep a simple binary target column.
    df["target"] = data.target.astype(int)
    X = df[data.feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Required test-data file for the Streamlit app.
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    # Save the exact feature order used by the models.
    with open(BASE_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_column": "target",
                "feature_names": list(data.feature_names),
                "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
                "rows": int(df.shape[0]),
                "features": int(X.shape[1]),
            },
            f,
            indent=2,
        )

    models = build_models()
    results = []

    print(f"Dataset shape: {df.shape}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print("\nModel results:\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)

        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

        row = {"ML Model Name": name, **metrics}
        results.append(row)

        print(name)
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df.to_csv(BASE_DIR / "metrics.csv", index=False)

    winner = results_df.loc[results_df["F1"].idxmax(), "ML Model Name"]
    with open(BASE_DIR / "winner.txt", "w", encoding="utf-8") as f:
        f.write(str(winner))

    print("Files created successfully.")
    print("Run the Streamlit app with:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()

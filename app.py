from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEST_FILE = BASE_DIR / "test_data.csv"
INFO_FILE = BASE_DIR / "feature_info.json"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if model_path.exists():
            models[model_name] = joblib.load(model_path)

    return models


# ============================================================
# LOAD DATASET INFORMATION
# ============================================================

@st.cache_data
def load_dataset_info():

    if not INFO_FILE.exists():
        return None

    try:

        with open(
            INFO_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


# ============================================================
# LOAD PROVIDED TEST DATA
# ============================================================

@st.cache_data
def load_provided_test_data():

    if not TEST_FILE.exists():
        return None

    try:

        return pd.read_csv(TEST_FILE)

    except Exception:

        return None


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    if hasattr(
        model,
        "predict_proba",
    ):

        scores = model.predict_proba(
            X_test
        )[:, 1]

    elif hasattr(
        model,
        "decision_function",
    ):

        scores = model.decision_function(
            X_test
        )

    else:

        scores = predictions

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "AUC": roc_auc_score(
            y_test,
            scores,
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y_test,
            predictions,
        ),
    }

    return metrics, predictions


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    models,
    X_test,
    y_test,
):

    results = []

    predictions_dict = {}

    for model_name, model in models.items():

        metrics, predictions = calculate_metrics(
            model,
            X_test,
            y_test,
        )

        results.append(
            {
                "ML Model Name": model_name,
                "Accuracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "MCC": metrics["MCC"],
            }
        )

        predictions_dict[model_name] = predictions

    return (
        pd.DataFrame(results),
        predictions_dict,
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "📊 Machine Learning Classification Dashboard"
)

st.write(
    "Interactive evaluation of six classification models "
    "using the Breast Cancer Wisconsin (Diagnostic) dataset."
)

st.divider()


# ============================================================
# LOAD MODELS AND INFORMATION
# ============================================================

models = load_models()

dataset_info = load_dataset_info()


if not models:

    st.error(
        "No saved models were found."
    )

    st.write(
        "Please make sure the following folder exists:"
    )

    st.code(
        str(MODEL_DIR)
    )

    st.write(
        "Then run:"
    )

    st.code(
        "python train.py"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Evaluation Setup"
)

st.sidebar.write(
    "Choose your test data and classification model."
)

st.sidebar.divider()


# ------------------------------------------------------------
# TEST DATA SOURCE
# ------------------------------------------------------------

st.sidebar.subheader(
    "1. Test Data"
)

data_source = st.sidebar.radio(
    "Select data source",
    [
        "Use provided test_data.csv",
        "Upload another CSV",
    ],
)


uploaded_file = None


if data_source == "Upload another CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload Test CSV",
        type=["csv"],
    )

else:

    if TEST_FILE.exists():

        st.sidebar.success(
            "Provided test_data.csv found."
        )

    else:

        st.sidebar.error(
            "test_data.csv not found."
        )


# ------------------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------------------

st.sidebar.subheader(
    "2. Classification Model"
)

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys()),
)


st.sidebar.divider()


# ------------------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------------------

if dataset_info:

    st.sidebar.subheader(
        "Dataset Information"
    )

    st.sidebar.write(
        f"Dataset: "
        f"{dataset_info.get('dataset_name', 'N/A')}"
    )

    st.sidebar.write(
        f"Instances: "
        f"{dataset_info.get('rows', 'N/A')}"
    )

    st.sidebar.write(
        f"Features: "
        f"{dataset_info.get('features', 'N/A')}"
    )

    st.sidebar.write(
        "Task: Binary Classification"
    )


# ============================================================
# PREPARE EVALUATION
# ============================================================

st.header(
    "1. Prepare Evaluation"
)

step1, step2, step3 = st.columns(
    3
)


with step1:

    st.subheader(
        "Step 1"
    )

    st.write(
        "**Choose your test dataset**"
    )

    if data_source == "Use provided test_data.csv":

        st.info(
            "Using the provided test_data.csv"
        )

    else:

        if uploaded_file is not None:

            st.success(
                f"Uploaded: {uploaded_file.name}"
            )

        else:

            st.warning(
                "Please upload a CSV file."
            )


with step2:

    st.subheader(
        "Step 2"
    )

    st.write(
        "**Select model**"
    )

    st.info(
        selected_model
    )


with step3:

    st.subheader(
        "Step 3"
    )

    st.write(
        "**Process the test file**"
    )

    st.write(
        "Click the button below."
    )


st.divider()


# ============================================================
# PROCESS BUTTON
# ============================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    process_clicked = st.button(
        "▶ PROCESS TEST FILE",
        use_container_width=True,
    )


# ============================================================
# WAITING STATE
# ============================================================

if not process_clicked:

    st.info(
        "Ready to process. "
        "Choose your test data and model, "
        "then click PROCESS TEST FILE."
    )

    st.subheader(
        "Dashboard Summary"
    )

    summary1, summary2, summary3 = st.columns(
        3
    )

    with summary1:

        st.metric(
            "Available Models",
            len(models),
        )

    with summary2:

        st.metric(
            "Required Metrics",
            6,
        )

    with summary3:

        st.metric(
            "Test Data",
            (
                "Provided CSV"
                if data_source
                == "Use provided test_data.csv"
                else "Custom Upload"
            ),
        )

    st.stop()


# ============================================================
# LOAD TEST DATA
# ============================================================

if data_source == "Use provided test_data.csv":

    test_df = load_provided_test_data()

    if test_df is None:

        st.error(
            "Could not load test_data.csv."
        )

        st.write(
            "Make sure test_data.csv is in the same "
            "folder as app.py."
        )

        st.stop()

    source_name = "Provided test_data.csv"

else:

    if uploaded_file is None:

        st.error(
            "Please upload a CSV file before processing."
        )

        st.stop()

    try:

        test_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            f"Could not read the uploaded CSV: {exc}"
        )

        st.stop()

    source_name = uploaded_file.name


# ============================================================
# VALIDATE TEST DATA
# ============================================================

st.header(
    "2. Test Data"
)

st.write(
    f"Source: **{source_name}**"
)


if "target" not in test_df.columns:

    st.error(
        "The test CSV must contain a 'target' column."
    )

    st.stop()


if dataset_info:

    expected_features = dataset_info.get(
        "feature_names",
        [],
    )

else:

    expected_features = []


# ------------------------------------------------------------
# FEATURE VALIDATION
# ------------------------------------------------------------

if expected_features:

    missing_features = [
        column
        for column in expected_features
        if column not in test_df.columns
    ]

    extra_features = [
        column
        for column in test_df.columns
        if column not in expected_features
        and column != "target"
    ]

    if missing_features:

        st.error(
            "Missing required features:"
        )

        st.write(
            missing_features
        )

        st.stop()

    feature_columns = expected_features

    if extra_features:

        st.warning(
            "The following extra columns will be ignored:"
        )

        st.write(
            extra_features
        )

else:

    feature_columns = [
        column
        for column in test_df.columns
        if column != "target"
    ]


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(test_df) == 0:

    st.error(
        "The test file contains no rows."
    )

    st.stop()


X_test = test_df[
    feature_columns
]

y_test = test_df[
    "target"
]


# ------------------------------------------------------------
# TARGET VALIDATION
# ------------------------------------------------------------

unique_targets = set(
    pd.unique(y_test)
)

if not unique_targets.issubset(
    {0, 1}
):

    st.error(
        "The target column must contain only 0 and 1."
    )

    st.stop()


# ============================================================
# SHOW TEST DATA
# ============================================================

st.subheader(
    "Uploaded Test Data"
)

data_col1, data_col2, data_col3 = st.columns(
    3
)

with data_col1:

    st.metric(
        "Rows",
        len(test_df),
    )

with data_col2:

    st.metric(
        "Features",
        len(feature_columns),
    )

with data_col3:

    st.metric(
        "Target Classes",
        len(unique_targets),
    )


st.dataframe(
    test_df.head(10),
    use_container_width=True,
)


# ============================================================
# PROCESS MODELS
# ============================================================

with st.spinner(
    "Processing test data..."
):

    results_df, predictions_dict = evaluate_models(
        models,
        X_test,
        y_test,
    )


# ============================================================
# FIND WINNERS
# ============================================================

winner_f1 = results_df.loc[
    results_df["F1"].idxmax(),
    "ML Model Name",
]

winner_auc = results_df.loc[
    results_df["AUC"].idxmax(),
    "ML Model Name",
]


selected_metrics = results_df[
    results_df[
        "ML Model Name"
    ]
    == selected_model
].iloc[0]


selected_predictions = predictions_dict[
    selected_model
]


# ============================================================
# PROCESSING SUCCESS
# ============================================================

st.success(
    f"Processing completed successfully. "
    f"Selected model: {selected_model}. "
    f"Best F1 model: {winner_f1}."
)


# ============================================================
# SELECTED MODEL METRICS
# ============================================================

st.header(
    "3. Selected Model Performance"
)

metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(
    6
)


with metric1:

    st.metric(
        "Accuracy",
        f"{selected_metrics['Accuracy']:.4f}",
    )


with metric2:

    st.metric(
        "AUC",
        f"{selected_metrics['AUC']:.4f}",
    )


with metric3:

    st.metric(
        "Precision",
        f"{selected_metrics['Precision']:.4f}",
    )


with metric4:

    st.metric(
        "Recall",
        f"{selected_metrics['Recall']:.4f}",
    )


with metric5:

    st.metric(
        "F1 Score",
        f"{selected_metrics['F1']:.4f}",
    )


with metric6:

    st.metric(
        "MCC",
        f"{selected_metrics['MCC']:.4f}",
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header(
    "4. Model Comparison"
)

display_df = results_df.copy()

for metric in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]:

    display_df[
        metric
    ] = display_df[
        metric
    ].round(4)


st.dataframe(
    display_df,
    use_container_width=True,
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

st.header(
    "5. Performance Comparison"
)

chart_df = results_df.set_index(
    "ML Model Name"
)[
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
]


st.bar_chart(
    chart_df
)


# ============================================================
# OVERALL WINNERS
# ============================================================

st.subheader(
    "Overall Performance"
)

winner_col1, winner_col2 = st.columns(
    2
)


with winner_col1:

    st.success(
        f"Best F1 Model: {winner_f1}"
    )


with winner_col2:

    st.success(
        f"Best AUC Model: {winner_auc}"
    )


# ============================================================
# SELECTED MODEL ANALYSIS
# ============================================================

st.header(
    "6. Selected Model Analysis"
)


analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(
    [
        "Confusion Matrix",
        "Classification Report",
        "Predictions",
    ]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

with analysis_tab1:

    cm = confusion_matrix(
        y_test,
        selected_predictions,
    )

    cm_col1, cm_col2 = st.columns(
        2
    )

    with cm_col1:

        fig, ax = plt.subplots(
            figsize=(5, 4)
        )

        ax.imshow(
            cm,
            cmap="Blues",
        )

        ax.set_xlabel(
            "Predicted Class"
        )

        ax.set_ylabel(
            "Actual Class"
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model}"
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Class 0",
                "Class 1",
            ]
        )

        ax.set_yticklabels(
            [
                "Class 0",
                "Class 1",
            ]
        )

        for row in range(
            cm.shape[0]
        ):

            for col in range(
                cm.shape[1]
            ):

                ax.text(
                    col,
                    row,
                    str(
                        cm[
                            row,
                            col
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


    with cm_col2:

        tn, fp, fn, tp = cm.ravel()

        st.subheader(
            "Classification Outcome"
        )

        st.metric(
            "True Negatives",
            int(tn),
        )

        st.metric(
            "False Positives",
            int(fp),
        )

        st.metric(
            "False Negatives",
            int(fn),
        )

        st.metric(
            "True Positives",
            int(tp),
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with analysis_tab2:

    report = classification_report(
        y_test,
        selected_predictions,
        target_names=[
            "Class 0",
            "Class 1",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df = report_df.round(
        4
    )

    st.dataframe(
        report_df,
        use_container_width=True,
    )


# ============================================================
# PREDICTIONS
# ============================================================

with analysis_tab3:

    prediction_df = X_test.copy()

    prediction_df[
        "Actual"
    ] = y_test.values

    prediction_df[
        "Predicted"
    ] = selected_predictions

    prediction_df[
        "Correct"
    ] = (
        prediction_df[
            "Actual"
        ]
        == prediction_df[
            "Predicted"
        ]
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.header(
    "7. Export Results"
)


download_col1, download_col2 = st.columns(
    2
)


with download_col1:

    st.download_button(
        "Download Model Comparison",
        data=results_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


with download_col2:

    prediction_csv = (
        prediction_df.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        "Download Predictions",
        data=prediction_csv,
        file_name=(
            selected_model
            .lower()
            .replace(
                " ",
                "_",
            )
            + "_predictions.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Learning Assignment 2 | "
    "Classification Model Performance Dashboard"
)
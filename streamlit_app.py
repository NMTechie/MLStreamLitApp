"""
Streamlit app: Mobile Price Range Classification
Loads pre-trained models and evaluates them against user-uploaded test data.
"""

import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

MODELS_DIR = "models"
TARGET_COLUMN = "price_range"

# Maps dropdown labels to their saved model file names
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}

# Each model owns its own scaler file, paired with its own model file.
SCALER_FILES = {
    "Logistic Regression": "logistic_regression_scaler.pkl",
    "Decision Tree": "decision_tree_scaler.pkl",
    "K-Nearest Neighbors": "knn_scaler.pkl",
    "Naive Bayes": "naive_bayes_scaler.pkl",
    "Support Vector Machine": "svm_scaler.pkl",
}


def load_pickle_file(file_path):
    """Load a pickled object, raising a clear error if the file is missing."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)


def load_selected_model(model_name):
    """Load the chosen model along with its own scaler, fit during that model's training."""
    model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILES[model_name])
    model = load_pickle_file(model_path)
    scaler = load_pickle_file(scaler_path)
    return model, scaler


def validate_uploaded_data(dataframe):
    """Confirm the uploaded CSV has the label column and no missing values."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Uploaded file must include the '{TARGET_COLUMN}' column "
            "so evaluation metrics can be calculated."
        )
    # This step may need to be optimized for data pre-processing pipeline.
    if dataframe.isnull().values.any():
        raise ValueError(
            "Uploaded file contains missing values. Please clean the data first.")


def split_features_and_target(dataframe):
    """Separate predictor columns from the true label column."""
    feature_columns = dataframe.drop(columns=[TARGET_COLUMN])
    true_labels = dataframe[TARGET_COLUMN]
    return feature_columns, true_labels


def compute_evaluation_metrics(true_labels, predicted_labels, predicted_probabilities):
    """Calculate the six required evaluation metrics for this multi-class problem."""
    metrics = {
        "Accuracy": accuracy_score(true_labels, predicted_labels),
        "AUC Score": roc_auc_score(
            true_labels, predicted_probabilities, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "Recall": recall_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "F1 Score": f1_score(true_labels, predicted_labels, average="macro", zero_division=0),
        "MCC Score": matthews_corrcoef(true_labels, predicted_labels),
    }
    return metrics


def plot_confusion_matrix(true_labels, predicted_labels, model_name):
    """Create a confusion matrix plot for the given true and predicted labels."""
    matrix = confusion_matrix(true_labels, predicted_labels)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axis)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_title(f"Confusion Matrix — {model_name}")
    return figure


def build_classification_report_table(true_labels, predicted_labels):
    """Convert the classification report into a properly aligned table."""
    report_dict = classification_report(
        true_labels, predicted_labels, output_dict=True)
    report_table = pd.DataFrame(report_dict).transpose()
    return report_table.round(4)


def main():
    # Custom styling to override the default red tab highlight/underline color.
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            color: #1f77b4;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("📱 Mobile Price Range Classifier")
    st.write(
        "Compares 5 classification models — Logistic Regression, Decision Tree, "
        "KNN, Naive Bayes, and SVM — trained on the Mobile Price Classification "
        "dataset (20 features, 4 price-range classes). Upload labeled test data "
        "in the sidebar to evaluate any of the pre-trained models."
    )
    st.set_page_config(
        page_title="Mobile Price Range Classifier",
        page_icon="📱",
        layout="wide",
    )
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))
        uploaded_file = st.file_uploader(
            "Upload test dataset (CSV)", type=["csv"])
        st.caption("Upload only labeled test data — training happens offline.")

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin evaluation.")
        return

    try:
        test_data = pd.read_csv(uploaded_file)
        validate_uploaded_data(test_data)
        st.subheader("Uploaded Data Preview")
        st.dataframe(test_data.head())

        feature_columns, true_labels = split_features_and_target(test_data)

        model, scaler = load_selected_model(model_name)
        scaled_features = scaler.transform(feature_columns)

        predicted_labels = model.predict(scaled_features)
        predicted_probabilities = model.predict_proba(scaled_features)

        metrics = compute_evaluation_metrics(
            true_labels, predicted_labels, predicted_probabilities)

        tab_metrics, tab_matrix, tab_report = st.tabs(
            ["Evaluation Metrics", "Confusion Matrix", "Classification Report"])

        with tab_metrics:
            st.subheader(f"Evaluation Metrics — {model_name}")
            metric_columns = st.columns(3)
            for index, (metric_name, metric_value) in enumerate(metrics.items()):
                metric_columns[index % 3].metric(
                    metric_name, f"{metric_value:.4f}")

        with tab_matrix:
            st.subheader(f"Confusion Matrix — {model_name}")
            st.pyplot(plot_confusion_matrix(
                true_labels, predicted_labels, model_name))

        with tab_report:
            st.subheader("Classification Report")
            report_table = build_classification_report_table(
                true_labels, predicted_labels)
            st.dataframe(report_table, use_container_width=True)

    except FileNotFoundError as error:
        st.error(f"Model file missing: {error}")
    except ValueError as error:
        st.error(f"Data validation error: {error}")
    except Exception as error:
        st.error(f"Unexpected error while evaluating the model: {error}")


if __name__ == "__main__":
    main()

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Define color constants using ANSI codes for decorated terminal output
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"


TARGET_COLUMN = "price_range"

# Shared across every model so every algorithm trains on the exact same rows.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_training_data_set(training_file_path: str) -> pd.DataFrame:
    """Read the labeled dataset used to train and validate models."""
    if not os.path.exists(training_file_path):
        raise FileNotFoundError(
            f"Training data not found: {training_file_path}")
    return pd.read_csv(training_file_path)


def features_and_target_column_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictor columns from the true label column."""
    feature_columns = dataframe.drop(columns=[TARGET_COLUMN])
    true_labels = dataframe[TARGET_COLUMN]
    return feature_columns, true_labels


def prepare_training_data(training_file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return only the training portion, unscaled."""
    data = load_training_data_set(training_file_path)
    features, labels = features_and_target_column_split(data)

    # chosen stratified split to ensure class distribution is preserved in train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    return train_features, train_labels


def save_pickle_file(fitted_object: object, file_path: str):
    """Persist a fitted model or scaler to disk, creating the folder if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_handle:
        pickle.dump(fitted_object, file_handle)

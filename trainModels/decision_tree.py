import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from training_utils import save_pickle_file, RED, GREEN, BLUE, RESET


class DecisionTreeTrainer:
    """Trains a Decision Tree classifier and saves it with its own scaler."""

    def __init__(self, model_file_path: str, scaler_file_path: str, random_state: int = 42):
        self.model_file_path = model_file_path
        self.scaler_file_path = scaler_file_path
        self.random_state = random_state

    def train_decision_tree(self, train_features: pd.DataFrame, train_labels: pd.Series):
        """Fit a scaler and the model on the training data, then save both to disk."""
        try:
            scaler = StandardScaler()
            scaled_train_features = scaler.fit_transform(train_features)

            model = DecisionTreeClassifier(random_state=self.random_state)
            model.fit(scaled_train_features, train_labels)

            training_accuracy = model.score(
                scaled_train_features, train_labels)
            print(
                f"{BLUE}Decision Tree training accuracy: {training_accuracy:.4f}{RESET}")

            save_pickle_file(model, self.model_file_path)
            save_pickle_file(scaler, self.scaler_file_path)
            print(f"Saved model to {self.model_file_path}")
            print(f"Saved scaler to {self.scaler_file_path}")

        except Exception as error:
            print(f"Unexpected error while training Decision Tree: {error}")

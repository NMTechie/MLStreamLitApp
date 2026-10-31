import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from training_utils import save_pickle_file, BLUE, RESET


class KnnTrainer:
    """Trains a K-Nearest Neighbors classifier and saves it with its own scaler."""

    def __init__(self, model_file_path: str, scaler_file_path: str, n_neighbors: int = 5):
        self.model_file_path = model_file_path
        self.scaler_file_path = scaler_file_path
        self.n_neighbors = n_neighbors

    def train_knn(self, train_features: pd.DataFrame, train_labels: pd.Series):
        """Fit a scaler and the model on the training data, then save both to disk."""
        try:
            scaler = StandardScaler()
            scaled_train_features = scaler.fit_transform(train_features)

            model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            model.fit(scaled_train_features, train_labels)

            training_accuracy = model.score(
                scaled_train_features, train_labels)
            print(
                f"{BLUE}K-Nearest Neighbors training accuracy: {training_accuracy:.4f}{RESET}")

            save_pickle_file(model, self.model_file_path)
            save_pickle_file(scaler, self.scaler_file_path)
            print(f"Saved model to {self.model_file_path}")
            print(f"Saved scaler to {self.scaler_file_path}")

        except Exception as error:
            print(
                f"Unexpected error while training K-Nearest Neighbors: {error}")

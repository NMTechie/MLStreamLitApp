import os

from logistics_regression import LogisticRegressionTrainer
from decision_tree import DecisionTreeTrainer
from knn import KnnTrainer
from training_utils import prepare_training_data, RED, GREEN, BLUE, RESET


# Shared across every model so the train/test split and scaler stay consistent.
# This RANDOM_STATE seed is used to ensure same set of splitted train/test data for every run.
# This ensures every algo being trained uses the same train/test split.
# This provides consistency in evaluation across different models.
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITERATION = 1000

# Setting up file and directory paths so relative references work as expected
TRAIN_MODEL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.dirname(TRAIN_MODEL_SCRIPT_DIR)
TRAIN_DATA_PATH = os.path.join(REPO_ROOT_DIR, "mobile_price_class_train.csv")
SAVE_MODELS_TO_DIR = os.path.join(REPO_ROOT_DIR, "models")

# Each model owns its own scaler file, paired with its own model file.
MODEL_FILES = {
    1: "logistic_regression.pkl",
    2: "decision_tree.pkl",
    3: "knn.pkl",
    4: "naive_bayes.pkl",
    5: "svm.pkl",
}
SCALER_FILES = {
    1: "logistic_regression_scaler.pkl",
    2: "decision_tree_scaler.pkl",
    3: "knn_scaler.pkl",
    4: "naive_bayes_scaler.pkl",
    5: "svm_scaler.pkl",
}


def main():
    print(f"{BLUE}Choose which Algorithm you want to run to train the model...{RESET}")
    try:
        choice = int(input(
            f"{GREEN}Enter your choice \n1 --> Logistic Regression \n" +
            "2 --> Decision Tree\n" +
            "3 --> K-Nearest Neighbors\n" +
            "4 --> Naive Bayes\n" +
            "5 --> Support Vector Machine:\n" + RESET))
        if choice not in [1, 2, 3, 4, 5]:
            raise ValueError(
                "Invalid choice for question. Please enter a number between 1 and 5.")

        train_features, train_labels = prepare_training_data(TRAIN_DATA_PATH)
        model_file_path = os.path.join(SAVE_MODELS_TO_DIR, MODEL_FILES[choice])
        scaler_file_path = os.path.join(
            SAVE_MODELS_TO_DIR, SCALER_FILES[choice])

        match choice:
            case 1:
                print(f"{GREEN}Training Logistic Regression model...{RESET}")
                trainer = LogisticRegressionTrainer(
                    model_file_path=model_file_path, scaler_file_path=scaler_file_path, random_state=RANDOM_STATE, max_iter=MAX_ITERATION)
                trainer.train_logistic_regression(train_features, train_labels)
            case 2:
                print(f"{GREEN}Training Decision Tree model...{RESET}")
                trainer = DecisionTreeTrainer(
                    model_file_path=model_file_path, scaler_file_path=scaler_file_path, random_state=RANDOM_STATE)
                trainer.train_decision_tree(train_features, train_labels)
            case 3:
                print(f"{GREEN}Training K-Nearest Neighbors model...{RESET}")
                valueof_kneighbour = int(input(
                    f"{GREEN}Enter the number of neighbors for K-Nearest Neighbors Algorithm: {RESET}"))
                if valueof_kneighbour <= 0:
                    raise ValueError(
                        "Number of neighbors must be a positive integer.")
                trainer = KnnTrainer(
                    model_file_path=model_file_path, scaler_file_path=scaler_file_path, n_neighbors=valueof_kneighbour)
                trainer.train_knn(train_features, train_labels)
            case 4:
                print(f"{GREEN}Training Naive Bayes model...{RESET}")
            case 5:
                print(f"{GREEN}Training Support Vector Machine model...{RESET}")

    except ValueError as error:
        print(f"{RED}{error}{RESET}")
        return
    except FileNotFoundError as error:
        print(f"{RED}{error}{RESET}")
        return
    except Exception as e:
        print(f"{RED}An unexpected error occurred: {e}{RESET}")
        return
    else:
        print(f"{GREEN} \n Algo training executed successfully.{RESET}")
    finally:
        print(f"{BLUE}The execution of the program has ended.{RESET}")


if __name__ == "__main__":
    main()

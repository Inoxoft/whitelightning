from binary_classifier.model import run_training
from binary_classifier.model_executor import run_model
from utils.benchmark import test_model_accuracy

if __name__ == "__main__":
    run_training()
    run_model()
    test_model_accuracy()
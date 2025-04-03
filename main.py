from binary_classifier.model_executor import run_predictions
from binary_classifier.classifier import run_training
from utils.benchmark import test_model_accuracy

if __name__ == "__main__":
    model_type = ("tensorflow")
    run_training(model_type=model_type)
    run_predictions(model_type=model_type)
    test_model_accuracy()
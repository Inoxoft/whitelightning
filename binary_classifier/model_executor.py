import pandas as pd
import logging
from .classifier import BinaryTextClassifier
from .strategies import TensorFlowStrategy
from .strategies import PyTorchStrategy
from .strategies import ScikitLearnStrategy
from settings import (
    MODEL_PREFIX,
    MODELS_PATH,
    TESTING_DATA_PATH,
    RESULTS_PATH,
    DATA_COLUMN_NAME,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_predictions(model_type: str):
    strategies = {
        "tensorflow": TensorFlowStrategy(input_dim=5000),
        "pytorch": PyTorchStrategy(input_dim=5000),
        "scikit": ScikitLearnStrategy(),
    }
    strategy = strategies[model_type]

    classifier = BinaryTextClassifier(strategy)
    classifier.load(f"{MODELS_PATH}/{MODEL_PREFIX}")

    df = pd.read_csv(f"{TESTING_DATA_PATH}{MODEL_PREFIX}_dataset.csv", encoding="utf-8")
    df["prediction"] = df[DATA_COLUMN_NAME].apply(lambda x: classifier.predict([x])[0])

    output_path = f"{RESULTS_PATH}{MODEL_PREFIX}_predictions.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

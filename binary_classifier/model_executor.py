import pandas as pd
import torch
import pickle
from binary_classifier.model import BinaryClassifier
from settings import MODEL_PREFIX, MODELS_PATH, TESTING_DATA_PATH, RESULTS_PATH, DATA_COLUMN_NAME, LABEL_COLUMN_NAME, POSITIVE_LABEL, NEGATIVE_LABEL

def run_model():
    classifier = BinaryClassifier()

    classifier.load_model(f'{MODELS_PATH}/{MODEL_PREFIX}')
    df = pd.read_csv(f"{TESTING_DATA_PATH}{MODEL_PREFIX}_dataset.csv", encoding="utf-8")
    df["prediction"] = df.apply(lambda x: classifier.predict(x[DATA_COLUMN_NAME])[0], axis=1)
    df.to_csv(f"{RESULTS_PATH}{MODEL_PREFIX}_predictions.csv", index=False)


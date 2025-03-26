import logging
from typing import Tuple, Union, List, Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import joblib

from binary_classifier.strategies import TensorFlowStrategy, PyTorchStrategy, ScikitLearnStrategy, \
    TextClassifierStrategy
from settings import (
    MODEL_PREFIX,
    DATA_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    TRAINING_DATA_PATH,
    MODELS_PATH,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BinaryTextClassifier:
    def __init__(self, strategy: TextClassifierStrategy, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 4)):
        self.strategy = strategy
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.scaler = StandardScaler()
        self._is_trained = False

    def load_and_preprocess(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if DATA_COLUMN_NAME not in df.columns or LABEL_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{DATA_COLUMN_NAME}' and '{LABEL_COLUMN_NAME}' columns")

        X = df[DATA_COLUMN_NAME]
        y = df[LABEL_COLUMN_NAME]
        if y.dtype == 'object':
            y = y.replace({'negative': 0, 'positive': 1}).astype(np.float32)

        X_processed = self.extract_features(X, training=True)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def extract_features(self, X: Union[pd.Series, List[str]], training: bool = True) -> np.ndarray:
        if isinstance(X, list):
            X = pd.Series(X)
        if training:
            X_tfidf = self.vectorizer.fit_transform(X).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_scaled = self.scaler.transform(X_tfidf)
        return X_scaled.astype(np.float32)

    def train(self, csv_path: str) -> Dict:
        X_train, X_test, y_train, y_test = self.load_and_preprocess(csv_path)
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        metrics = self.strategy.train(X_train, X_test, y_train, y_test)
        self._is_trained = True
        return metrics

    def predict(self, new_data: Union[str, List[str]]) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        X_processed = self.extract_features(new_data, training=False)
        return self.strategy.predict(X_processed)

    def save(self, filename_prefix: str):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        self.strategy.save_model(filename_prefix)
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")

    def load(self, filename_prefix: str):
        self.strategy.load_model(filename_prefix)
        self.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        self._is_trained = True

def run_training(*, model_type: str, **kwargs):
    if model_type == "tensorflow":
        strategy = TensorFlowStrategy(input_dim=5000)
    elif model_type == "pytorch":
        strategy = PyTorchStrategy(input_dim=5000)
    elif model_type == "scikit":
        strategy = ScikitLearnStrategy()

    # Train and save
    classifier = BinaryTextClassifier(strategy)
    metrics = classifier.train(f'{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv')
    classifier.save(f"{MODELS_PATH}/{MODEL_PREFIX}")

    # Log metrics
    logging.info(f"Metrics: {metrics}")
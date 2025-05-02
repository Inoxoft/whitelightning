import logging
from typing import Tuple, Union, List, Dict

import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import joblib

from binary_classifier.strategies import TextClassifierStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BinaryTextClassifier:
    def __init__(
        self,
        strategy: TextClassifierStrategy,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 3),
    ):
        self.strategy = strategy
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self.scaler = StandardScaler()
        self._is_trained = False

    def load_and_preprocess(
        self, csv_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV must contain 'text' and 'label' columns")

        X = df["text"]
        y = df["label"]
        if y.dtype == "object":
            y = y.replace({"negative": 0, "positive": 1}).astype(np.float32)

        X_processed = self.extract_features(X, training=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def extract_features(
        self, X: Union[pd.Series, List[str]], training: bool = True
    ) -> np.ndarray:
        if isinstance(X, list):
            X = pd.Series(X)
        if training:
            X_tfidf = self.vectorizer.fit_transform(X).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_scaled = self.scaler.transform(X_tfidf)
        return X_scaled.astype(np.float32)

    def train(self, csv_path: Path) -> Dict:
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

        import json

        with open(f"{filename_prefix}_vocab.json", "w") as f:
            vocab = {
                str(word): int(idx) for word, idx in self.vectorizer.vocabulary_.items()
            }
            idf = [float(x) for x in self.vectorizer.idf_]
            json.dump({"vocab": vocab, "idf": idf}, f)

        with open(f"{filename_prefix}_scaler.json", "w") as f:
            scaler_params = {
                "mean": [float(x) for x in self.scaler.mean_],
                "scale": [float(x) for x in self.scaler.scale_],
            }
            json.dump(scaler_params, f)

        logging.info(
            f"Preprocessing parameters saved to {filename_prefix}_vocab.json and {filename_prefix}_scaler.json"
        )

    def load(self, filename_prefix: str):
        self.strategy.load_model(filename_prefix)
        self.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        self._is_trained = True

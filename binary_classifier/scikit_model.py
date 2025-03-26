import logging
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

from settings import (
    MODEL_PREFIX,
    DATA_COLUMN_NAME, LABEL_COLUMN_NAME, TRAINING_DATA_PATH, MODELS_PATH
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class BinaryClassifier:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 4)):
        logging.info("Initializing BinaryClassifier")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_iter_no_change=10,
            verbose=1
        )
        self._is_trained = False

    def load_and_preprocess(self, csv_path: str) -> Tuple[pd.Series, pd.Series]:
        """Load and preprocess the CSV data"""
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if DATA_COLUMN_NAME not in df.columns or LABEL_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{DATA_COLUMN_NAME}' and '{LABEL_COLUMN_NAME}' columns")

        return df[DATA_COLUMN_NAME], df[LABEL_COLUMN_NAME]

    def extract_features(self, X: pd.Series, training: bool = True) -> np.ndarray:
        """Convert text data to numerical features"""
        if training:
            X_tfidf = self.vectorizer.fit_transform(X).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_scaled = self.scaler.transform(X_tfidf)

        return X_scaled

    def train(self, csv_path: str, test_size: float = 0.2) -> dict:
        """Train the model and return metrics"""
        logging.info("Starting model training")
        X, y = self.load_and_preprocess(csv_path)
        X_processed = self.extract_features(X, training=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )

        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")

        self.model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        logging.info(f"Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_processed, y, cv=5)
        logging.info(f"Cross-validation scores: {cv_scores}")
        logging.info(f"Average CV score: {cv_scores.mean():.4f}")

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'classification_report': classification_report(y_test, test_pred)
        }

        return metrics

    def predict(self, new_data: Union[str, List[str]]) -> np.ndarray:
        """Make predictions on new data"""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        if isinstance(new_data, str):
            new_data = [new_data]

        X_processed = self.extract_features(new_data, training=False)
        predictions = self.model.predict(X_processed)

        return predictions

    def save_model(self, filename_prefix: str):
        """Save the trained model and preprocessors"""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        logging.info(f"Saving model to {filename_prefix}")
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")
        joblib.dump(self.model, f"{filename_prefix}_model.pkl")

    def load_model(self, filename_prefix: str):
        """Load a trained model and preprocessors"""
        logging.info(f"Loading model from {filename_prefix}")
        self.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        self.model = joblib.load(f"{filename_prefix}_model.pkl")
        self._is_trained = True

    def export_to_onnx(self, output_path: str, input_features: int):
        """
        Export the trained model to ONNX format

        Args:
            output_path: Path where to save the ONNX model
            input_features: Number of input features (same as vectorizer.max_features)
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")

        logging.info(f"Converting model to ONNX format")

        # Define input type
        initial_type = [('float_input', FloatTensorType([None, input_features]))]

        # Convert the model to ONNX
        onnx_model = convert_sklearn(
            self.model,
            initial_types=initial_type,
            target_opset=13
        )

        # Save the ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logging.info(f"ONNX model saved to {output_path}")

        # Verify the model
        sess = rt.InferenceSession(output_path)
        input_name = sess.get_inputs()[0].name
        logging.info(f"Model verified. Input name: {input_name}")


def run_training():
    classifier = BinaryClassifier()
    classifier.train(f'{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv')

    classifier.save_model(f"{MODELS_PATH}/{MODEL_PREFIX}")
    onnx_path = f"{MODELS_PATH}/{MODEL_PREFIX}_model.onnx"
    classifier.export_to_onnx(
        output_path=onnx_path,
        input_features=classifier.vectorizer.max_features
    )
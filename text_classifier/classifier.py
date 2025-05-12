import logging
from typing import Tuple, Union, List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
from enum import Enum

try:
    import text_classifier.settings as settings
    from text_classifier.strategies import (  # Updated import
        TextClassifierStrategy,  # Import base for type hint
    )
except ModuleNotFoundError:  # Handle if running script directly from its dir
    import settings
    from text_classifier.strategies import TextClassifierStrategy  # Keep path for now

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelType(Enum):
    BINARY = "binary"  # Simple binary with probability output
    MULTICLASS_SIGMOID = "multiclass_sigmoid"  # Multiclass with sigmoid output
    MULTICLASS_SOFTMAX = "multiclass_softmax"  # Multiclass with softmax output

class TextClassifier:  # Renamed from BinaryTextClassifier
    def __init__(
        self,
        strategy: TextClassifierStrategy,
        class_labels: List[str],
        model_type: ModelType,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 3),
    ):
        if not class_labels or len(class_labels) < 2:
            raise ValueError("Must provide at least two class labels.")

        if model_type == ModelType.BINARY and len(class_labels) != 2:
            raise ValueError("Binary model type requires exactly two class labels.")

        self.strategy = strategy
        self.class_labels = sorted(list(set(class_labels)))
        self.num_classes = len(self.class_labels)
        self.model_type = model_type

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_labels)

        self._is_trained = False
        logging.info(
            f"TextClassifier initialized for {self.num_classes} classes: {self.class_labels} with model type: {self.model_type.value}"
        )

    def load_and_preprocess(
        self, csv_path: Path
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # Added X_val, y_val
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")

        X = df["text"].astype(str)  # Ensure text is string
        y_str = df["label"].astype(str)  # Labels from CSV are strings

        # Convert string labels to integer indices
        y_int = self.label_encoder.transform(y_str)

        logging.info(f"Integer mapped labels: {np.unique(y_int, return_counts=True)}")

        # Feature extraction
        # X_processed needs to be fit on combined train+val or only train for TF-IDF
        # For simplicity, let's fit on the whole X before splitting, then transform separately
        # This is common, though strictly, TF-IDF should be fit only on training data.
        # Let's adjust to fit only on train.

        # Split data first
        # Stratify by y_int to ensure class distribution is similar in splits
        X_train_text, X_temp_text, y_train_int, y_temp_int = train_test_split(
            X, y_int, test_size=0.3, random_state=42, stratify=y_int
        )
        X_val_text, X_test_text, y_val_int, y_test_int = train_test_split(
            X_temp_text, y_temp_int, test_size=0.5, random_state=42, stratify=y_temp_int
        )

        logging.info("Extracting features (TF-IDF and Scaling)...")
        X_train_processed = self.extract_features(X_train_text, training=True)
        X_val_processed = self.extract_features(X_val_text, training=False)
        X_test_processed = self.extract_features(X_test_text, training=False)

        # Convert y to float32 for Keras/TF if it's not already (it's int now)
        # The strategy will handle one-hot encoding if needed.
        y_train = y_train_int.astype(
            np.int32
        )  # Keep as int for PyTorch CrossEntropy, TF will one-hot
        y_val = y_val_int.astype(np.int32)
        y_test = y_test_int.astype(np.int32)

        logging.info(
            f"Shapes: X_train: {X_train_processed.shape}, y_train: {y_train.shape}"
        )
        logging.info(f"Shapes: X_val: {X_val_processed.shape}, y_val: {y_val.shape}")
        logging.info(
            f"Shapes: X_test: {X_test_processed.shape}, y_test: {y_test.shape}"
        )

        return (
            X_train_processed,
            X_val_processed,
            X_test_processed,
            y_train,
            y_val,
            y_test,
        )

    def extract_features(
        self, X_text: Union[pd.Series, List[str]], training: bool = True
    ) -> np.ndarray:
        if isinstance(X_text, list):
            X_text = pd.Series(X_text)  # TF-IDF expects iterables of strings

        # Ensure all elements are strings, replace NaN with empty string
        X_text_cleaned = X_text.fillna("").astype(str)

        if training:
            X_tfidf = self.vectorizer.fit_transform(X_text_cleaned).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            # Check if vectorizer is fitted
            if (
                not hasattr(self.vectorizer, "vocabulary_")
                or not self.vectorizer.vocabulary_
            ):
                raise RuntimeError(
                    "Vectorizer has not been fitted. Call with training=True first."
                )
            if not hasattr(self.scaler, "mean_") or self.scaler.mean_ is None:
                raise RuntimeError(
                    "Scaler has not been fitted. Call with training=True first."
                )
            X_tfidf = self.vectorizer.transform(X_text_cleaned).toarray()
            X_scaled = self.scaler.transform(X_tfidf)
        return X_scaled.astype(np.float32)

    def train(self, csv_path: Path) -> Dict:
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess(
            csv_path
        )

        # The strategy's train method needs to accept X_val, y_val
        logging.info(
            f"Training with {self.num_classes} classes. Strategy: {type(self.strategy).__name__}"
        )
        metrics = self.strategy.train(X_train, X_val, X_test, y_train, y_val, y_test)
        self._is_trained = True
        return metrics

    def predict_proba(self, new_data: Union[str, List[str]]) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        if isinstance(new_data, str):
            new_data = [new_data]

        X_processed = self.extract_features(new_data, training=False)
        # Strategy's predict method should return probabilities
        return self.strategy.predict(X_processed)

    def predict(self, new_data: Union[str, List[str]]) -> List[str]:
        probabilities = self.predict_proba(new_data)
        # Get class index with highest probability
        predicted_indices = np.argmax(probabilities, axis=1)
        # Convert indices back to string labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_indices)
        return predicted_labels.tolist()

    def save(self, filename_prefix: str):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        self.strategy.save_model(filename_prefix)
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{filename_prefix}_label_encoder.pkl")

        # Save metadata including class_labels, num_classes, and model_type
        metadata = {
            "class_labels": self.class_labels,
            "num_classes": self.num_classes,
            "model_type": self.model_type.value,
            "max_features": self.vectorizer.max_features,
            "ngram_range": self.vectorizer.ngram_range,
            "strategy_class": type(self.strategy).__name__,
        }
        with open(f"{filename_prefix}_{settings.CLASSIFIER_META_FILENAME}", "w") as f:
            json.dump(metadata, f, indent=2)

        logging.info(
            f"Model, preprocessors, and metadata saved with prefix: {filename_prefix}"
        )
        self._save_onnx_compat_params(filename_prefix)

    def _save_onnx_compat_params(self, filename_prefix: str):
        """Saves TF-IDF vocabulary and scaler params in JSON format for potential ONNX/JS usage."""
        try:
            with open(f"{filename_prefix}_vectorizer_vocab.json", "w") as f:
                vocab = {
                    str(word): int(idx)
                    for word, idx in self.vectorizer.vocabulary_.items()
                }
                idf = (
                    [float(x) for x in self.vectorizer.idf_]
                    if hasattr(self.vectorizer, "idf_")
                    else []
                )
                json.dump(
                    {
                        "vocabulary": vocab,
                        "idf": idf,
                        "ngram_range": self.vectorizer.ngram_range,
                    },
                    f,
                    indent=2,
                )

            with open(f"{filename_prefix}_scaler_params.json", "w") as f:
                scaler_params = {
                    "mean": (
                        [float(x) for x in self.scaler.mean_]
                        if hasattr(self.scaler, "mean_")
                        and self.scaler.mean_ is not None
                        else []
                    ),
                    "scale": (
                        [float(x) for x in self.scaler.scale_]
                        if hasattr(self.scaler, "scale_")
                        and self.scaler.scale_ is not None
                        else []
                    ),
                }
                json.dump(scaler_params, f, indent=2)
            logging.info(
                f"ONNX/JS compatible preprocessor params saved (vocab & scaler)."
            )
        except Exception as e:
            logging.warning(f"Could not save ONNX/JS compatible params: {e}")

    @classmethod
    def load(
        cls,
        filename_prefix: str,
        strategy_instance: Optional[TextClassifierStrategy] = None,
    ) -> "TextClassifier":
        logging.info(f"Loading model and preprocessors from prefix: {filename_prefix}")

        with open(f"{filename_prefix}_{settings.CLASSIFIER_META_FILENAME}", "r") as f:
            metadata = json.load(f)

        class_labels = metadata["class_labels"]
        model_type = ModelType(metadata.get("model_type", "binary"))  # Default to binary for backward compatibility
        max_features = metadata.get("max_features", 5000)
        ngram_range_tuple = tuple(metadata.get("ngram_range", [1, 3]))

        if strategy_instance is None:
            raise ValueError(
                "strategy_instance must be provided to TextClassifier.load() "
                "or strategy re-instantiation logic needs to be implemented based on metadata."
            )

        classifier = cls(
            strategy=strategy_instance,
            class_labels=class_labels,
            model_type=model_type,
            max_features=max_features,
            ngram_range=ngram_range_tuple,
        )

        classifier.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        classifier.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        classifier.label_encoder = joblib.load(f"{filename_prefix}_label_encoder.pkl")

        classifier.strategy.load_model(filename_prefix)
        classifier._is_trained = True
        logging.info("Model, preprocessors, and metadata loaded successfully.")
        return classifier

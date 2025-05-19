# text_classifier/classifier.py

import json
from pathlib import Path
from typing import List, Dict, Any, Type, Optional  # Added Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from .strategies import (
    TextClassifierStrategy,
    ScikitLearnStrategy,
    TensorFlowStrategy,
    PyTorchStrategy,
)

import logging

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "ScikitLearnStrategy": ScikitLearnStrategy,
    "TensorFlowStrategy": TensorFlowStrategy,
    "PyTorchStrategy": PyTorchStrategy,
}
CLASS_TO_STRATEGY_NAME_MAP = {v: k for k, v in STRATEGY_MAP.items()}


class TextClassifier:
    def __init__(
        self,
        strategy: TextClassifierStrategy,
        class_labels: List[str],
        max_features: int = 5000,
        language: str = "english",
    ):

        if not isinstance(strategy, TextClassifierStrategy):
            raise ValueError("Strategy must be an instance of TextClassifierStrategy.")

        self.strategy = strategy
        self.class_labels = sorted(list(set(class_labels)))
        self.num_classes = len(self.class_labels)

        self.target_max_features = max_features
        self.fitted_input_features: Optional[int] = (
            None  # Will be set after vectorizer is fit
        )

        self.language = language.lower()
        self.training_metrics: Dict[str, Any] = {}

        actual_stop_words = "english" if self.language == "english" else None
        if self.language not in ["english", "none"] and actual_stop_words is None:
            logger.warning(
                f"Language '{self.language}' for TfidfVectorizer stop_words. Assuming None (no stop words)."
            )

        self.vectorizer = TfidfVectorizer(
            max_features=self.target_max_features, stop_words=actual_stop_words
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_labels)

        # The strategy instance is passed in. Its input_dim may be based on target_max_features.
        # This will be corrected in train() if necessary.
        # For clarity, explicitly align strategy's initial input_dim with target_max_features if it's being set up fresh.
        # If a strategy is passed from elsewhere, its input_dim could be different (e.g., from loading).
        # TextClassifier.train() is the authority for setting the correct fitted_input_features.
        logger.info(
            f"TextClassifier __init__: Strategy initial input_dim: {self.strategy.input_dim}, target_max_features: {self.target_max_features}"
        )
        # self.strategy.input_dim = self.target_max_features # Let's remove this, strategy init should take precedence or load should handle it. Train is king.

        if self.strategy.num_classes != self.num_classes:
            raise ValueError(
                f"Strategy's num_classes ({self.strategy.num_classes}) != TextClassifier's ({self.num_classes})."
            )

        logger.info(
            f"TextClassifier initialized with {self.strategy.__class__.__name__}, {self.num_classes} classes. Target max_features: {self.target_max_features}"
        )

    def train(self, dataset_path: str) -> Dict[str, Any]:
        logger.info(f"TextClassifier.train(): Starting for dataset {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
            if not ("text" in df.columns and "label" in df.columns):
                raise ValueError("Dataset CSV must contain 'text' and 'label' columns.")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            raise

        texts = df["text"].astype(str).tolist()
        str_labels = df["label"].astype(str).tolist()

        # Filter out data with labels not in self.class_labels
        df_filtered = df[df["label"].isin(self.class_labels)]
        if len(df_filtered) < len(df):
            logger.warning(
                f"Filtered out {len(df) - len(df_filtered)} rows with labels not in {self.class_labels}."
            )
        if len(df_filtered) == 0:
            raise ValueError(
                f"Dataset empty after filtering for known labels {self.class_labels}."
            )
        texts = df_filtered["text"].astype(str).tolist()
        str_labels = df_filtered["label"].astype(str).tolist()

        logger.info("TextClassifier.train(): Fitting TfidfVectorizer...")
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.fitted_input_features = X_tfidf.shape[1]
        logger.info(
            f"TextClassifier.train(): TfidfVectorizer fitted. Actual features: {self.fitted_input_features} (target was {self.target_max_features})."
        )

        # CRITICAL STEP: Update strategy's input dimension and rebuild model if necessary
        if self.strategy.input_dim != self.fitted_input_features:
            logger.info(
                f"TextClassifier.train(): Updating strategy's input_dim from {self.strategy.input_dim} to actual fitted features: {self.fitted_input_features}"
            )
            self.strategy.input_dim = self.fitted_input_features
            self.strategy.model = (
                None  # Invalidate pre-existing model structure if input_dim changed
            )
            logger.info(
                "TextClassifier.train(): Strategy model invalidated due to input_dim change."
            )

        # Ensure model is (re)built with the correct input_dim, especially for NN strategies
        if self.strategy.model is None:  # If invalidated or never built
            logger.info(
                "TextClassifier.train(): Strategy model is None. Calling strategy.build_model()."
            )
            self.strategy.build_model()  # This will use the (potentially updated) self.strategy.input_dim
        elif not isinstance(
            self.strategy, ScikitLearnStrategy
        ):  # Always rebuild NN if not None but input_dim might have changed (covered by None check now)
            # This case might be redundant if input_dim change always sets model to None
            logger.info(
                "TextClassifier.train(): Non-Scikit strategy and model exists. Ensuring build. This might be redundant if input_dim change always sets model to None."
            )
            # self.strategy.build_model() # To be safe, let's comment this out as the None check should cover it.

        y_integers = self.label_encoder.transform(str_labels)

        logger.info(
            f"TextClassifier.train(): Calling strategy.train(). Strategy input_dim: {self.strategy.input_dim}, Fitted features: {self.fitted_input_features}"
        )
        self.training_metrics = self.strategy.train(X_tfidf, y_integers)
        logger.info("TextClassifier.train(): Training complete.")
        return self.training_metrics

    def predict(self, texts: List[str]) -> List[str]:
        # ... (unchanged from previous correct version)
        if not texts:
            return []
        logger.debug(f"Predicting labels for {len(texts)} texts.")
        X_tfidf = self.vectorizer.transform(texts)
        if X_tfidf.shape[1] != self.strategy.input_dim:
            logger.error(
                f"Dimension mismatch in predict: TF-IDF output {X_tfidf.shape[1]} features, "
                f"strategy expects {self.strategy.input_dim} (fitted: {self.fitted_input_features})."
            )
        int_predictions = self.strategy.predict(X_tfidf)
        str_predictions = self.label_encoder.inverse_transform(int_predictions)
        return str_predictions.tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        # ... (unchanged from previous correct version)
        if not texts:
            return np.array([])
        logger.debug(f"Predicting probabilities for {len(texts)} texts.")
        X_tfidf = self.vectorizer.transform(texts)
        if X_tfidf.shape[1] != self.strategy.input_dim:
            logger.error(
                f"Dimension mismatch in predict_proba: TF-IDF output {X_tfidf.shape[1]} features, "
                f"strategy expects {self.strategy.input_dim} (fitted: {self.fitted_input_features})."
            )
        probabilities = self.strategy.predict_proba(X_tfidf)
        if probabilities.shape[1] != self.num_classes:
            logger.error(
                f"Probabilities shape mismatch: Got {probabilities.shape[1]} classes, expected {self.num_classes}"
            )
            raise ValueError(
                "Probability output dimension does not match number of classes."
            )
        return probabilities

    def save(self, path_prefix: str):
        # ... (ensure fitted_input_features is saved, unchanged from previous correct version)
        p_prefix = Path(path_prefix)
        p_prefix.parent.mkdir(parents=True, exist_ok=True)
        vec_path = f"{path_prefix}_vectorizer.joblib"
        le_path = f"{path_prefix}_label_encoder.joblib"
        meta_path = f"{path_prefix}_classifier_metadata.json"
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.label_encoder, le_path)
        self.strategy.save(path_prefix)
        strategy_name = CLASS_TO_STRATEGY_NAME_MAP.get(
            self.strategy.__class__, self.strategy.__class__.__name__
        )
        metadata = {
            "class_labels": self.class_labels,
            "num_classes": self.num_classes,
            "target_max_features": self.target_max_features,
            "fitted_input_features": self.fitted_input_features,  # Crucial for loading
            "language": self.language,
            "strategy_type": strategy_name,
            "model_type_from_strategy": self.strategy.model_type,
            "training_metrics": self.training_metrics,
            "vectorizer_path_suffix": Path(vec_path).name,
            "label_encoder_path_suffix": Path(le_path).name,
            "model_files_prefix_suffix": Path(path_prefix).name,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Classifier saved with prefix: {path_prefix}. Fitted features: {self.fitted_input_features}"
        )

    @classmethod
    def load(cls: Type["TextClassifier"], path_prefix: str) -> "TextClassifier":
        # ... (ensure strategy is initialized with fitted_input_features, unchanged from previous correct version)
        logger.info(
            f"TextClassifier.load(): Loading classifier with prefix: {path_prefix}"
        )
        meta_path = f"{path_prefix}_classifier_metadata.json"
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata {meta_path}: {e}")
            raise

        vec_path = str(Path(path_prefix).parent / metadata["vectorizer_path_suffix"])
        le_path = str(Path(path_prefix).parent / metadata["label_encoder_path_suffix"])
        vectorizer = joblib.load(vec_path)
        label_encoder = joblib.load(le_path)

        strategy_type_name = metadata["strategy_type"]
        StrategyClass = STRATEGY_MAP.get(strategy_type_name)
        if not StrategyClass:
            raise ValueError(f"Unknown strategy type: {strategy_type_name}")

        actual_model_input_dim = metadata.get("fitted_input_features")
        if actual_model_input_dim is None:
            if hasattr(vectorizer, "vocabulary_"):
                actual_model_input_dim = len(vectorizer.vocabulary_)
                logger.warning(
                    f"'fitted_input_features' not in metadata. Using vectorizer vocab size: {actual_model_input_dim}"
                )
            else:
                raise ValueError(
                    "Cannot determine model input dim: 'fitted_input_features' missing & vectorizer not evidently fitted."
                )

        # Instantiate strategy with the CORRECT input_dim the model was trained with
        logger.info(
            f"TextClassifier.load(): Instantiating strategy {strategy_type_name} with input_dim={actual_model_input_dim}, num_classes={metadata['num_classes']}"
        )
        strategy = StrategyClass(
            input_dim=actual_model_input_dim, num_classes=metadata["num_classes"]
        )
        strategy.load(
            path_prefix
        )  # This appropriately loads the model (TF loads arch, PyTorch needs build_model then load_state_dict)

        target_max_features_from_meta = metadata.get(
            "target_max_features", actual_model_input_dim
        )
        classifier = cls(
            strategy=strategy,
            class_labels=metadata["class_labels"],
            max_features=target_max_features_from_meta,
            language=metadata.get("language", "english"),
        )
        classifier.vectorizer = vectorizer
        classifier.label_encoder = label_encoder
        classifier.training_metrics = metadata.get("training_metrics", {})
        classifier.fitted_input_features = (
            actual_model_input_dim  # Set this crucial attribute
        )

        # Post-load sanity check
        logger.info(
            f"TextClassifier.load(): Classifier loaded. Fitted input features: {classifier.fitted_input_features}. Strategy input_dim: {strategy.input_dim}."
        )
        if strategy.input_dim != actual_model_input_dim:
            logger.error(
                f"CRITICAL LOAD INCONSISTENCY: Strategy input_dim {strategy.input_dim} != loaded actual_model_input_dim {actual_model_input_dim}"
            )

        logger.info("Classifier loaded successfully.")
        return classifier

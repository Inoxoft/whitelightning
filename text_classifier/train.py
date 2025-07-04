import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from text_classifier.strategies.binary import (  # Assuming binary strategies are in binary_strategies.py
        TensorFlowStrategyBinary,
        PyTorchStrategyBinary,
        ScikitLearnStrategyBinary,
    )
    from text_classifier.strategies.multiclass import (  # Assuming multiclass strategies are in multiclass_strategies.py
        PyTorchLSTMStrategy,
        TensorFlowLSTMStrategy,
        ScikitLearnTFIDFStrategy,
    )
    from text_classifier.strategies.multilabel import (  # Assuming multilabel strategies are in multilabel_strategies.py
        TensorFlowStrategyMultiLabel,
        PyTorchStrategyMultiLabel,
        ScikitLearnStrategyMultiLabel,
    )
except ModuleNotFoundError:  # Handle if running script directly from its dir
    from strategies.binary import (  # Assuming binary strategies are in binary_strategies.py
        TensorFlowStrategyBinary,
        PyTorchStrategyBinary,
        ScikitLearnStrategyBinary,
    )
    from strategies.multiclass import (  # Assuming multiclass strategies are in multiclass_strategies.py
        PyTorchLSTMStrategy,
        TensorFlowLSTMStrategy,
        ScikitLearnTFIDFStrategy,
    )
    from strategies.multilabel import (  # Assuming multilabel strategies are in multilabel_strategies.py
        TensorFlowStrategyMultiLabel,
        PyTorchStrategyMultiLabel,
        ScikitLearnStrategyMultiLabel,
    )


DEFAULT_MAX_FEATURES = 5000


class TextClassifierRunner:
    def __init__(
        self,
        train_path: str,
        test_path: str,
        labels: List[str],
        library_type: str,
        data_type: str,
        output_path: str,
    ):
        self.library_type = library_type
        self.data_type = data_type.split("_")[0]
        self.train_path = train_path
        self.test_path = test_path
        self.labels = labels
        self.output_path = output_path
        self.strategy = None

    def preprocess_binary_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, int]:
        """
        Preprocess CSV data for binary classification using TF-IDF features.

        Returns:
            tuple: (X_train, X_test, y_train, y_test, vocab_data, scaler_data)
                - X_train, X_test: TF-IDF feature arrays (shape: n_samples, 5000).
                - y_train, y_test: Binary label arrays (shape: n_samples,).
                - vocab_data: Dictionary with TF-IDF vocabulary and IDF values.
                - scaler_data: Dictionary with mean, scale, and label mapping.
        """
        # Load CSV files
        train_df = pd.read_csv(self.train_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )
        test_df = pd.read_csv(self.test_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )

        X_train_text = train_df["text"].values
        X_test_text = test_df["text"].values
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df["label"].values)
        y_test = label_encoder.transform(test_df["label"].values)

        vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES)
        self.vectorizer = vectorizer

        X_train = vectorizer.fit_transform(X_train_text).toarray()
        X_test = vectorizer.transform(X_test_text).toarray()

        actual_features = len(vectorizer.vocabulary_)
        # Compute scaler parameters
        mean = X_train.mean(axis=0)
        scale = X_train.std(axis=0) + 1e-8  # Avoid division by zero

        # Prepare vocab and scaler data
        vocab_data = {
            "vocab": {
                str(word): int(idx) for word, idx in vectorizer.vocabulary_.items()
            },
            "idf": [float(x) for x in vectorizer.idf_.tolist()],
        }
        scaler_data = {
            "mean": [float(x) for x in mean],
            "scale": [float(x) for x in scale],
        }

        return X_train, X_test, y_train, y_test, vocab_data, scaler_data, actual_features

    def preprocess_multiclass_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, int]:
        """
        Preprocess CSV data for multiclass classification using tokenized sequences or TF-IDF.
        Returns:
            tuple: (X_train, X_test, y_train, y_test, tokenizer_data, scaler_data)
                - X_train, X_test: Tokenized sequences (shape: n_samples, max_len) or TF-IDF features.
                - y_train, y_test: Encoded label arrays (shape: n_samples,).
                - tokenizer_data: Dictionary with tokenizer word index or TF-IDF vocabulary.
                - scaler_data: Dictionary with label mapping.
        """
        # Load CSV files
        train_df = pd.read_csv(self.train_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )
        test_df = pd.read_csv(self.test_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )

        # Filter by allowed labels
        train_df = train_df[train_df["label"].isin(self.labels)]
        test_df = test_df[test_df["label"].isin(self.labels)]

        # Extract text and labels
        X_train_text = train_df["text"].values
        X_test_text = test_df["text"].values
        y_train = train_df["label"].values
        y_test = test_df["label"].values

        return X_train_text, X_test_text, y_train, y_test, {}, {}, 0

    def preprocess_multilabel_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict, int]:
        """
        Preprocess CSV data for multilabel classification using TF-IDF features.
        Returns:
            tuple: (X_train, X_test, y_train, y_test, vocab_data, scaler_data)
                - X_train, X_test: TF-IDF feature arrays (shape: n_samples, 5000).
                - y_train, y_test: Multilabel binary arrays (shape: n_samples, num_labels).
                - vocab_data: Dictionary with TF-IDF vocabulary and IDF values.
                - scaler_data: Dictionary with mean, scale, and label mapping.
        """
        # Load CSV files
        train_df = pd.read_csv(self.train_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )
        test_df = pd.read_csv(self.test_path, on_bad_lines="skip").dropna(
            subset=["text", "label"]
        )

        # Parse comma-separated labels into binary matrix
        def parse_labels(label_str: str, allowed_labels: List[str]) -> np.ndarray:
            label_list = [l.strip() for l in label_str.split(",") if l.strip()]
            return np.array(
                [1 if label in label_list else 0 for label in allowed_labels],
                dtype=np.float32,
            )

        # Apply label parsing
        y_train = np.array(
            [parse_labels(label, self.labels) for label in train_df["label"]]
        )
        y_test = np.array(
            [parse_labels(label, self.labels) for label in test_df["label"]]
        )

        # Keep multi-hot encoding for multilabel classification
        # Do NOT use argmax as it destroys the multilabel nature!

        # Extract text
        X_train_text = train_df["text"].values
        X_test_text = test_df["text"].values

        # Compute TF-IDF features
        vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES)

        self.vectorizer = vectorizer
        X_train = vectorizer.fit_transform(X_train_text).toarray()
        X_test = vectorizer.transform(X_test_text).toarray()

        actual_features = len(vectorizer.vocabulary_)

        # Compute scaler parameters
        mean = X_train.mean(axis=0)
        scale = X_train.std(axis=0) + 1e-8  # Avoid division by zero

        # Prepare vocab and scaler data - include the actual vectorizer object
        vocab_data = {
            "vocab": {
                str(word): int(idx) for word, idx in vectorizer.vocabulary_.items()
            },
            "idf": [float(x) for x in vectorizer.idf_.tolist()],
            "vectorizer": vectorizer,  # Add the actual vectorizer object
        }
        scaler_data = {
            "mean": [float(x) for x in mean],
            "scale": [float(x) for x in scale],
            **{str(i): label for i, label in enumerate(self.labels)},
        }

        return X_train, X_test, y_train, y_test, vocab_data, scaler_data, actual_features

    def get_strategy_class(self):
        """
        Get the strategy class based on the data type and library type.

        Returns:
            class: The strategy class corresponding to the data type and library type.

        Raises:
            ValueError: If the combination of data_type and library_type is invalid.
        """
        strategy_map = {
            "binary": {
                "torch": PyTorchStrategyBinary,
                "tensorflow": TensorFlowStrategyBinary,
                "sklearn": ScikitLearnStrategyBinary,
            },
            "multiclass": {
                "torch": PyTorchLSTMStrategy,
                "tensorflow": TensorFlowLSTMStrategy,
                "sklearn": ScikitLearnTFIDFStrategy,
            },
            "multilabel": {
                "torch": PyTorchStrategyMultiLabel,
                "tensorflow": TensorFlowStrategyMultiLabel,
                "sklearn": ScikitLearnStrategyMultiLabel,
            },
        }

        try:
            return strategy_map[self.data_type][self.library_type]
        except KeyError:
            raise ValueError(
                f"Invalid combination: {self.data_type} with {self.library_type}"
            )

    def train_model(
        self,
    ):
        """
        Train a model of the specified type (binary, multiclass, multilabel) using the given library.
        """
        if self.library_type not in ["torch", "tensorflow", "sklearn"]:
            raise ValueError("library_type must be 'torch', 'tensorflow', or 'sklearn'")
        if self.data_type not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(
                "data_type must be 'binary', 'multiclass', or 'multilabel'"
            )

        # Preprocess data based on model type
        if self.data_type == "binary":
            X_train, X_test, y_train, y_test, vocab_data, scaler_data, input_dim = (
                self.preprocess_binary_data()
            )
        elif self.data_type == "multiclass":
            X_train, X_test, y_train, y_test, vocab_data, scaler_data, input_dim = (
                self.preprocess_multiclass_data()
            )
        else:  # multilabel
            X_train, X_test, y_train, y_test, vocab_data, scaler_data, input_dim = (
                self.preprocess_multilabel_data()
            )

        strategy_cls = self.get_strategy_class()
        strategy = strategy_cls(
            num_classes=len(self.labels) if isinstance(self.labels, list) else 1,
            input_dim=input_dim,
            vocab=vocab_data,
            scaler=scaler_data,
            output_path=self.output_path,
        )

        strategy.train(X_train, y_train, X_test, y_test)

        strategy.save_model()

        self.strategy = strategy

    def predict(self, inputs: List[str]) -> List[str]:
        """
        Predict labels for a list of input texts using the trained strategy.

        Args:
            inputs (List[str]): List of input texts to predict.

        Returns:
            List[str]: Predicted labels for the input texts.
        """
        if not self.strategy:
            raise ValueError("No strategy is initialized. Train a model first.")

        if not inputs:
            return []

        # Preprocess inputs based on the strategy type
        if self.data_type == "binary" or self.data_type == "multilabel":
            # Use TF-IDF preprocessing for binary and multilabel strategies
            X_inputs = self.vectorizer.transform(inputs).toarray()
        elif self.data_type == "multiclass":
            # Use tokenized sequences for multiclass strategies
            strategy = self.strategy
            if isinstance(strategy, (PyTorchLSTMStrategy, TensorFlowLSTMStrategy)):
                # Use the strategy's tokenizer and parameters
                sequences = strategy.tokenizer.texts_to_sequences(inputs)
                X_inputs = pad_sequences(sequences, maxlen=strategy.max_len, padding="post")
                return strategy.predict(X_inputs if isinstance(strategy, TensorFlowLSTMStrategy) else inputs)
            elif isinstance(strategy, ScikitLearnTFIDFStrategy):
                # Scikit-learn expects raw text
                return strategy.predict(np.array(inputs))
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

        # Call the strategy's predict method
        predictions = self.strategy.predict(X_inputs)

        return predictions

import logging
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, classification_report
import tf2onnx
import onnxruntime as rt

from settings import (
    MODEL_PREFIX,
    DATA_COLUMN_NAME,
    LABEL_COLUMN_NAME,
    TRAINING_DATA_PATH,
    MODELS_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class BinaryClassifier:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 4)):
        logging.info("Initializing BinaryClassifierTF")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.scaler = StandardScaler()
        self.model = self._build_model(max_features)
        self._is_trained = False

    def _build_model(self, input_dim: int) -> Model:
        """Build a TensorFlow model using Functional API"""
        inputs = Input(shape=(input_dim,), name='float_input')
        x = Dense(512, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_and_preprocess(self, csv_path: str) -> Tuple[pd.Series, pd.Series]:
        """Load and preprocess the CSV data"""
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if DATA_COLUMN_NAME not in df.columns or LABEL_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{DATA_COLUMN_NAME}' and '{LABEL_COLUMN_NAME}' columns")

        X = df[DATA_COLUMN_NAME]
        y = df[LABEL_COLUMN_NAME]

        # Convert labels to numeric if they aren't already
        if y.dtype == 'object':
            logging.info("Converting labels to numeric")
            y = y.replace({'negative': 0, 'positive': 1})  # Adjust mapping based on your data
            if y.isnull().any():
                raise ValueError("Labels contain null values after conversion")
            y = pd.to_numeric(y, errors='raise')

        return X, y

    def extract_features(self, X: pd.Series, training: bool = True) -> np.ndarray:
        """Convert text data to numerical features"""
        if training:
            X_tfidf = self.vectorizer.fit_transform(X).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_scaled = self.scaler.transform(X_tfidf)
        return X_scaled.astype(np.float32)  # Ensure float32 for TensorFlow

    def train(self, csv_path: str, test_size: float = 0.2, epochs: int = 10, batch_size: int = 32) -> dict:
        """Train the model and return metrics"""
        logging.info("Starting model training")
        X, y = self.load_and_preprocess(csv_path)
        X_processed = self.extract_features(X, training=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )

        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")

        # Ensure y_train and y_test are float32 for TensorFlow
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        self._is_trained = True

        # Evaluate
        train_pred = (self.model.predict(X_train) > 0.5).astype(int)
        test_pred = (self.model.predict(X_test) > 0.5).astype(int)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        logging.info(f"Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
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
        predictions = (self.model.predict(X_processed) > 0.5).astype(int)
        return predictions

    def save_model(self, filename_prefix: str):
        """Save the trained model and preprocessors"""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        logging.info(f"Saving model to {filename_prefix}")
        self.model.save(f"{filename_prefix}_model.h5")
        import joblib
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")

    def load_model(self, filename_prefix: str):
        """Load a trained model and preprocessors"""
        logging.info(f"Loading model from {filename_prefix}")
        self.model = tf.keras.models.load_model(f"{filename_prefix}_model.h5")
        import joblib
        self.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")
        self._is_trained = True

    def export_to_onnx(self, output_path: str):
        """Export the trained model to ONNX format"""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")

        logging.info(f"Converting model to ONNX format")
        spec = (tf.TensorSpec((None, 5000), tf.float32, name="float_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)

        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        logging.info(f"ONNX model saved to {output_path}")

        # Verify the model
        sess = rt.InferenceSession(output_path)
        input_name = sess.get_inputs()[0].name
        output_names = [output.name for output in sess.get_outputs()]
        logging.info(f"Model verified. Input name: {input_name}, Output names: {output_names}")


# After training in run_training()
def export_preprocessing_data(classifier, output_dir):
    import json
    import numpy as np

    # Convert vocabulary keys (words) to str and values (indices) from int64 to int
    vocab = {str(word): int(idx) for word, idx in classifier.vectorizer.vocabulary_.items()}

    # Convert IDF to list (if available), ensuring float conversion
    idf = classifier.vectorizer.idf_.tolist() if hasattr(classifier.vectorizer, 'idf_') else None

    # Convert scaler mean and scale to lists, ensuring float conversion
    mean = classifier.vectorizer.mean_.tolist() if hasattr(classifier.vectorizer,
                                                           'mean_') else classifier.scaler.mean_.tolist()
    scale = classifier.vectorizer.scale_.tolist() if hasattr(classifier.vectorizer,
                                                             'scale_') else classifier.scaler.scale_.tolist()

    preprocessing_data = {
        'vocabulary': vocab,
        'idf': idf,
        'mean': mean,
        'scale': scale,
        'max_features': 5000
    }

    with open(f"{output_dir}_preprocessing_data.json", 'w') as f:
        json.dump(preprocessing_data, f)

def run_training():
    classifier = BinaryClassifier()
    classifier.train(f'{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv')
    classifier.save_model(f"{MODELS_PATH}/{MODEL_PREFIX}")
    onnx_path = f"{MODELS_PATH}/{MODEL_PREFIX}_model.onnx"
    classifier.export_to_onnx(onnx_path)
    export_preprocessing_data(classifier, f'{MODELS_PATH}/{MODEL_PREFIX}')


def test_model():
    classifier = BinaryClassifier()
    classifier.train(f'{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv')

    onnx_path = f"{MODELS_PATH}/{MODEL_PREFIX}_model.onnx"
    classifier.export_to_onnx(onnx_path)

    # Load ONNX model
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")

    # Test with dummy input
    dummy_input = np.zeros((1, 5000), dtype=np.float32)
    feeds = {input_name: dummy_input}
    results = sess.run(None, feeds)
    print("Test output:", results)
# text_classifier/strategies.py

import abc
import json
from pathlib import Path

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import tensorflow as tf
    import tf2onnx

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class TextClassifierStrategy(abc.ABC):
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim  # This is the KEY variable for model input shape
        self.num_classes = num_classes
        self.model = None
        self.model_type = "base"
        logger.info(
            f"Strategy {self.__class__.__name__} initialized with input_dim: {self.input_dim}, num_classes: {self.num_classes}"
        )

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save(self, path_prefix: str):
        pass

    @abc.abstractmethod
    def load(self, path_prefix: str):
        pass

    @abc.abstractmethod
    def export_to_onnx(self, output_path: str, X_sample: np.ndarray):
        pass


class ScikitLearnStrategy(TextClassifierStrategy):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes)
        self.model_type = "scikit"
        self.model = LogisticRegression(
            multi_class="ovr", solver="liblinear", C=1.0, random_state=42, max_iter=1000
        )

    def build_model(self):
        logger.info(
            f"ScikitLearnStrategy: Model already configured in __init__. Current input_dim: {self.input_dim} (Note: Scikit models adapt or expect this implicitly)."
        )
        pass  # Model is ready

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        logger.info(
            f"Training Scikit-learn model (data shape: {X_train.shape}, strategy input_dim: {self.input_dim}, num_classes: {self.num_classes})"
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        try:
            loss = log_loss(
                y_train,
                y_proba,
                labels=np.arange(self.num_classes if self.num_classes > 1 else 2),
            )
        except ValueError as e:
            logger.warning(
                f"Could not calculate log_loss for scikit-learn: {e}. Using NaN."
            )
            loss = float("nan")
        metrics = {"accuracy": accuracy, "loss": loss}
        logger.info(f"Scikit-learn training complete. Metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path_prefix: str):
        model_path = f"{path_prefix}_sklearn_model.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Scikit-learn model saved to {model_path}")

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_sklearn_model.joblib"
        self.model = joblib.load(model_path)
        # After loading, its internal state reflects the data it was trained on.
        # self.input_dim is the one from metadata used for initialization.
        # For scikit-learn, predict/transform will work if data matches original fitted data features.
        logger.info(
            f"Scikit-learn model loaded from {model_path}. Strategy input_dim: {self.input_dim}"
        )

    def export_to_onnx(self, output_path: str, X_sample: np.ndarray):
        if self.model is None:
            raise ValueError("Scikit-learn model is not trained or loaded yet.")
        logger.info(
            f"Exporting Scikit-learn model to ONNX: {output_path}. Sample input shape: {X_sample.shape}, Strategy input_dim: {self.input_dim}"
        )
        if X_sample.shape[1] != self.input_dim:
            logger.warning(
                f"ONNX X_sample dim ({X_sample.shape[1]}) differs from strategy input_dim ({self.input_dim}). Problems might occur if a fixed-size model was expected based on strategy input_dim."
            )
        initial_type = [
            ("float_input", FloatTensorType([None, self.input_dim]))
        ]  # Use strategy's actual input_dim
        options = {id(self.model): {"zipmap": False}}
        try:
            onnx_model = convert_sklearn(
                self.model, initial_types=initial_type, target_opset=12, options=options
            )
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(
                f"Scikit-learn model successfully exported to ONNX: {output_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to export Scikit-learn model to ONNX: {e}", exc_info=True
            )
            raise


class TensorFlowStrategy(TextClassifierStrategy):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
        self.model_type = "tensorflow"
        self.epochs = 10
        self.batch_size = 32

    def build_model(self):
        # THIS IS THE CRITICAL POINT FOR TENSORFLOW MODEL DEFINITION
        logger.info(
            f"TensorFlowStrategy: BUILD_MODEL called. Current self.input_dim: {self.input_dim}, self.num_classes: {self.num_classes}"
        )
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=(self.input_dim,), name="float_input"
                ),  # Uses current self.input_dim
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info("TensorFlow Keras model built and compiled.")
        if self.model:
            logger.info(
                f"TF STRATEGY BUILD_MODEL: Model details: {self.model.summary(print_fn=logger.debug)}"
            )  # print_fn to logger.debug
            logger.info(
                f"TF STRATEGY BUILD_MODEL: Model's expected input shape from Keras: {self.model.input_shape}"
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        if self.model is None:
            logger.warning(
                "TensorFlowStrategy.train(): self.model is None. Calling build_model(). This should ideally be handled by TextClassifier."
            )
            self.build_model()  # Build if not already built (e.g. direct use of strategy)

        if hasattr(X_train, "toarray"):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train

        logger.info(
            f"TensorFlowStrategy.train(): Starting training. Data shape: {X_train_dense.shape}, Model's expected input_shape: {self.model.input_shape if self.model else 'N/A'}, Strategy's self.input_dim: {self.input_dim}"
        )

        if self.model.input_shape[-1] != X_train_dense.shape[-1]:
            logger.error(
                f"CRITICAL DIM MISMATCH DETECTED IN TensorFlowStrategy.train(): "
                f"Model expects {self.model.input_shape[-1]} features, data has {X_train_dense.shape[-1]} features. "
                f"Strategy's self.input_dim is {self.input_dim}."
            )
            # This indicates TextClassifier failed to properly orchestrate input_dim update and rebuild.
            raise ValueError(
                f"Dimension mismatch: Model expects {self.model.input_shape[-1]}, Data has {X_train_dense.shape[-1]}"
            )

        history = self.model.fit(
            X_train_dense,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )
        metrics = {
            k: [float(val) for val in v] if isinstance(v, list) else float(v)
            for k, v in history.history.items()
        }
        logger.info(
            f"TensorFlow training complete. Final epoch metrics: {{'loss': {metrics.get('loss', [-1])[-1]:.4f}, 'accuracy': {metrics.get('accuracy', [-1])[-1]:.4f}}}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("TF Model not available.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        y_pred_proba = self.model.predict(X)
        return np.argmax(y_pred_proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("TF Model not available.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model.predict(X)

    def save(self, path_prefix: str):
        if not self.model:
            raise ValueError("TF Model not available for saving.")
        model_path = f"{path_prefix}_tf_model.keras"
        self.model.save(model_path)
        logger.info(f"TensorFlow model saved to {model_path}")

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_tf_model.keras"
        self.model = tf.keras.models.load_model(model_path)
        # After loading, self.model.input_shape is determined by the saved model.
        # self.input_dim (from strategy init via metadata) should match this.
        logger.info(
            f"TensorFlow model loaded from {model_path}. Strategy's self.input_dim: {self.input_dim}. Loaded model's input_shape: {self.model.input_shape}"
        )
        if self.model.input_shape[-1] != self.input_dim:
            logger.warning(
                f"Loaded TF model input dim ({self.model.input_shape[-1]}) mismatch with strategy's input_dim ({self.input_dim}). "
                "Strategy's input_dim will be updated to reflect loaded model."
            )
            self.input_dim = self.model.input_shape[-1]

    def export_to_onnx(self, output_path: str, X_sample: np.ndarray):
        if not self.model:
            raise ValueError("TF Model not available for ONNX export.")

        logger.info(
            f"Exporting TensorFlow model to ONNX: {output_path}. "
            f"Sample input shape: {X_sample.shape}, "
            f"Strategy input_dim: {self.input_dim}, "
            f"Model input_shape: {self.model.input_shape}"
        )

        if X_sample.shape[1] != self.input_dim:
            # This is a strong indicator of a problem if they don't match.
            # The sample shape MUST match the model's expected input dimension.
            logger.error(
                f"ONNX X_sample dim ({X_sample.shape[1]}) differs from strategy's/model's "
                f"expected input_dim ({self.input_dim}). This will likely cause export to fail or lead to a faulty ONNX model."
            )
            # You might want to raise an error here if this condition is critical
            # raise ValueError(f"ONNX sample dimension mismatch: sample {X_sample.shape[1]} vs model {self.input_dim}")

        if hasattr(X_sample, "toarray"):
            X_sample_dense = X_sample.toarray()
        else:
            X_sample_dense = X_sample

        # Define input signature using self.input_dim (which *should* match model.input_shape[-1])
        input_signature = [
            tf.TensorSpec(
                shape=[None, self.input_dim], dtype=tf.float32, name="float_input"
            )
        ]

        # Define output names explicitly. Since it's a single output tensor (probabilities for classes)
        output_node_names = ["output"]  # Common practice for a single output

        try:
            logger.info(
                f"Attempting to convert Keras model to ONNX. Input signature: {input_signature}, Output names: {output_node_names}"
            )

            # tf2onnx.convert.from_keras will write to output_path if specified.
            # The first return value is model_proto, second is external_tensor_storage (usually None for smaller models)
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=input_signature,
                opset=13,  # Or your desired opset
                output_path=output_path,
            )

            # If model_proto is None here, it means tf2onnx wrote directly to output_path and doesn't return the proto
            # when output_path is given. Check if file exists as confirmation.
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                logger.info(
                    f"TensorFlow model successfully exported to ONNX: {output_path}"
                )
            else:
                logger.error(
                    f"TensorFlow model export to ONNX failed. File not created or empty at {output_path}."
                )
                # Depending on tf2onnx version, if output_path is set, model_proto might be None.
                # If it returned a proto, we could write it:
                # if model_proto:
                #    with open(output_path, "wb") as f:
                #        f.write(model_proto.SerializeToString())
                #    logger.info(f"TensorFlow model successfully exported to ONNX (manually written): {output_path}")
                # else:
                #    logger.error(f"TensorFlow model export to ONNX failed. tf2onnx did not return a model_proto and did not write to {output_path}.")
                #    raise RuntimeError(f"ONNX export failed for TensorFlow model.")

        except AttributeError as ae:
            logger.error(
                f"AttributeError during TensorFlow ONNX export: {ae}", exc_info=True
            )
            logger.error(
                "This might be due to an issue with tf2onnx expecting certain attributes on the Keras model. "
                "Ensure your TensorFlow and tf2onnx versions are compatible."
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to export TensorFlow model to ONNX: {e}", exc_info=True
            )
            raise


class PyTorchStrategy(TextClassifierStrategy):
    class _Net(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            # Logging within _Net can be tricky, do it before instantiation
            self.fc1 = nn.Linear(input_dim, 128)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes)
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")
        self.model_type = "pytorch"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch using device: {self.device}")
        self.epochs = 10
        self.batch_size = 32
        self.lr = 1e-3

    def build_model(self):
        logger.info(
            f"PyTorchStrategy: BUILD_MODEL called. Current self.input_dim: {self.input_dim}, self.num_classes: {self.num_classes}"
        )
        self.model = self._Net(self.input_dim, self.num_classes).to(
            self.device
        )  # Uses current self.input_dim
        logger.info("PyTorch model built.")
        logger.info(str(self.model))
        # Can log first layer's input features if needed: self.model.fc1.in_features

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        if self.model is None:
            logger.warning(
                "PyTorchStrategy.train(): self.model is None. Calling build_model(). This should ideally be handled by TextClassifier."
            )
            self.build_model()

        if hasattr(X_train, "toarray"):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train

        logger.info(
            f"PyTorchStrategy.train(): Starting training. Data shape: {X_train_dense.shape}, Model's first layer expects: {self.model.fc1.in_features if self.model else 'N/A'}, Strategy's self.input_dim: {self.input_dim}"
        )

        if self.model.fc1.in_features != X_train_dense.shape[-1]:
            logger.error(
                f"CRITICAL DIM MISMATCH DETECTED IN PyTorchStrategy.train(): "
                f"Model expects {self.model.fc1.in_features} features, data has {X_train_dense.shape[-1]} features. "
                f"Strategy's self.input_dim is {self.input_dim}."
            )
            raise ValueError(
                f"Dimension mismatch: Model expects {self.model.fc1.in_features}, Data has {X_train_dense.shape[-1]}"
            )

        X_tensor = torch.from_numpy(X_train_dense).float().to(self.device)
        y_tensor = torch.from_numpy(y_train).long().to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        epoch_losses, epoch_accuracies = [], []
        for epoch in range(self.epochs):
            running_loss, correct_predictions, total_samples = 0.0, 0, 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch_X.size(0)
                _, predicted_labels = torch.max(outputs, 1)
                correct_predictions += (predicted_labels == batch_y).sum().item()
                total_samples += batch_y.size(0)
            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
            )
        metrics = {"loss": epoch_losses, "accuracy": epoch_accuracies}
        logger.info(
            f"PyTorch training complete. Final epoch: Loss {metrics['loss'][-1]:.4f}, Acc {metrics['accuracy'][-1]:.4f}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("PyTorch Model not available.")
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        _, predicted_labels = torch.max(outputs, 1)
        return predicted_labels.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("PyTorch Model not available.")
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def save(self, path_prefix: str):
        if not self.model:
            raise ValueError("PyTorch Model not available for saving.")
        model_path = f"{path_prefix}_pytorch_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"PyTorch model saved to {model_path}")

    def load(self, path_prefix: str):
        logger.info(
            f"PyTorchStrategy: LOAD called. Initializing model structure with self.input_dim: {self.input_dim} before loading state_dict."
        )
        # Build model structure first, according to self.input_dim from metadata
        self.build_model()
        model_path = f"{path_prefix}_pytorch_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(
            f"PyTorch model loaded from {model_path}. Model's fc1.in_features: {self.model.fc1.in_features}, Strategy input_dim {self.input_dim}"
        )
        # Ensure loaded model's actual input dim matches strategy's, update strategy if there's discrepancy (shouldn't happen if saved correctly)
        if self.model.fc1.in_features != self.input_dim:
            logger.warning(
                f"Loaded PyTorch model input dim ({self.model.fc1.in_features}) mismatch with strategy's input_dim ({self.input_dim}). "
                "Strategy's input_dim will be updated."
            )
            self.input_dim = self.model.fc1.in_features

    def export_to_onnx(self, output_path: str, X_sample: np.ndarray):
        if not self.model:
            raise ValueError("PyTorch Model not available for ONNX export.")
        logger.info(
            f"Exporting PyTorch model to ONNX: {output_path}. Sample input shape: {X_sample.shape}, Strategy input_dim: {self.input_dim}, Model fc1.in_features: {self.model.fc1.in_features}"
        )
        if X_sample.shape[1] != self.input_dim:
            logger.error(
                f"ONNX X_sample dim ({X_sample.shape[1]}) differs from strategy input_dim ({self.input_dim}). This will likely fail."
            )

        self.model.eval()
        if hasattr(X_sample, "toarray"):
            X_sample_dense = X_sample.toarray()
        else:
            X_sample_dense = X_sample
        dummy_input = torch.from_numpy(X_sample_dense).float().to(self.device)

        # For ONNX export of probabilities
        onnx_export_model = nn.Sequential(self.model, nn.Softmax(dim=1)).to(self.device)
        onnx_export_model.eval()
        try:
            torch.onnx.export(
                onnx_export_model,
                dummy_input,
                output_path,
                input_names=["float_input"],
                output_names=["output"],
                opset_version=12,
                dynamic_axes={
                    "float_input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            logger.info(f"PyTorch model successfully exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export PyTorch model to ONNX: {e}", exc_info=True)
            raise

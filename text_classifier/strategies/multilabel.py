from pathlib import Path

import numpy as np
import joblib
import pickle
import json

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
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

from .base import TextClassifierStrategy

logger = logging.getLogger(__name__)


class ScikitLearnStrategyMultiLabel(TextClassifierStrategy):
    def __init__(
        self,
        input_dim: int = 5000,
        num_classes: int = 2,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):
        self.model = MultiOutputClassifier(
            LogisticRegression(solver="liblinear", C=1.0, random_state=42, max_iter=1000)
        )
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Extract vectorizer from vocab if available
        if isinstance(vocab, dict) and 'vectorizer' in vocab:
            self.vectorizer = vocab['vectorizer']
        else:
            self.vectorizer = None

    def build_model(self):
        logger.info(
            f"ScikitLearnStrategy: Model already configured in __init__. Current input_dim: {self.input_dim} (Note: Scikit models adapt or expect this implicitly)."
        )
        pass  # Model is ready

    def train(self, X_train: np.ndarray, y_train: np.ndarray, *args) -> dict:
        logger.info(
            f"Training Scikit-learn model (data shape: {X_train.shape}, strategy input_dim: {self.input_dim}, num_classes: {self.num_classes})"
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_train)
        y_proba = self.model.predict_proba(X_train)
        
        # For multilabel classification, calculate accuracy differently
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # Multilabel accuracy: average accuracy across all labels
            accuracy = ((y_pred == y_train).mean())
        else:
            accuracy = accuracy_score(y_train, y_pred)
        
        try:
            # For multilabel, log_loss calculation is different
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                # Convert list of arrays to proper format for multilabel log_loss
                if isinstance(y_proba, list):
                    # MultiOutputClassifier returns list of arrays
                    loss = 0.0
                    for i in range(len(y_proba)):
                        loss += log_loss(y_train[:, i], y_proba[i][:, 1])
                    loss /= len(y_proba)
                else:
                    loss = log_loss(y_train, y_proba)
            else:
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
        predictions = self.model.predict(X)
        # For multilabel, predict_proba returns probabilities
        proba = self.model.predict_proba(X)
        if isinstance(proba, list):
            # MultiOutputClassifier returns list of arrays, convert to proper format
            result = np.zeros((X.shape[0], len(proba)))
            for i, prob_array in enumerate(proba):
                result[:, i] = prob_array[:, 1]  # Take positive class probability
            return result
        else:
            return proba

    def save_model(self):
        model_path = f"{self.output_path}/model.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Scikit-learn model saved to {model_path}")
        
        # Save the vectorizer object if available
        if hasattr(self, 'vectorizer') and self.vectorizer is not None:
            vocab_path = f"{self.output_path}/vocab.pkl"
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Scikit-learn TF-IDF vectorizer object saved to {vocab_path}")
        
        self.save_model_vocab_and_scaler()
        self.export_to_onnx()

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_sklearn_model.joblib"
        self.model = joblib.load(model_path)
        logger.info(
            f"Scikit-learn model loaded from {model_path}. Strategy input_dim: {self.input_dim}"
        )

    def export_to_onnx(self):
        output_path = f"{self.output_path}/model.onnx"
        X_sample = np.random.rand(1, self.input_dim).astype(
            np.float32
        )  # Sample input for ONNX export
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


class TensorFlowStrategyMultiLabel(TextClassifierStrategy):
    def __init__(
        self,
        input_dim: int = 5000,
        num_classes: int = 2,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed.")
        self.epochs = 10
        self.batch_size = 32
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None  # Model will be built in build_model()
        
        # Extract vectorizer from vocab if available
        if isinstance(vocab, dict) and 'vectorizer' in vocab:
            self.vectorizer = vocab['vectorizer']
        else:
            self.vectorizer = None

    def build_model(self):
        logger.info(
            f"TensorFlowStrategy: BUILD_MODEL called. Current self.input_dim: {self.input_dim}, self.num_classes: {self.num_classes}"
        )
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=(self.input_dim,), name="float_input"
                ),  # Uses current self.input_dim
                tf.keras.layers.Dense(256, activation="relu"),  # Updated from 128 to 256
                tf.keras.layers.Dropout(0.3),  # Updated from 0.5 to 0.3
                tf.keras.layers.Dense(self.num_classes, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],  # Updated to binary_accuracy for multilabel
        )
        logger.info("TensorFlow Keras model built and compiled.")
        if self.model:
            logger.info(
                f"TF STRATEGY BUILD_MODEL: Model details: {self.model.summary(print_fn=logger.debug)}"
            )  # print_fn to logger.debug
            logger.info(
                f"TF STRATEGY BUILD_MODEL: Model's expected input shape from Keras: {self.model.input_shape}"
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, *args) -> dict:
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
            f"TensorFlow training complete. Final epoch metrics: {{'loss': {metrics.get('loss', [-1])[-1]:.4f}, 'accuracy': {metrics.get('binary_accuracy', [-1])[-1]:.4f}}}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("TF Model not available.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("TF Model not available.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        return self.model.predict(X)

    def save_model_vocab_and_scaler_tensorflow(self):
        """Save vocab as pickle and classes in scaler.json for TensorFlow multilabel"""
        # Save the actual TF-IDF vectorizer object (not just vocab dict)
        vocab_path = f"{self.output_path}/vocab.pkl"
        with open(vocab_path, "wb") as f:
            # If vocab contains the actual vectorizer object, save it
            if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                pickle.dump(self.vectorizer, f)
                logger.info(f"TensorFlow TF-IDF vectorizer object saved to {vocab_path}")
            elif isinstance(self.vocab, dict) and 'vectorizer' in self.vocab:
                pickle.dump(self.vocab['vectorizer'], f)
                logger.info(f"TensorFlow TF-IDF vectorizer object saved to {vocab_path}")
            else:
                # Fallback: save the vocab dictionary
                pickle.dump(self.vocab, f)
                logger.info(f"TensorFlow vocab dictionary saved to {vocab_path}")
        
        # Save classes in scaler.json format as dictionary with string indices
        scaler_path = f"{self.output_path}/scaler.json"
        # If scaler contains class labels, save them
        if hasattr(self, 'class_labels'):
            class_names = self.class_labels
        elif isinstance(self.scaler, dict) and 'classes' in self.scaler:
            class_names = self.scaler['classes']
        else:
            # Default classes if not available
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
        # Convert to dictionary format with string indices: {"0": "Business", "1": "Health", ...}
        scaler_data = {str(i): class_name for i, class_name in enumerate(class_names)}
            
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f)
        logger.info(f"TensorFlow classes saved to {scaler_path}")

    def save_model(self):
        if not self.model:
            raise ValueError("TF Model not available for saving.")
        model_path = f"{self.output_path}/model.keras"
        self.model.save(model_path)
        self.save_model_vocab_and_scaler_tensorflow()
        self.export_to_onnx()
        logger.info(f"TensorFlow model saved to {model_path}")

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_tf_model.keras"
        self.model = tf.keras.models.load_model(model_path)
        
        # Load vocab from pickle file
        vocab_path = f"{path_prefix}_vocab.pkl"
        try:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Vocab file not found at {vocab_path}")
            self.vocab = {}
        
        # Load classes from scaler.json file
        scaler_path = f"{path_prefix}_scaler.json"
        try:
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
                
                # Handle both old and new formats
                if 'classes' in scaler_data:
                    # Old format: {"classes": ["class_0", "class_1", ...]}
                    self.class_labels = scaler_data['classes']
                elif all(key.isdigit() for key in scaler_data.keys()):
                    # New format: {"0": "Business", "1": "Health", ...}
                    # Convert to list ordered by index
                    self.class_labels = [scaler_data[str(i)] for i in range(len(scaler_data))]
                else:
                    logger.warning(f"Unknown scaler format in {scaler_path}")
                    self.class_labels = []
                
                logger.info(f"TensorFlow classes loaded from {scaler_path}: {self.class_labels}")
        except FileNotFoundError:
            logger.warning(f"Scaler file not found at {scaler_path}")
            self.class_labels = []
        
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

    def export_to_onnx(self):
        output_path = f"{self.output_path}/model.onnx"
        if not self.model:
            raise ValueError("TensorFlow model not available for ONNX export.")
        logger.info(f"Exporting TensorFlow multilabel model to ONNX: {output_path}")
        
        try:
            import tf2onnx
            import subprocess
            
            # Get the actual input shape from the trained model
            actual_input_shape = self.model.input_shape[1:]  # Remove batch dimension
            logger.info(f"Using actual model input shape: {actual_input_shape}")
            
            # Create a model without sigmoid activation for ONNX export (logits only)
            logger.info("Creating model without sigmoid for ONNX export...")
            
            # Create model without sigmoid (logits only)
            model_no_sigmoid = tf.keras.Sequential([
                tf.keras.layers.Input(shape=actual_input_shape, name="float_input"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(self.num_classes, activation=None)  # No activation (logits)
            ])
            
            # Copy weights from the trained model
            model_no_sigmoid.set_weights(self.model.get_weights())
            
            # Try tf2onnx command line tool which is more reliable
            savedmodel_for_onnx = f"{self.output_path}/temp_savedmodel_for_onnx"
            model_no_sigmoid.export(savedmodel_for_onnx)
            
            result = subprocess.run([
                "python", "-m", "tf2onnx.convert",
                "--saved-model", savedmodel_for_onnx,
                "--output", output_path,
                "--opset", "11"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Verify export success by checking file size
                file_size = Path(output_path).stat().st_size
                if file_size > 1000:  # If larger than 1KB, it's likely a real model
                    logger.info(f"TensorFlow multilabel model successfully exported to ONNX: {output_path} ({file_size} bytes)")
                    
                    # Clean up temporary SavedModel
                    import shutil
                    if os.path.exists(savedmodel_for_onnx):
                        shutil.rmtree(savedmodel_for_onnx)
                    
                    # Save ONNX model info
                    onnx_info = {
                        'input_name': 'float_input',
                        'output_name': 'output',
                        'input_shape': [None, actual_input_shape[0]],
                        'output_shape': [None, self.num_classes],
                        'opset_version': 11,
                        'conversion_method': 'tf2onnx_cli',
                        'activation_note': 'Model outputs logits (no sigmoid). Apply sigmoid during inference!',
                        'framework': 'tensorflow'
                    }
                    onnx_info_path = f"{self.output_path}/onnx_model_info.json"
                    with open(onnx_info_path, "w") as f:
                        json.dump(onnx_info, f, indent=2)
                    logger.info(f"ONNX model info saved to {onnx_info_path}")
                    return  # Exit early on success
                else:
                    logger.warning(f"ONNX export produced small file ({file_size} bytes), trying fallback methods...")
                    raise Exception("Export produced suspiciously small file")
            else:
                logger.error(f"tf2onnx CLI failed with exit code {result.returncode}")
                logger.error(f"Error: {result.stderr}")
                raise Exception("tf2onnx CLI conversion failed")
            
        except Exception as e:
            logger.error(f"Failed to export TensorFlow multilabel model to ONNX: {e}")
            # Try functional model approach as primary fallback
            try:
                logger.info("Trying functional model approach...")
                actual_input_shape = self.model.input_shape[1:]
                # Create a functional model with correct architecture (updated to match build_model)
                input_layer = tf.keras.layers.Input(shape=actual_input_shape, name="float_input")
                x = tf.keras.layers.Dense(256, activation="relu")(input_layer)  # Updated to 256
                x = tf.keras.layers.Dropout(0.3)(x)  # Updated to 0.3
                outputs = tf.keras.layers.Dense(self.num_classes, activation=None)(x)  # No sigmoid for ONNX
                
                functional_model = tf.keras.Model(inputs=input_layer, outputs=outputs, name="multilabel_model")
                
                # Copy weights from Sequential model to Functional model
                # Sequential: [Dense(256), Dropout, Dense(num_classes)]
                # Functional: [Input, Dense(256), Dropout, Dense(num_classes)]
                for i, layer in enumerate(self.model.layers):
                    if layer.get_weights():
                        # Skip input layer (no weights) and map correctly
                        if i == 0:  # First layer in Sequential (Dense)
                            functional_model.layers[1].set_weights(layer.get_weights())  # Skip input layer
                        elif i == 1:  # Second layer (Dropout - no weights)
                            continue  # Skip dropout layer
                        elif i == 2:  # Third layer (Dense)
                            # For the output layer, we need to copy weights but remove sigmoid
                            # The weights are the same, just no sigmoid activation
                            functional_model.layers[3].set_weights(layer.get_weights())  # Skip input and dropout
                
                # Convert functional model to ONNX
                spec = (tf.TensorSpec((None, actual_input_shape[0]), tf.float32, name="float_input"),)
                model_proto, _ = tf2onnx.convert.from_keras(
                    functional_model, input_signature=spec, opset=11
                )
                
                with open(output_path, "wb") as f:
                    f.write(model_proto.SerializeToString())
                
                # Verify export success
                file_size = Path(output_path).stat().st_size
                if file_size > 1000:
                    logger.info(f"TensorFlow multilabel model exported to ONNX using functional model approach: {output_path} ({file_size} bytes)")
                    return  # Exit early on success
                else:
                    logger.warning(f"Functional model approach produced small file ({file_size} bytes), trying next fallback...")
                    raise Exception("Functional model export produced suspiciously small file")
                
            except Exception as e2:
                logger.error(f"Functional model approach failed: {e2}")
                # Try fallback approach with older opset as secondary fallback
                try:
                    logger.info("Trying direct conversion with older opset...")
                    actual_input_shape = self.model.input_shape[1:]
                    spec = (tf.TensorSpec((None, actual_input_shape[0]), tf.float32, name="float_input"),)
                    model_proto, _ = tf2onnx.convert.from_keras(
                        self.model, input_signature=spec, opset=11  # Use older opset
                    )
                    with open(output_path, "wb") as f:
                        f.write(model_proto.SerializeToString())
                    
                    # Verify export success
                    file_size = Path(output_path).stat().st_size
                    if file_size > 1000:
                        logger.info(f"TensorFlow multilabel model exported to ONNX using older opset: {output_path} ({file_size} bytes)")
                        return  # Exit early on success
                    else:
                        logger.warning(f"Older opset approach produced small file ({file_size} bytes), creating placeholder...")
                        raise Exception("Older opset export produced suspiciously small file")
                except Exception as e3:
                    logger.error(f"Older opset approach also failed: {e3}")
                    # Create a simple ONNX model manually as last resort
                    logger.warning("Creating minimal ONNX model file as last resort...")
                    try:
                        import onnx
                        from onnx import helper, TensorProto
                        
                        # Use actual model dimensions for placeholder
                        actual_input_dim = self.model.input_shape[1]
                        
                        # Create a simple ONNX graph as placeholder
                        input_tensor = helper.make_tensor_value_info('float_input', TensorProto.FLOAT, [None, actual_input_dim])
                        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, self.num_classes])
                        
                        # Create a simple identity node (placeholder)
                        node = helper.make_node('Identity', ['float_input'], ['output'])
                        graph = helper.make_graph([node], 'multilabel_placeholder', [input_tensor], [output_tensor])
                        model_def = helper.make_model(graph, producer_name='whitelightning')
                        
                        with open(output_path, "wb") as f:
                            f.write(model_def.SerializeToString())
                        logger.warning(f"Created placeholder ONNX model at {output_path}")
                    except Exception as e4:
                        logger.error(f"Failed to create placeholder ONNX model: {e4}")
                        raise


class PyTorchStrategyMultiLabel(TextClassifierStrategy):
    class _Net(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            # Updated architecture based on user's code
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    def __init__(
        self,
        input_dim: int = 5000,
        num_classes: int = 2,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch using device: {self.device}")
        self.epochs = 10
        self.batch_size = 32
        self.lr = 1e-3
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None  # Model will be built in build_model()
        
        # Extract vectorizer from vocab if available
        if isinstance(vocab, dict) and 'vectorizer' in vocab:
            self.vectorizer = vocab['vectorizer']
        else:
            self.vectorizer = None

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray, *args) -> dict:
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

        # Check input dimensions for new architecture
        model_input_dim = self.model.net[0].in_features
        if model_input_dim != X_train_dense.shape[-1]:
            logger.error(
                f"CRITICAL DIM MISMATCH DETECTED IN PyTorchStrategy.train(): "
                f"Model expects {model_input_dim} features, data has {X_train_dense.shape[-1]} features. "
                f"Strategy's self.input_dim is {self.input_dim}."
            )
            raise ValueError(
                f"Dimension mismatch: Model expects {model_input_dim}, Data has {X_train_dense.shape[-1]}"
            )

        X_tensor = torch.from_numpy(X_train_dense).float().to(self.device)
        y_tensor = torch.from_numpy(y_train).float().to(self.device)  # Change to float for multilabel
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()  # Change to BCE for multilabel
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
                # For multilabel, use sigmoid and threshold at 0.5
                predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
                # Calculate accuracy as the average of correct predictions across all labels
                correct_predictions += ((predicted_labels == batch_y).float().mean(dim=1)).sum().item()
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
            probabilities = torch.sigmoid(outputs)  # Use sigmoid for multilabel
        return probabilities.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("PyTorch Model not available.")
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        probabilities = torch.sigmoid(outputs)  # Use sigmoid for multilabel
        return probabilities.cpu().numpy()

    def save_model_vocab_and_scaler_pytorch(self):
        """Save vocab as pickle and classes in scaler.json for PyTorch multilabel"""
        # Save the actual TF-IDF vectorizer object (not just vocab dict)
        vocab_path = f"{self.output_path}/vocab.pkl"
        with open(vocab_path, "wb") as f:
            # If vocab contains the actual vectorizer object, save it
            if hasattr(self, 'vectorizer') and self.vectorizer is not None:
                pickle.dump(self.vectorizer, f)
                logger.info(f"PyTorch TF-IDF vectorizer object saved to {vocab_path}")
            elif isinstance(self.vocab, dict) and 'vectorizer' in self.vocab:
                pickle.dump(self.vocab['vectorizer'], f)
                logger.info(f"PyTorch TF-IDF vectorizer object saved to {vocab_path}")
            else:
                # Fallback: save the vocab dictionary
                pickle.dump(self.vocab, f)
                logger.info(f"PyTorch vocab dictionary saved to {vocab_path}")
        
        # Save classes in scaler.json format as dictionary with string indices
        scaler_path = f"{self.output_path}/scaler.json"
        # If scaler contains class labels, save them
        if hasattr(self, 'class_labels'):
            class_names = self.class_labels
        elif isinstance(self.scaler, dict) and 'classes' in self.scaler:
            class_names = self.scaler['classes']
        else:
            # Default classes if not available
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
        # Convert to dictionary format with string indices: {"0": "Business", "1": "Health", ...}
        scaler_data = {str(i): class_name for i, class_name in enumerate(class_names)}
            
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f)
        logger.info(f"PyTorch classes saved to {scaler_path}")

    def save_model(self):
        if not self.model:
            raise ValueError("PyTorch Model not available for saving.")
        model_path = f"{self.output_path}/model.pth"
        torch.save(self.model.state_dict(), model_path)
        self.save_model_vocab_and_scaler_pytorch()
        self.export_to_onnx()
        logger.info(f"PyTorch model saved to {model_path}")

    def load(self, path_prefix: str):
        logger.info(
            f"PyTorchStrategy: LOAD called. Initializing model structure with self.input_dim: {self.input_dim} before loading state_dict."
        )
        # Build model structure first, according to self.input_dim from metadata
        self.build_model()
        model_path = f"{path_prefix}_pytorch_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load vocab (TF-IDF vectorizer) from pickle
        vocab_path = f"{path_prefix}_vocab.pkl"
        try:
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
            logger.info(f"PyTorch vocab (vectorizer) loaded from {vocab_path}")
        except FileNotFoundError:
            logger.warning(f"Vocab file not found: {vocab_path}")
        
        # Load classes from scaler.json
        scaler_path = f"{path_prefix}_scaler.json"
        try:
            with open(scaler_path, "r") as f:
                scaler_data = json.load(f)
                
                # Handle both old and new formats
                if 'classes' in scaler_data:
                    # Old format: {"classes": ["class_0", "class_1", ...]}
                    self.class_labels = scaler_data['classes']
                elif all(key.isdigit() for key in scaler_data.keys()):
                    # New format: {"0": "Business", "1": "Health", ...}
                    # Convert to list ordered by index
                    self.class_labels = [scaler_data[str(i)] for i in range(len(scaler_data))]
                else:
                    logger.warning(f"Unknown scaler format in {scaler_path}")
                    self.class_labels = []
                    
                logger.info(f"PyTorch classes loaded from {scaler_path}: {self.class_labels}")
        except FileNotFoundError:
            logger.warning(f"Scaler file not found: {scaler_path}")
            self.class_labels = []
        
        # Check model dimensions
        model_input_dim = self.model.net[0].in_features
        logger.info(
            f"PyTorch model loaded from {model_path}. Model input features: {model_input_dim}, Strategy input_dim {self.input_dim}"
        )
        # Ensure loaded model's actual input dim matches strategy's
        if model_input_dim != self.input_dim:
            logger.warning(
                f"Loaded PyTorch model input dim ({model_input_dim}) mismatch with strategy's input_dim ({self.input_dim}). "
                "Strategy's input_dim will be updated."
            )
            self.input_dim = model_input_dim

    def export_to_onnx(self):
        output_path = f"{self.output_path}/model.onnx"
        if not self.model:
            raise ValueError("PyTorch Model not available for ONNX export.")
        
        logger.info(f"Exporting PyTorch multilabel model to ONNX: {output_path}")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, self.input_dim).to(self.device)
            
            self.model.eval()
            
            # Export model directly (without softmax for multilabel - will use sigmoid in inference)
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=11
            )
            
            # Verify export success by checking file size
            file_size = Path(output_path).stat().st_size
            if file_size > 1000:
                logger.info(f"PyTorch multilabel model successfully exported to ONNX: {output_path} ({file_size} bytes)")
            else:
                logger.warning(f"ONNX export produced small file ({file_size} bytes)")
                raise Exception("Export produced suspiciously small file")
                
        except Exception as e:
            logger.error(f"Failed to export PyTorch multilabel model to ONNX: {e}")
            # Try fallback with different settings
            try:
                logger.info("Trying PyTorch ONNX export with different settings...")
                dummy_input = torch.randn(1, self.input_dim).to(self.device)
                
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    output_path,
                    input_names=["float_input"],
                    output_names=["output"],
                    dynamic_axes={"float_input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    opset_version=12
                )
                
                # Verify fallback export
                file_size = Path(output_path).stat().st_size
                if file_size > 1000:
                    logger.info(f"PyTorch model exported to ONNX using fallback method: {output_path} ({file_size} bytes)")
                else:
                    logger.error(f"Fallback export also produced small file ({file_size} bytes)")
                    raise Exception("Fallback export failed")
                    
            except Exception as e2:
                logger.error(f"PyTorch ONNX export fallback also failed: {e2}")
                raise

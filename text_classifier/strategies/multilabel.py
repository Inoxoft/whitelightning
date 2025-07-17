from pathlib import Path

import numpy as np
import joblib
import pickle
import json

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
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
        
       
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
           
            accuracy = ((y_pred == y_train).mean())
        else:
            accuracy = accuracy_score(y_train, y_pred)
        
        try:
           
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
               
                if isinstance(y_proba, list):
                  
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
        
        proba = self.model.predict_proba(X)
        if isinstance(proba, list):
            
            result = np.zeros((X.shape[0], len(proba)))
            for i, prob_array in enumerate(proba):
                result[:, i] = prob_array[:, 1] 
            return result
        else:
            return proba

    def save_model(self):
        model_path = f"{self.output_path}/model.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Scikit-learn model saved to {model_path}")
        
       
        if hasattr(self, 'vectorizer') and self.vectorizer is not None:
            vectorizer_data = {
                'vocabulary': {word: int(idx) for word, idx in self.vectorizer.vocabulary_.items()},
                'idf': self.vectorizer.idf_.tolist(),
                'max_features': self.vectorizer.max_features,
                'vocabulary_size': len(self.vectorizer.vocabulary_)
            }
            vocab_path = f"{self.output_path}/vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vectorizer_data, f, indent=2)
            logger.info(f"Scikit-learn TF-IDF vectorizer data saved to {vocab_path}")
        
        
        scaler_path = f"{self.output_path}/scaler.json"
        
       
        if isinstance(self.scaler, dict):
            
            label_keys = [k for k in self.scaler.keys() if k.isdigit()]
            if label_keys:
              
                sorted_keys = sorted(label_keys, key=int)
               
                base_labels = sorted_keys[:self.num_classes]
                class_names = [self.scaler[k] for k in base_labels]
            else:
                
                class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
           
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
       
        scaler_data = {str(i): class_name for i, class_name in enumerate(class_names)}
        
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f)
        logger.info(f"Scikit-learn classes saved to {scaler_path}")
        
        self.export_to_onnx()

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_sklearn_model.joblib"
        self.model = joblib.load(model_path)
        logger.info(
            f"Scikit-learn model loaded from {model_path}. Strategy input_dim: {self.input_dim}"
        )
        
        
        vocab_path = f"{path_prefix}_vocab.json"
        try:
            with open(vocab_path, 'r') as f:
                vectorizer_data = json.load(f)
            
            
            self.vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
            self.vectorizer.vocabulary_ = vectorizer_data['vocabulary']
            self.vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
            logger.info(f"Scikit-learn TF-IDF vectorizer reconstructed from {vocab_path}")
        except FileNotFoundError:
            logger.warning(f"Vocab file not found at {vocab_path}")
            self.vectorizer = None
        except Exception as e:
            logger.warning(f"Error loading vectorizer data: {e}")
            self.vectorizer = None

    def export_to_onnx(self):
        output_path = f"{self.output_path}/model.onnx"
        X_sample = np.random.rand(1, self.input_dim).astype(
            np.float32
        )  
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
        ] 
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
        
       
        tf.random.set_seed(42)
        np.random.seed(42)
        
        self.epochs = 10  
        self.batch_size = 32
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None  
        self.history = None  
        
       
        if isinstance(vocab, dict) and 'vectorizer' in vocab:
            self.vectorizer = vocab['vectorizer']
        else:
            self.vectorizer = None
        
       
        try:
            import tf2onnx
            self.ONNX_AVAILABLE = True
        except ImportError:
            self.ONNX_AVAILABLE = False
            logger.warning("tf2onnx not available. ONNX export will be skipped.")

    def build_model(self):
        """Create a multi-label classification model with optimal sigmoid training"""
        logger.info(
            f"TensorFlowStrategy: BUILD_MODEL called. Current self.input_dim: {self.input_dim}, self.num_classes: {self.num_classes}"
        )
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,), name="float_input"),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='sigmoid') 
        ])
        
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  
            metrics=['binary_accuracy', 'precision', 'recall']
        )
        
        logger.info("TensorFlow Keras model built and compiled with enhanced architecture.")
        if self.model:
            logger.info(f"TF STRATEGY BUILD_MODEL: Model summary:")
            self.model.summary(print_fn=logger.info)
            logger.info(f"TF STRATEGY BUILD_MODEL: Model's expected input shape: {self.model.input_shape}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, *args) -> dict:
        if self.model is None:
            logger.warning(
                "TensorFlowStrategy.train(): self.model is None. Calling build_model()."
            )
            self.build_model()

        if hasattr(X_train, "toarray"):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train.copy()

        logger.info(
            f"TensorFlowStrategy.train(): Starting training. Data shape: {X_train_dense.shape}, "
            f"Model's expected input_shape: {self.model.input_shape if self.model else 'N/A'}, "
            f"Strategy's self.input_dim: {self.input_dim}"
        )

        if self.model.input_shape[-1] != X_train_dense.shape[-1]:
            logger.error(
                f"CRITICAL DIM MISMATCH DETECTED: "
                f"Model expects {self.model.input_shape[-1]} features, data has {X_train_dense.shape[-1]} features."
            )
            raise ValueError(
                f"Dimension mismatch: Model expects {self.model.input_shape[-1]}, Data has {X_train_dense.shape[-1]}"
            )

       
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_dense, y_train, test_size=0.2, random_state=42
        )

       
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

       
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{self.output_path}/best_model_tf.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        
        self.history = self.model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )

        
        eval_results = self.model.evaluate(X_val, y_val, verbose=0)
        val_loss = eval_results[0]
        val_binary_accuracy = eval_results[1]
        val_precision = eval_results[2]
        val_recall = eval_results[3]

        
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Binary Accuracy: {val_binary_accuracy:.4f}")
        logger.info(f"Validation Precision: {val_precision:.4f}")
        logger.info(f"Validation Recall: {val_recall:.4f}")
        logger.info(f"Validation F1 Score: {val_f1:.4f}")

       
        metrics = {
            'val_loss': float(val_loss),
            'val_binary_accuracy': float(val_binary_accuracy),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1)
        }

       
        if self.history:
            history_metrics = {
                k: [float(val) for val in v] if isinstance(v, list) else float(v)
                for k, v in self.history.history.items()
            }
            metrics.update(history_metrics)

        logger.info(f"TensorFlow training complete. Final metrics: {metrics}")
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
        """Save vocab as JSON and classes in scaler.json for TensorFlow multilabel"""
        
        if hasattr(self, 'vectorizer') and self.vectorizer is not None:
            vectorizer_data = {
                'vocabulary': {word: int(idx) for word, idx in self.vectorizer.vocabulary_.items()},
                'idf': self.vectorizer.idf_.tolist(),
                'max_features': self.vectorizer.max_features,
                'vocabulary_size': len(self.vectorizer.vocabulary_)
            }
            vocab_path = f"{self.output_path}/vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vectorizer_data, f, indent=2)
            logger.info(f"TensorFlow TF-IDF vectorizer data saved to {vocab_path}")
        elif isinstance(self.vocab, dict) and 'vectorizer' in self.vocab:
            vectorizer = self.vocab['vectorizer']
            vectorizer_data = {
                'vocabulary': {word: int(idx) for word, idx in vectorizer.vocabulary_.items()},
                'idf': vectorizer.idf_.tolist(),
                'max_features': vectorizer.max_features,
                'vocabulary_size': len(vectorizer.vocabulary_)
            }
            vocab_path = f"{self.output_path}/vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vectorizer_data, f, indent=2)
            logger.info(f"TensorFlow TF-IDF vectorizer data saved to {vocab_path}")
        else:
            logger.warning("TensorFlow: No vectorizer available to save")
        
        
        scaler_path = f"{self.output_path}/scaler.json"
        
       
        if isinstance(self.scaler, dict):
           
            label_keys = [k for k in self.scaler.keys() if k.isdigit()]
            if label_keys:
               
                sorted_keys = sorted(label_keys, key=int)
                
                base_labels = sorted_keys[:self.num_classes]
                class_names = [self.scaler[k] for k in base_labels]
            else:
               
                class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
       
        scaler_data = {str(i): class_name for i, class_name in enumerate(class_names)}
        
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
        logger.info(f"TensorFlow classes saved to {scaler_path}")

       
        if self.history:
            history_dict = {
                'loss': self.history.history['loss'],
                'val_loss': self.history.history['val_loss'],
                'binary_accuracy': self.history.history['binary_accuracy'],
                'val_binary_accuracy': self.history.history['val_binary_accuracy'],
                'precision': self.history.history['precision'],
                'val_precision': self.history.history['val_precision'],
                'recall': self.history.history['recall'],
                'val_recall': self.history.history['val_recall']
            }
            history_path = f"{self.output_path}/training_history_tf.json"
            with open(history_path, "w") as f:
                json.dump(history_dict, f, indent=2)
            logger.info(f"Training history saved to {history_path}")

    def save_model(self):
        if not self.model:
            raise ValueError("TF Model not available for saving.")
        
       
        keras_model_path = f"{self.output_path}/model.keras"
        self.model.save(keras_model_path)
        logger.info(f"TensorFlow model saved as {keras_model_path}")

       
        self.save_model_vocab_and_scaler_tensorflow()
        
       
        self.export_to_onnx()

    def load(self, path_prefix: str):
        model_path = f"{path_prefix}_model.keras"
        self.model = tf.keras.models.load_model(model_path)
        
       
        vocab_path = f"{path_prefix}_vocab.json"
        try:
            with open(vocab_path, 'r') as f:
                vectorizer_data = json.load(f)
            
           
            self.vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
            self.vectorizer.vocabulary_ = vectorizer_data['vocabulary']
            self.vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
            logger.info(f"TensorFlow TF-IDF vectorizer reconstructed from {vocab_path}")
        except FileNotFoundError:
            logger.warning(f"Vocab file not found at {vocab_path}")
            self.vectorizer = None
        except Exception as e:
            logger.warning(f"Error loading vectorizer data: {e}")
            self.vectorizer = None
        
       
        scaler_path = f"{path_prefix}_scaler.json"
        try:
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
                
             
                if isinstance(scaler_data, dict) and all(key.isdigit() for key in scaler_data.keys()):
                  
                    self.class_labels = [scaler_data[str(i)] for i in range(len(scaler_data))]
                else:
                   
                    self.class_labels = scaler_data
                
                logger.info(f"TensorFlow classes loaded from {scaler_path}: {self.class_labels}")
        except FileNotFoundError:
            logger.warning(f"Classes file not found at {scaler_path}")
            self.class_labels = []
        
       
        logger.info(
            f"TensorFlow model loaded from {model_path}. Strategy's self.input_dim: {self.input_dim}. "
            f"Loaded model's input_shape: {self.model.input_shape}"
        )
        if self.model.input_shape[-1] != self.input_dim:
            logger.warning(
                f"Loaded TF model input dim ({self.model.input_shape[-1]}) mismatch with strategy's input_dim ({self.input_dim}). "
                "Strategy's input_dim will be updated to reflect loaded model."
            )
            self.input_dim = self.model.input_shape[-1]

    def export_to_onnx(self):
        """Export TensorFlow model to ONNX format with sigmoid activation"""
        if not self.ONNX_AVAILABLE:
            logger.warning("ONNX export skipped - tf2onnx not available")
            return

        output_path = f"{self.output_path}/model.onnx"
        if not self.model:
            raise ValueError("TensorFlow model not available for ONNX export.")
        
        logger.info(f"Converting TensorFlow model to ONNX: {output_path}")
        
        try:
            
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                savedmodel_path = f"{temp_dir}/temp_savedmodel"
                self.model.export(savedmodel_path)
                
               
                import subprocess
                result = subprocess.run([
                    "python", "-m", "tf2onnx.convert",
                    "--saved-model", savedmodel_path,
                    "--output", output_path,
                    "--opset", "11"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    file_size = Path(output_path).stat().st_size
                    logger.info(f"ONNX model saved as {output_path} ({file_size} bytes)")
                    
                   
                    try:
                        import onnxruntime as ort
                        
                        
                        onnx_session = ort.InferenceSession(output_path)
                        onnx_input_name = onnx_session.get_inputs()[0].name
                        onnx_output_name = onnx_session.get_outputs()[0].name
                        
                        logger.info(f"ONNX Input name: {onnx_input_name}")
                        logger.info(f"ONNX Output name: {onnx_output_name}")
                        
                    
                        onnx_info = {
                            'input_name': onnx_input_name,
                            'output_name': onnx_output_name,
                            'input_shape': [None, self.input_dim],
                            'output_shape': [None, self.num_classes],
                            'opset_version': 11,
                            'conversion_method': 'tf2onnx_cli',
                            'activation_note': 'Model outputs probabilities (with sigmoid activation).',
                            'classes': getattr(self, 'class_labels', [f"class_{i}" for i in range(self.num_classes)]),
                            'vocab_size': len(self.vectorizer.vocabulary_) if self.vectorizer else self.input_dim,
                            'framework': 'tensorflow'
                        }
                        onnx_info_path = f"{self.output_path}/onnx_model_info.json"
                        with open(onnx_info_path, "w") as f:
                            json.dump(onnx_info, f, indent=2)
                        logger.info(f"ONNX model info saved as {onnx_info_path}")
                        
                    except ImportError:
                        logger.warning("onnxruntime not available for testing")
                    except Exception as e:
                        logger.warning(f"ONNX testing failed: {e}")
                else:
                    logger.error(f"ONNX export failed with exit code {result.returncode}")
                    logger.error(f"Error: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"Failed to export TensorFlow model to ONNX: {e}")

    def create_inference_function(self):
        """Create inference function for multi-label classification"""
        def predict_multilabel(texts, model_path=None, vectorizer_path=None, classes_path=None):
            """
            Inference function for multi-label classification
            
            Args:
                texts: List of text strings to classify
                model_path: Path to saved TensorFlow model
                vectorizer_path: Path to saved TF-IDF vectorizer JSON data
                classes_path: Path to saved class names
            
            Returns:
                List of dictionaries with predictions for each text
            """
           
            if model_path is None:
                model_path = f"{self.output_path}/model.keras"
            if vectorizer_path is None:
                vectorizer_path = f"{self.output_path}/vocab.json"
            if classes_path is None:
                classes_path = f"{self.output_path}/scaler.json"
            
          
            model = tf.keras.models.load_model(model_path)
            
          
            with open(vectorizer_path, 'r') as f:
                vectorizer_data = json.load(f)
            
           
            vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
            vectorizer.vocabulary_ = vectorizer_data['vocabulary']
            vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
            with open(classes_path, 'r') as f:
                classes_data = json.load(f)
            
            
            if isinstance(classes_data, dict) and all(key.isdigit() for key in classes_data.keys()):
               
                classes = [classes_data[str(i)] for i in range(len(classes_data))]
            else:
              
                classes = classes_data
            
           
            X = vectorizer.transform(texts).toarray()
            
           
            predictions = model.predict(X, verbose=0)
            
           
            results = []
            for i, text in enumerate(texts):
                result = {
                    'text': text,
                    'predictions': {cls: float(prob) for cls, prob in zip(classes, predictions[i])},
                    'predicted_labels': [cls for cls, prob in zip(classes, predictions[i]) if prob > 0.5]
                }
                results.append(result)
            
            return results
        
        return predict_multilabel


class PyTorchStrategyMultiLabel(TextClassifierStrategy):
    class _Net(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
          
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
                nn.Sigmoid() 
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
        
       
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PyTorch using device: {self.device}")
        
        self.epochs = 50  
        self.batch_size = 32
        self.lr = 1e-3
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.model = None 
        self.history = None  
        
       
        if isinstance(vocab, dict) and 'vectorizer' in vocab:
            self.vectorizer = vocab['vectorizer']
        else:
            self.vectorizer = None

    def build_model(self):
        logger.info(
            f"PyTorchStrategy: BUILD_MODEL called. Current self.input_dim: {self.input_dim}, self.num_classes: {self.num_classes}"
        )
        self.model = self._Net(self.input_dim, self.num_classes).to(self.device)
        logger.info("PyTorch model built with enhanced architecture.")
        logger.info(str(self.model))
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def calculate_accuracy(self, preds, targets, threshold=0.5):
        """Calculate element-wise accuracy for multi-label classification"""
        preds_binary = (preds >= threshold).float()
        correct = (preds_binary == targets).sum().item()
        total = torch.numel(targets)
        return correct / total * 100

    def evaluate_model(self, val_loader, criterion, threshold=0.5):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                probs = self.model(batch_X)  
                loss = criterion(probs, batch_y)
                total_loss += loss.item()
                
                preds = (probs >= threshold).float()
                total_correct += (preds == batch_y).sum().item()
                total_samples += torch.numel(batch_y)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples * 100
        return avg_loss, accuracy

    def train(self, X_train: np.ndarray, y_train: np.ndarray, *args) -> dict:
        if self.model is None:
            logger.warning(
                "PyTorchStrategy.train(): self.model is None. Calling build_model()."
            )
            self.build_model()

        if hasattr(X_train, "toarray"):
            X_train_dense = X_train.toarray()
        else:
            X_train_dense = X_train.copy()

        logger.info(
            f"PyTorchStrategy.train(): Starting training. Data shape: {X_train_dense.shape}, "
            f"Model's first layer expects: {self.model.net[0].in_features if self.model else 'N/A'}, "
            f"Strategy's self.input_dim: {self.input_dim}"
        )

       
        model_input_dim = self.model.net[0].in_features
        if model_input_dim != X_train_dense.shape[-1]:
            logger.error(
                f"CRITICAL DIM MISMATCH DETECTED: "
                f"Model expects {model_input_dim} features, data has {X_train_dense.shape[-1]} features."
            )
            raise ValueError(
                f"Dimension mismatch: Model expects {model_input_dim}, Data has {X_train_dense.shape[-1]}"
            )

        
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_dense, y_train, test_size=0.2, random_state=42
        )

    
        X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

       
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")

       
        criterion = nn.BCELoss()  
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        threshold = 0.5

        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        logger.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                probs = self.model(batch_X)  
                loss = criterion(probs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

               
                with torch.no_grad():
                    preds = (probs >= threshold).float()
                    total_correct += (preds == batch_y).sum().item()
                    total_samples += torch.numel(batch_y)
            
          
            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_samples * 100
            
           
            val_loss, val_acc = self.evaluate_model(val_loader, criterion, threshold)
            
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1:2d}/{self.epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            
           
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
              
                torch.save(self.model.state_dict(), f"{self.output_path}/best_model_pytorch.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

       
        self.model.load_state_dict(torch.load(f"{self.output_path}/best_model_pytorch.pth"))
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")

       
        self.history = history

      
        final_val_loss, final_val_acc = self.evaluate_model(val_loader, criterion, threshold)
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.2f}%")

       
        metrics = {
            'val_loss': float(final_val_loss),
            'val_accuracy': float(final_val_acc),
            'best_val_loss': float(best_val_loss),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss_history': history['val_loss'],
            'val_acc_history': history['val_acc']
        }

        logger.info(f"PyTorch training complete. Final metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("PyTorch Model not available.")
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            probabilities = self.model(X_tensor)  
        return probabilities.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            raise ValueError("PyTorch Model not available.")
        self.model.eval()
        if hasattr(X, "toarray"):
            X = X.toarray()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            probabilities = self.model(X_tensor)  
        return probabilities.cpu().numpy()

    def save_model_vocab_and_scaler_pytorch(self):
        """Save vocab as JSON and classes in scaler.json for PyTorch multilabel"""
       
        if hasattr(self, 'vectorizer') and self.vectorizer is not None:
            vectorizer_data = {
                'vocabulary': {word: int(idx) for word, idx in self.vectorizer.vocabulary_.items()},
                'idf': self.vectorizer.idf_.tolist(),
                'max_features': self.vectorizer.max_features,
                'vocabulary_size': len(self.vectorizer.vocabulary_)
            }
            vocab_path = f"{self.output_path}/vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vectorizer_data, f, indent=2)
            logger.info(f"PyTorch TF-IDF vectorizer data saved to {vocab_path}")
        elif isinstance(self.vocab, dict) and 'vectorizer' in self.vocab:
            vectorizer = self.vocab['vectorizer']
            vectorizer_data = {
                'vocabulary': {word: int(idx) for word, idx in vectorizer.vocabulary_.items()},
                'idf': vectorizer.idf_.tolist(),
                'max_features': vectorizer.max_features,
                'vocabulary_size': len(vectorizer.vocabulary_)
            }
            vocab_path = f"{self.output_path}/vocab.json"
            with open(vocab_path, "w") as f:
                json.dump(vectorizer_data, f, indent=2)
            logger.info(f"PyTorch TF-IDF vectorizer data saved to {vocab_path}")
        else:
            logger.warning("PyTorch: No vectorizer available to save")
        
      
        scaler_path = f"{self.output_path}/scaler.json"
        
      
        if isinstance(self.scaler, dict):
           
            label_keys = [k for k in self.scaler.keys() if k.isdigit()]
            if label_keys:
                
                sorted_keys = sorted(label_keys, key=int)
               
                base_labels = sorted_keys[:self.num_classes]
                class_names = [self.scaler[k] for k in base_labels]
            else:
               
                class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        
       
        scaler_data = {str(i): class_name for i, class_name in enumerate(class_names)}
        
        with open(scaler_path, "w") as f:
            json.dump(scaler_data, f, indent=2)
        logger.info(f"PyTorch classes saved to {scaler_path}")

       
        if self.history:
            history_path = f"{self.output_path}/training_history_pytorch.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
            logger.info(f"Training history saved to {history_path}")

       
        model_config = {
            'input_dim': self.input_dim,
            'output_dim': self.num_classes,
            'architecture': 'Linear(512) + ReLU + Dropout(0.3) + Linear(256) + ReLU + Dropout(0.3) + Linear(num_classes) + Sigmoid',
            'classes': class_names,
            'framework': 'pytorch'
        }
        config_path = f"{self.output_path}/model_config_pytorch.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Model configuration saved to {config_path}")

    def save_model(self):
        if not self.model:
            raise ValueError("PyTorch Model not available for saving.")
        
     
        model_path = f"{self.output_path}/model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"PyTorch model weights saved to {model_path}")

       
        complete_model_path = f"{self.output_path}/model_complete.pth"
        torch.save(self.model, complete_model_path)
        logger.info(f"Complete PyTorch model saved to {complete_model_path}")

      
        self.save_model_vocab_and_scaler_pytorch()
        
       
        self.export_to_onnx()

    def load(self, path_prefix: str):
        
        config_path = f"{path_prefix}_model_config_pytorch.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Model configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}")
            config = {'input_dim': self.input_dim, 'output_dim': self.num_classes}

       
        self.model = self._Net(config['input_dim'], config['output_dim']).to(self.device)
        
       
        model_path = f"{path_prefix}_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"PyTorch model loaded from {model_path}")
        
       
        vocab_path = f"{path_prefix}_vocab.json"
        try:
            with open(vocab_path, 'r') as f:
                vectorizer_data = json.load(f)
            
          
            self.vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
            self.vectorizer.vocabulary_ = vectorizer_data['vocabulary']
            self.vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
            logger.info(f"PyTorch TF-IDF vectorizer reconstructed from {vocab_path}")
        except FileNotFoundError:
            logger.warning(f"Vocab file not found at {vocab_path}")
            self.vectorizer = None
        except Exception as e:
            logger.warning(f"Error loading vectorizer data: {e}")
            self.vectorizer = None
        
       
        scaler_path = f"{path_prefix}_scaler.json"
        try:
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
                
               
                if isinstance(scaler_data, dict) and all(key.isdigit() for key in scaler_data.keys()):
                    
                    self.class_labels = [scaler_data[str(i)] for i in range(len(scaler_data))]
                else:
                    
                    self.class_labels = scaler_data
                
                logger.info(f"PyTorch classes loaded from {scaler_path}: {self.class_labels}")
        except FileNotFoundError:
            logger.warning(f"Classes file not found at {scaler_path}")
            self.class_labels = []
        
      
        if config['input_dim'] != self.input_dim:
            logger.warning(
                f"Loaded PyTorch model input dim ({config['input_dim']}) mismatch with strategy's input_dim ({self.input_dim}). "
                "Strategy's input_dim will be updated to reflect loaded model."
            )
            self.input_dim = config['input_dim']

    def export_to_onnx(self):
        """Export PyTorch model to ONNX format"""
        if not self.model:
            raise ValueError("PyTorch model not available for ONNX export.")
        
        output_path = f"{self.output_path}/model.onnx"
        logger.info(f"Exporting PyTorch model to ONNX: {output_path}")
        
        try:
           
            dummy_input = torch.randn(1, self.input_dim).to(self.device)
            
           
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=11
            )
            
            file_size = Path(output_path).stat().st_size
            logger.info(f"ONNX model saved as {output_path} ({file_size} bytes)")
            
           
            try:
                import onnxruntime as ort
                
                
                onnx_session = ort.InferenceSession(output_path)
                onnx_input_name = onnx_session.get_inputs()[0].name
                onnx_output_name = onnx_session.get_outputs()[0].name
                
                logger.info(f"ONNX Input name: {onnx_input_name}")
                logger.info(f"ONNX Output name: {onnx_output_name}")
                
               
                onnx_info = {
                    'input_name': onnx_input_name,
                    'output_name': onnx_output_name,
                    'input_shape': [None, self.input_dim],
                    'output_shape': [None, self.num_classes],
                    'opset_version': 11,
                    'activation_note': 'Model outputs probabilities (with sigmoid activation).',
                    'classes': getattr(self, 'class_labels', [f"class_{i}" for i in range(self.num_classes)]),
                    'vocab_size': len(self.vectorizer.vocabulary_) if self.vectorizer else self.input_dim,
                    'framework': 'pytorch'
                }
                onnx_info_path = f"{self.output_path}/onnx_model_info.json"
                with open(onnx_info_path, "w") as f:
                    json.dump(onnx_info, f, indent=2)
                logger.info(f"ONNX model info saved as {onnx_info_path}")
                
            except ImportError:
                logger.warning("onnxruntime not available for testing")
            except Exception as e:
                logger.warning(f"ONNX testing failed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to export PyTorch model to ONNX: {e}")

    def create_inference_function(self):
        """Create inference function for PyTorch multi-label classification"""
        def predict_multilabel_pytorch(texts, model_path=None, vectorizer_path=None, 
                                      classes_path=None, config_path=None):
            """
            Inference function for PyTorch multi-label classification
            
            Args:
                texts: List of text strings to classify
                model_path: Path to saved PyTorch model weights
                vectorizer_path: Path to saved TF-IDF vectorizer JSON data
                classes_path: Path to saved class names
                config_path: Path to saved model configuration
            
            Returns:
                List of dictionaries with predictions for each text
            """
           
            if model_path is None:
                model_path = f"{self.output_path}/model.pth"
            if vectorizer_path is None:
                vectorizer_path = f"{self.output_path}/vocab.json"
            if classes_path is None:
                classes_path = f"{self.output_path}/scaler.json"
            if config_path is None:
                config_path = f"{self.output_path}/model_config_pytorch.json"
            
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            
            model = self._Net(config['input_dim'], config['output_dim'])
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            
            with open(vectorizer_path, 'r') as f:
                vectorizer_data = json.load(f)
            
            
            vectorizer = TfidfVectorizer(max_features=vectorizer_data['max_features'])
            vectorizer.vocabulary_ = vectorizer_data['vocabulary']
            vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
          
            with open(classes_path, 'r') as f:
                classes_data = json.load(f)
            
           
            if isinstance(classes_data, dict) and all(key.isdigit() for key in classes_data.keys()):
               
                classes = [classes_data[str(i)] for i in range(len(classes_data))]
            else:
                
                classes = classes_data
            
            
            X = vectorizer.transform(texts).toarray()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
          
            with torch.no_grad():
                predictions = model(X_tensor) 
            
          
            results = []
            for i, text in enumerate(texts):
                result = {
                    'text': text,
                    'predictions': {cls: float(prob) for cls, prob in zip(classes, predictions[i])},
                    'predicted_labels': [cls for cls, prob in zip(classes, predictions[i]) if prob > 0.5]
                }
                results.append(result)
            
            return results
        
        return predict_multilabel_pytorch

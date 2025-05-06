import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.utils import to_categorical  # For one-hot encoding
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TextClassifierStrategy:
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        self._is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,  # Added validation set
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,  # Added validation set
        y_test: np.ndarray,
    ) -> Dict:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # Should return probabilities
        raise NotImplementedError

    def save_model(self, filename_prefix: str):
        raise NotImplementedError

    def load_model(self, filename_prefix: str):
        raise NotImplementedError

    def export_to_onnx(self, output_path: str, X_sample: Optional[np.ndarray] = None):
        if (
            X_sample is None
        ):  # Add a default sample if not provided, using self.input_dim
            X_sample = np.zeros((1, self.input_dim), dtype=np.float32)
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")
        self._export_to_onnx_internal(output_path, X_sample)

    def _export_to_onnx_internal(self, output_path: str, X_sample: np.ndarray):
        raise NotImplementedError


class TensorFlowStrategy(TextClassifierStrategy):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes)
        self.model = self._build_model()  # Build model upon initialization

    def _build_model(self) -> Model:
        inputs = Input(shape=(self.input_dim,), name="float_input")
        x = Dense(512, activation="relu")(inputs)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.3)(x)
        # Output layer changes
        if self.num_classes == 2:  # Technically binary can be 1 output with sigmoid
            # For consistency with multiclass expecting probabilities for each class,
            # let's use num_classes output units and softmax even for binary.
            # Or, handle binary as a special case with 1 unit/sigmoid.
            # Using 2 units/softmax is cleaner for generic multiclass handling.
            outputs = Dense(self.num_classes, activation="softmax", name="output")(x)
            loss_function = "categorical_crossentropy"
        else:  # Multiclass
            outputs = Dense(self.num_classes, activation="softmax", name="output")(x)
            loss_function = "categorical_crossentropy"

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])
        return model

    def train(self, X_train, X_val, X_test, y_train_int, y_val_int, y_test_int) -> Dict:
        # One-hot encode labels for categorical_crossentropy
        y_train_cat = to_categorical(y_train_int, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val_int, num_classes=self.num_classes)
        y_test_cat = to_categorical(y_test_int, num_classes=self.num_classes)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = self.model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=20,  # Increased epochs, with early stopping
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping],
        )
        self._is_trained = True

        # Evaluate on test set
        loss, accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        logging.info(
            f"Test set evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        # Predictions for classification report (on original integer labels)
        y_pred_proba_test = self.model.predict(X_test)
        y_pred_int_test = np.argmax(y_pred_proba_test, axis=1)

        y_pred_proba_train = self.model.predict(X_train)
        y_pred_int_train = np.argmax(y_pred_proba_train, axis=1)

        return {
            "train_accuracy": accuracy_score(y_train_int, y_pred_int_train),
            "val_accuracy": (
                history.history["val_accuracy"][-1]
                if "val_accuracy" in history.history
                else None
            ),
            "test_accuracy": accuracy_score(y_test_int, y_pred_int_test),
            "test_loss": loss,
            "classification_report_test": classification_report(
                y_test_int,
                y_pred_int_test,
                zero_division=0,  # target_names=[str(i) for i in range(self.num_classes)]
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:  # Returns probabilities
        if not self._is_trained or self.model is None:
            raise RuntimeError("Model must be trained/loaded before predicting")
        return self.model.predict(X)

    def save_model(self, filename_prefix: str):
        if not self._is_trained or self.model is None:
            raise RuntimeError("Cannot save untrained model")
        self.model.save(f"{filename_prefix}_model.h5")

    def load_model(self, filename_prefix: str):
        self.model = tf.keras.models.load_model(f"{filename_prefix}_model.h5")
        # Re-derive input_dim and num_classes from loaded model if possible, or assume they were set correctly
        loaded_input_shape = self.model.input_shape[1]
        loaded_output_shape = self.model.output_shape[1]
        if self.input_dim != loaded_input_shape:
            logging.warning(
                f"Loaded model input_dim {loaded_input_shape} differs from strategy {self.input_dim}. Using loaded model's."
            )
            self.input_dim = loaded_input_shape
        if self.num_classes != loaded_output_shape:
            logging.warning(
                f"Loaded model num_classes {loaded_output_shape} differs from strategy {self.num_classes}. Using loaded model's."
            )
            self.num_classes = loaded_output_shape
        self._is_trained = True

    def _export_to_onnx_internal(self, output_path: str, X_sample: np.ndarray):
        import tf2onnx

        # Ensure X_sample matches the model's expected input dimension
        if X_sample.shape[1] != self.input_dim:
            X_sample = np.zeros((1, self.input_dim), dtype=np.float32)

        spec = (tf.TensorSpec((None, self.input_dim), tf.float32, name="float_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            opset=13,
            output_path=output_path,  # tf2onnx can write directly
        )
        # with open(output_path, "wb") as f: # Not needed if output_path is passed to from_keras
        #     f.write(model_proto.SerializeToString())
        logging.info(f"TensorFlow ONNX model saved to {output_path}")


class PyTorchStrategy(TextClassifierStrategy):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(input_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Module:
        class PTClassifier(nn.Module):
            def __init__(self, D_in, D_out):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(D_in, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, D_out),
                    # No Softmax here if using nn.CrossEntropyLoss
                    # Softmax will be applied in predict() or export_to_onnx()
                )

            def forward(self, x):
                return self.layers(x)

        return PTClassifier(self.input_dim, self.num_classes)

    def train(self, X_train, X_val, X_test, y_train_int, y_val_int, y_test_int) -> Dict:
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_int).to(
            self.device
        )  # CrossEntropyLoss expects LongTensor indices
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_int).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        # y_test_tensor for loss calculation if needed, otherwise use y_test_int for sklearn metrics

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
        optimizer = optim.Adam(self.model.parameters())

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = 3  # For early stopping

        for epoch in range(20):  # Max epochs
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)  # Raw logits
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    outputs_val = self.model(batch_X_val)
                    loss_val = criterion(outputs_val, batch_y_val)
                    val_loss += loss_val.item()
                    _, predicted_val = torch.max(outputs_val.data, 1)
                    total_val += batch_y_val.size(0)
                    correct_val += (predicted_val == batch_y_val).sum().item()

            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_val / total_val
            logging.info(
                f"Epoch {epoch + 1}/20, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save best model state (optional, or do it at the end)
                torch.save(self.model.state_dict(), f"temp_best_model.pt")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after epoch {epoch + 1}.")
                self.model.load_state_dict(
                    torch.load("temp_best_model.pt")
                )  # Restore best
                break

        if (
            epochs_no_improve < patience
        ):  # if not early stopped, ensure best model is loaded if saved per epoch
            if Path("temp_best_model.pt").exists():
                self.model.load_state_dict(torch.load("temp_best_model.pt"))
                Path("temp_best_model.pt").unlink(missing_ok=True)

        self._is_trained = True
        self.model.eval()
        with torch.no_grad():
            # Train metrics
            train_outputs = self.model(X_train_tensor)
            train_pred_probs = torch.softmax(train_outputs, dim=1)
            train_pred_int = torch.argmax(train_pred_probs, dim=1).cpu().numpy()

            # Test metrics
            test_outputs = self.model(X_test_tensor)
            test_pred_probs = torch.softmax(test_outputs, dim=1)
            test_pred_int = torch.argmax(test_pred_probs, dim=1).cpu().numpy()

            # Recalculate final val accuracy with best model
            final_val_outputs = self.model(X_val_tensor)
            final_val_pred_probs = torch.softmax(final_val_outputs, dim=1)
            final_val_pred_int = torch.argmax(final_val_pred_probs, dim=1).cpu().numpy()
            final_val_accuracy = accuracy_score(y_val_int, final_val_pred_int)

        return {
            "train_accuracy": accuracy_score(y_train_int, train_pred_int),
            "val_accuracy": final_val_accuracy,
            "test_accuracy": accuracy_score(y_test_int, test_pred_int),
            "test_loss": log_loss(
                y_test_int,
                test_pred_probs.cpu().numpy(),
                labels=np.arange(self.num_classes),
            ),
            # Calculate log loss for test
            "classification_report_test": classification_report(
                y_test_int, test_pred_int, zero_division=0
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:  # Returns probabilities
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)  # Raw logits
            probabilities = torch.softmax(
                outputs, dim=1
            )  # Apply softmax for probabilities
            return probabilities.cpu().numpy()

    def save_model(self, filename_prefix: str):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        torch.save(self.model.state_dict(), f"{filename_prefix}_model.pt")

    def load_model(self, filename_prefix: str):
        # Rebuild model structure first, then load state dict
        # self.model = self._build_model().to(self.device) # Should be already built by __init__
        self.model.load_state_dict(
            torch.load(f"{filename_prefix}_model.pt", map_location=self.device)
        )
        self.model.to(self.device)  # Ensure it's on the correct device
        # Infer input_dim and num_classes from loaded model if architecture allows
        # For nn.Sequential, it's harder to inspect directly without knowing layer names/indices
        # Assume self.input_dim and self.num_classes passed to __init__ are correct for the saved model
        self._is_trained = True
        self.model.eval()

    def _export_to_onnx_internal(self, output_path: str, X_sample: np.ndarray):
        self.model.eval()  # Ensure model is in eval mode

        # Create a wrapper model if the main model doesn't include softmax (for CrossEntropyLoss)
        class ModelWithSoftmax(nn.Module):
            def __init__(self, model_core, dim_softmax=1):
                super().__init__()
                self.model_core = model_core
                self.dim_softmax = dim_softmax

            def forward(self, x):
                logits = self.model_core(x)
                return torch.softmax(logits, dim=self.dim_softmax)

        onnx_export_model = ModelWithSoftmax(self.model).to(self.device)
        onnx_export_model.eval()

        dummy_input = torch.from_numpy(X_sample.astype(np.float32)).to(self.device)
        if (
            dummy_input.shape[1] != self.input_dim
        ):  # Ensure correct shape for dummy input
            dummy_input = torch.zeros(1, self.input_dim, dtype=torch.float32).to(
                self.device
            )

        torch.onnx.export(
            onnx_export_model,  # Export the wrapper with softmax
            dummy_input,
            output_path,
            input_names=["float_input"],
            output_names=["output"],  # Output is now probabilities
            dynamic_axes={
                "float_input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=13,
        )
        logging.info(f"PyTorch ONNX model saved to {output_path}")


class ScikitLearnStrategy(TextClassifierStrategy):
    def __init__(
        self, input_dim: int, num_classes: int
    ):  # num_classes not strictly needed by GBC but good for interface
        super().__init__(input_dim, num_classes)
        self.model = GradientBoostingClassifier(
            n_estimators=300,  # Reduced for faster example, adjust as needed
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            # n_iter_no_change=10, # For early stopping behavior
            # validation_fraction=0.1, # For n_iter_no_change
            # tol=1e-4,
            verbose=0,  # Set to 1 for more verbosity
        )

    def train(self, X_train, X_val, X_test, y_train_int, y_val_int, y_test_int) -> Dict:
        # Scikit-learn's GBC can use validation set for early stopping if configured,
        # but simpler fit just uses X_train, y_train. Let's use default fit.
        # For a more robust GBC, consider using X_val with `early_stopping_rounds` in XGBoost/LightGBM
        # or `n_iter_no_change` with `validation_fraction` in scikit-learn's GBC.
        # For now, we'll train on X_train and evaluate on X_val and X_test.

        self.model.fit(X_train, y_train_int)
        self._is_trained = True

        # Predictions (integer labels)
        train_pred_int = self.model.predict(X_train)
        val_pred_int = self.model.predict(X_val)
        test_pred_int = self.model.predict(X_test)

        # Probabilities for log loss
        train_pred_proba = self.model.predict_proba(X_train)
        val_pred_proba = self.model.predict_proba(X_val)
        test_pred_proba = self.model.predict_proba(X_test)

        # CV scores on combined data for a general performance idea
        # Note: This re-trains the model multiple times, can be slow.
        # Consider if this is essential here or can be optional.
        # combined_X = np.vstack((X_train, X_val, X_test))
        # combined_y = np.hstack((y_train_int, y_val_int, y_test_int))
        # cv_scores = cross_val_score( # Use a fresh model for CV
        #     GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        #     combined_X, combined_y, cv=3, scoring='accuracy' # Use a smaller CV fold
        # )
        # logging.info(f"Cross-validation scores (accuracy, 3-fold on combined data): {cv_scores}, Mean: {cv_scores.mean():.4f}")

        return {
            "train_accuracy": accuracy_score(y_train_int, train_pred_int),
            "val_accuracy": accuracy_score(y_val_int, val_pred_int),
            "test_accuracy": accuracy_score(y_test_int, test_pred_int),
            "test_loss": log_loss(
                y_test_int, test_pred_proba, labels=np.arange(self.num_classes)
            ),
            # "cv_mean_accuracy": cv_scores.mean(),
            "classification_report_test": classification_report(
                y_test_int, test_pred_int, zero_division=0
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:  # Returns probabilities
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        return self.model.predict_proba(X)  # Return probabilities

    def save_model(self, filename_prefix: str):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump(self.model, f"{filename_prefix}_model.pkl")

    def load_model(self, filename_prefix: str):
        self.model = joblib.load(f"{filename_prefix}_model.pkl")
        # Infer num_classes from loaded model if possible
        if hasattr(self.model, "n_classes_") and self.model.n_classes_ is not None:
            loaded_n_classes = self.model.n_classes_
            if (
                self.num_classes != loaded_n_classes and self.num_classes != 1
            ):  # num_classes might be 1 for binary in some contexts
                # For GBC, n_classes_ is the number of classes. If binary, it's 2.
                # If the model was trained as binary (1 output unit internally sometimes), it might show 2 here.
                # Be careful with this check if binary models are sometimes treated as num_classes=1.
                # Safest is that num_classes in strategy matches what model expects.
                if not (
                    self.num_classes == 2 and loaded_n_classes == 1
                ):  # GBC n_classes_ is 1 for binary in some internal logic
                    if self.num_classes != loaded_n_classes:
                        logging.warning(
                            f"Loaded Scikit-learn model n_classes_ {loaded_n_classes} differs from strategy's num_classes {self.num_classes}. Using model's."
                        )
                        self.num_classes = loaded_n_classes
        elif hasattr(self.model, "classes_"):  # Another way to get num_classes
            loaded_n_classes = len(self.model.classes_)
            if self.num_classes != loaded_n_classes:
                logging.warning(
                    f"Loaded Scikit-learn model found {loaded_n_classes} classes, differs from strategy's num_classes {self.num_classes}. Using model's."
                )
                self.num_classes = loaded_n_classes
        self._is_trained = True

    def _export_to_onnx_internal(self, output_path: str, X_sample: np.ndarray):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Ensure X_sample matches the model's expected input dimension
        # Scikit-learn models store n_features_in_
        if (
            hasattr(self.model, "n_features_in_")
            and X_sample.shape[1] != self.model.n_features_in_
        ):
            X_sample = np.zeros((1, self.model.n_features_in_), dtype=np.float32)
        elif X_sample.shape[1] != self.input_dim:  # Fallback
            X_sample = np.zeros((1, self.input_dim), dtype=np.float32)

        initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]

        # For GBC, target_opset >= 12 is generally better.
        # skl2onnx converts predict_proba output to be ZipMap if not handled.
        # We want raw probabilities usually.
        # Options for output: 'predict_proba', 'label'. We want probabilities.
        options = {id(self.model): {"zipmap": False}}  # To get raw probabilities array

        onnx_model = convert_sklearn(
            self.model, initial_types=initial_type, target_opset=13, options=options
        )
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logging.info(f"Scikit-learn ONNX model saved to {output_path}")

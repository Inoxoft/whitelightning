import json
import logging
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
import joblib

from .base import TextClassifierStrategy


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TensorFlowStrategyBinary(TextClassifierStrategy):
    def __init__(
        self,
        input_dim: int = 5000,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):
        self.model = self._build_model(input_dim)
        self._is_trained = False
        self.input_dim = input_dim  # Store input dimension for ONNX export
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path

    def _build_model(self, input_dim: int) -> Model:
        inputs = Input(shape=(input_dim,), name="float_input")
        x = Dense(512, activation="relu")(inputs)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation="sigmoid", name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        history = self.model.fit(
            X_train,
            y_train.astype(np.float32),
            validation_data=(X_test, y_test.astype(np.float32)),
            epochs=10,
            batch_size=32,
            verbose=1,
        )
        self._is_trained = True

        train_pred = (self.model.predict(X_train) > 0.5).astype(int)
        test_pred = (self.model.predict(X_test) > 0.5).astype(int)

        return {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "classification_report": classification_report(y_test, test_pred),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        return (self.model.predict(X)).astype(float)

    def save_model(self):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        self.model.save(f"{self.output_path}/model.h5")
        self.save_model_vocab_and_scaler()
        self.export_to_onnx()

    def load_model(self, filename_prefix: str):
        self.model = tf.keras.models.load_model(f"{filename_prefix}_model.h5")
        self._is_trained = True

    def export_to_onnx(self):
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")
        import tf2onnx

        spec = (tf.TensorSpec((None, self.input_dim), tf.float32, name="float_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=spec, opset=13
        )
        with open(self.output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        logging.info(f"ONNX model saved to {self.output_path}")


class PyTorchStrategyBinary(TextClassifierStrategy):
    def __init__(
        self,
        input_dim: int = 5000,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_dim).to(self.device)
        self._is_trained = False
        self.input_dim = input_dim  # Store input dimension for ONNX export
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path

    def _build_model(self, input_dim: int) -> nn.Module:
        class TextClassifier(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.layers(x)

        return TextClassifier(input_dim)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(10):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logging.info(
                f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}"
            )

        self._is_trained = True

        self.model.eval()
        with torch.no_grad():
            train_pred = (self.model(X_train_tensor) > 0.5).cpu().numpy().astype(int)
            test_pred = (self.model(X_test_tensor) > 0.5).cpu().numpy().astype(int)

        return {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "classification_report": classification_report(y_test, test_pred),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy().astype(float)

    def save_model(self):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        torch.save(self.model.state_dict(), f"{self.output_path}/model.pt")
        self.save_model_vocab_and_scaler()
        self.export_to_onnx()

    def load_model(self, filename_prefix: str):
        self.model.load_state_dict(torch.load(f"{filename_prefix}_model.pt"))
        self.model.to(self.device)
        self._is_trained = True

    def export_to_onnx(self):
        output_path = f"{self.output_path}/model.onnx"
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")
        self.model.eval()
        dummy_input = torch.zeros(1, self.input_dim).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["float_input"],
            output_names=["output"],
            dynamic_axes={
                "float_input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=13,
        )
        logging.info(f"ONNX model saved to {output_path}")


class ScikitLearnStrategyBinary(TextClassifierStrategy):
    def __init__(
        self,
        input_dim: int = 5000,
        vocab: dict = {},
        scaler: dict = {},
        output_path: str = "",
        **kwargs,
    ):
        self.model = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_iter_no_change=10,
            verbose=1,
        )
        self._is_trained = False
        self.input_dim = input_dim
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        self.model.fit(X_train, y_train)
        self._is_trained = True

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        cv_scores = cross_val_score(
            self.model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=5
        )

        return {
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "classification_report": classification_report(y_test, test_pred),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before predicting")
        return self.model.predict(X)

    def save_model(self):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump(self.model, f"{self.output_path}/model.pkl")
        self.save_model_vocab_and_scaler()
        self.export_to_onnx()

    def load_model(self, filename_prefix: str):
        self.model = joblib.load(f"{filename_prefix}_model.pkl")
        self._is_trained = True

    def export_to_onnx(self):
        output_path = f"{self.output_path}_model.onnx"
        if not self._is_trained:
            raise RuntimeError("Model must be trained before exporting to ONNX")
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, self.input_dim]))]
        onnx_model = convert_sklearn(
            self.model, initial_types=initial_type, target_opset=13
        )
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logging.info(f"ONNX model saved to {output_path}")

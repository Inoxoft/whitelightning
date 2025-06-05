import abc
import json

import numpy as np


class TextClassifierStrategy(abc.ABC):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        vocab: dict,
        scaler: dict,
        output_path: str,
    ):
        self.input_dim = input_dim  # This is the KEY variable for model input shape
        self.num_classes = num_classes
        self.model = None
        self.vocab = vocab
        self.scaler = scaler
        self.output_path = output_path

    def build_model(self):
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self):
        pass

    def load(self, path_prefix: str):
        pass

    def export_to_onnx(self):
        pass

    def save_model_vocab_and_scaler(self):
        with open(f"{self.output_path}/vocab.json", "w") as f:
            json.dump(self.vocab, f)
        with open(f"{self.output_path}/scaler.json", "w") as f:
            json.dump(self.scaler, f)

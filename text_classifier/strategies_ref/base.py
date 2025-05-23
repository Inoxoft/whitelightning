import abc

import numpy as np


class TextClassifierStrategy(abc.ABC):
    def __init__(self, input_dim: int, num_classes: int):
        self.input_dim = input_dim  # This is the KEY variable for model input shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path_prefix: str):
        pass

    def load(self, path_prefix: str):
        pass

    def export_to_onnx(self, output_path: str, X_sample: np.ndarray):
        pass

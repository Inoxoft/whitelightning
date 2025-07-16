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
        self.input_dim = input_dim  
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

    def load_vocab_and_scaler(self, path_prefix: str):
        """Load vocabulary and scaler from saved JSON files"""
        try:
            with open(f"{path_prefix}/vocab.json", "r") as f:
                self.vocab = json.load(f)
            with open(f"{path_prefix}/scaler.json", "r") as f:
                self.scaler = json.load(f)
            
            # Update input_dim based on loaded vocabulary
            if self.vocab and "vocab" in self.vocab:
                actual_vocab_size = len(self.vocab["vocab"])
                if actual_vocab_size > 0 and actual_vocab_size != self.input_dim:
                    print(f"Updating input_dim from {self.input_dim} to {actual_vocab_size} based on loaded vocabulary")
                    self.input_dim = actual_vocab_size
        except FileNotFoundError as e:
            print(f"Warning: Could not load vocab/scaler files: {e}")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse vocab/scaler JSON: {e}")

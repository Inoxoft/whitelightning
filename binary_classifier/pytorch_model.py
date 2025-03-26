import logging
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

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


class BaseTextClassifier:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 4)):
        logging.info("Initializing BaseTextClassifier")
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_trained = False

    def load_and_preprocess(self, csv_path: str) -> Tuple[pd.Series, pd.Series]:
        logging.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if DATA_COLUMN_NAME not in df.columns or LABEL_COLUMN_NAME not in df.columns:
            raise ValueError(f"CSV must contain '{DATA_COLUMN_NAME}' and '{LABEL_COLUMN_NAME}' columns")

        X = df[DATA_COLUMN_NAME]
        y = df[LABEL_COLUMN_NAME]

        if y.dtype == 'object':
            logging.info("Converting labels to numeric")
            y = y.replace({'negative': 0, 'positive': 1})
            if y.isnull().any():
                raise ValueError("Labels contain null values after conversion")
            y = pd.to_numeric(y, errors='raise')

        return X, y

    def extract_features(self, X: pd.Series, training: bool = True) -> np.ndarray:
        if training:
            X_tfidf = self.vectorizer.fit_transform(X).toarray()
            X_scaled = self.scaler.fit_transform(X_tfidf)
        else:
            X_tfidf = self.vectorizer.transform(X).toarray()
            X_scaled = self.scaler.transform(X_tfidf)
        return X_scaled.astype(np.float32)

    def save_preprocessors(self, filename_prefix: str):
        import joblib
        joblib.dump(self.vectorizer, f"{filename_prefix}_vectorizer.pkl")
        joblib.dump(self.scaler, f"{filename_prefix}_scaler.pkl")

    def load_preprocessors(self, filename_prefix: str):
        import joblib
        self.vectorizer = joblib.load(f"{filename_prefix}_vectorizer.pkl")
        self.scaler = joblib.load(f"{filename_prefix}_scaler.pkl")


# PyTorch-specific binary classifier
class BinaryClassifierPytorch(BaseTextClassifier):
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 4)):
        super().__init__(max_features, ngram_range)
        logging.info("Initializing BinaryClassifierPytorch")
        self.model = self._build_model(max_features).to(self.device)

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
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.layers(x)

        return TextClassifier(input_dim)

    def train(self, csv_path: str, test_size: float = 0.2, epochs: int = 10, batch_size: int = 32) -> dict:
        logging.info("Starting model training")
        X, y = self.load_and_preprocess(csv_path)
        X_processed = self.extract_features(X, training=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )

        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(self.device)

        # Create DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self._is_trained = True

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            train_pred = (self.model(X_train_tensor) > 0.5).cpu().numpy().astype(int)
            test_pred = (self.model(X_test_tensor) > 0.5).cpu().numpy().astype(int)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        logging.info(f"Training Accuracy: {train_accuracy:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")

        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, test_pred)
        }

    def predict(self, new_data: Union[str, List[str]]) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        if isinstance(new_data, str):
            new_data = [new_data]

        X_processed = self.extract_features(new_data, training=False)
        X_tensor = torch.FloatTensor(X_processed).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = (self.model(X_tensor) > 0.5).cpu().numpy().astype(int)
        return predictions

    def save_model(self, filename_prefix: str):
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        logging.info(f"Saving model to {filename_prefix}")
        torch.save(self.model.state_dict(), f"{filename_prefix}_model.pt")
        self.save_preprocessors(filename_prefix)

    def load_model(self, filename_prefix: str):
        logging.info(f"Loading model from {filename_prefix}")
        self.model.load_state_dict(torch.load(f"{filename_prefix}_model.pt"))
        self.model.to(self.device)
        self.load_preprocessors(filename_prefix)
        self._is_trained = True


def run_training():
    classifier = BinaryClassifierPytorch()
    classifier.train(f'{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv')
    classifier.save_model(f"{MODELS_PATH}/{MODEL_PREFIX}")

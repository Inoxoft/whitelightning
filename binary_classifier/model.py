import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import pickle
from settings import MODEL_PREFIX, MODELS_PATH, TRAINING_DATA_PATH, DATA_COLUMN_NAME, LABEL_COLUMN_NAME

class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        self.vocab = {word: i + 1 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
        self.vocab["<UNK>"] = len(self.vocab) + 1

    def encode(self, text):
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.lower().split()]

    def __len__(self):
        return len(self.vocab)


class DatasetLoader(Dataset):
    def __init__(self, texts, labels, vocab, max_len=20):
        self.texts = [vocab.encode(text)[:max_len] + [0] * (max_len - len(vocab.encode(text))) for text in texts]
        self.labels = [1 if label == "spam" else 0 for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


class BinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        return self.fc(x).squeeze()


def train_model(model, train_loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


def run_training(batch_size=64, epochs=50, test_size=0.01, embed_dim=64, hidden_dim=128, custom_models_path=None):
    df = pd.read_csv(f"{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv")
    train_texts, test_texts, train_labels, test_labels = train_test_split(df[DATA_COLUMN_NAME], df[LABEL_COLUMN_NAME], test_size=test_size,
                                                                          random_state=42)
    vocab = Vocab(train_texts)
    train_data = DatasetLoader(train_texts, train_labels, vocab)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = BinaryClassifier(len(vocab) + 1, embed_dim, hidden_dim)
    train_model(model, train_loader, epochs)

    torch.save(model.state_dict(), f"{custom_models_path or MODELS_PATH}{MODEL_PREFIX}.pth")

    with open(f"{custom_models_path or MODELS_PATH}{MODEL_PREFIX}vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("Model and vocabulary saved successfully.")

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import onnx
import tensorflow as tf
import tf2onnx
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

def load_and_clean(path, allowed_labels):
    df = pd.read_csv(path, on_bad_lines="skip").dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.replace('"', "").str.replace("'", "").str.capitalize()
    df = df[df["label"].isin([label.capitalize() for label in allowed_labels])]
    return df

def load_data(train_path, test_path, labels):
    train_df = load_and_clean(train_path, labels)
    test_df = load_and_clean(test_path, labels)
    label_encoder = LabelEncoder()
    train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
    test_df["label_enc"] = label_encoder.transform(test_df["label"])
    return train_df, test_df, label_encoder

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class CustomClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(CustomClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def train_torch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_torch(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            out = model(input_ids)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)

def export_model(model, example_input, language, text_type):
    os.makedirs("models", exist_ok=True)
    onnx_path = f"models/{text_type}_classifier({language}).onnx"
    torch.onnx.export(
        model,
        example_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17
    )
    print(f"✅ Exported ONNX model to {onnx_path}")

def train_tensorflow(X_train, X_test, y_train, y_test, vocab_size, max_len, num_classes, language, text_type):
    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{text_type}_classifier({language}).h5")

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input")
    outputs = model(inputs)
    full_model = tf.keras.Model(inputs, outputs)
    spec = (tf.TensorSpec((None, max_len), tf.int32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(full_model, input_signature=spec, opset=17)

    with open(f"models/{text_type}_classifier({language}).onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("✅ TensorFlow model exported")

def train_sklearn(train_df, test_df, label_encoder, language, text_type):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(train_df["text"], train_df["label_enc"])
    preds = pipeline.predict(test_df["text"])
    acc = accuracy_score(test_df["label_enc"], preds)
    print(f"✅ Sklearn Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    dump(pipeline, f"models/{text_type}_classifier({language})_sklearn_model.joblib")

    with open(f"models/{text_type}_classifier({language})_scaler.json", "w") as f:
        json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

    vocab = {k: int(v) for k, v in pipeline.named_steps["tfidf"].vocabulary_.items()}
    with open(f"models/{text_type}_classifier({language})_tokenizer.json", "w") as f:
        json.dump(vocab, f)

    initial_type = [("input", StringTensorType([None]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    with open(f"models/{text_type}_classifier({language}).onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("✅ Sklearn model exported to ONNX")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--platform", choices=["torch", "tensorflow", "sklearn"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_model_from_params(
        language=args.language,
        text_type=args.type,
        labels=[l.strip() for l in args.labels.split(",") if l.strip()],
        platform=args.platform,
        epochs=args.epochs,
    )

def train_model_from_params(language, text_type, labels, platform, epochs):
    train_path = f"training_data/{text_type}_train_{language}.csv"
    test_path = f"testing_data/{text_type}_test_{language}.csv"
    train_df, test_df, label_encoder = load_data(train_path, test_path, labels)

    if platform == "sklearn":
        train_sklearn(train_df, test_df, label_encoder, language, text_type)
        return

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(pd.concat([train_df["text"], test_df["text"]]))
    max_len = 30

    def tokenize(df):
        return pad_sequences(tokenizer.texts_to_sequences(df["text"]), maxlen=max_len, padding='post')

    X_train, X_test = tokenize(train_df), tokenize(test_df)
    y_train, y_test = train_df["label_enc"].values, test_df["label_enc"].values

    if platform == "tensorflow":
        train_tensorflow(X_train, X_test, y_train, y_test, vocab_size=10000, max_len=max_len, num_classes=len(label_encoder.classes_), language=language, text_type=text_type)
    else:
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CustomClassifier(vocab_size=10000, embedding_dim=64, hidden_dim=64, num_classes=len(label_encoder.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            loss = train_torch(model, train_loader, optimizer, criterion, device)
            acc = evaluate_torch(model, test_loader, device)
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        torch.save(model.state_dict(), f"models/{text_type}_classifier({language}).pt")
        export_model(model, torch.tensor(X_test[:1], dtype=torch.long), language, text_type)

    with open(f"models/{text_type}_classifier({language})_tokenizer.json", "w") as f:
        json.dump(tokenizer.word_index, f)
    with open(f"models/{text_type}_classifier({language})_scaler.json", "w") as f:
        json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

if __name__ == "__main__":
    main()
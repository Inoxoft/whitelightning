import argparse
import os
import json
import tensorflow as tf
import tf2onnx
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

def load_and_clean(path, allowed_labels):
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.replace('"', "").str.replace("'", "")
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].str.capitalize()
    df = df[df["label"].isin([label.capitalize() for label in allowed_labels])]

    return df

def prepare_data(train_df, test_df, vocab_size, max_len):
    label_encoder = LabelEncoder()
    train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
    test_df["label_enc"] = label_encoder.transform(test_df["label"])

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(pd.concat([train_df["text"], test_df["text"]]))

    def tokenize(df):
        return pad_sequences(tokenizer.texts_to_sequences(df["text"]), maxlen=max_len, padding='post')

    X_train = tokenize(train_df)
    X_test = tokenize(test_df)
    y_train = train_df["label_enc"].values
    y_test = test_df["label_enc"].values

    return X_train, X_test, y_train, y_test, tokenizer, label_encoder

def build_model(vocab_size, max_len, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def export_all(model, tokenizer, label_encoder, language, text_type, max_len):
    os.makedirs("models", exist_ok=True)

    h5_path = f"models/{text_type}_classifier({language}).h5"
    model.save(h5_path)
    print(f"💾 Saved model: {h5_path}")

    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input")
    outputs = model(inputs)
    full_model = tf.keras.Model(inputs, outputs)
    spec = (tf.TensorSpec((None, max_len), tf.int32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(full_model, input_signature=spec, opset=13)

    onnx_path = f"models/{text_type}_classifier({language}).onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"💾 Saved ONNX model: {onnx_path}")

    tokenizer_path = f"models/{text_type}_classifier({language})_tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(tokenizer.word_index, f)
    print(f"💾 Saved tokenizer: {tokenizer_path}")

    label_map_path = f"models/{text_type}_classifier({language})_scaler.json"
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)
    print(f"💾 Saved label map: {label_map_path}")

def main():
    parser = argparse.ArgumentParser(description="Train classification model from CSV with dynamic labels")
    parser.add_argument("--language", required=True, help="Language of the dataset")
    parser.add_argument("--type", required=True, help="Type of text: e.g. news, review, comment")
    parser.add_argument("--labels", required=True, help="Comma-separated list of classification labels")

    args = parser.parse_args()

    language = args.language
    text_type = args.type
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    print(f"✅ Using labels: {labels}")

    TRAIN_PATH = f"training_data/{text_type}_train_{language}.csv"
    TEST_PATH = f"testing_data/{text_type}_test_{language}.csv"

    VOCAB_SIZE = 10000
    MAX_LEN = 30

    train_df = load_and_clean(TRAIN_PATH, labels)
    test_df = load_and_clean(TEST_PATH, labels)

    X_train, X_test, y_train, y_test, tokenizer, label_encoder = prepare_data(train_df, test_df, VOCAB_SIZE, MAX_LEN)
    model = build_model(VOCAB_SIZE, MAX_LEN, len(label_encoder.classes_))
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    export_all(model, tokenizer, label_encoder, language, text_type, MAX_LEN)

if __name__ == "__main__":
    main()

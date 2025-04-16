from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

def build_model(vocab_size: int, max_len: int, num_classes: int):
    return Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

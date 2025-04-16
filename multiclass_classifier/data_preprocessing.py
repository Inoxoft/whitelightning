from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def prepare_data(train_df, test_df, vocab_size: int, max_len: int):
    label_encoder = LabelEncoder()
    train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
    test_df["label_enc"] = label_encoder.transform(test_df["label"])

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(pd.concat([train_df["text"], test_df["text"]]))

    def tokenize(df):
        return pad_sequences(
            tokenizer.texts_to_sequences(df["text"]),
            maxlen=max_len, padding='post', truncating='post'
        )

    return (
        tokenize(train_df), tokenize(test_df),
        train_df["label_enc"].values, test_df["label_enc"].values,
        tokenizer, label_encoder
    )


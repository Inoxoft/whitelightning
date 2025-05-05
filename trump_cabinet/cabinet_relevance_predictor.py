import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

MODEL_PATH = "cabinet_model.h5"
TOKENIZER_PATH = "cabinet_tokenizer.json"
MAX_SEQUENCE_LENGTH = 200

print("🔄 Loading model and tokenizer...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())

cabinet_keys = [
    "JD_Vance", "Marco_Rubio", "Scott_Bessent", "Pete_Hegseth", "Pam_Bondi",
    "Doug_Burgum", "Brooke_Rollins", "Howard_Lutnick", "Lori_Chavez_DeRemer",
    "Robert_F_Kennedy_Jr", "Scott_Turner", "Sean_Duffy", "Chris_Wright",
    "Linda_McMahon", "Doug_Collins", "Kristi_Noem", "Lee_Zeldin", "Tulsi_Gabbard",
    "John_Ratcliffe", "Jamieson_Greer", "Kelly_Loeffler", "Russell_Vought", "Susie_Wiles"
]


def predict_relevance(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    preds = model.predict(padded)[0]
    return {k: float(f"{v:.2f}") for k, v in zip(cabinet_keys, preds)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cabinet Relevance Model")
    parser.add_argument("--text", "-t", type=str, required=True, help="News article text")
    args = parser.parse_args()

    print("📰 Input:", args.text)
    results = predict_relevance(args.text)

    print("\n📊 Relevance Scores:")
    for k, v in results.items():
        print(f"{k:25}: {v}")


import pandas as pd
import torch
import pickle
from binary_classifier.model import BinaryClassifier
from settings import MODEL_PREFIX, MODELS_PATH, TESTING_DATA_PATH, RESULTS_PATH, DATA_COLUMN_NAME, LABEL_COLUMN_NAME, POSITIVE_LABEL, NEGATIVE_LABEL

def predict(model, vocab, text):
    if not isinstance(text, str):
        text = ""
    with torch.no_grad():
        encoded_text = vocab.encode(text)[:20] + [0] * (20 - len(vocab.encode(text)))
        input_tensor = torch.tensor([encoded_text], dtype=torch.long)
        output = model(input_tensor).item()
        return POSITIVE_LABEL if output > 0.5 else NEGATIVE_LABEL
        #return f"{label} output: {output:.4f}"

def run_model(custom_models_path=None, custom_results_path=None, embed_dim=64, hidden_dim=128):
    with open(f"{custom_models_path or MODELS_PATH}{MODEL_PREFIX}vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = BinaryClassifier(len(vocab) + 1, embed_dim, hidden_dim)
    model.load_state_dict(torch.load(f"{custom_models_path or MODELS_PATH}{MODEL_PREFIX}.pth"))
    model.eval()
    df = pd.read_csv(f"{TESTING_DATA_PATH}{MODEL_PREFIX}_dataset.csv", encoding="utf-8")
    df["prediction"] = df.apply(lambda x: predict(model, vocab, x[DATA_COLUMN_NAME]), axis=1)
    df.to_csv(f"{custom_results_path or RESULTS_PATH}{MODEL_PREFIX}_predictions.csv", index=False)
    print(f"Predictions saved to {MODEL_PREFIX}_predictions.csv")

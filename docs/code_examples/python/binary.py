import json
import numpy as np
import onnxruntime as ort


# --- Preprocessing: TF-IDF + Scaling ---
def preprocess_text(text, vocab_file, scaler_file):
    # Load vocabulary and IDF weights
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    with open(scaler_file, "r") as f:
        scaler = json.load(f)
    idf = vocab["idf"]
    word2idx = vocab["vocab"]
    mean = np.array(scaler["mean"], dtype=np.float32)
    scale = np.array(scaler["scale"], dtype=np.float32)

    # Compute term frequency (TF)
    tf = np.zeros(len(word2idx), dtype=np.float32)
    words = text.lower().split()
    for word in words:
        idx = word2idx.get(word)
        if idx is not None:
            tf[idx] += 1
    if tf.sum() > 0:
        tf = tf / tf.sum()  # Normalize TF

    # TF-IDF
    tfidf = tf * np.array(idf, dtype=np.float32)

    # Standardize
    tfidf_scaled = (tfidf - mean) / scale
    return tfidf_scaled.astype(np.float32)


# Example usage
text = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize!"  # Example text
vector = preprocess_text(
    text,
    "../../../text_classifier/models/spam-sms-tensorflow_classifier(en)_vocab.json",
    "../../../text_classifier/models/spam-sms-tensorflow_classifier(en)_scaler.json",
)  # 5000-dim float32

# --- ONNX Inference ---
session = ort.InferenceSession(
    "../../../text_classifier/models/spam-sms-tensorflow_classifier(en).onnx"
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_data = vector.reshape(1, -1)
outputs = session.run([output_name], {input_name: input_data})

probability = outputs[0][0][0]  # Probability of positive class
print(f"Python ONNX output: Probability = {probability:.4f}")

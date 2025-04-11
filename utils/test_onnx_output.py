import json
import numpy as np
import onnxruntime as ort

from settings import MODELS_PATH, MODEL_PREFIX


def preprocess_text(text, vocab_file, scaler_file):
    with open(vocab_file, "r") as f:
        tfidf_data = json.load(f)
    vocab, idf = tfidf_data["vocab"], np.array(tfidf_data["idf"])

    with open(scaler_file, "r") as f:
        scaler_data = json.load(f)
    mean, scale = np.array(scaler_data["mean"]), np.array(scaler_data["scale"])

    vector = np.zeros(5000, dtype=np.float32)
    words = text.lower().split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    for word, count in word_counts.items():
        if word in vocab:
            vector[vocab[word]] = count * idf[vocab[word]]

    # Scale
    vector = (vector - mean) / scale
    return vector


# Test
test_string = [
        "Don’t you think racial pride strengthens us?",
        "Why do people resist admitting Judaism is elite?",
        "Isn’t it obvious that men conquer while women nurture?",
        "Since race shapes fate, why fight it?",
        "Don’t you agree that faith makes life worth living?",
        "What’s your favorite way to spend a weekend?",
        "How do you feel about the legalization of recreational marijuana?",
        "What are some benefits of meditation you’ve noticed?",
        "Do you think religion influences politics too much?",
        "How do you define success in life?",
        "What’s your opinion on gender roles in modern society?",
        "How does race affect job opportunities in your experience?",
        "What’s the best book you’ve read recently?",
    ]
for text in test_string:
    vector = preprocess_text(text, f"../{MODELS_PATH}{MODEL_PREFIX}_vocab.json", f"../{MODELS_PATH}{MODEL_PREFIX}_scaler.json")
    session = ort.InferenceSession(f"../{MODELS_PATH}{MODEL_PREFIX}_pytorch.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_data = vector.reshape(1, 5000).astype(np.float32)
    outputs = session.run([output_name], {input_name: input_data})
    print("Python ONNX output:", outputs[0][0][0])
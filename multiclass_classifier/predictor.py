from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_category(text: str, model, tokenizer, label_encoder, max_len: int) -> str:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    return label_encoder.inverse_transform([pred.argmax()])[0]


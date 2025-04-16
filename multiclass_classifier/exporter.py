import tensorflow as tf
import tf2onnx
import json

def save_model_onnx(model, max_len, language):
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input")
    outputs = model(inputs)
    model_func = tf.keras.Model(inputs, outputs)
    spec = (tf.TensorSpec((None, max_len), tf.int32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model_func, input_signature=spec, opset=13)
    with open(f"../multiclas_models/news_classifier({language}).onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

def save_label_map(label_encoder, language):
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(f"../multiclas_models/news_classifier({language})_scaler.json", "w") as f:
        json.dump(label_map, f)

def save_vocab_tokens(tokenizer, language):
    with open(f"../multiclas_models/news_classifier({language})_vocab.json", "w") as f:
        json.dump(tokenizer.word_index, f)
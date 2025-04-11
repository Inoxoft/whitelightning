import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout # type: ignore
import tf2onnx
train_df = pd.read_csv("Backend/whitelightning.ai/training_data/news_train.csv")
test_df = pd.read_csv("Backend/whitelightning.ai/testing_data/news_test.csv")


def clean_labels(df):
    df["label"] = df["label"].astype(str).str.strip().str.replace('"', '', regex=False).str.replace("'", '', regex=False)

clean_labels(train_df)
clean_labels(test_df)

all_texts = pd.concat([train_df["text"], test_df["text"]])


label_encoder = LabelEncoder()
train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])
test_df["label_enc"] = label_encoder.transform(test_df["label"])


vocab_size = 10000
max_len = 30  


tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(all_texts)


X_train_seq = tokenizer.texts_to_sequences(train_df["text"])
X_test_seq = tokenizer.texts_to_sequences(test_df["text"])

X_train = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

y_train = train_df["label_enc"].values
y_test = test_df["label_enc"].values

print(y_train)
print(y_test)


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 класів
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Progress")
plt.grid(True)
plt.show()

model.save("news_classifier_model.h5")
model = tf.keras.models.load_model("news_classifier_model.h5")
inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input")
outputs = model(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
spec = (tf.TensorSpec((None, max_len), tf.int32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
with open("news_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

import json
with open("news_classifier_vocab.json", "w") as f:
    json.dump(tokenizer.word_index, f)


label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

with open("news_classifier_scaler.json", "w") as f:
    json.dump(label_map, f)

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("news_classifier.onnx")
input_name = session.get_inputs()[0].name

text = "The new Netflix series has taken the world by storm."

seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
result = session.run(None, {input_name: padded.astype(np.int32)})
class_labels = [
    "Business", "Education", "Entertainment", "Environment", "Health",
    "Politics", "Science", "Sports", "Technology", "World"
]

output = result[0]
predicted_class = np.argmax(output, axis=1)[0]
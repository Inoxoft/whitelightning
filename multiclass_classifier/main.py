from data_loader import load_and_clean,ALLOWED_LABELS
from data_preprocessing import prepare_data
from model_builder import build_model
from predictor import predict_category
from exporter import save_model_onnx, save_label_map,save_vocab_tokens
import tensorflow as tf

language = "hindi"

TRAIN_PATH = f"../training_data/news_train_{language}.csv"
TEST_PATH = f"../testing_data/news_test_{language}.csv"

VOCAB_SIZE = 10000
MAX_LEN = 30
NUM_CLASSES = len(ALLOWED_LABELS)

train_df = load_and_clean(TRAIN_PATH)
test_df = load_and_clean(TEST_PATH)

X_train, X_test, y_train, y_test, tokenizer, label_encoder = prepare_data(train_df, test_df, VOCAB_SIZE, MAX_LEN)

model = build_model(VOCAB_SIZE, MAX_LEN, NUM_CLASSES)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save(f"../multiclas_models/news_classifier({language})_model.h5")
model = tf.keras.models.load_model(f"../multiclas_models/news_classifier({language})_model.h5")

save_model_onnx(model, MAX_LEN, language)
save_label_map(label_encoder, language)
save_vocab_tokens(tokenizer, language)

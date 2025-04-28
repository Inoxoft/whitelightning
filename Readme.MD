🧠 Multiclass Classifier Agent
Overview
The Multiclass Classifier Agent is a modular CLI pipeline for generating datasets and training text classification models (multiclass) in multiple frameworks: PyTorch, TensorFlow, or sklearn.
It supports multilingual data, ONNX export, tokenizer and label mapping for easy deployment.
Data generation is fully automatic via LLM by parsing a simple prompt!

🚀 Workflow
Prompt Parsing (extract text type and labels via LLM)

Dataset Generation (via LLM API)

Model Training (Torch, TensorFlow, or Sklearn)

ONNX Export + Artifacts (tokenizer, label map)

🛠️ CLI Usage
bash
Копіювати
Редагувати
python -m agents.multiclass_classifier_agent \
  -p="Detect news topic in tree types: Sport, Business, Health"
Optional advanced parameters:


Name	Description	Default
--lang	Language of dataset generation	english
--platform	Framework to train (torch, tensorflow, sklearn)	tensorflow
--batches	Total number of batches to generate	300
--batch-size	Number of parallel LLM requests at once	20
--epochs	Number of training epochs	10
🧩 How It Works
-p prompt is parsed automatically by LLM into:

text_type (e.g., news, review, comment, or fallback generic)

labels (list of classes)

The agent generates a synthetic dataset accordingly.

It trains a model on selected framework.

It exports the model to ONNX for easy deployment.

It saves a tokenizer and label map for inference.

📦 Output Structure
pgsql
Копіювати
Редагувати
training_data/
  └── {type}_train_{language}.csv
testing_data/
  └── {type}_test_{language}.csv
models/
├── {type}_classifier({language}).onnx
├── {type}_classifier({language}).h5              # (TensorFlow only)
├── {type}_classifier({language}).pt              # (Torch only)
├── {type}_classifier({language})_tokenizer.json
├── {type}_classifier({language})_scaler.json
├── {type}_classifier({language})_sklearn_model.joblib (if sklearn)
✨ Features
✅ End-to-end pipeline (generate ➔ train ➔ export)

🌍 Multilingual support (any LLM-supported language)

🧩 Framework-agnostic training: Torch / TensorFlow / Sklearn

📤 ONNX model export for browser or server deployment

🧠 Automatic tokenizer + label mapping export

🛠 Fast CLI prototyping for any multiclass text classification task

🧪 Examples
Simple News Classifier
bash
Копіювати
Редагувати
python -m agents.multiclass_classifier_agent \
  -p="Detect news topic in tree types: Politics, Sports, Health"
Hate Speech Classifier in Ukrainian (Torch)
bash
Копіювати
Редагувати
python -m agents.multiclass_classifier_agent \
  -p="Detect hate speech in tree types: Hate, Not_Hate" \
  --lang ukrainian \
  --platform torch \
  --batches 400 \
  --batch-size 20 \
  --epochs 10
Review Sentiment Classifier (TensorFlow)
bash
Копіювати
Редагувати
python -m agents.multiclass_classifier_agent \
  -p="Detect review sentiment in tree types: Positive, Negative, Neutral" \
  --platform tensorflow
Comment Tone Classifier (Sklearn)
bash
Копіювати
Редагувати
python -m agents.multiclass_classifier_agent \
  -p="Detect comment tone in tree types: Sarcastic, Serious, Playful" \
  --platform sklearn
📤 Deployment Tips
Use the exported .onnx model with ONNXRuntime for inference.

Load the .json tokenizer and scaler for text preprocessing.

Deploy easily in FastAPI, React, or any environment supporting ONNX models.

Fine-tune or retrain with real-world data for production readiness.

📝 Notes
The labels extracted by the LLM must match between dataset generation and model training.

TensorFlow and PyTorch models are automatically converted to ONNX.

Sklearn uses a fast pipeline (TF-IDF + Logistic Regression) and also exports to ONNX.

Parallel batch generation speeds up dataset creation drastically.


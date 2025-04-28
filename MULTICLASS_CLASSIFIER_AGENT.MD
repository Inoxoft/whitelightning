# 🧠 Multiclass Classifier Agent

## Overview
The **Multiclass Classifier Agent** is a modular CLI pipeline for generating datasets and training text classification models (multiclass) in multiple frameworks: **PyTorch**, **TensorFlow**, or **sklearn**. It supports multilingual data, ONNX export, and tokenization/label mapping for deployment

---

## 🚀 Workflow

1. **Data Generation** (via LLM)
2. **Model Training** (`--platform`: `torch`, `tensorflow`, or `sklearn`)
3. **ONNX Export + Artifacts** (tokenizer, label map)

---

## 🛠️ CLI Usage

### 1. Generate Dataset
```bash
python -m agents.multiclass_classifier_agent \
  -p="Detect news topic in tree types: Sport, Business, Health"
```

### 2. Optional advanced parameters:
```bash
python -m agents.multiclass_classifier_agent \
  --p="Detect news topic in tree types: Sport, Business, Health"\
  --lang="german" \
  --platform="torch" \
  --epochs=10
```

---

## 🔧 Parametes

| Name            | Description                                              | Required | Default           | Example                                          |
|-----------------|----------------------------------------------------------|----------|-------------------|--------------------------------------------------|
| `-p, --prompt`  | Task description to extract text type and labels from     | ✅       | -                 | `"Detect news topic in tree types: Sport, Business, Health"` |
| `--lang`        | Language of the dataset generation                        | ❌       | `english`         | `english`, `ukrainian`, `german`                 |
| `--platform`    | Framework for model training                              | ❌       | `tensorflow`      | `torch`, `tensorflow`, `sklearn`                 |
| `--batches`     | Total number of batches to generate                       | ❌       | `300`             | `400`, `500`                                     |
| `--batch-size`  | Number of parallel LLM requests per group                 | ❌       | `20`              | `20`, `30`, `50`                                 |
| `--epochs`      | Number of training epochs                                 | ❌       | `10`              | `5`, `10`, `15`                                  |
| `--model`       | LLM model to use for dataset generation                   | ❌       | `gpt-3.5-turbo`   | `gpt-3.5-turbo`, `gpt-4`, etc.                   |



## 📦 Output Structure

```
training_data/
  └── {type}_train_{language}.csv
testing_data/
  └── {type}_test_{language}.csv
models/
├── {type}_classifier({language}).onnx
├── {type}_classifier({language}).h5              # (TF only)
├── {type}_classifier({language}).pt              # (Torch only)
├── {type}_classifier({language})_tokenizer.json
├── {type}_classifier({language})_scaler.json
├── {type}_classifier({language})_sklearn_model.joblib (if sklearn)
```

---

## ✨ Features

- ✅ Framework-agnostic training: PyTorch, TensorFlow, sklearn
- 🌐 Language-agnostic (LLM-generated dataset)
- 📤 Export to ONNX for browser or server inference
- 🧩 Tokenizer & LabelMap for deployment
- 🔍 Fast prototyping & plug-and-play CLI

---

## 🧪 Examples

### Hate Speech Classifier in Ukrainian (Torch)
```bash
python -m agents.multiclass_classifier_agent \
  -p="Detect hate speech in tree types: Hate, Not_Hate" \
  --lang ukrainian \
  --platform torch \
  --batches 400 \
  --batch-size 20 \
  --epochs 10
```

### Review Sentiment Classifier (TensorFlow)
```bash
python -m agents.multiclass_classifier_agent \
  -p="Detect review sentiment in tree types: Positive, Negative, Neutral" \
  --platform tensorflow
```

### Comment Tone Classifier (Sklearn)
```bash
python -m agents.multiclass_classifier_agent \
  -p="Detect comment tone in tree types: Sarcastic, Serious, Playful" \
  --platform sklearn
```

---

## 📤 Deployment Tips

1. Use the exported `.onnx` model with ONNXRuntime (Python, browser).
2. Load `.json` tokenizer + scaler for text preprocessing.
3. Package model into backend or frontend (e.g., FastAPI, React).
4. Retrain on real-world or multilingual datasets for better generalization.

---






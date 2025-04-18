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
python -m agents.multiclass_generate_dataset \
  --language="english" \
  --type="comment" \
  --labels="Toxic,Neutral,Helpful" \
  --batches=300
```

### 2. Train Model
```bash
python -m agents.multiclass_train_model \
  --language="english" \
  --type="comment" \
  --labels="Toxic,Neutral,Helpful" \
  --platform="torch" \
  --epochs=10
```

---

## 🔧 Parametes

| Name             | Description                                | Required | Example                         |
|------------------|--------------------------------------------|----------|---------------------------------|
| `--language`     | Language of the dataset                    | ✅       | `english`, `japanese`           |
| `--type`         | Text type                                  | ✅       | `comment`, `news`, `review`     |
| `--labels`       | Comma-separated class labels               | ✅       | `Positive,Negative,Neutral`     |
| `--platform`     | Framework to train on                      | ✅       | `torch`, `tensorflow`, `sklearn`|
| `--epochs`       | Number of training epochs (if supported)   | ❌       | `10`                           |

---

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

### News Topic Classifier (PyTorch)
```bash
--type="news" --labels="Politics,Business,Science" --platform="torch"
```

### Sentiment Review Model (TensorFlow)
```bash
--type="review" --labels="Positive,Negative,Neutral" --platform="tensorflow"
```

### Toxicity Classifier (sklearn)
```bash
--type="comment" --labels="Positive,Negative,Neutral" --platform="sklearn"
```

---

## 📤 Deployment Tips

1. Use the exported `.onnx` model with ONNXRuntime (Python, browser).
2. Load `.json` tokenizer + scaler for text preprocessing.
3. Package model into backend or frontend (e.g., FastAPI, React).
4. Retrain on real-world or multilingual datasets for better generalization.

---

## 📝 Notes

- The `--labels` list must be **exactly the same** between dataset generation and training.
- TensorFlow and PyTorch models are converted to ONNX automatically.
- Sklearn uses TF-IDF and Logistic Regression for fast baselines.

---

🔗 *Ideal for rapid prototyping and deployment of text classifiers powered by LLM-generated dataets.*


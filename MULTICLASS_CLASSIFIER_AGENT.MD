
# Multiclass Classifier Agent - Workflow Documentation

## Overview

The Multiclass Classifier Agent is a modular tool designed for generating synthetic datasets and training ONNX-ready multiclass classification models using large language models (LLMs). It supports various text types (e.g., news, reviews, comments) and allows flexible configuration of labels and languages.

## Workflow

1. **Data Generation**: Generates balanced examples for each label using an LLM based on user-defined language, text type, and class list.
2. **Model Training**: Trains a TensorFlow-based neural network on the generated data and exports it to ONNX.
3. **Tokenizer & Label Map Export**: Saves preprocessing artifacts required for deployment.

---

## Command-Line Usage

### 1. Dataset Generation
```bash
python -m agents.multiclass_generate_dataset \
  --language="english" \
  --type="comment" \
  --labels="Toxic,Neutral,Helpful" \
  --batches=300
```

### 2. Model Training
```bash
python -m agents.multiclass_train_model \
  --language="english" \
  --type="comment" \
  --labels="Toxic,Neutral,Helpful"
```

---

## Parameters

| Parameter        | Description                                                 | Required | Example                                      |
|------------------|-------------------------------------------------------------|----------|----------------------------------------------|
| `--language`     | Target language of the data                                 | Yes      | `english`, `japanese`, `hindi`               |
| `--type`         | Type of text to classify                                    | Yes      | `news`, `review`, `comment`                 |
| `--labels`       | Comma-separated list of class labels                       | Yes      | `Politics,Sports,Health`                    |
| `--batches`      | Number of generation batches                               | Optional | `300`                                       |
| `--model`        | LLM model to use for generation                            | Optional | `gpt-3.5-turbo`                             |
| `--batch-size`   | Parallel generation batch size                             | Optional | `20`                                        |

---

## Output Structure

```
training_data/{type}_train_{language}.csv
  # e.g. training_data/news_train_english.csv
testing_data/{type}_test_{language}.csv
models/
├── {type}_classifier({language}).onnx
├── {type}_classifier({language}).h5
├── {type}_classifier({language})_tokenizer.json
├── {type}_classifier({language})_scaler.json
```

---

## Key Features

1. **Flexible Label Input**: Use any custom labels via `--labels`
2. **Multilingual Support**: Train models in any supported language
3. **ONNX Export**: Easy deployment with ONNX and TF tokenizer/scaler artifacts
4. **Composable CLI**: Decoupled generation and training for reusability

---

## File Descriptions

- **Train/Test CSVs**: Labeled synthetic text examples
- **ONNX Model**: Exported neural network for inference
- **H5 Model**: Saved Keras model (optional fallback)
- **Tokenizer JSON**: Word index used for encoding
- **Scaler JSON**: Label mapping `{index: label}`

---

## Example Use Cases

- **News Topic Classifier**:
```bash
--type="news" --labels="Politics,Business,Science"
```

- **Sentiment Analysis**:
```bash
--type="review" --labels="Positive,Negative,Neutral"
```

- **Toxicity Detection**:
```bash
--type="comment" --labels="Toxic,Clean"
```

---

## Next Steps After Training

1. Use exported ONNX model with any ONNX runtime (e.g., in-browser, on server)
2. Load tokenizer and label map for preprocessing
3. Wrap inference into an API or frontend
4. Optionally retrain with real data or additional labels

---

## Notes

- You must ensure that the label list used during generation and training **match exactly**
- All models and metadata are saved under the `models/` folder
- Ideal for training fast prototypes or LLM-bootstrapped classifiers


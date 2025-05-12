# Multiclass Classifier

## Overview

The Multiclass Classifier supports two types of classification tasks:
1. **Multi-label** (Sigmoid): When a text can belong to multiple categories simultaneously
2. **Single-label** (Softmax): When a text can only belong to one category

## Use Cases

### Multi-label (Sigmoid)
- Topic tagging (e.g., a text can be both "sports" and "politics")
- Emotion detection (e.g., a text can be both "happy" and "excited")
- Content categorization (e.g., a text can be both "funny" and "informative")

### Single-label (Softmax)
- News categorization (e.g., a text can only be "sports", "politics", or "entertainment")
- Intent classification (e.g., a text can only be "question", "statement", or "command")
- Language detection (e.g., a text can only be in one language)

## Usage

```bash
python -m text_classifier.agent -p="Your classification task description"
```

The agent will automatically:
1. Analyze your task and determine the appropriate model type (sigmoid/softmax)
2. Generate appropriate training data
3. Train and evaluate the model
4. Export the model in ONNX format

## Parameters

| Parameter                  | Description                                                             | Default              |
|----------------------------|-------------------------------------------------------------------------|----------------------|
| `-p --problem_description` | Description of your classification problem                              | (Required)           |
| `-m --model`               | LLM to use for training data generation                                 | `openai/gpt-4o-mini` |
| `-l --library`             | ML library to use (pytorch/tensorflow/scikit-learn)                     | `tensorflow`         |
| `-o --output-path`         | Directory to save model and data                                        | `./models`           |
| `--lang`                   | Language for training data                                              | `english`            |

## Output

The trained model and associated files will be saved in the specified output directory:

```
./models/[model_prefix]/
├── config.json                    # Configuration and analysis
├── data/
│   ├── training_data.csv          # Generated training data
│   └── edge_case_data.csv         # Challenging test cases
├── model.onnx                     # ONNX model file
├── model_scaler.json              # StandardScaler parameters
└── model_vocab.json               # TF-IDF vocabulary
```

## Model Details

### Multi-label (Sigmoid)
- **Type**: Multi-label classification
- **Input**: Text data
- **Output**: Independent probabilities for each class
- **Activation**: Sigmoid (for independent class probabilities)
- **Framework Support**: TensorFlow, PyTorch, Scikit-learn

### Single-label (Softmax)
- **Type**: Single-label classification
- **Input**: Text data
- **Output**: Mutually exclusive probabilities for each class
- **Activation**: Softmax (for mutually exclusive probabilities)
- **Framework Support**: TensorFlow, PyTorch, Scikit-learn

## Example

```python
from text_classifier import TextClassifier, ModelType

# Initialize multi-label classifier
classifier = TextClassifier(
    strategy=strategy,
    class_labels=["sports", "politics", "entertainment"],
    model_type=ModelType.MULTICLASS_SIGMOID
)

# Initialize single-label classifier
classifier = TextClassifier(
    strategy=strategy,
    class_labels=["sports", "politics", "entertainment"],
    model_type=ModelType.MULTICLASS_SOFTMAX
)

# Train model
classifier.train("path/to/training_data.csv")

# Make predictions
probabilities = classifier.predict_proba("Your text here")
predictions = classifier.predict("Your text here")
```

## Best Practices

1. **Model Selection**:
   - Use sigmoid for multi-label tasks where classes are independent
   - Use softmax for single-label tasks where classes are mutually exclusive

2. **Data Quality**:
   - Ensure balanced representation of all classes
   - Include examples of class combinations (for sigmoid)
   - Include clear examples of each class (for softmax)

3. **Evaluation**:
   - For sigmoid: Use per-class metrics and micro/macro averages
   - For softmax: Use accuracy, confusion matrix, and per-class metrics

4. **Deployment**:
   - Export to ONNX for cross-platform compatibility
   - Consider class thresholds for sigmoid outputs
   - Use argmax for softmax outputs 
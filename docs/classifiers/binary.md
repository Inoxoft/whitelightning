# Binary Classifier

## Overview

The Binary Classifier is designed for tasks where you need to classify text into exactly two categories with probability output. It's ideal for simple yes/no or true/false classification tasks.

## Use Cases

- Sentiment analysis (positive/negative)
- Spam detection (spam/ham)
- Content moderation (appropriate/inappropriate)
- Any other binary classification task

## Usage

```bash
python -m text_classifier.agent -p="Your classification task description"
```

The agent will automatically:
1. Analyze your task and determine if it's suitable for binary classification
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

- **Type**: Binary classification with probability output
- **Input**: Text data
- **Output**: Probability of the positive class
- **Activation**: Sigmoid (for probability output)
- **Framework Support**: TensorFlow, PyTorch, Scikit-learn

## Example

```python
from text_classifier import TextClassifier, ModelType

# Initialize classifier
classifier = TextClassifier(
    strategy=strategy,
    class_labels=["negative", "positive"],
    model_type=ModelType.BINARY
)

# Train model
classifier.train("path/to/training_data.csv")

# Make predictions
probabilities = classifier.predict_proba("Your text here")
predictions = classifier.predict("Your text here")
```

## Best Practices

1. **Data Quality**: Ensure your training data is balanced and representative
2. **Edge Cases**: Include challenging examples in your training data
3. **Evaluation**: Use appropriate metrics (accuracy, precision, recall, F1)
4. **Deployment**: Export to ONNX for cross-platform compatibility 
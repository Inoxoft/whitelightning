# pyth-still - LLM Distillation Tool

This project implements a tool for distilling large language models (LLMs) into simple binary classifiers. It can be used to classify text data into binary categories (e.g., "spam" vs "ham").

## Model Architecture

The model uses a Gradient Boosting Classifier with TF-IDF features:

1. **Feature Extraction**:
   - TF-IDF Vectorizer for converting text into numerical features
   - Configurable n-gram range and maximum features
   - Standard scaling of TF-IDF features

2. **Binary Classifier (`BinaryClassifier` class)**:
   - Gradient Boosting Classifier with configurable parameters
   - Includes preprocessing pipeline (TF-IDF + StandardScaler)
   - Cross-validation for robust performance evaluation
   - Model persistence functionality

3. **Training and Evaluation**:
   - Automatic train-test splitting
   - Performance metrics including accuracy and classification report
   - Cross-validation scores for model validation
   - Model artifacts saving and loading capabilities

## Environment Variables

The project requires the following environment variables (defined in `settings.py`):

- `OPEN_ROUTER_API_KEY`: API key for openrouter.ai
- `PROMPT_POSITIVE`: Prompt for generating positive label data
- `PROMPT_NEGATIVE`: Prompt for generating negative label data
- `DATA_COLUMN_NAME`: Column name for text data in dataset
- `LABEL_COLUMN_NAME`: Column name for labels in dataset
- `MODEL_PREFIX`: Prefix for model-related files
- `POSITIVE_LABEL`: Label for positive class
- `NEGATIVE_LABEL`: Label for negative class

## Project Structure

- `models/`: Saved model artifacts
- `results/`: Model evaluation results
- `training_data/`: Training datasets
- `testing_data/`: Test datasets

## Usage

1. **Set up environment**:
   - Create and configure `.env` file based on `.env-example`

2. **Generate dataset**:
   - Use `data_generator.py` to generate training data using LLM
   - Multiple runs can be performed to increase dataset size

3. **Train and test model**:
   - Run `main.py` to:
     - Train the model
     - Execute model predictions
     - Test model accuracy

## Example

Train and evaluate the model:

```bash
python main.py
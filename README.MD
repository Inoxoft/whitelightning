# WhiteLightning - LLM Distillation Tool

This project implements a tool for distilling large language models (LLMs) into simple binary text classifiers. It supports classifying text data into binary categories (e.g., "positive" vs "negative" or "spam" vs "ham") using multiple machine learning frameworks and enables deployment across various platforms and languages via ONNX.

## Model Architecture

The tool provides a flexible architecture with three interchangeable strategies for binary classification, all built on top of TF-IDF features:

1. **Feature Extraction**:
   - **TF-IDF Vectorizer**: Converts text into numerical features with a configurable maximum of 5000 features and n-gram range (default: 1-3).
   - **StandardScaler**: Scales TF-IDF features to zero mean and unit variance for consistent input to the classifiers.

2. **Binary Classifier (`BinaryTextClassifier` class)**:
   - **Strategies**:
     - **TensorFlowStrategy**: A neural network with two hidden layers (512 and 256 units, ReLU activation, 0.3 dropout) and a sigmoid output.
     - **PyTorchStrategy**: A similar neural network implemented in PyTorch, with the same architecture and sigmoid output.
     - **ScikitLearnStrategy**: A Gradient Boosting Classifier with configurable parameters (e.g., 500 estimators, max depth 5).
   - **Preprocessing Pipeline**: Integrated TF-IDF and scaling within the `BinaryTextClassifier` class.
   - **Training**: Automatic train-test splitting (80-20 split), with metrics including accuracy, classification report, and optional cross-validation (Scikit-learn only).
   - **Persistence**: Save/load functionality for models, vectorizer, and scaler, plus export to ONNX for cross-platform use.

3. **ONNX Export**:
   - All strategies support exporting trained models to ONNX format for deployment in Python, Web (JavaScript), C, C++, and Rust.
   - Preprocessing parameters (TF-IDF vocabulary, IDF weights, scaler mean, and scale) are exported as JSON for consistent feature extraction across languages.

4. **Training and Evaluation**:
   - Train-test split with performance metrics (accuracy, classification report).
   - Cross-validation scores (Scikit-learn only) for robust validation.
   - Model artifacts saved in native formats (`.h5`, `.pt`, `.pkl`) and ONNX (`.onnx`).

## Environment Variables

The project uses environment variables defined in `settings.py` (loaded from a `.env` file):

- `OPEN_ROUTER_API_KEY`: API key for openrouter.ai (used in data generation).
- `PROMPT_POSITIVE`: Prompt for generating positive label data.
- `PROMPT_NEGATIVE`: Prompt for generating negative label data.
- `DATA_COLUMN_NAME`: Column name for text data in the dataset (e.g., "text").
- `LABEL_COLUMN_NAME`: Column name for labels in the dataset (e.g., "label").
- `MODEL_PREFIX`: Prefix for model-related files (e.g., "text_classifier").
- `POSITIVE_LABEL`: Label for the positive class (e.g., "positive" or 1).
- `NEGATIVE_LABEL`: Label for the negative class (e.g., "negative" or 0).
- `TRAINING_DATA_PATH`: Directory for training datasets (e.g., "training_data/").
- `MODELS_PATH`: Directory for saved model artifacts (e.g., "models/").

## Project Structure

- `models/`: Saved model artifacts (`.h5`, `.pt`, `.pkl`, `.onnx`, `_vectorizer.pkl`, `_scaler.pkl`, `_vocab.json`, `_scaler.json`).
- `results/`: Model evaluation results (not explicitly used yet but reserved).
- `training_data/`: Training datasets (e.g., CSV files).
- `testing_data/`: Test datasets (e.g., CSV files).
- `binary_classifier/`: Core module with `strategies.py` (strategy implementations).
- `settings.py`: Configuration and environment variable handling.
- `data_generator.py`: Script for generating training data using an LLM.
- `main.py`: Main script for training and testing.

## Usage

1. **Set Up Environment**:
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     - Required packages: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `torch`, `onnxruntime`, `tf2onnx`, `skl2onnx`, `joblib`, `python-dotenv`.
   - Create and configure a `.env` file based on `.env-example`:
     ```
     OPEN_ROUTER_API_KEY=your_api_key
     PROMPT_POSITIVE="Generate positive text examples"
     PROMPT_NEGATIVE="Generate negative text examples"
     DATA_COLUMN_NAME=text
     LABEL_COLUMN_NAME=label
     MODEL_PREFIX=text_classifier
     POSITIVE_LABEL=positive
     NEGATIVE_LABEL=negative
     TRAINING_DATA_PATH=training_data/
     MODELS_PATH=models/
     ```

2. **Generate Dataset**:
   - Run `data_generator.py` to create a training dataset using an LLM via openrouter.ai:
     ```bash
     python data_generator.py
     ```
   - Output: A CSV file (e.g., `training_data/text_classifier_dataset.csv`) with text and labels.
   - Multiple runs can append to the dataset for increased size.

3. **Train and Test Model**:
   - Run `main.py` to train, evaluate, and export the model:
     ```bash
     python main.py
     ```
   - **Steps**:
     - Loads data from `training_data/text_classifier_dataset.csv`.
     - Trains the model using the specified strategy (e.g., PyTorch).
     - Saves model artifacts and exports to ONNX.
     - Logs performance metrics (accuracy, classification report).
   - **Customization**: Modify `run_training` in `main.py` to choose the strategy (`tensorflow`, `pytorch`, `scikit`).

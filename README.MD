# pyth-still - LLM Distillation Tool

This project implements a tool for distilling large language models (LLMs) into simple binary classifiers. This is an example of usage - classify text messages as either "spam" or "ham" (not spam).

## Model Architecture

The model architecture consists of the following components:

1. **Vocabulary (`Vocab` class)**:
   - Encodes text messages into sequences of integers based on word frequency.
   - Handles unknown words by mapping them to a special `<UNK>` token.

2. **Spam Dataset (`SpamDataset` class)**:
   - Prepares the dataset for training by encoding text messages and converting labels to binary format.
   - Pads or truncates text sequences to a fixed length (20 tokens).

3. **Spam Classifier (`SpamClassifier` class)**:
   - A neural network with an embedding layer, a fully connected layer, and a sigmoid activation function.
   - The embedding layer converts word indices into dense vectors.
   - The fully connected layer processes the concatenated embeddings to produce a binary output (spam or ham).

4. **Training and Evaluation**:
   - The model is trained using the Adam optimizer and binary cross-entropy loss.
   - The training process includes saving the trained model and vocabulary for later use.
   - The evaluation process involves predicting labels for a test dataset and calculating match rates for accuracy assessment.

## Environment Variables

The project requires the following environment variables, which are defined in the `settings.py` file:

- `MODEL_PREFIX`: A prefix used for naming model-related files (e.g., vocabulary and model weights).
- `POSITIVE_LABEL`: The label used for positive class (e.g., "spam").
- `NEGATIVE_LABEL`: The label used for negative class (e.g., "ham").

## Usage

1. **Training the Model**:
   - Run the `run_training` function from the `model.py` file to train the model on the training dataset.
   - The trained model and vocabulary will be saved to the `models/` directory.

2. **Running the Model**:
   - Run the `run_model` function from the `model_executor.py` file to make predictions on the test dataset.
   - The predictions will be saved to the `results/` directory.

3. **Testing Model Accuracy**:
   - Run the `test_model_accuracy` function from the `utils/benchmark.py` file to calculate and print the match rates for the positive and negative labels.

## Example

To train the model, run predictions, and test accuracy, execute the `main.py` script:

```bash
python main.py
```

This script will sequentially train the model, run predictions, and test the model's accuracy, saving the results to the appropriate directories.
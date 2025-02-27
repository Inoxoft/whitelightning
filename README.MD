# pyth-still - LLM Distillation Tool

This project implements a tool for distilling large language models (LLMs) into simple binary classifiers. This is an example of usage - classify text messages as either "spam" or "ham" (not spam).

## Model Architecture

The model architecture consists of the following components:

1. **Vocabulary (`Vocab` class)**:
   - Encodes text messages into sequences of integers based on word frequency.
   - Handles unknown words by mapping them to a special `<UNK>` token.

2. **Dataset (`SpamDataset` class)**:
   - Prepares the dataset for training by encoding text messages and converting labels to binary format.
   - Pads or truncates text sequences to a fixed length (20 tokens).

3. **Binary Classifier (`BinaryClassifier` class)**:
   - A neural network with an embedding layer, a fully connected layer, and a sigmoid activation function.
   - The embedding layer converts word indices into dense vectors.
   - The fully connected layer processes the concatenated embeddings to produce a binary output (spam or ham).

4. **Training and Evaluation**:
   - The model is trained using the Adam optimizer and binary cross-entropy loss.
   - The training process includes saving the trained model and vocabulary for later use.
   - The evaluation process involves predicting labels for a test dataset and calculating match rates for accuracy assessment.

## Environment Variables

The project requires the following environment variables, which are defined in the `settings.py` file:

- `OPEN_ROUTER_API_KEY`: The API key for openrouter.ai.
- `PROMPT_POSITIVE`: Prompt that will be used to generate data with positive labels.
- `PROMPT_NEGATIVE`: Prompt that will be used to generate data with negative labels.
- `DATA_COLUMN_NAME` Column name for the text data in the dataset.
- `LABEL_COLUMN_NAME` Column name for the label data in the dataset.
- `MODEL_PREFIX`: A prefix used for naming model-related files (e.g., vocabulary and model weights).
- `POSITIVE_LABEL`: The label used for positive class (e.g., "spam").
- `NEGATIVE_LABEL`: The label used for negative class (e.g., "ham").

## Usage

1. **Set up .env variables**:
   - Define the environment variables in the `.env` file, see .env-example.

2. **Gather dataset**:
   - Run the `data_generator.py` file to ask LLM for training data. Run it as many times as needed. One cycle produces ~200 entries of data, depending on the model.

3. **Train and test model**:
   - Run the `main.py` file to train model, calculate and print the match rates for the positive and negative labels.

## Example

To train the model, run predictions, and test accuracy, execute the `main.py` script:

```bash
python main.py
```

This script will sequentially train the model, run predictions, and test the model's accuracy, saving the results to the appropriate directories.
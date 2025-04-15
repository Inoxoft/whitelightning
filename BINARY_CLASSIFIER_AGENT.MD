# Binary Classifier Training Agent - Workflow Documentation

## Overview

The Binary Classifier Training Agent is a comprehensive tool for creating and training binary classification models using LLM-generated synthetic data. This document outlines the general workflow, available commands, and parameters.

## Workflow

1. **Configuration Generation**: The agent first analyzes your problem description and creates a config with appropriate prompts.
2. **Data Generation & Prompt Refinement**: The agent generates synthetic data and iteratively improves the prompts based on data quality.
3. **Edge Case Generation**: The agent creates challenging test cases to thoroughly evaluate the model.
4. **Model Performance Analysis**: The agent analyzes the model's performance and provides improvement suggestions.
5. **Documentation**: All configurations, prompts, and analysis are saved for future reference.

## Command-Line Usage

```bash
python -m agents.binary_classifier_agent -p="Your classification problem description" [options]
```

## Parameters

| Parameter                  | Description                                                             | Default              |
|----------------------------|-------------------------------------------------------------------------|----------------------|
| `-p --problem_description` | Description of your classification problem                              | (Required)           |
| `-m --model`               | LLM to use for training data generation                                 | `openai/gpt-4o-mini` |
| `--config-model`           | LLM to use for prompting and config                                     | `scikit-learn`       |
| `-l --library`             | ML library to use (pytorch/tensorflow/scikit-learn)                     | `tensorflow`         |
| `-o --output-path`         | Directory to save model and data                                        | `./models`           |
| `--refinement-cycles`      | Number of prompt improvement iterations                                 | `1`                  |
| `--edge-cases`             | Use config LLM to generate more sophisticated testing data for feedback | True                 |
| `--batch-size`             | Number of parallel requests to run when generating training data        | `10`                 |

## Output Structure

```
./models/[model_prefix]/
├── config.json                    # Complete configuration and analysis
├── data/
│   ├── api_requests/              # Folder with all API requests and responses
│   └── edge_case_data.csv         # Challenging test cases
│   └── edge_case_predictions.csv  # Challenging test cases predictions
│   └── generation_config.json     # Propmts used for data generation and training and final verdict with recommendations
│   └── model.onnx                 # ONNX model file
│   └── model_scaler.json          # StandardScaler parameters
│   └── model_vocab.json           # TF-IDF vocabulary
│   └── training_data.csv          # Generated training data
│   * + native model files, depending on the library
```

## Key Features

1. **Automatic Configuration**: Generates appropriate prompts based on your problem description.
2. **Prompt Evolution**: Iteratively improves prompts based on data quality feedback.
3. **Edge Case Testing**: Identifies challenging scenarios that might cause model failure.
4. **Performance Analysis**: Provides insights on model strengths and weaknesses.
5. **Comprehensive Documentation**: Tracks the entire process from initial configuration to final thoughts.

## File Documentation

The `generation_config.json` file contains:
- Problem summary and description
- Data generation prompts (initial and refined versions)
- Model configuration
- Prompt evolution history with quality evaluations
- Training data statistics
- Edge case information
- Performance analysis and recommendations

This allows you to review the entire development process and understand the strengths and limitations of your model.

## Example Workflow

```bash
# Basic usage with required parameters
python -m agents.binary_classifier_agent -p="Detect phishing emails that attempt to steal user credentials"

# Advanced usage with custom settings
python -m agents.binary_classifier_agent -p="Identify toxic comments in online discussions" \
  --model="anthropic/claude-3.7-sonnet" \
  --library="pytorch" \
  --output-path="./customer_models" \
  --refinement-cycles=3
```

## Next Steps After Training

1. Review the configuration file to understand the model's strengths and weaknesses
2. Examine edge cases to identify potential areas for improvement
3. Use the generated data for training with your preferred ML library
4. Deploy the model in your application
5. Monitor performance and iterate as needed
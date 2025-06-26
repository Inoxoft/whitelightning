# Using Your Own Dataset with WhiteLightning

WhiteLightning now supports using your own datasets instead of generating synthetic data with LLMs. This is perfect when you already have labeled data and want to create optimized ONNX models for deployment.

## 🚀 Quick Start with Own Dataset

### 1. Prepare Your Dataset

Place your dataset file in the `own_data/` directory. WhiteLightning supports:
- **CSV files** (comma or tab-separated)
- **JSON files** (regular JSON or JSONL format)
- **TXT files** (tab or comma-separated)

Example dataset formats:

**CSV Format:**
```csv
text,label
"This product is amazing!",positive
"Terrible service, very disappointed",negative
"Great quality and fast shipping",positive
```

**JSONL Format:**
```json
{"text": "This product is amazing!", "category": "positive"}
{"text": "Terrible service, very disappointed", "category": "negative"}
{"text": "Great quality and fast shipping", "category": "positive"}
```

### 2. Run with Docker

```bash
# Create the own_data directory and add your dataset
mkdir -p own_data
cp your_dataset.csv own_data/

# Run WhiteLightning with your own dataset
docker run \
    --rm \
    -v $(pwd):/app/models \
    -v $(pwd)/own_data:/app/own_data \
    -e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
    ghcr.io/whitelightning-ai/whitelightning:latest \
    -p="Classify customer feedback sentiment" \
    --use-own-dataset="/app/own_data/your_dataset.csv" \
    --model-type="tensorflow"
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements/base.txt

# Set your API key (only needed for LLM analysis of your dataset structure)
export OPEN_ROUTER_API_KEY="your_key_here"

# Run with your dataset
python -m text_classifier.agent \
    -p="Classify customer feedback sentiment" \
    --use-own-dataset="own_data/your_dataset.csv" \
    --model-type="tensorflow"
```

## 📊 What Happens When You Use Your Own Dataset

1. **Automatic Analysis**: WhiteLightning uses an LLM to analyze your dataset structure and identify:
   - Text columns (content to classify)
   - Label columns (classification targets)
   - Task type (binary, multiclass, or multilabel)

2. **Data Preparation**: Your dataset is automatically:
   - Cleaned (text normalized to single lines)
   - Converted to standard format (`text`, `label` columns)
   - Limited to 20,000 samples (configurable)
   - Validated for quality

3. **Model Training**: A specialized model is trained using your data with:
   - TF-IDF feature extraction
   - Optimized architecture for your task type
   - ONNX export for cross-platform deployment

## 📁 Output Files

When using your own dataset, you'll get:

```
models/
├── user_dataset_multiclass.onnx      # ONNX model file
├── model_vocab.json                   # TF-IDF vocabulary
├── model_scaler.json                  # Feature scaling parameters
├── training_data.csv                  # Processed training data
├── generation_config.json             # Complete configuration
└── your_dataset_analysis_report.json  # Dataset analysis results
```

## 🎯 Supported Dataset Types

### Binary Classification
```csv
text,sentiment
"Love this product!",positive
"Worst purchase ever",negative
```

### Multiclass Classification
```csv
text,category
"Breaking: Election results announced",politics
"New smartphone released",technology
"Stock market hits record high",business
```

### Multilabel Classification
```csv
text,tags
"Eco-friendly solar panels for homes","environment,technology,home"
"Organic food delivery service","food,health,delivery"
```

## ⚙️ Advanced Options

```bash
# Use your dataset with custom settings
python -m text_classifier.agent \
    -p="Classify news articles by topic" \
    --use-own-dataset="own_data/news_data.csv" \
    --model-type="sklearn" \
    --max-features=10000 \
    --output-path="my_models"
```

## 🔧 Troubleshooting

### Common Issues

1. **"Could not identify text/label columns"**
   - Ensure your dataset has clear column names like `text`, `content`, `label`, `category`
   - Check that your data is properly formatted

2. **"Encoding error"**
   - WhiteLightning automatically tries multiple encodings (UTF-8, Latin-1, etc.)
   - Ensure your file is saved in a standard encoding

3. **"No API key found"**
   - Even when using your own dataset, an API key is needed for dataset analysis
   - Set `OPEN_ROUTER_API_KEY` in your environment or `.env` file

### Dataset Requirements

- **Minimum**: 10 samples per class
- **Recommended**: 100+ samples per class
- **Maximum**: 20,000 samples (automatically limited)
- **Text length**: Any length (automatically cleaned)

## 🚀 Benefits of Using Your Own Dataset

- **No LLM costs** for data generation (only small cost for dataset analysis)
- **Real data quality** instead of synthetic data
- **Domain-specific accuracy** using your actual use case data
- **Faster training** with pre-labeled data
- **Privacy-friendly** - your data stays local during training

## 🔄 Migration from LLM-Generated Data

If you've been using WhiteLightning's LLM data generation and want to switch to your own data:

1. Export your existing generated data: `training_data.csv`
2. Use it as your own dataset: `--use-own-dataset="training_data.csv"`
3. Skip the data generation costs and get consistent results

This is perfect for production workflows where you want reproducible model training! 
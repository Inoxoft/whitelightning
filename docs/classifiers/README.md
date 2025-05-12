# WhiteLightning Classifiers

This directory contains documentation for WhiteLightning's text classification models. We currently support two types of classifiers:

## Binary Classifier
The Binary Classifier is designed for tasks where you need to classify text into exactly two categories. Common use cases include:
- Spam detection (spam vs. not spam)
- Sentiment analysis (positive vs. negative)
- Content moderation (appropriate vs. inappropriate)

[Read the Binary Classifier Documentation](binary.md)

## Multiclass Classifier
The Multiclass Classifier supports classification into three or more categories. This is useful for:
- Topic classification (sports, business, technology, etc.)
- Intent detection (purchase, inquiry, complaint, etc.)
- Emotion analysis (happy, sad, angry, neutral, etc.)

[Read the Multiclass Classifier Documentation](multiclass.md)

## Common Features
Both classifiers share these key features:
- LLM-powered synthetic data generation
- Multiple framework support (PyTorch, TensorFlow, scikit-learn)
- ONNX export for cross-platform deployment
- Multilingual support
- Comprehensive documentation and analysis
- Edge case testing and validation

## Getting Started
To get started with either classifier:
1. Install the required dependencies
2. Choose the appropriate classifier for your use case
3. Follow the specific documentation for your chosen classifier
4. Train and export your model
5. Deploy using ONNX runtime 
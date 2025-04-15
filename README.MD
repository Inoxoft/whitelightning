# WhiteLightning - LLM Distillation Tool

WhiteLightning is a powerful tool designed to distill large language models (LLMs) into lightweight, efficient classifiers. By leveraging advanced techniques, it simplifies the process of creating text classifiers that can run anywhere, from cloud environments to edge devices, using the ONNX format for cross-platform compatibility.

## What is LLM Distillation?

LLM distillation is the process of transforming large, complex language models into smaller, task-specific models. WhiteLightning focuses on binary and multiclass text classification, enabling users to classify text data into categories like "spam vs ham" or multiple classes with ease. This approach ensures high performance while significantly reducing computational requirements.

## Why ONNX?

WhiteLightning uses ONNX (Open Neural Network Exchange) to export trained models, making them deployable across a wide range of platforms and programming languages. With ONNX, you can run your models in Python, JavaScript, C++, Rust, and more, ensuring flexibility and scalability for your applications.

## Key Features

- **LLM Distillation**: Converts large language models into efficient binary or multiclass classifiers.
- **Cross-Platform Deployment**: Export models to ONNX for use in diverse environments.
- **Lightweight and Fast**: Optimized for performance with minimal resource usage.
- **Customizable**: Supports multiple machine learning frameworks, including TensorFlow, PyTorch, and Scikit-learn.

## Available Models

WhiteLightning currently supports the following classifiers:

1. **Binary Classifier**: Classifies text into two categories (e.g., "positive" vs "negative").  
   [Read the Binary Classifier Documentation](BINARY_CLASSIFIER_AGENT.MD)

2. **Multiclass Classifier**: Classifies text into multiple categories.  
   [Read the Multiclass Classifier Documentation](MULTICLASS_CLASSIFIER_AGENT.MD)  

With WhiteLightning, you can easily train, evaluate, and deploy these models for your specific use case.
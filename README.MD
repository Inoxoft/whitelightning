# WhiteLightning - LLM Distillation Tool

WhiteLightning is a powerful tool designed to distill large language models (LLMs) into lightweight, efficient text classifiers. By leveraging advanced techniques, it simplifies the process of creating text classifiers that can run anywhere, from cloud environments to edge devices, using the ONNX format for cross-platform compatibility.

<p align="center">
   <img src="media/moonshiner_floppy.jpeg" width="300" height="300" alt="Moonshiner">
   <br><br>
  <a href="https://github.com/whitelightning-ai/whitelightning/actions"><img src="https://img.shields.io/github/actions/workflow/status/whitelightning-ai/whitelightning/ci.yml?branch=main&style=flat-square" alt="Build Status"></a>
  <a href="https://github.com/whitelightning-ai/whitelightning/stargazers"><img src="https://img.shields.io/github/stars/whitelightning-ai/whitelightning?style=flat-square" alt="Stars"></a>
  <a href="https://github.com/whitelightning-ai/whitelightning/network/members"><img src="https://img.shields.io/github/forks/whitelightning-ai/whitelightning?style=flat-square" alt="Forks"></a>
  <a href="https://github.com/whitelightning-ai/whitelightning/issues"><img src="https://img.shields.io/github/issues/whitelightning-ai/whitelightning?style=flat-square" alt="Issues"></a>
  <a href="https://github.com/whitelightning-ai/whitelightning/blob/main/LICENSE"><img src="https://img.shields.io/github/license/whitelightning-ai/whitelightning?style=flat-square" alt="License"></a>
</p>

<p align="center">
   <img src="media/demo.gif" width="700" alt="CLI Usage">
</p>

## What is LLM Distillation?

LLM distillation is the process of transforming large, complex language models into smaller, task-specific models. WhiteLightning focuses on text classification, enabling users to create efficient classifiers for various use cases. This approach ensures high performance while significantly reducing computational requirements.

## Why ONNX?

WhiteLightning uses ONNX (Open Neural Network Exchange) to export trained models, making them deployable across a wide range of platforms and programming languages. With ONNX, you can run your models in Python, JavaScript, C++, Rust, and more, ensuring flexibility and scalability for your applications.

## Key Features

- **Multiple Model Types**: Support for binary and multiclass classification with different activation types
- **Cross-Platform Deployment**: Export models to ONNX for use in diverse environments
- **Lightweight and Fast**: Optimized for performance with minimal resource usage
- **Customizable**: Supports multiple machine learning frameworks (TensorFlow, PyTorch, Scikit-learn)
- **Multilingual Support**: Generate training data in multiple languages
- **Automatic Configuration**: Smart prompt generation and refinement based on your task

## Model Types

WhiteLightning supports three types of text classifiers:

1. **Binary Classifier**
   - Simple yes/no or true/false classification
   - Probability output for single class
   - Ideal for: Sentiment analysis, spam detection, content moderation
   - [Learn More](docs/classifiers/binary.md)

2. **Multiclass Sigmoid**
   - Multi-label classification
   - Independent class probabilities
   - Multiple classes can be active simultaneously
   - Ideal for: Topic tagging, emotion detection, content categorization
   - [Learn More](docs/classifiers/multiclass.md)

3. **Multiclass Softmax**
   - Single-label classification
   - Mutually exclusive class probabilities
   - Only one class can be active at a time
   - Ideal for: News categorization, intent classification, language detection
   - [Learn More](docs/classifiers/multiclass.md)

## Quick Start

1. Pull the Docker image:
```bash
docker pull ghcr.io/whitelightning-ai/whitelightning:latest
```

2. Set up your environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the classifier agent:
```bash
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/.env:/app/.env ghcr.io/whitelightning-ai/whitelightning:latest -p="Classify customer feedback as positive or negative sentiment" --model-type="tensorflow" --refinement-cycles=1 --generate-edge-cases="true" --lang="english"
```

## OR

Check out the [Docker Setup](docs/docker/README.md) for a building a local image of WhiteLightning.

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

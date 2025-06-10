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

## What do you mean by distillation?

We use large, complex frotier models and use them to train  smaller, task-specific models. WhiteLightning focuses on text classification, enabling users to create efficient classifiers for various use cases. This approach ensures high performance while significantly reducing computational requirements.

<p align="center">
   <img src="media/wl-metaphor.png" width="800" alt="The white lightning metaphor">
</p>

## How are the models saved?

WhiteLightning uses ONNX (Open Neural Network Exchange, see [onnx.ai](https://onnx.ai/)) to export trained models, making them deployable across a wide range of platforms and programming languages. With ONNX, you can run your models in Python, JavaScript, C++, Rust, and more, ensuring flexibility and scalability for your applications.

## Key Features

- **Multiple Model Types**: Support for binary and multiclass classification with different activation types
- **Cross-Platform Deployment**: Export models to ONNX for use in diverse environments
- **Lightweight and Fast**: Optimized for performance with minimal resource usage
- **Customizable**: Supports multiple machine learning frameworks (TensorFlow, PyTorch, Scikit-learn)
- **Multilingual Support**: Generate training data in multiple languages
- **Automatic Configuration**: Smart prompt generation and refinement based on your task

## Quick Start

1. Get an OpenRouter API key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)

1. Pull and Run the Docker image:

```bash
docker run \
	--rm \
	-v $(pwd):/app/models \
	-e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
	ghcr.io/whitelightning-ai/whitelightning:latest \
	-p="Classify customer feedback as positive or negative sentiment" \
	--refinement-cycles=1 \
	--generate-edge-cases="true" \
	--lang="english"
```

your output should look something like this...

<img src="media/demo.gif" width="500" alt="CLI Usage">

when you are done your model files will be available to you, if you ```ls``` you'll see something like this: 

```
api_requests  edge_case_data.csv  generation_config.json  model.h5  model.onnx  scaler.json  sentiment_classifier_edge_case_predictions.csv  training_data.csv  vocab.json
```

see the complete documentation here for a guide on how to use these files in the language or project of your choosing (Android, iOS, Rust, C, Arduino, Java etc... whatever you want!!!!)

## Advanced Setup

Check out the [Local Setup](docs/docker/README.md) for a building a local image of WhiteLightning.

## Complete Documentation

see [https://whitelightning.ai](https://whitelightning.ai)

## Contributing

We welcome contributions! We recommend joining our [Discord](https://discord.com/invite/QDj8NS2yDt) and chatting with us before submitting your first PR.

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

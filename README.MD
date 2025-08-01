<p align="center">
  <img src="media/20250522_1503_Steampunk Distillery Awakens_simple_compose_01jvvy1hy8e4w9fq5648qt450z.gif" width="450" alt="WhiteLightning Mascot">
  <h1 align="center">WhiteLightning</h1>
  <p align="center">
    The LLM Distillation Tool
  </p>
  <p align="center">
    <a href="https://whitelightning.ai"><img src="https://img.shields.io/badge/Documentation-whitelightning.ai-28a745?style=for-the-badge&logo=readme&logoColor=white" alt="Documentation"></a>
  </p>
</p>

<p align="center">
    <a href="https://github.com/whitelightning-ai/whitelightning/actions"><img src="https://img.shields.io/github/actions/workflow/status/whitelightning-ai/whitelightning/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white" alt="Build Status"></a>
    <a href="https://discord.com/invite/QDj8NS2yDt"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord"></a>
    <a href="https://github.com/whitelightning-ai/whitelightning/stargazers"><img src="https://img.shields.io/github/stars/whitelightning-ai/whitelightning?style=for-the-badge&logo=github&color=gold" alt="Stars"></a>
    <a href="https://github.com/whitelightning-ai/whitelightning/blob/main/LICENSE"><img src="https://img.shields.io/github/license/whitelightning-ai/whitelightning?style=for-the-badge&color=blue" alt="License"></a>
</p>

**WhiteLightning** distills massive, state-of-the-art language models into lightweight, hyper-efficient text classifiers. It's a command-line tool that lets you create specialized models that run anywhere—from the cloud to the edge—using the universal **ONNX** format for maximum compatibility.

---

### What do we mean by "Distillation"?

We use large, powerful frontier models as "teachers" to train much smaller, task-specific "student" models. WhiteLightning automates this process for text classification, allowing you to create high-performance classifiers with a fraction of the computational footprint.

<p align="center">
  <img src="media/openart-image_m8sOEHVQ_1753429527461_raw.png" width="800" alt="The WhiteLightning metaphor: from a complex still to a pure, potent product.">
</p>

### How are the models saved?

WhiteLightning exports every trained model to **ONNX (Open Neural Network Exchange)**. This standard format makes your models instantly portable. Run them natively in Python, JavaScript, C++, Rust, Java, and more, ensuring total flexibility for any project. Learn more at [onnx.ai](https://onnx.ai/).

---

### ⚡ Cross-Platform Compatibility

WhiteLightning is designed as a **"generic" Docker image** that works seamlessly across **macOS**, **Linux**, and **Windows** with identical commands:

* **Zero Configuration**: No need for complex `--user` flags or platform-specific commands
* **Automatic Permission Handling**: Intelligently detects your system and sets correct file ownership
* **Universal Commands**: Same `docker run` command works everywhere
* **Smart User Management**: Internally manages user creation and permission mapping
* **Secure by Default**: Always runs as non-root user with proper privilege dropping

---

### Key Features

* **Multiple Model Architectures**: Generate models for binary and multiclass classification with different activation functions.
* **Instant Cross-Platform Deployment**: Export to ONNX for use in any environment or language.
* **Lightweight & Incredibly Fast**: Optimized for high-speed inference with minimal resource consumption.
* **Framework Agnostic**: The final ONNX model has **zero dependencies** on TensorFlow or PyTorch. It's pure, portable compute.
* **Multilingual Support**: Generate training data and classifiers in a wide variety of languages.
* **Smart & Automatic**: Intelligently generates and refines prompts based on your classification task.

---

### 🚀 Quick Start

1.  Clone the repository:
    ```bash
    git clone https://github.com/Inoxoft/whitelightning.git
    cd whitelightning
    ```

2.  Get an OpenRouter API key at **[openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)**.

3.  Run the Docker image:

    **Mac:**
    ```bash
    docker run --rm \
        -v "$(pwd)":/app/models \
        -e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
        ghcr.io/inoxoft/whitelightning:latest \
        python -m text_classifier.agent \
        -p "Categorize customer reviews as positive, neutral, or negative"
    ```

    **Linux:**
    ```bash
    docker run --rm \
        -v "$(pwd)":/app/models \
        -e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
        ghcr.io/inoxoft/whitelightning:latest \
        python -m text_classifier.agent \
        -p "Categorize customer reviews as positive, neutral, or negative"
    ```

    **Windows (PowerShell):**
    ```powershell
    docker run --rm \
        -v "${PWD}:/app/models" \
        -e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
        ghcr.io/inoxoft/whitelightning:latest \
        python -m text_classifier.agent \
        -p "Categorize customer reviews as positive, neutral, or negative"
    ```

4.  That's it! You'll see the generation process in your terminal.

    <img src="media/demo.gif" width="600" alt="WhiteLightning CLI Demo">

    When it's finished, list the files in your directory (`ls -l`). You'll find all the assets for your new model, ready to go:

    **🎮 Try your trained model right here:** [WhiteLightning Playground](https://whitelightning.ai/playground.html)

### 📊 Use Your Own Data

**NEW!** Skip LLM data generation and train directly on your existing datasets. WhiteLightning automatically analyzes your data structure and creates optimized models from real domain data.

```bash
# Create folder for your data
mkdir own_data
cp your_dataset.csv own_data/

# Train on your data (faster, cheaper, more accurate!)
docker run --rm \
    -v "$(pwd)":/app/models \
    -e OPEN_ROUTER_API_KEY="YOUR_OPEN_ROUTER_KEY_HERE" \
    ghcr.io/inoxoft/whitelightning:latest \
    python -m text_classifier.agent \
    -p "Categorize customer reviews as positive, neutral, or negative" \
    --use-own-dataset="/app/models/own_data/your_dataset.csv"
```

**Benefits:**
- ⚡ **3-5x Faster**: No data generation needed
- 💰 **95% Cheaper**: Only uses LLM for data analysis (~$0.01 vs $1-10)
- 🎯 **Higher Accuracy**: Real domain data vs synthetic
- 📁 **Multiple Formats**: Supports CSV, JSON, JSONL, and TXT files
- 🔍 **Auto-Detection**: Automatically identifies text/label columns and classification type



```sh
config.json                # Configuration and analysis
training_data.csv          # Generated training data
edge_case_data.csv         # Challenging test cases
model.onnx                 # ONNX model file
model_scaler.json          # StandardScaler parameters
model_vocab.json           # TF-IDF vocabulary
```

See our **[Complete Documentation](https://whitelightning.ai)** for guides on how to use these files in your language of choice (C++, Rust, iOS, Android, and more).

---

### 💡 Making It Your Own: Example Prompts

The power of WhiteLightning is the `-p` (prompt) argument. You can create a classifier for almost anything just by describing it. Here are some ideas to get you started:

* **Spam Filter**:
    `-p "Classify emails as 'spam' or 'not_spam'"`

* **Topic Classifier**:
    `-p "Determine if a news headline is about 'tech', 'sports', 'world_news', or 'finance'"`

* **Toxicity Detector**:
    `-p "Detect whether a user comment is 'toxic' or 'safe'"`

* **Urgency Detection**:
    `-p "Categorize a support ticket's urgency as 'high', 'medium', or 'low'"`
    
* **Intent Recognition**:
    `-p "Classify the user's intent as 'book_flight', 'check_status', or 'customer_support'"`

The possibilities are endless. For more inspiration and advanced prompt engineering techniques, check out our **[Complete Documentation](https://whitelightning.ai)**.

---

### 🔧 Docker Command Generator

Don't want to manually construct Docker commands? Use our **[Interactive Command Generator](https://whitelightning.ai/command-generator.html)** to build your personalized WhiteLightning commands with a user-friendly interface:

* **📝 Simple Configuration**: Enter your API key and describe your classification task
* **⚙️ Advanced Options**: Configure model type, activation functions, language settings, and more
* **🖥️ Platform Detection**: Automatically generates the correct command format for macOS, Linux, or Windows
* **📋 One-Click Copy**: Copy the generated command directly to your clipboard
* **💡 Smart Defaults**: Intelligent parameter suggestions based on your task description

**Features:**
- **Model Type Selection**: Choose between TensorFlow, PyTorch, or Scikit-learn
- **Activation Functions**: Auto-detect or manually select sigmoid/softmax
- **Custom Datasets**: Easy configuration for using your own data files
- **Language Support**: Set primary language for multilingual classification
- **Performance Tuning**: Adjust batch size, refinement cycles, and feature limits

**Perfect for:**
- First-time users who want guided setup
- Complex configurations with multiple parameters
- Teams sharing standardized commands
- Quick experimentation with different settings

---

### 🧪 Testing & Validation

Want to test your ONNX models across multiple programming languages? Check out our **[WhiteLightning Test Framework](https://github.com/whitelightning-ai/whitelightning-test)** - a comprehensive cross-language testing suite that validates your models in:

* **8 Programming Languages**: Python, Java, C++, C, Node.js, Rust, Dart, and Swift
* **Performance Benchmarking**: Detailed timing, memory usage, and throughput analysis
* **Automated Testing**: GitHub Actions workflows for continuous validation
* **Real-world Scenarios**: Test with custom inputs and edge cases

Perfect for ensuring your WhiteLightning models work consistently across all target platforms and deployment environments.

### 🌐 Documentation & Website

Need comprehensive guides and documentation? Check out our **[WhiteLightning Site](https://github.com/whitelightning-ai/whitelightning-site)** - this repository hosts the official website for WhiteLightning at https://whitelightning.ai, a cutting-edge LLM distillation tool with detailed documentation, tutorials, and implementation guides.

### 📚 Model Library

Looking for pre-trained models or want to share your own? Visit our **[WhiteLightning Model Library](https://github.com/whitelightning-ai/whitelightning-model-library)** - a centralized repository for uploading, downloading, and managing trained machine learning models. Perfect for sharing community contributions and accessing ready-to-use classifiers.

### 🚀 GitHub Actions Integration

Train your models directly in GitHub Actions! This repository includes a pre-configured workflow that lets you:

* **🤖 Train Models in the Cloud**: No local setup required - train directly in GitHub's infrastructure
* **⚙️ Customizable Parameters**: Set classification prompt, refinement cycles, language, and mock mode
* **🔧 Manual Triggers**: Run training on-demand via GitHub's "Run workflow" button
* **📦 Automatic Artifacts**: Generated models (ONNX, vocab, scaler) are automatically saved as downloadable artifacts
* **✅ Built-in Validation**: ONNX model validation and inference testing included

**To use:**
2. Go to Actions → "Test Model Training" → "Run workflow""
3. Customize training parameters or use defaults
4. Download generated models from the workflow artifacts

Perfect for teams, CI/CD pipelines, or when you need cloud-based model training!

---

### 🔧 Troubleshooting

**File Permissions:**
WhiteLightning automatically handles all file permission issues across platforms. Generated files will have correct ownership on your host system without any additional configuration.

**Windows Path Issues:**
Use PowerShell and `${PWD}` instead of `$(pwd)` in your commands.

**Container Access Issues:**
If you encounter any Docker-related issues, ensure Docker is running and you have proper permissions to run Docker commands.

### Advanced Setup

Want to build from source or customize the Docker image? Check out the **[Local Setup Guide](docs/docker/README.md)**.

### Contributing

We welcome all contributions! The best way to start is by joining our **[Discord Server](https://discord.com/invite/QDj8NS2yDt)** and chatting with the team. We're happy to help you get started.

### License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## 🚀 Quick Start

```bash
# Basic usage (automatic activation detection)
python text_classifier/agent.py -p "Classify movie reviews as positive, negative, or neutral"

# Using your own dataset (automatic detection)
python text_classifier/agent.py -p "Emotion classifier" --use-own-dataset=data/emotions.csv

# Override activation function (advanced users)
python text_classifier/agent.py -p "Emotion classifier" --use-own-dataset=data/emotions.csv --activation sigmoid

# Available activation options
--activation auto      # Smart automatic detection (default)
--activation sigmoid   # For multi-label classification
--activation softmax   # For single-label classification
```

## 🎯 Activation Function Guide

### When to Use Each Activation

**Sigmoid** (`--activation sigmoid`):
- ✅ Multi-label: One sample can have multiple labels
- ✅ Independent classes: `"action,comedy,drama"`
- ✅ Tags, symptoms, characteristics
- ✅ Example: Movie genres, article tags, medical symptoms

**Softmax** (`--activation softmax`):
- ✅ Single-label: One sample has exactly one label
- ✅ Mutually exclusive: `"positive"` OR `"negative"` OR `"neutral"`
- ✅ Categories, emotions, languages
- ✅ Example: Sentiment analysis, document categories

**Auto** (`--activation auto`):
- 🤖 System analyzes your data structure
- 🔍 Detects comma-separated labels → sigmoid
- 📊 Detects single labels → softmax
- 💡 Shows reasoning and alternatives

## 📊 Overview of the Dataset Creation and Model Training Process

### 1. 🧠 Dataset Creation: Using LLMs to Generate Realistic and Diverse Data

We leveraged a **Large Language Model (LLM)** to generate a high-quality dataset tailored for a **multi-label relevance classification task**.

One of the most critical components in this process is designing a **high-quality, precise prompt**. A well-constructed prompt directly influences the quality, diversity, and consistency of the generated data. Poor prompt design can lead to repetitive, vague, or irrelevant samples that weaken the training process.

---

### 🎯 Why the Prompt Matters

- Clearly defines **data structure and format** (JSON with text and labels).
- Controls the **topic variety**, avoiding bias or over-representation.
- Encourages **contextual relevance** and realistic scenarios.
- Guides the LLM to avoid **mentioning specific entities** unless needed.

> ✅ **Prompt engineering is not optional — it’s a core part of the dataset design pipeline.**

---


### 2. 🧹 Data Cleaning: Ensuring High-Quality Inputs

Before training, we performed **rigorous filtering** and validation of the generated data to ensure quality and consistency.

---

### ✅ Validation Criteria

Each entry in the dataset had to meet the following conditions:

- `article_text` must be a valid, non-empty string.
- `labels` must be a dictionary with **exactly 23 keys**, matching the Cabinet member list.
- All label values must be **floats between 0 and 1**.

--- 
### 3. 🧠 Model Training: Deep Learning with TensorFlow (LSTM)

We trained a deep learning model using the **TensorFlow** framework with an **LSTM-based** architecture designed for multi-label classification.

---

### 📌 Preprocessing

- **Text vectorization**: Converts raw news article text into tokenized sequences.
- **Train/Test split**: We used an 80/20 split to balance training and validation data.

---

### 🧩 Model Architecture

- **Input**: Raw article text (`string`)
- **TextVectorization Layer**  
- **Embedding Layer**:  
  - Vocabulary size: `10,000 tokens`  
  - Embedding dimension: `128`
- **Bidirectional LSTM**: `64 units`
- **GlobalAveragePooling1D + Dense layers**
- **Output Layer**: `23 sigmoid units`  
  (One per U.S. Cabinet member for multi-label prediction)

---

### 📊 Training Configuration

| Parameter      | Value                                 |
|----------------|---------------------------------------|
| Loss           | `BinaryCrossentropy(label_smoothing=0.1)` |
| Metric         | `AUC (multi-label)`                   |
| Optimizer      | `Adam`                                |
| Epochs         | `10`                                  |
| Batch size     | `32`                                  |

---

This architecture enables the model to understand text context and predict **relevance scores** (0 to 1) for **each Cabinet member** simultaneously.

### 📈 4. Evaluation: Model Metrics

To monitor the model's performance and detect overfitting, we tracked and visualized two key metrics:

- **AUC (Area Under Curve)** — for measuring multi-label classification performance.
- **Loss** — to understand convergence during training.

---

### 🔍 Metrics Tracked

- **Train AUC vs Validation AUC**  
  Indicates how well the model is learning to distinguish relevance across multiple Cabinet members.

- **Train Loss vs Validation Loss**  
  Shows whether the model is overfitting or underfitting.

### 🔮 5. Prediction: Inference on New Articles

After training, we used the model to evaluate **relevance scores** for new unseen news articles. The process involves:

### 🧾 Input
A raw English-language article (1–3 sentences) about any global topic.

### ⚙️ Processing Steps

1. The raw article is passed as a string input.
2. It is transformed into a numerical sequence using the `TextVectorization` layer.
3. The model predicts a **23-dimensional sigmoid output**, representing how relevant the article is to each U.S. Cabinet member.

---
### 💾 6. Saving the Trained Model

To reuse the trained model for future inference or deployment, we saved it in HDF5 format using Keras.
# 💼 Business Value

| **Feature**               | **Benefit**                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| **LLM Generation**         | Fast, scalable, and domain-aware dataset creation                            |
| **Multi-Label Output**     | Enables relevance prediction across multiple stakeholder roles               |
| **Automated**              | Eliminates manual labeling, saving hundreds of annotation hours              |
| **Adaptable**              | Easily extendable to new domains, entities, and use cases                    |
| **ONNX Conversion**        | Model can be exported to ONNX format for lightweight deployment              |
| **Cross-Platform Inference** | ONNX allows the model to run efficiently on **web**, **mobile (iOS/Android)**, and **edge devices (e.g., Jetson, Raspberry Pi)** |

---


## 💰 Cost Estimate: Generating 10,000 Examples with Gemini 2.5 Flash Preview

### 📌 Model Information (via OpenRouter):
| Parameter      | Value                                      |
|----------------|---------------------------------------------|
| Model          | `google/gemini-2.5-flash-preview`           |
| Input Price    | $0.15 per 1M tokens                         |
| Output Price   | $0.60 per 1M tokens                         |
| Context Window | 1,048,576 tokens (~1M+)                     |

---

### 📊 Assumptions:
- Average input prompt length: **~300 tokens**
- Average output (JSON with article + 23-label dict): **~500 tokens**
- **Total per example**: ~800 tokens

---

### 🔹 Token Breakdown for 10,000 Examples:
- **Total tokens**: 800 × 10,000 = **8,000,000 tokens**
  - Input: ~3,000,000 tokens
  - Output: ~5,000,000 tokens

---

### 💵 Cost Calculation:
- **Input**:  
  `3,000,000 × $0.15 / 1,000,000 = $0.45`

- **Output**:  
  `5,000,000 × $0.60 / 1,000,000 = $3.00`

---

### ✅ **Total Estimated Cost: `$3.45 USD`**

---



# 🧾  Conclusion

This project showcases how combining **Large Language Models (LLMs)** with **deep learning architectures** enables us to automate complex classification pipelines at scale.

### ✅ Key Takeaways:

- **Scalable**: Can generate and process thousands of examples with minimal human input  
- **Cost-Effective**: Reduces the need for manual annotation and data collection  
- **Customizable**: Easily adapted to other industries (e.g., legal, healthcare, finance) and tasks (e.g., sentiment, topic classification)

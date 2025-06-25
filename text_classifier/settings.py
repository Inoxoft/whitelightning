import os
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration ---
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "YOUR_OPEN_ROUTER_API_KEY")
OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Default Models ---
DEFAULT_CONFIG_MODEL = (
    "x-ai/grok-3-beta"  # "anthropic/claude-3-opus" # More capable model
)
DEFAULT_DATA_GEN_MODEL = "mistralai/mistral-nemo"  # Less restrictive for content moderation data

# --- Default Paths ---
DEFAULT_OUTPUT_PATH = "models"  # Changed for differentiation
RAW_RESPONSES_DIR = "api_requests"
TRAINING_DATASET_FILENAME = "training_data.csv"
EDGE_CASE_DATASET_FILENAME = "edge_case_data.csv"
CONFIG_FILENAME = "generation_config.json"
CLASSIFIER_META_FILENAME = "classifier_metadata.json"  # For num_classes, class_labels

# --- Data Generation Parameters ---
DEFAULT_TRAINING_DATA_VOLUME = 1000
DATA_GEN_BATCH_SIZE = 10  # Number of parallel requests
MIN_DATA_LINE_LENGTH = 10  # Minimum characters for a valid data line
DEFAULT_EDGE_CASE_VOLUME = 100
PROMPT_REFINEMENT_BATCH_SIZE = 1  # Number of samples per class for refinement

# --- Labels (These are now more like symbolic placeholders, actual labels come from config) ---
# POSITIVE_LABEL = 1 # Will be deprecated for specific class names
# NEGATIVE_LABEL = 0 # Will be deprecated for specific class names

# --- Feature Control ---
DEFAULT_PROMPT_REFINEMENT_CYCLES = 1  # How many times to refine prompts
DEFAULT_GENERATE_EDGE_CASES = True

# --- Data Quality Control ---
DUPLICATE_RATE_THRESHOLD = 5.0  # Percentage threshold for duplicate rate warnings

# --- Prompts ---
CONFIG_SYSTEM_PROMPT = "You are an expert AI assistant specializing in data generation and configuration for machine learning. Follow instructions precisely and provide output in the requested JSON format."

CONFIG_USER_PROMPT_TEMPLATE = """
Given the following problem description: "{problem_description}"

1.  **Problem Analysis:**
    *   Summarize the core classification task in one sentence.
    *   Determine the appropriate model type:
        - `binary_sigmoid`: For simple yes/no or true/false classification with probability output
        - `multilabel_sigmoid`: For multi-label classification where classes are not mutually exclusive
        - `multiclass_softmax`: For single-label classification where classes are mutually exclusive
    *   List the distinct class labels as an array of strings (e.g., ["spam", "ham", "promotional"]) for multiclass.
      For binary classification, use a simple "1" or "0".

2.  **Data Generation Prompts:**
    *   For *each* class label identified, create a specific, detailed prompt to generate synthetic text data representative of that class. Ensure prompts encourage diversity and realism.
    *   The prompts should be structured as a JSON object where keys are the class labels and values are the prompt strings.

3.  **Configuration:**
    *   Suggest a short, snake_case `model_prefix` for file naming.
    *   Recommend a `training_data_volume` (e.g., 500, 1000 ... 10000) for the total dataset size (it will be split among classes).

Return *only* JSON format:
{{
  "summary": "...",
  "classification_type": "binary_sigmoid|multilabel_sigmoid|multiclass_softmax",
  "class_labels": ["label1", "label2", ...],
  "prompts": {{
    "label1": "Prompt for label1...",
    "label2": "Prompt for label2..."
  }},
  "model_prefix": "...",
  "training_data_volume": 1000
}}
"""

PROMPT_REFINEMENT_TEMPLATE = """
**Goal:** Refine prompts for generating classification training data.

**Problem Description:** {problem_description}
**Classification Type:** {classification_type}
**Class Labels:** {class_labels_str}

**Current Prompts:**
{current_prompts_json_str}

**Sample Data Generated (using current prompts):**
{samples_by_class_str}

**Task:**
1.  **Evaluate:** Assess the quality and relevance of the generated samples for *each class* based on the Problem Description. Are they distinct? Diverse? Do they accurately represent their respective classes?
2.  **Suggest Improvements:** Provide specific, actionable suggestions on how to modify *each* prompt to generate data that is more diverse, more accurately reflects the class nuances, and might lead to a better classifier. Focus on clarity, specificity, and covering potential variations for each class.

**Output Format (Return *only* JSON):**
{{
  "evaluation_summary": "Your brief overall evaluation of the current prompts and sample data.",
  "refined_prompts": {{
    "label1": "Your improved prompt for label1.",
    "label2": "Your improved prompt for label2."
  }}
}}
"""

# Edge case prompts now take class_label as an argument
# We might need a more sophisticated way to generate edge cases for multiclass,
# e.g., "text that is class_X but looks like class_Y". For now, we'll generate hard examples *for a specific class*.

EDGE_CASE_PROMPT_TEMPLATE = """
**Goal:** Generate challenging examples for the class "{class_label}" for testing a {classification_type} classifier.

**Problem Description:** {problem_description}
**All Class Labels:** {all_class_labels_str}

**Task:** Generate diverse text samples that ARE examples of "{class_label}" according to the Problem Description, but are intentionally designed to be difficult for a classifier to identify correctly. Focus on:
*   Borderline cases that barely meet the "{class_label}" criteria.
*   Examples disguised to look like other classes (e.g., subtle "{class_label}" signals).
*   Samples using unusual phrasing, jargon, or obfuscation related to the "{class_label}" class.
*   Ambiguous examples that require careful reading to identify as "{class_label}".

Generate only the text samples, in json format using numbers as keys. Do not add labels or explanations. Samples should be in {language} language.
"""

# This general negative edge case prompt might still be useful if we want "none of the above" hard examples.
# Or we adapt it to be "hard negative for class X" (i.e., looks like X but is not).
# For simplicity now, the above template is per-class. Let's defer a "universal negative" edge case.

PERFORMANCE_ANALYSIS_PROMPT_TEMPLATE = """
**Goal:** Analyze classifier performance and suggest improvements based on test results.

**Problem Description:** {problem_description}
**Classification Type:** {classification_type}
**Class Labels:** {class_labels_str}

The model was trained on data generated using these prompts:
**Final Data Generation Prompts Used:**
{final_prompts_json_str}

**Test Performance Summary:**
*Sample Misclassifications (or challenging cases):*
(Format: "Text Snippet", True Label, Predicted Label/Probabilities)
{test_results_summary}

**Task:**
1.  **Analyze Weaknesses:** Based on the problem description, class definitions, and the sample test results (especially errors or low-confidence correct predictions), identify the likely weaknesses of the model. Which classes are confused? What types of examples does it struggle with for each class?
2.  **Identify Strengths:** What does the model seem to handle well for each class?
3.  **Suggest Improvements:** Provide concrete recommendations focused on *improving the data generation process* for future iterations. Should specific class prompts be changed further? Do we need more specific types of data (e.g., more borderline cases between class_X and class_Y)? Suggest specific prompt modifications or data augmentation ideas if possible for individual classes.

**Output format:** Free-form text analysis. Start with a summary.
"""

DATA_GEN_SYSTEM_PROMPT = "Users will request specific data. Respond only with realistic, simple text examples in {language} language. Generate news headlines or short article excerpts (1-2 sentences max). Do not include labels or metadata. Provide 50 entries, one per line. Each line should be a simple, clean text sample without quotes or formatting."

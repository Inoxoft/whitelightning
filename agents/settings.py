# settings.py
import os
from dotenv import load_dotenv

load_dotenv() # Load .env file if you use one for secrets

# --- API Configuration ---
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY", "YOUR_OPEN_ROUTER_API_KEY")
OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Default Models ---
# Model for generating configuration and performing analysis/refinement. Needs to be capable.
DEFAULT_CONFIG_MODEL = "anthropic/claude-3.5-sonnet"
# Default model for generating bulk training/edge-case data samples
DEFAULT_DATA_GEN_MODEL = "deepseek/deepseek-coder"

# --- Default Paths ---
DEFAULT_OUTPUT_PATH = "generated_data"
RAW_RESPONSES_DIR = "api_requests"
TRAINING_DATASET_FILENAME = "training_data.csv"
EDGE_CASE_DATASET_FILENAME = "edge_case_data.csv" # New
CONFIG_FILENAME = "generation_config.json" # New

# --- Data Generation Parameters ---
DEFAULT_TRAINING_DATA_VOLUME = 1000
DATA_GEN_BATCH_SIZE = 100
MIN_DATA_LINE_LENGTH = 15
DEFAULT_EDGE_CASE_VOLUME = 100
PROMPT_REFINEMENT_BATCH_SIZE = 4

# --- Labels ---
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# --- Feature Control ---
DEFAULT_PROMPT_REFINEMENT_CYCLES = 1
DEFAULT_GENERATE_EDGE_CASES = True

# --- Prompts ---

# System prompt for Config/Analysis/Refinement Model
CONFIG_SYSTEM_PROMPT = "You are an expert AI assistant specializing in data generation for machine learning. Follow instructions precisely and provide output in the requested format (usually JSON)."

# User prompt template for initial configuration generation
CONFIG_USER_PROMPT_TEMPLATE = """
Given the following problem description: "{problem_description}"

1. Summarize the problem in one sentence.
2. Create initial prompts for generating synthetic data:
   - A "positive" prompt for data representing the problem. Be specific and encourage diversity.
   - A "negative" prompt for data *not* representing the problem. Also encourage diversity.
3. Suggest a short, snake_case `model_prefix`.
4. Recommend a `training_data_volume` (e.g., 500).

Return *only* JSON format:
{{
  "summary": "...",
  "prompts": {{ "positive": "...", "negative": "..." }},
  "model_prefix": "...",
  "training_data_volume": 500
}}
"""

# New: Prompt Template for Refining Data Generation Prompts
PROMPT_REFINEMENT_TEMPLATE = """
**Goal:** Refine prompts for generating binary classification training data.

**Problem Description:** {problem_description}

**Current Positive Prompt:**
{current_positive_prompt}

**Current Negative Prompt:**
{current_negative_prompt}

**Sample Data Generated (using current prompts):**

*Positive Samples:*
{positive_samples}

*Negative Samples:*
{negative_samples}

**Task:**
1.  **Evaluate:** Assess the quality and relevance of the generated samples based on the Problem Description. Are they diverse? Do they accurately represent the positive/negative classes?
2.  **Suggest Improvements:** Provide specific, actionable suggestions on how to modify *both* the Positive and Negative prompts to generate data that is more diverse, more accurately reflects the problem nuances, and might lead to a better classifier. Focus on clarity, specificity, and covering potential variations.

**Output Format (Return *only* JSON):**
{{
  "evaluation_summary": "Your brief evaluation of the current prompts and sample data.",
  "refined_positive_prompt": "Your improved positive prompt text.",
  "refined_negative_prompt": "Your improved negative prompt text."
}}
"""

POSITIVE_EDGE_CASE_PROMPT_TEMPLATE = """
**Goal:** Generate challenging POSITIVE examples for testing a binary classifier.

**Problem Description:** {problem_description}

**Classifier Task:** Distinguish between POSITIVE examples (matching the description) and NEGATIVE examples.

**Task:** Generate diverse text samples that ARE POSITIVE examples according to the Problem Description, but are intentionally designed to be difficult for a classifier to identify correctly. Focus on:
*   Borderline cases that barely meet the positive criteria.
*   Examples disguised to look like negative cases (e.g., subtle positive signals).
*   Samples using unusual phrasing, jargon, or obfuscation related to the positive class.
*   Ambiguous examples that require careful reading.

Generate only the text samples, one per line. Do not add labels or explanations.
"""

NEGATIVE_EDGE_CASE_PROMPT_TEMPLATE = """
**Goal:** Generate challenging NEGATIVE examples for testing a binary classifier.

**Problem Description:** {problem_description}

**Classifier Task:** Distinguish between POSITIVE examples (matching the description) and NEGATIVE examples.

**Task:** Generate diverse text samples that ARE NEGATIVE examples (do NOT match the description), but are intentionally designed to be difficult for a classifier. Focus on:
*   Examples that closely mimic positive cases but lack the key defining elements.
*   Borderline negative cases that are almost positive.
*   Samples using phrasing or topics often associated with the positive class, but used in a negative context.
*   "False positives" - things that might trick a simple classifier.

Generate only the text samples, one per line. Do not add labels or explanations.
"""

PERFORMANCE_ANALYSIS_PROMPT_TEMPLATE = """
**Goal:** Analyze classifier performance and suggest improvements based on test results.

**Problem Description:** {problem_description}
The model was trained on data generated using prompts aiming to solve this problem.

**Final Data Generation Prompts Used:**
*Positive:* {final_positive_prompt}
*Negative:* {final_negative_prompt}

**Test Performance Summary:**
*Overall Accuracy (Optional):* {accuracy:.2f}
*Sample Misclassifications (or challenging cases):*
(Format: "Text Snippet", True Label, Predicted Probability/Label)
{test_results_summary}

**Task:**
1.  **Analyze Weaknesses:** Based on the problem description and the sample test results (especially errors), identify the likely weaknesses of the model. What types of examples does it struggle with?
2.  **Identify Strengths:** What does the model seem to handle well?
3.  **Suggest Improvements:** Provide concrete recommendations focused on *improving the data generation process* for future iterations. Should the prompts be changed further? Do we need more specific types of data (e.g., more borderline cases)? Suggest specific prompt modifications or data augmentation ideas if possible.

**Output format:** Free-form text analysis. Start with a summary.
"""

DATA_GEN_SYSTEM_PROMPT = 'Users will request specific data. Respond only with realistic, generated examples that resemble real-world datasets. Do not include numbering, labels, or extra text beyond the examples. Example should be complete, without meta-labels and incomplete patterns. Provide up to 100 entries per request, each enclosed in double quotes, separated by "\n".'

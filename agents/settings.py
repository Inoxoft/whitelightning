# settings.py
import os
from dotenv import load_dotenv

load_dotenv() # Load .env file if you use one for secrets

# --- API Configuration ---
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY") # Replace with your key or env variable
OPEN_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Default Models ---
# Model used for generating the configuration (prompts, prefix, etc.)
# Choose a capable model like Claude Sonnet, GPT-4o, Grok etc.
DEFAULT_CONFIG_MODEL = "anthropic/claude-3.5-sonnet"
# Default model for generating the actual training data samples
DEFAULT_DATA_GEN_MODEL = "deepseek/deepseek-coder" # Or any other preferred model like openai/gpt-4o-mini

# --- Default Paths ---
DEFAULT_OUTPUT_PATH = "generated_data"
RAW_RESPONSES_DIR = "api_requests" # Subdirectory within output_path for raw json
DATASET_FILENAME = "training_data.csv"

# --- Data Generation Parameters ---
DEFAULT_TRAINING_DATA_VOLUME = 1000
DATA_GEN_BATCH_SIZE = 100
MIN_DATA_LINE_LENGTH = 15

# --- Labels ---
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# --- Prompts --- (Moved here from the main script)
# System prompt for the LLM generating the configuration
CONFIG_SYSTEM_PROMPT = "You are an expert assistant helping set up configurations for machine learning projects. Follow the user's instructions precisely and return JSON output."

# User prompt template for generating the configuration
# Note: Added instructions for generating the DATA_GEN_SYSTEM_PROMPT as well.
CONFIG_USER_PROMPT_TEMPLATE = """
Given the following problem description from a user: "{problem_description}"

1. Summarize the problem in one sentence.
2. Create two prompts for generating synthetic training data:
   - A "positive" prompt for gathering dummy data representing the problem. The data should be diverse in style, case, and length. (e.g., for a spam classifier: "Generate diverse messages that look like spam using different obfuscation techniques, sales pitches, urgency tactics, and styles. Examples: 'B3ST D3ALS!!!', 'URGENT: Action Required!', fake lottery wins, phishing attempts.")
   - A "negative" prompt for gathering dummy data that does *not* represent the problem. This should also be diverse. (e.g., for a spam classifier: "Generate diverse, generic, non-spam messages in different styles, cases, and lengths. Examples: casual chats, formal emails, short greetings, work-related discussions, technical questions.")
3. Based on the positive/negative prompts, create a concise system prompt for the *data generation* model, telling it its role is to generate data based on the user prompts it will receive. (e.g., "You are a data generation assistant. Create text samples according to the user's prompt.")
4. Suggest a short, descriptive, snake_case prefix for naming resources related to this problem (e.g., "spam_classifier", "sentiment_analyzer").
5. Recommend a reasonable starting volume of training data samples (e.g., 500, 1000). Consider the complexity.

Return the output *only* in the following JSON format, with no preamble or explanation:
{{
  "summary": "Concise summary of the problem.",
  "prompts": {{
    "positive": "Positive prompt text for data generation.",
    "negative": "Negative prompt text for data generation."
  }},
  "data_gen_system_prompt": "System prompt for the data generation model.",
  "model_prefix": "suggested_prefix",
  "training_data_volume": 1000
}}
"""

DATA_GEN_SYSTEM_PROMPT = 'Users will request specific data. Respond only with realistic, generated examples that resemble real-world datasets. Do not include numbering, labels, or extra text beyond the examples. Example should be complete, without meta-labels and incomplete patterns. Provide up to 100 entries per request, each enclosed in double quotes, separated by "\n".'

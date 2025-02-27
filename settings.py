import os

from dotenv import load_dotenv

load_dotenv()

POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL")
NEGATIVE_LABEL = os.environ.get("NEGATIVE_LABEL")
OPEN_ROUTER_API_KEY = os.environ.get("OPEN_ROUTER_API_KEY")

MODEL_PREFIX = os.environ.get("MODEL_PREFIX")
PROMPT_POSITIVE = os.environ.get("PROMPT_POSITIVE")
PROMPT_NEGATIVE = os.environ.get("PROMPT_NEGATIVE")
DATA_COLUMN_NAME = os.environ.get("DATA_COLUMN_NAME")
LABEL_COLUMN_NAME = os.environ.get("LABEL_COLUMN_NAME")
SYSTEM_PROMPT = "Users will request specific data. Respond only with realistic, generated examples that resemble real-world datasets. Provide up to 100 entries per request, each enclosed in double quotes, separated by \n (endline character). Do not include numbering, labels, or extra text beyond the examples."

MODELS_PATH = "models/"
RESULTS_PATH = "results/"
TRAINING_DATA_PATH = "training_data/"
TESTING_DATA_PATH = "testing_data/"

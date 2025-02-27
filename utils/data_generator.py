from openai import OpenAI
import json
from datetime import datetime
from settings import PROMPT_POSITIVE, PROMPT_NEGATIVE, SYSTEM_PROMPT, MODEL_PREFIX, \
    TRAINING_DATA_PATH, NEGATIVE_LABEL, POSITIVE_LABEL, OPEN_ROUTER_API_KEY


class OpenRouterModels:
    GPT4omini = "openai/gpt-4o-mini"
    DeepSeek = "deepseek/deepseek-r1:free"
    Grok2 = "x-ai/grok-2-1212"
    Claude = "anthropic/claude-3.7-sonnet"
    GeminiFlash = "google/gemini-2.0-flash-lite-001"


def ask_openai(prompt, label, model):
    client = OpenAI(
        api_key=OPEN_ROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10000,
    )

    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"../api_requests/{MODEL_PREFIX}{model.replace('/', '-')}{current_datetime}.json"

    result = {
        "question": prompt,
        "answer": completion.choices[0].message.content
    }

    with open(filename, 'w') as file:
        json.dump(result, file, indent=4)

    print(f"Result saved to {filename}")

    with open(f"../{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv", 'a') as file:
        for line in result["answer"].split('\n'):
           if len(line) > 15:
               file.write(f"{line.strip(' ')},{label}\n")

    return result


for i in range(1):
    ask_openai(PROMPT_POSITIVE, POSITIVE_LABEL, OpenRouterModels.GeminiFlash)
    ask_openai(PROMPT_NEGATIVE, NEGATIVE_LABEL, OpenRouterModels.GeminiFlash)

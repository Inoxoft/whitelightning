from openai import AsyncOpenAI
import json
from datetime import datetime
import asyncio
import logging
from settings import PROMPT_POSITIVE, PROMPT_NEGATIVE, SYSTEM_PROMPT, MODEL_PREFIX, \
    TRAINING_DATA_PATH, NEGATIVE_LABEL, POSITIVE_LABEL, OPEN_ROUTER_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class OpenRouterModels:
    GPT4omini = "openai/gpt-4o-mini"
    DeepSeek = "deepseek/deepseek-r1:free"
    Grok2 = "x-ai/grok-2-1212"
    Claude = "anthropic/claude-3.7-sonnet"
    GeminiFlash = "google/gemini-2.0-flash-lite-001"


async def ask_openai_async(prompt, label, model):
    try:
        client = AsyncOpenAI(
            api_key=OPEN_ROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

        completion = await client.chat.completions.create(
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

        try:
            with open(filename, 'w') as file:
                json.dump(result, file, indent=4)
            logging.info(f"Result saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save result to file: {str(e)}")
            raise

        try:
            with open(f"../{TRAINING_DATA_PATH}{MODEL_PREFIX}_dataset.csv", 'a') as file:
                for line in result["answer"].split('\n'):
                    if len(line) > 15:
                        file.write(f"{line.strip(' ')},{label}\n")
            logging.info(f"Data appended to dataset for model {model}")
        except IOError as e:
            logging.error(f"Failed to append to dataset: {str(e)}")
            raise

        return result

    except Exception as e:
        logging.error(f"Error in ask_openai_async: {str(e)}")
        raise


async def make_parallel_requests():
    total_successful = 0
    total_failed = 0

    for iteration in range(5):
        tasks = []
        logging.info(f"Starting iteration {iteration + 1}")

        for i in range(5):
            prompt = PROMPT_POSITIVE if i % 2 == 0 else PROMPT_NEGATIVE
            label = POSITIVE_LABEL if i % 2 == 0 else NEGATIVE_LABEL
            tasks.append(ask_openai_async(prompt, label, OpenRouterModels.DeepSeek))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Request {idx + 1} in batch {iteration + 1} failed: {str(result)}")
                    total_failed += 1
                else:
                    total_successful += 1

            logging.info(f"Completed batch {iteration + 1} of 5 requests")
        except Exception as e:
            logging.error(f"Batch {iteration + 1} failed: {str(e)}")
            total_failed += 5

    logging.info(f"All requests completed. Successful: {total_successful}, Failed: {total_failed}")


if __name__ == "__main__":
    try:
        asyncio.run(make_parallel_requests())
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    except Exception as e:
        logging.critical(f"Application failed: {str(e)}")

import argparse
import asyncio
import os
import csv
import random
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
open_router_key = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=open_router_key,
)

SYSTEM_PROMPT = """You are an expert data generation assistant for machine learning tasks. Your job is to create clean, diverse, and realistic synthetic data for training classification models. Always follow formatting instructions exactly. Use natural language that reflects real-world usage. Ensure class balance and label clarity."""

def build_prompt(language: str) -> str:
    return f"""Generate 10 short news headlines in {language}. Each headline should clearly belong to one of the following categories:
        Politics, Sports, Business, World, Technology, Entertainment, Science, Health, Education, Environment.

        Return the results as plain CSV text with two columns: text and label.

        Each row should have one headline and its corresponding label.

        Make sure that all 10 categories are covered exactly once."""

async def get_examples_batch(prompt: str, model: str):
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
    )
    return completion.choices[0].message.content

async def fetch_batch(index: int, prompt: str, model: str):
    try:
        print(f"🔄 Starting batch {index}...")
        csv_text = await get_examples_batch(prompt, model)
        lines = csv_text.strip().split("\n")
        lines = [line for line in lines if not line.lower().strip().startswith("text,label")]
        rows = [line.split(",", 1) for line in lines if "," in line]
        print(f"✅ Batch {index}: {len(rows)} rows")
        return rows
    except Exception as e:
        print(f"❌ Batch {index} failed: {e}")
        return []

async def generate_dataset(language: str, model: str, num_batches: int, batch_size: int):
    prompt = build_prompt(language)
    all_rows = []
    total_examples = 0

    for start in range(0, num_batches, batch_size):
        end = min(start + batch_size, num_batches)
        print(f"🚀 Launching batch group {start + 1} to {end}...")

        tasks = [fetch_batch(i + 1, prompt, model) for i in range(start, end)]
        results = await asyncio.gather(*tasks)
        batch_rows = [row for batch in results for row in batch]
        all_rows.extend(batch_rows)
        total_examples += len(batch_rows)

        print(f"✅ Group {start + 1}-{end} collected: {len(batch_rows)} rows")

    print(f"🧮 Total collected examples: {total_examples}")
    random.shuffle(all_rows)

    train_size = int(len(all_rows) * 0.8)
    train_rows = all_rows[:train_size]
    test_rows = all_rows[train_size:]

    train_path = f"training_data/news_train_{language}.csv"
    test_path = f"testing_data/news_test_{language}.csv"
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("testing_data", exist_ok=True)

    for path, rows in [(train_path, train_rows), (test_path, test_rows)]:
        with open(path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, quotechar="'", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["text", "label"])
            writer.writerows(rows)
        print(f"💾 Saved {len(rows)} rows to: {path}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic news classification dataset")
    parser.add_argument("--language", default="hindi", help="Language for dataset")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model")
    parser.add_argument("--batches", type=int, default=2500, help="Total number of batches")
    parser.add_argument("--batch-size", type=int, default=20, help="Concurrent batch size")

    args = parser.parse_args()
    asyncio.run(generate_dataset(args.language, args.model, args.batches, args.batch_size))

if __name__ == "__main__":
    main()


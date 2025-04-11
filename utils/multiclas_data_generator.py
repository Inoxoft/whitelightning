import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import csv
import random

load_dotenv()
api_key = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openai/gpt-4o"

SYSTEM_PROMPT = ("""You are an expert data generation assistant for machine learning tasks. Your job is to create clean, diverse, and realistic synthetic data for training classification models. Always follow formatting instructions exactly. Use natural language that reflects real-world usage. Ensure class balance and label clarity."""
                 
)
prompt = ("""Generate 10 short news headlines. Each headline should clearly belong to one of the following categories:
                Politics, Sports, Business, World, Technology, Entertainment, Science, Health, Education, Environment.

                Return the results as plain CSV text with two columns: text and label.

                Each row should have one headline and its corresponding label.

                Make sure that all 10 categories are covered exactly once."""
    
)

TRAIN_FILE = "news_train.csv"
TEST_FILE = "news_test.csv"

WRITE_HEADER = True

async def get_examples_batch():
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        extra_headers={
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "NER Dataset Generator",
        }
    )
    return completion.choices[0].message.content

async def fetch_batch(index: int):
    try:
        print(f"🔄 Starting batch {index}...")
        csv_text = await get_examples_batch()
        lines = csv_text.strip().split("\n")

        lines = [line for line in lines if not line.lower().strip().startswith("text,label")]

        rows = [line.split(",", 1) for line in lines if "," in line]

        print(f"✅ Batch {index}: {len(rows)} rows")
        return rows

    except Exception as e:
        print(f"❌ Batch {index} failed: {e}")
        return []


async def main():
    num_batches = 10000
    batch_size = 500
    total_examples = 0
    all_rows = []

    for start in range(0, num_batches, batch_size):
        end = min(start + batch_size, num_batches)
        print(f"🚀 Launching batch group {start + 1} to {end}...")

        tasks = [fetch_batch(i + 1) for i in range(start, end)]
        results = await asyncio.gather(*tasks)

        batch_rows = [row for batch in results for row in batch]
        all_rows.extend(batch_rows)
        total_examples += len(batch_rows)

        print(f"✅ Batch group {start + 1}-{end} completed: {len(batch_rows)} examples collected.")

    print(f"🧮 Total examples collected: {total_examples}")

    random.shuffle(all_rows)

    train_ratio = 0.8
    train_size = int(len(all_rows) * train_ratio)

    train_rows = all_rows[:train_size]
    test_rows = all_rows[train_size:]

    for path, rows in [(TRAIN_FILE, train_rows), (TEST_FILE, test_rows)]:
        with open(path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, quotechar="'", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["text", "label"])
            writer.writerows(rows)
        print(f"✅ Saved {len(rows)} rows to: {path}")

if __name__ == "__main__":
    asyncio.run(main())


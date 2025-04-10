import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
api_key = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openai/gpt-4o"
SYSTEM_PROMPT = (
    "You are an expert linguistic data generator trained to create high-quality datasets for Named Entity Recognition (NER) tasks. "
    "Your job is to generate realistic sentences in English and annotate each word with its corresponding NER tag using the BIO format. "
    "The supported entity types are: PERSON, ORGANIZATION, LOCATION, PRODUCT. Always return data in JSON format with 'tokens' and 'tags'. "
    "Avoid repetition and use a wide range of vocabulary. Return only the JSON response without any additional text or explanation. "
    "Do not include the word 'json'."
)
prompt = (
    "Generate 10 short sentences, each containing at least one named entity. For each sentence, return a list of tokens and corresponding NER tags using the BIO format. "
    "Use the following entity types: PERSON, ORGANIZATION, LOCATION, PRODUCT. The output should be in JSON format as a list of objects with two fields: "
    "'tokens' (a list of words) and 'tags' (a list of BIO tags). Avoid punctuation tokens unless necessary. "
    "Return only the JSON response without any extra text. Do not mention the word json."
)

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
        data = await get_examples_batch()
        ready_data = data.replace("json", "").strip()

        if ready_data.startswith('"'):
            ready_data = json.loads(ready_data)

        start = ready_data.find('[')
        end = ready_data.rfind(']')
        json_part = ready_data[start:end+1]
        batch_data = json.loads(json_part)

        print(f"✅ Batch {index}: {len(batch_data)} examples")
        return batch_data

    except Exception as e:
        print(f"❌ Batch {index} failed: {e}")
        print(f"📝 Preview:\n{str(data)[:200] if 'data' in locals() else 'No response received'}")
        return []

async def main():
    num_batches = 200
    output_file = "ner_dataset_2000.json"

    print(f"🚀 Launching {num_batches} parallel batch requests...")

    tasks = [fetch_batch(i+1) for i in range(num_batches)]
    results = await asyncio.gather(*tasks)

    all_data = [example for batch in results for example in batch]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Dataset saved to: {output_file}")
    print(f"✅ Total examples generated: {len(all_data)}")

if __name__ == "__main__":
    asyncio.run(main())

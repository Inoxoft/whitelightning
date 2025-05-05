import asyncio
import os
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

NUM_EXAMPLES = 100
PARALLEL_LIMIT = 20
OUTPUT_FILE = "tr_data/cabinet_dataset.json"

load_dotenv()
api_key = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openai/gpt-4o"

cabinet_keys = [
    "JD_Vance", "Marco_Rubio", "Scott_Bessent", "Pete_Hegseth", "Pam_Bondi",
    "Doug_Burgum", "Brooke_Rollins", "Howard_Lutnick", "Lori_Chavez_DeRemer",
    "Robert_F_Kennedy_Jr", "Scott_Turner", "Sean_Duffy", "Chris_Wright",
    "Linda_McMahon", "Doug_Collins", "Kristi_Noem", "Lee_Zeldin", "Tulsi_Gabbard",
    "John_Ratcliffe", "Jamieson_Greer", "Kelly_Loeffler", "Russell_Vought",
    "Susie_Wiles"
]

cabinet_prompt = "\n".join([
    f"\u2022 {key.replace('_', ' ')}" for key in cabinet_keys
])

SYSTEM_PROMPT = (
    "You are an expert data generation assistant for machine learning tasks. "
    "Your job is to create clean, diverse, and realistic synthetic data. "
    "Always follow formatting instructions exactly."
)

PROMPT_TEMPLATE = f"""
Generate one realistic and concise **unique** news article in English (1–3 sentences max). 
It must not repeat or rephrase previously used content. Each article should be fresh and contextually distinct.

The article can be about any national or international topic — including politics, economy, health, technology, environment, social issues, or global events — but it should NOT be focused on any specific U.S. Cabinet member.

Then, return a JSON object with:

- \"article_text\": the text of the news article
- \"labels\": a dictionary mapping each U.S. Cabinet member key to a float between 0 and 1, representing how relevant the article is to their role (0 = not relevant, 1 = highly relevant)

Use this list of Cabinet members:
{cabinet_prompt}

Instructions:
- Do NOT focus the article on any Cabinet member
- The topic must be realistic, **unique**, and context-rich
- Avoid rephrasing or duplicating earlier examples
- Assign relevance scores for each Cabinet member based on how the topic may relate to their domain
- If the article is pertinent to all, assign 1 to all; if irrelevant to all, assign 0 to all
- Return only valid JSON in this exact format:

{{
  \"article_text\": \"...\",
  \"labels\": {{
    \"Marco_Rubio\": 0.7,
    \"Robert_F_Kennedy_Jr\": 0.1,
    ...
  }}
}}
"""

async def get_example():
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE}
        ],
        max_tokens=2000,
        extra_headers={
            "HTTP-Referer": "https://your-site.com",
            "X-Title": "Cabinet Dataset Generator",
        }
    )
    return completion.choices[0].message.content

async def fetch_example(index: int):
    try:
        print(f"🔄 Generating example {index}...")
        data = await get_example()

        json_start = data.find("{")
        json_end = data.rfind("}") + 1
        json_data = data[json_start:json_end]

        example = json.loads(json_data)
        print(f"✅ Example {index} generated")
        return [example]

    except Exception as e:
        print(f"❌ Example {index} failed: {e}")
        print(f"📝 Preview:\n{str(data)[:500] if 'data' in locals() else 'No response received'}")
        return []

async def main():
    print(f"🚀 Generating {NUM_EXAMPLES} examples (parallel={PARALLEL_LIMIT})...")
    all_data = []

    for i in range(0, NUM_EXAMPLES, PARALLEL_LIMIT):
        batch_range = list(range(i, min(i + PARALLEL_LIMIT, NUM_EXAMPLES)))
        tasks = [fetch_example(index + 1) for index in batch_range]
        results = await asyncio.gather(*tasks)
        for batch in results:
            all_data.extend(batch)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Dataset saved to: {OUTPUT_FILE}")
    print(f"✅ Total examples generated: {len(all_data)}")

if __name__ == "__main__":
    asyncio.run(main())

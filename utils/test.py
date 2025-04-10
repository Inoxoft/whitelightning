import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import json

load_dotenv()
api_key = os.getenv("OPEN_ROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openai/gpt-4o"
SYSTEM_PROMPT = (
    "helpful assistant"
)
prompt = (
    "Generate 10 short English sentences with clearly Positive sentiment. These sentences should sound like real-world opinions or reviews, expressing satisfaction, happiness, or appreciation. Each sentence should be unique and emotionally positive."
    
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
    print( completion.choices[0].message.content)



if __name__ == "__main__":
    
    asyncio.run(get_examples_batch())

import argparse
import asyncio
import json
import re
from agents.multiclass_generate_dataset import generate_dataset, client
from agents.multiclass_train_model import train_model_from_params

def print_error(message: str):
    print(f"\033[91m{message}\033[0m")

async def get_prompt_arguments(prompt: str):
    rules = """
                You are a system that extracts the task type and class labels from a user prompt.

                Rules:
                - Respond ONLY in pure JSON format without any explanations, markdown, code blocks, or text.
                - JSON format must be exactly: {"text_type": "task type", "labels": ["label1", "label2", ...]}.
                - "labels" must contain ONLY clean label names — single words or short phrases, without descriptions or sentences.
                - No extra text before or after JSON.
                - Always lowercase "text_type" value if possible.

                Example input:
                Prompt: Detect news topic in tree types: Sport, Business, Health
                Expected output: {"text_type": "news", "labels": ["Sport", "Business", "Health"]}
                """

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": rules.strip()
            },
            {
                "role": "user",
                "content": f"""
{rules.strip()}

Prompt: {prompt}
"""
            }
        ],
        max_tokens=400,
    )
    return completion.choices[0].message.content


async def main():
    parser = argparse.ArgumentParser(description="Generate dataset and train model in one command.")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt describing the task, including text type and labels")
    parser.add_argument("--lang", default="english", help="Language for dataset (default: english)")
    parser.add_argument("--platform", default="tensorflow", choices=["tensorflow", "torch", "sklearn"], help="Framework to train (default: tensorflow)")
    parser.add_argument("--batches", type=int, default=40, help="Total number of batches to generate (default: 300)")
    parser.add_argument("--batch-size", type=int, default=20, help="Parallel requests at once (default: 20)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model to use for generation (default: gpt-3.5-turbo)")

    args = parser.parse_args()

    print("🔍 Asking LLM to parse the prompt...")
    response = await get_prompt_arguments(args.prompt)

    try:
        response = response.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(response)
        text_type = parsed["text_type"].lower()  
        labels = parsed["labels"]
    except Exception as e:
        print_error(f"❌ Error parsing LLM response: {e}")
        print_error(f"Received response: {response}")
        exit(1)


    print(f"📰 Text type: {text_type}")
    print(f"🏷️ Labels: {labels}")

    print("🛠️ Generating dataset...")
    await generate_dataset(
        language=args.lang,
        model=args.model,
        text_type=text_type,
        labels=labels,
        num_batches=args.batches,
        batch_size=args.batch_size,
    )

    print("🏗️ Starting model training...")

    train_model_from_params(
        language=args.lang,
        text_type=text_type,
        labels=labels,
        platform=args.platform,
        epochs=args.epochs
    )

    print("✅ Done: Dataset generated and model trained!")

if __name__ == "__main__":
    asyncio.run(main())

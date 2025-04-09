import asyncio
from agents.text import BinaryClassifierDataGenerator

async def main():
    generator = BinaryClassifierDataGenerator(
        problem_description="Detect urgency of the customer feedback",
        selected_data_gen_model="openai/gpt-4o-mini",
        output_path="feedback_urgency_classifier"
    )
    await generator.generate_data_async()

if __name__ == "__main__":
     asyncio.run(main())

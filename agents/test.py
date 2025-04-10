import asyncio
from agents.text import BinaryClassifierDataGenerator

async def main():
    generator = BinaryClassifierDataGenerator(
        problem_description="Detect urgency of the customer feedback",
        selected_data_gen_model="openai/gpt-4o-mini",
        output_path="models"
    )
    await generator.generate_data_and_train_model_async(model_type="tensorflow")

if __name__ == "__main__":
     asyncio.run(main())

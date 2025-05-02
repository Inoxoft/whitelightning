import asyncio
from agents.binary_classifier_agent import BinaryClassifierDataGenerator


async def main():
    generator = BinaryClassifierDataGenerator(
        problem_description="I need a model to detect pirate speech in text.",
        selected_data_gen_model="openai/gpt-4o-mini",
        output_path="models",
    )
    await generator.generate_data_and_train_model_async(model_type="tensorflow")


if __name__ == "__main__":
    asyncio.run(main())

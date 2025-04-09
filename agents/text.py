import asyncio
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal

from openai import AsyncOpenAI, OpenAIError # Use OpenAIError for broad exceptions

# Import settings - ensure settings.py is in the same directory or Python path
import agents.settings as settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class BinaryClassifierDataGenerator:
    """
    Generates configuration and synthetic training data for a binary classifier
    based on a user-provided problem description using an LLM.
    """

    def __init__(
        self,
        problem_description: str,
        selected_data_gen_model: str = settings.DEFAULT_DATA_GEN_MODEL,
        library: Optional[Literal['pytorch', 'tensorflow', 'scikit']] = None,
        output_path: str = settings.DEFAULT_OUTPUT_PATH,
        config_model: str = settings.DEFAULT_CONFIG_MODEL,
        api_key: Optional[str] = settings.OPEN_ROUTER_API_KEY,
        api_base_url: str = settings.OPEN_ROUTER_BASE_URL,
        batch_size: int = settings.DATA_GEN_BATCH_SIZE
    ):
        """
        Initializes the data generator.

        Args:
            problem_description: User's description of the classification problem.
            selected_data_gen_model: Name of the OpenRouter model for generating data samples.
            library: The target ML library (optional, stored but not used in data generation).
            output_path: Base directory to save generated files.
            config_model: Name of the OpenRouter model for generating the configuration.
            api_key: OpenRouter API key.
            api_base_url: OpenRouter API base URL.
            batch_size: Number of data generation requests to run in parallel.
        """
        if not problem_description:
            raise ValueError("Problem description cannot be empty.")
        if not api_key or api_key == "YOUR_OPEN_ROUTER_API_KEY":
            raise ValueError("OpenRouter API key is not configured in settings.py or environment variables.")

        self.problem_description = problem_description
        self.selected_data_gen_model = selected_data_gen_model
        self.library = library # Stored for potential future use
        self.output_base_path = Path(output_path)
        self.config_model = config_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.batch_size = batch_size

        self.config: Optional[Dict[str, Any]] = None
        self.model_output_path: Optional[Path] = None
        self.raw_responses_path: Optional[Path] = None
        self.dataset_path: Optional[Path] = None

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )
        logger.info(f"DataGenerator initialized for model '{self.selected_data_gen_model}'")

    async def _call_llm_async(self, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 4000) -> str:
        """Helper function to call the LLM API."""
        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7, # Add some temperature for diversity
            )
            response_content = completion.choices[0].message.content
            if not response_content:
                raise ValueError("LLM returned an empty response.")
            return response_content

        except OpenAIError as e:
            logger.error(f"API error calling model {model}: {e}", exc_info=True)
            raise # Re-raise to be handled by caller
        except Exception as e:
            logger.error(f"Unexpected error calling model {model}: {e}", exc_info=True)
            raise # Re-raise

    async def _generate_config_async(self) -> Dict[str, Any]:
        """Generates the configuration JSON using the config_model LLM."""
        logger.info(f"Generating configuration using model: {self.config_model}")
        user_prompt = settings.CONFIG_USER_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description
        )

        try:
            response_content = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            # Clean potential markdown code fences
            if response_content.strip().startswith("```json"):
                response_content = response_content.strip()[7:]
            if response_content.strip().endswith("```"):
                response_content = response_content.strip()[:-3]

            config_data = json.loads(response_content.strip())
            logger.info(f"Successfully generated configuration: {config_data}")

            # Basic validation
            required_keys = ["summary", "prompts", "model_prefix", "training_data_volume", "data_gen_system_prompt"]
            if not all(key in config_data for key in required_keys):
                raise ValueError(f"Generated config is missing required keys. Got: {config_data.keys()}")
            if not isinstance(config_data["prompts"], dict) or "positive" not in config_data["prompts"] or "negative" not in config_data["prompts"]:
                raise ValueError("Generated config 'prompts' key is malformed.")

            return config_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from config model: {e}\nRaw response:\n{response_content}", exc_info=True)
            raise ValueError("Failed to generate valid configuration JSON from LLM.") from e
        except (ValueError, OpenAIError, Exception) as e: # Catch specific and general errors
             logger.error(f"Failed to generate configuration: {e}", exc_info=True)
             raise # Re-raise to stop the process

    def _prepare_output_directory(self):
        """Creates the necessary output directories based on the model prefix."""
        if not self.config or 'model_prefix' not in self.config:
            raise ValueError("Configuration with model_prefix must be generated before preparing directories.")

        model_prefix = self.config['model_prefix']
        self.model_output_path = self.output_base_path / model_prefix
        self.raw_responses_path = self.model_output_path / settings.RAW_RESPONSES_DIR
        self.dataset_path = self.model_output_path / settings.DATASET_FILENAME

        try:
            # Create all necessary directories, including parents
            self.model_output_path.mkdir(parents=True, exist_ok=True)
            self.raw_responses_path.mkdir(exist_ok=True)
            logger.info(f"Output directory structure prepared at: {self.model_output_path}")
        except OSError as e:
            logger.error(f"Failed to create output directories: {e}", exc_info=True)
            raise

    def _save_raw_response(self, prompt: str, response: str, label: int, model: str):
        """Saves the raw LLM request/response to a JSON file."""
        if not self.raw_responses_path:
             logger.warning("Raw responses path not set, skipping save.")
             return

        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label_str = "positive" if label == settings.POSITIVE_LABEL else "negative"
        # Sanitize model name for filename
        safe_model_name = model.replace('/', '_').replace(':', '_')
        filename = self.raw_responses_path / f"{label_str}_{safe_model_name}_{current_datetime}.json"

        result = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model,
            "label": label,
            "prompt": prompt,
            "response": response
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            # logger.debug(f"Raw response saved to {filename}") # Use debug level
        except IOError as e:
            logger.error(f"Failed to save raw response to {filename}: {e}", exc_info=False) # Don't need full trace usually
        except Exception as e:
             logger.error(f"Unexpected error saving raw response: {e}", exc_info=True)

    def _append_to_dataset(self, text_data: str, label: int):
        """Appends valid lines from text_data to the dataset CSV file."""
        if not self.dataset_path:
             logger.error("Dataset path not set, cannot append data.")
             return 0 # Return count of lines added

        lines_added = 0
        try:
            # Use 'a' mode (append) and ensure newline='' for csv compatibility
            with open(self.dataset_path, 'a', encoding='utf-8', newline='') as f:
                for line in text_data.split('\n'):
                    cleaned_line = line.strip()
                    # Basic cleaning: remove potential csv-breaking chars like quotes at ends
                    if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
                        cleaned_line = cleaned_line[1:-1]
                    cleaned_line = cleaned_line.replace('"', "'") # Replace internal quotes

                    if len(cleaned_line) >= settings.MIN_DATA_LINE_LENGTH:
                        # Simple CSV format: text,label
                        f.write(f'"{cleaned_line}",{label}\n')
                        lines_added += 1
            # logger.debug(f"Appended {lines_added} lines to {self.dataset_path}")
            return lines_added
        except IOError as e:
            logger.error(f"Failed to append data to dataset {self.dataset_path}: {e}", exc_info=False)
            return 0
        except Exception as e:
             logger.error(f"Unexpected error appending to dataset: {e}", exc_info=True)
             return 0

    async def _generate_data_batch_async(self, prompts_labels: list[tuple[str, int]]) -> list[tuple[str, int, str, str]]:
        """Generates a batch of data samples concurrently."""
        tasks = []

        for prompt, label in prompts_labels:
             tasks.append(
                 self._call_llm_async(
                     model=self.selected_data_gen_model,
                     system_prompt=settings.DATA_GEN_SYSTEM_PROMPT,
                     user_prompt=prompt
                 )
             )

        # Gather results, returning exceptions for failed tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            prompt, label = prompts_labels[i]
            if isinstance(result, Exception):
                logger.error(f"Data generation request failed for label {label} prompt: '{prompt[:50]}...'. Error: {result}")
                # Optionally save failure info
                # self._save_raw_response(prompt, f"ERROR: {result}", label, self.selected_data_gen_model)
                processed_results.append((prompt, label, f"ERROR: {result}", "failed"))
            else:
                # Success case result is the response string
                self._save_raw_response(prompt, result, label, self.selected_data_gen_model)
                processed_results.append((prompt, label, result, "success"))

        return processed_results

    async def generate_data_async(self):
        """Orchestrates the entire data generation process."""
        try:
            # 1. Generate Configuration
            self.config = await self._generate_config_async()
            if not self.config: return # Error logged in _generate_config_async

            # 2. Prepare Output Directory
            self._prepare_output_directory()
            if not self.dataset_path: return # Error logged in _prepare_output_directory

            # Write header to CSV if file is new/empty
            if not self.dataset_path.exists() or self.dataset_path.stat().st_size == 0:
                 try:
                     with open(self.dataset_path, 'w', encoding='utf-8', newline='') as f:
                         f.write("text,label\n")
                 except IOError as e:
                     logger.error(f"Failed to write header to dataset file {self.dataset_path}: {e}", exc_info=True)
                     return # Cannot proceed without dataset file

            # 3. Generate Data
            positive_prompt = self.config['prompts']['positive']
            negative_prompt = self.config['prompts']['negative']
            target_volume = self.config.get('training_data_volume', settings.DEFAULT_TRAINING_DATA_VOLUME)

            logger.info(f"Starting data generation. Target samples: ~{target_volume}. Batch size: {self.batch_size}")
            logger.info(f"Using model: {self.selected_data_gen_model}")
            logger.info(f"Positive Prompt: {positive_prompt[:100]}...")
            logger.info(f"Negative Prompt: {negative_prompt[:100]}...")

            total_samples_generated = 0
            total_requests_made = 0
            total_failed_requests = 0

            estimated_lines_per_request = 5
            required_requests = max(1, round(target_volume / estimated_lines_per_request))
            num_batches = max(1, (required_requests + self.batch_size - 1) // self.batch_size) # Ceiling division

            logger.info(f"Estimated required API calls: {required_requests} ({num_batches} batches)")

            current_request_count = 0
            for batch_num in range(num_batches):
                if total_samples_generated >= target_volume:
                    logger.info(f"Target volume {target_volume} reached. Stopping generation.")
                    break

                logger.info(f"Starting data generation batch {batch_num + 1}/{num_batches}")
                prompts_labels_batch = []
                for i in range(self.batch_size):
                    # Alternate between positive and negative prompts
                    if (current_request_count + i) % 2 == 0:
                        prompts_labels_batch.append((positive_prompt, settings.POSITIVE_LABEL))
                    else:
                        prompts_labels_batch.append((negative_prompt, settings.NEGATIVE_LABEL))

                current_request_count += len(prompts_labels_batch)
                total_requests_made += len(prompts_labels_batch)

                batch_results = await self._generate_data_batch_async(prompts_labels_batch)

                batch_samples_added = 0
                for prompt, label, response_content, status in batch_results:
                    if status == "success":
                        lines_added = self._append_to_dataset(response_content, label)
                        batch_samples_added += lines_added
                    else:
                        total_failed_requests += 1

                total_samples_generated += batch_samples_added
                logger.info(f"Batch {batch_num + 1} completed. Added {batch_samples_added} samples. Total samples: {total_samples_generated}. Failed requests in batch: {sum(1 for r in batch_results if r[3]=='failed')}")

            logger.info("--------------------------------------------------")
            logger.info("Data Generation Summary:")
            logger.info(f"  Target samples: {target_volume}")
            logger.info(f"  Total valid samples generated: {total_samples_generated}")
            logger.info(f"  Total API requests made: {total_requests_made}")
            logger.info(f"  Failed API requests: {total_failed_requests}")
            logger.info(f"  Dataset saved to: {self.dataset_path}")
            logger.info(f"  Raw responses saved in: {self.raw_responses_path}")
            logger.info("--------------------------------------------------")

        except (ValueError, OpenAIError) as e:
            logger.critical(f"Data generation process failed: {e}", exc_info=True)
        except Exception as e:
            logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

# --- CLI Interface ---
async def cli_main():
    parser = argparse.ArgumentParser(description="Generate Training Data for a Binary Classifier")
    parser.add_argument(
        "-p", "--problem-description",
        required=True,
        help="Describe the problem you want to classify (e.g., 'Identify spam emails', 'Classify movie reviews as positive or negative')."
    )
    parser.add_argument(
        "-m", "--model",
        default=settings.DEFAULT_DATA_GEN_MODEL,
        help=f"OpenRouter model name for data generation (default: {settings.DEFAULT_DATA_GEN_MODEL})."
    )
    parser.add_argument(
        "-l", "--library",
        choices=['pytorch', 'tensorflow', 'scikit'],
        default=None,
        help="Specify the target ML library (optional)."
    )
    parser.add_argument(
        "-o", "--output-path",
        default=settings.DEFAULT_OUTPUT_PATH,
        help=f"Base directory for output (default: {settings.DEFAULT_OUTPUT_PATH}). A subdirectory based on model prefix will be created."
    )

    args = parser.parse_args()

    try:
        generator = BinaryClassifierDataGenerator(
            problem_description=args.problem_description,
            selected_data_gen_model=args.model,
            library=args.library,
            output_path=args.output_path,
        )
        await generator.generate_data_async()
    except ValueError as e:
         # Configuration or initialization errors
         logger.error(f"Configuration Error: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred during setup: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(cli_main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e:
        # Catch any unexpected errors at the top level
        logger.critical(f"Application failed unexpectedly: {e}", exc_info=True)

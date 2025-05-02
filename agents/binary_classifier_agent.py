import asyncio
import json
import logging
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal, List, Tuple
import pandas as pd
from openai import AsyncOpenAI, OpenAIError

import agents.settings as settings
from binary_classifier.classifier import BinaryTextClassifier
from binary_classifier.strategies import (
    TensorFlowStrategy,
    PyTorchStrategy,
    ScikitLearnStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BinaryClassifierDataGenerator:
    """
    Generates configuration, synthetic training data, edge cases, and analyzes
    performance for a binary classifier based on a user-provided problem description.
    """

    def __init__(
        self,
        problem_description: str,
        selected_data_gen_model: str = settings.DEFAULT_DATA_GEN_MODEL,
        library: Optional[Literal["pytorch", "tensorflow", "scikit"]] = None,
        output_path: str = settings.DEFAULT_OUTPUT_PATH,
        config_model: str = settings.DEFAULT_CONFIG_MODEL,
        api_key: Optional[str] = settings.OPEN_ROUTER_API_KEY,
        api_base_url: str = settings.OPEN_ROUTER_BASE_URL,
        batch_size: int = settings.DATA_GEN_BATCH_SIZE,
        prompt_refinement_cycles: int = settings.DEFAULT_PROMPT_REFINEMENT_CYCLES,
        generate_edge_cases: bool = settings.DEFAULT_GENERATE_EDGE_CASES,
        edge_case_volume: int = settings.DEFAULT_EDGE_CASE_VOLUME,
        analyze_performance_data_path: Optional[str] = None,
        language: Optional[str] = None,
    ):
        if not problem_description:
            raise ValueError("Problem description cannot be empty.")
        if not api_key or api_key == "YOUR_OPEN_ROUTER_API_KEY":
            raise ValueError("OpenRouter API key is not configured.")

        self.problem_description = problem_description
        self.selected_data_gen_model = selected_data_gen_model
        self.library = library
        self.output_base_path = Path(output_path)
        self.config_model = config_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.batch_size = batch_size
        self.prompt_refinement_cycles = prompt_refinement_cycles
        self.generate_edge_cases = generate_edge_cases
        self.edge_case_volume = edge_case_volume
        self.analyze_performance_data_path = (
            Path(analyze_performance_data_path)
            if analyze_performance_data_path
            else None
        )
        self.language = language

        self.initial_config: Optional[Dict[str, Any]] = None
        self.final_config: Optional[Dict[str, Any]] = None
        self.prompt_refinement_history: List[Dict[str, Any]] = []
        self.performance_analysis_result: Optional[str] = None
        self.model_output_path: Optional[Path] = None
        self.raw_responses_path: Optional[Path] = None
        self.dataset_path: Optional[Path] = None
        self.edge_case_dataset_path: Optional[Path] = None
        self.final_config_path: Optional[Path] = None

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
            timeout=60.0,
            max_retries=2,
        )
        logger.info(
            f"DataGenerator initialized. Config Model: '{self.config_model}', Data Gen Model: '{self.selected_data_gen_model}'"
        )
        logger.info(f"Prompt Refinement Cycles: {self.prompt_refinement_cycles}")
        logger.info(
            f"Generate Edge Cases: {self.generate_edge_cases} (Target Volume: {self.edge_case_volume})"
        )
        if self.analyze_performance_data_path:
            logger.info(
                f"Performance analysis requested using data from: {self.analyze_performance_data_path}"
            )

    async def _call_llm_async(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """Helper function to call the LLM API with retry and error handling."""
        try:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_content = completion.choices[0].message.content
            if not response_content:
                raise ValueError("LLM returned an empty response.")
            return response_content.strip()

        except OpenAIError as e:
            logger.error(f"API error calling model {model}: {e}", exc_info=True)
            raise  # Re-raise to be handled by caller
        except Exception as e:
            logger.error(f"Unexpected error calling model {model}: {e}", exc_info=True)
            raise  # Re-raise

    async def _generate_initial_config_async(self) -> bool:
        """Generates the initial configuration JSON."""
        logger.info(
            f"Generating initial configuration using model: {self.config_model}"
        )
        user_prompt = settings.CONFIG_USER_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description,
            DEFAULT_DATA_GEN_MODEL=settings.DEFAULT_DATA_GEN_MODEL,  # Pass default in case needed
        )
        try:
            response_content = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,  # Lower temp for consistent JSON structure
            )

            # Clean potential markdown code fences
            if "```json" in response_content:
                response_content = response_content.split("```json", 1)[1]
            if "```" in response_content:
                response_content = response_content.split("```", 1)[0]

            config_data = json.loads(response_content.strip())
            logger.info(f"Successfully generated initial configuration.")

            # Basic validation
            required_keys = [
                "summary",
                "prompts",
                "model_prefix",
                "training_data_volume",
            ]
            if not all(key in config_data for key in required_keys):
                raise ValueError(
                    f"Generated config missing keys. Expected: {required_keys}, Got: {config_data.keys()}"
                )
            if (
                not isinstance(config_data["prompts"], dict)
                or "positive" not in config_data["prompts"]
                or "negative" not in config_data["prompts"]
            ):
                raise ValueError("Generated config 'prompts' key is malformed.")

            self.initial_config = config_data
            self.final_config = config_data.copy()  # Start final config as a copy
            self.final_config["parameters"] = (
                self._get_init_parameters()
            )  # Store initial params
            return True

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response from config model: {e}\nRaw response:\n{response_content}",
                exc_info=True,
            )
            return False
        except (ValueError, OpenAIError, Exception) as e:
            logger.error(
                f"Failed to generate initial configuration: {e}", exc_info=True
            )
            return False

    def _get_init_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of the parameters used to initialize the class."""
        return {
            "problem_description": self.problem_description,
            "selected_data_gen_model": self.selected_data_gen_model,
            "library": self.library,
            "output_base_path": str(self.output_base_path),
            "config_model": self.config_model,
            "batch_size": self.batch_size,
            "prompt_refinement_cycles": self.prompt_refinement_cycles,
            "generate_edge_cases": self.generate_edge_cases,
            "edge_case_volume": self.edge_case_volume,
            "analyze_performance_data_path": (
                str(self.analyze_performance_data_path)
                if self.analyze_performance_data_path
                else None
            ),
        }

    def _prepare_output_directory(self):
        """Creates output directories based on the model prefix."""
        if not self.final_config or "model_prefix" not in self.final_config:
            raise ValueError("Config unavailable.")

        model_prefix = self.final_config["model_prefix"]
        self.model_output_path = self.output_base_path / model_prefix
        self.raw_responses_path = self.model_output_path / settings.RAW_RESPONSES_DIR
        self.dataset_path = self.model_output_path / settings.TRAINING_DATASET_FILENAME
        self.edge_case_dataset_path = (
            self.model_output_path / settings.EDGE_CASE_DATASET_FILENAME
        )
        self.final_config_path = self.model_output_path / settings.CONFIG_FILENAME

        try:
            self.model_output_path.mkdir(parents=True, exist_ok=True)
            self.raw_responses_path.mkdir(exist_ok=True)
            logger.info(
                f"Output directory structure prepared at: {self.model_output_path}"
            )
            return True
        except OSError as e:
            logger.error(f"Failed to create output directories: {e}", exc_info=True)
            return False

    def _save_raw_response(
        self,
        prompt_type: str,
        prompt: str,
        response: str,
        label: Optional[int],
        model: str,
        status: str = "success",
    ):
        """Saves raw LLM response to a JSON file."""
        if not self.raw_responses_path:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label_str = (
            f"_{settings.POSITIVE_LABEL}"
            if label == settings.POSITIVE_LABEL
            else (
                f"_{settings.NEGATIVE_LABEL}"
                if label == settings.NEGATIVE_LABEL
                else ""
            )
        )
        safe_model = model.replace("/", "_").replace(":", "_")
        filename = (
            self.raw_responses_path
            / f"{prompt_type}{label_str}_{safe_model}_{status}_{ts}.json"
        )

        data = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model,
            "prompt_type": prompt_type,
            "label": label,
            "status": status,
            "prompt": prompt,
            "response": response,
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(
                f"Failed to save raw response to {filename}: {e}", exc_info=False
            )

    def _append_to_dataset(self, text_data: str, label: int, target_path: Path) -> int:
        """Appends valid lines to the specified dataset CSV file."""
        if not target_path:
            logger.error("Target path not set, cannot append data.")
            return 0

        lines_added = 0
        try:
            write_header = not target_path.exists() or target_path.stat().st_size == 0
            with open(target_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                if write_header:
                    writer.writerow(["text", "label"])

                for line in text_data.split("\n"):
                    cleaned_line = line.strip()
                    # Basic cleaning - remove leading/trailing quotes if present
                    if (
                        len(cleaned_line) > 1
                        and cleaned_line.startswith('"')
                        and cleaned_line.endswith('"')
                    ):
                        cleaned_line = cleaned_line[1:-1]
                    # Replace internal quotes that might break CSV, ensure it's not empty
                    cleaned_line = cleaned_line.replace('"', "'").strip()

                    if len(cleaned_line) >= settings.MIN_DATA_LINE_LENGTH:
                        writer.writerow([cleaned_line, label])
                        lines_added += 1
            return lines_added
        except IOError as e:
            logger.error(f"IOError appending to {target_path}: {e}", exc_info=False)
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected error appending to {target_path}: {e}", exc_info=True
            )
            return 0

    async def _generate_text_samples_batch_async(
        self, prompts_labels: List[Tuple[str, int, str]], model: str, system_prompt: str
    ) -> List[Tuple[str, int, str, str]]:
        """Generates a batch of text samples concurrently using a specified model."""
        tasks = []
        for prompt, label, prompt_type in prompts_labels:
            tasks.append(
                self._call_llm_async(
                    model=model, system_prompt=system_prompt, user_prompt=prompt
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            prompt, label, prompt_type = prompts_labels[i]
            if isinstance(result, Exception):
                error_msg = f"ERROR: {result}"
                logger.error(
                    f"Data generation failed for {prompt_type} (Label: {label}) - {error_msg}"
                )
                self._save_raw_response(
                    prompt_type, prompt, error_msg, label, model, status="failed"
                )
                processed_results.append((prompt, label, error_msg, "failed"))
            else:
                self._save_raw_response(
                    prompt_type, prompt, result, label, model, status="success"
                )
                processed_results.append((prompt, label, result, "success"))
        return processed_results

    async def _run_prompt_refinement_cycle_async(self, cycle_num: int) -> bool:
        """Runs one cycle of prompt generation, sampling, and refinement."""
        if not self.final_config:
            return False
        logger.info(
            f"--- Starting Prompt Refinement Cycle {cycle_num + 1}/{self.prompt_refinement_cycles} ---"
        )

        current_prompts = self.final_config["prompts"]
        pos_prompt = current_prompts["positive"]
        neg_prompt = current_prompts["negative"]
        data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
            language=self.language
        )

        # 1. Generate small sample batch with current prompts
        logger.info("Generating sample data for refinement evaluation...")
        sample_prompts_labels = []
        # Generate half positive, half negative for the sample batch
        num_samples_per_type = settings.PROMPT_REFINEMENT_BATCH_SIZE // 2
        for _ in range(num_samples_per_type):
            sample_prompts_labels.append(
                (pos_prompt, settings.POSITIVE_LABEL, "refinement_sample")
            )
            sample_prompts_labels.append(
                (neg_prompt, settings.NEGATIVE_LABEL, "refinement_sample")
            )

        sample_results = await self._generate_text_samples_batch_async(
            sample_prompts_labels, self.selected_data_gen_model, data_gen_sys_prompt
        )

        pos_samples = [
            res[2]
            for res in sample_results
            if res[1] == settings.POSITIVE_LABEL and res[3] == "success"
        ]
        neg_samples = [
            res[2]
            for res in sample_results
            if res[1] == settings.NEGATIVE_LABEL and res[3] == "success"
        ]

        if not pos_samples and not neg_samples:
            logger.warning(
                f"Refinement Cycle {cycle_num + 1}: Failed to generate any sample data. Skipping refinement."
            )
            return False  # Cannot refine without samples

        pos_samples_str = "\n".join(
            [f"- {s[:200]}..." for s in pos_samples[:5]]
        )  # Show first 5 samples (truncated)
        neg_samples_str = "\n".join([f"- {s[:200]}..." for s in neg_samples[:5]])

        refinement_prompt = settings.PROMPT_REFINEMENT_TEMPLATE.format(
            problem_description=self.problem_description,
            current_positive_prompt=pos_prompt,
            current_negative_prompt=neg_prompt,
            positive_samples=(
                pos_samples_str if pos_samples else "No positive samples generated."
            ),
            negative_samples=(
                neg_samples_str if neg_samples else "No negative samples generated."
            ),
        )

        logger.info("Asking config model for prompt refinement suggestions...")
        try:
            refinement_response = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=refinement_prompt,
                temperature=0.5,  # Balance creativity and structure
            )

            # Clean potential markdown
            if "```json" in refinement_response:
                refinement_response = refinement_response.split("```json", 1)[1]
            if "```" in refinement_response:
                refinement_response = refinement_response.split("```", 1)[0]

            refinement_data = json.loads(refinement_response.strip())

            if (
                "evaluation_summary" not in refinement_data
                or "refined_positive_prompt" not in refinement_data
                or "refined_negative_prompt" not in refinement_data
            ):
                raise ValueError("Refinement response missing required keys.")

            # 3. Update Prompts and Log History
            new_pos_prompt = refinement_data["refined_positive_prompt"]
            new_neg_prompt = refinement_data["refined_negative_prompt"]

            logger.info(
                f"Refinement Cycle {cycle_num + 1} Evaluation: {refinement_data['evaluation_summary']}"
            )
            if new_pos_prompt != pos_prompt:
                logger.info("Refined Positive Prompt Updated.")
                self.final_config["prompts"]["positive"] = new_pos_prompt
            if new_neg_prompt != neg_prompt:
                logger.info("Refined Negative Prompt Updated.")
                self.final_config["prompts"]["negative"] = new_neg_prompt

            self.prompt_refinement_history.append(
                {
                    "cycle": cycle_num + 1,
                    "evaluation": refinement_data["evaluation_summary"],
                    "previous_positive_prompt": pos_prompt,
                    "previous_negative_prompt": neg_prompt,
                    "refined_positive_prompt": new_pos_prompt,
                    "refined_negative_prompt": new_neg_prompt,
                }
            )
            logger.info(f"--- Prompt Refinement Cycle {cycle_num + 1} Completed ---")
            return True

        except json.JSONDecodeError as e:
            logger.error(
                f"Refinement Cycle {cycle_num + 1}: Failed to parse JSON response from config model: {e}\nRaw response:\n{refinement_response}",
                exc_info=True,
            )
            return False
        except (ValueError, OpenAIError, Exception) as e:
            logger.error(
                f"Refinement Cycle {cycle_num + 1}: Failed: {e}", exc_info=True
            )
            return False

    async def _generate_training_data_async(self):
        """Generates the main training dataset using final prompts."""
        if not self.final_config or not self.dataset_path:
            return 0

        logger.info("--- Starting Bulk Training Data Generation ---")
        positive_prompt = self.final_config["prompts"]["positive"]
        negative_prompt = self.final_config["prompts"]["negative"]
        target_volume = self.final_config.get(
            "training_data_volume", settings.DEFAULT_TRAINING_DATA_VOLUME
        )
        data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
            language=self.language
        )

        logger.info(f"Target samples: ~{target_volume}. Batch size: {self.batch_size}")
        logger.info(f"Using Model: {self.selected_data_gen_model}")

        total_samples_generated = 0
        total_requests_made = 0
        total_failed_requests = 0

        estimated_lines_per_request = 5
        required_requests = max(1, round(target_volume / estimated_lines_per_request))
        num_batches = max(
            1, (required_requests + self.batch_size - 1) // self.batch_size
        )

        current_request_count = 0
        for batch_num in range(num_batches):
            if total_samples_generated >= target_volume:
                break

            logger.info(f"Starting training data batch {batch_num + 1}/{num_batches}")
            prompts_labels_batch = []
            for i in range(self.batch_size):
                label = (
                    settings.POSITIVE_LABEL
                    if (current_request_count + i) % 2 == 0
                    else settings.NEGATIVE_LABEL
                )
                prompt = (
                    positive_prompt
                    if label == settings.POSITIVE_LABEL
                    else negative_prompt
                )
                prompts_labels_batch.append((prompt, label, "training_data"))

            current_request_count += len(prompts_labels_batch)
            total_requests_made += len(prompts_labels_batch)

            batch_results = await self._generate_text_samples_batch_async(
                prompts_labels_batch, self.selected_data_gen_model, data_gen_sys_prompt
            )

            batch_samples_added = 0
            for _, label, response_content, status in batch_results:
                if status == "success":
                    lines_added = self._append_to_dataset(
                        response_content, label, self.dataset_path
                    )
                    batch_samples_added += lines_added
                else:
                    total_failed_requests += 1

            total_samples_generated += batch_samples_added
            logger.info(
                f"Batch {batch_num + 1} completed. Added {batch_samples_added} samples. Total samples: {total_samples_generated}. Failed requests in batch: {sum(1 for r in batch_results if r[3]=='failed')}"
            )
            await asyncio.sleep(0.5)  # Small delay between batches

        logger.info("--- Bulk Training Data Generation Finished ---")
        logger.info(f"Total training samples generated: {total_samples_generated}")
        logger.info(
            f"Total training API requests: {total_requests_made} (Failed: {total_failed_requests})"
        )
        logger.info(f"Training dataset saved to: {self.dataset_path}")
        return total_samples_generated  # Return actual count

    async def _generate_edge_cases_async(self):
        """Generates edge case data using the config model."""
        if not self.final_config or not self.edge_case_dataset_path:
            return 0
        if not self.generate_edge_cases:
            logger.info("Skipping edge case generation as per configuration.")
            return 0

        logger.info("--- Starting Edge Case Generation ---")
        target_volume = self.edge_case_volume
        # Use the more capable config model for edge cases
        edge_case_model = self.config_model
        # Edge cases don't need a specific system prompt beyond the user prompt's instructions
        edge_case_sys_prompt = "You are a data generation assistant specializing in creating challenging test cases."

        # Use final refined prompts for context in edge case generation prompts
        final_positive_prompt = self.final_config["prompts"]["positive"]
        final_negative_prompt = self.final_config["prompts"]["negative"]
        pos_edge_case_prompt = settings.POSITIVE_EDGE_CASE_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description,
            final_positive_prompt=final_positive_prompt,  # Provide context
            final_negative_prompt=final_negative_prompt,
            language=self.language,
        )
        neg_edge_case_prompt = settings.NEGATIVE_EDGE_CASE_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description,
            final_positive_prompt=final_positive_prompt,  # Provide context
            final_negative_prompt=final_negative_prompt,
            language=self.language,
        )

        # Add prompts to config for record-keeping
        self.final_config["edge_case_prompts"] = {
            "positive": pos_edge_case_prompt,
            "negative": neg_edge_case_prompt,
        }

        logger.info(f"Target edge cases: ~{target_volume}. Model: {edge_case_model}")
        logger.info(f"Positive Edge Case Prompt: {pos_edge_case_prompt[:150]}...")
        logger.info(f"Negative Edge Case Prompt: {neg_edge_case_prompt[:150]}...")

        total_samples_generated = 0
        total_requests_made = 0
        total_failed_requests = 0

        # Assume fewer lines per response for focused edge cases
        estimated_lines_per_request = 3
        required_requests = max(1, round(target_volume / estimated_lines_per_request))
        # Use smaller batches for potentially slower config model
        edge_batch_size = max(1, self.batch_size // 2)
        num_batches = max(
            1, (required_requests + edge_batch_size - 1) // edge_batch_size
        )

        logger.info(
            f"Estimated edge case API calls: {required_requests} ({num_batches} batches of size {edge_batch_size})"
        )

        current_request_count = 0
        for batch_num in range(num_batches):
            if total_samples_generated >= target_volume:
                logger.info(
                    f"Target edge case volume reached ({total_samples_generated}/{target_volume}). Stopping generation."
                )
                break

            logger.info(f"Starting edge case batch {batch_num + 1}/{num_batches}")
            prompts_labels_batch = []
            for i in range(edge_batch_size):
                label = (
                    settings.POSITIVE_LABEL
                    if (current_request_count + i) % 2 == 0
                    else settings.NEGATIVE_LABEL
                )
                prompt = (
                    pos_edge_case_prompt
                    if label == settings.POSITIVE_LABEL
                    else neg_edge_case_prompt
                )
                prompts_labels_batch.append((prompt, label, "edge_case"))

            current_request_count += len(prompts_labels_batch)
            total_requests_made += len(prompts_labels_batch)

            batch_results = await self._generate_text_samples_batch_async(
                prompts_labels_batch, edge_case_model, edge_case_sys_prompt
            )

            batch_samples_added = 0
            for _, label, response_content, status in batch_results:
                if status == "success":
                    lines_added = self._append_to_dataset(
                        response_content, label, self.edge_case_dataset_path
                    )
                    batch_samples_added += lines_added
                else:
                    total_failed_requests += 1

            total_samples_generated += batch_samples_added
            logger.info(
                f"Batch {batch_num + 1} completed. Added {batch_samples_added} edge cases. Total: {total_samples_generated}. Failed requests: {sum(1 for r in batch_results if r[3]=='failed')}"
            )
            await asyncio.sleep(1.0)  # Longer delay for potentially slower model

        logger.info("--- Edge Case Generation Finished ---")
        logger.info(f"Total edge case samples generated: {total_samples_generated}")
        logger.info(
            f"Total edge case API requests: {total_requests_made} (Failed: {total_failed_requests})"
        )
        logger.info(f"Edge case dataset saved to: {self.edge_case_dataset_path}")
        return total_samples_generated

    async def _analyze_performance_async(self):
        """Analyzes performance results using the config model."""
        if not self.analyze_performance_data_path:
            logger.info("Skipping performance analysis: No results file provided.")
            return
        if not self.final_config:
            logger.error("Cannot analyze performance: Configuration missing.")
            return

        logger.info(
            f"--- Starting Performance Analysis using {self.analyze_performance_data_path} ---"
        )

        try:
            if not self.analyze_performance_data_path.exists():
                raise FileNotFoundError(
                    f"Performance results file not found at {self.analyze_performance_data_path}"
                )

            results_summary = ""
            accuracy = float("nan")  # Placeholder
            sample_results_list = []
            limit_samples = 20

            try:
                df = pd.read_csv(self.analyze_performance_data_path)
                # Try to infer column names or use standard ones
                text_col = next(
                    (c for c in df.columns if "text" in c.lower()), df.columns[0]
                )
                true_label_col = next(
                    (
                        c
                        for c in df.columns
                        if "label" in c.lower() or "actual" in c.lower()
                    ),
                    df.columns[1],
                )
                pred_col = next(
                    (
                        c
                        for c in df.columns
                        if "pred" in c.lower() or "prob" in c.lower()
                    ),
                    df.columns[2],
                )
                # Calculate accuracy if possible
                if pd.api.types.is_numeric_dtype(
                    df[true_label_col]
                ) and pd.api.types.is_numeric_dtype(df[pred_col]):
                    # Simple accuracy assuming threshold 0.5 if probabilities, or direct comparison if labels
                    df["correct"] = (
                        (df[true_label_col] == df[pred_col])
                        if not (df[pred_col].between(0, 1, inclusive="neither")).any()
                        else (df[true_label_col] == (df[pred_col] > 0.5).astype(int))
                    )
                    if "correct" in df.columns:
                        accuracy = df["correct"].mean()
                        logger.info(f"Calculated Accuracy: {accuracy:.4f}")
                        # Prioritize showing errors
                        errors_df = df[~df["correct"]].head(limit_samples // 2)
                        correct_df = df[df["correct"]].head(
                            limit_samples - len(errors_df)
                        )
                        sample_df = pd.concat([errors_df, correct_df])
                    else:  # Fallback if 'correct' didn't work
                        sample_df = df.head(limit_samples)
                else:
                    sample_df = df.head(limit_samples)
                for _, row in sample_df.iterrows():
                    sample_results_list.append(
                        f'- "{str(row[text_col])[:150]}...", True: {row[true_label_col]}, Predicted: {row[pred_col]}'
                    )
            except Exception as e:
                logger.error(f"Error reading performance data file: {e}", exc_info=True)
                # Provide fallback message to LLM if reading fails
                sample_results_list.append(
                    "Error: Could not read or parse the performance data file."
                )

            results_summary = "\n".join(sample_results_list)

            analysis_prompt = settings.PERFORMANCE_ANALYSIS_PROMPT_TEMPLATE.format(
                problem_description=self.problem_description,
                final_positive_prompt=self.final_config["prompts"]["positive"],
                final_negative_prompt=self.final_config["prompts"]["negative"],
                test_results_summary=(
                    results_summary
                    if results_summary
                    else "No sample results could be extracted."
                ),
            )

            logger.info(
                "Asking config model for performance analysis and recommendations..."
            )
            analysis_response = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=analysis_prompt,
                max_tokens=1500,  # Allow longer response for analysis
                temperature=0.4,
            )

            self.performance_analysis_result = analysis_response
            self.final_config["performance_analysis"] = {
                "input_file": str(self.analyze_performance_data_path),
                "llm_analysis": analysis_response,
            }
            logger.info("--- Performance Analysis Finished ---")

        except FileNotFoundError as e:
            logger.error(f"Performance analysis failed: {e}")
        except Exception as e:
            logger.error(
                f"Performance analysis failed unexpectedly: {e}", exc_info=True
            )

    def _save_final_config(self):
        """Saves the consolidated configuration and results to JSON."""
        if not self.final_config or not self.model_output_path:
            logger.error(
                "Cannot save final config: Configuration or output path missing."
            )
            return False
        if not self.final_config_path:
            self.final_config_path = (
                self.model_output_path / settings.CONFIG_FILENAME
            )  # Ensure path exists

        logger.info(f"Saving final configuration to: {self.final_config_path}")

        # Consolidate all data into the final_config dictionary
        self.final_config["generation_timestamp"] = datetime.now().isoformat()
        self.final_config["initial_config"] = (
            self.initial_config
        )  # Store the very first config
        self.final_config["prompt_refinement_history"] = self.prompt_refinement_history

        # Add file paths
        self.final_config["output_paths"] = {
            "main_output_directory": str(self.model_output_path),
            "training_data": (
                str(self.dataset_path)
                if self.dataset_path and self.dataset_path.exists()
                else None
            ),
            "edge_case_data": (
                str(self.edge_case_dataset_path)
                if self.edge_case_dataset_path and self.edge_case_dataset_path.exists()
                else None
            ),
            "raw_api_responses": str(self.raw_responses_path),
            "final_config_file": str(self.final_config_path),
        }

        # Performance analysis already added if run

        try:
            with open(self.final_config_path, "w", encoding="utf-8") as f:
                json.dump(self.final_config, f, indent=2, ensure_ascii=False)
            logger.info("Final configuration saved successfully.")
            return True
        except TypeError as e:
            logger.error(
                f"Failed to serialize final config to JSON: {e}. Check for non-serializable types.",
                exc_info=True,
            )
            return False
        except IOError as e:
            logger.error(f"Failed to write final config file: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving final config: {e}", exc_info=True)
            return False

    def run_predictions(self, model_type: str):
        strategies = {
            "tensorflow": TensorFlowStrategy(input_dim=5000),
            "pytorch": PyTorchStrategy(input_dim=5000),
            "scikit": ScikitLearnStrategy(),
        }
        strategy = strategies[model_type]

        classifier = BinaryTextClassifier(strategy)
        classifier.load(f"{self.model_output_path}/model")

        df = pd.read_csv(self.edge_case_dataset_path, encoding="utf-8")
        df["prediction"] = df["text"].apply(lambda x: classifier.predict([x])[0])

        output_path = f"{self.model_output_path}/edge_case_predictions.csv"
        self.analyze_performance_data_path = Path(output_path)
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    async def generate_data_and_train_model_async(self, model_type: str):
        """Orchestrates the entire data generation process."""
        start_time = datetime.now()
        logger.info(
            f"=== Starting Data Generation Process for: '{self.problem_description}' ==="
        )

        try:
            # 1. Generate Initial Configuration
            if not await self._generate_initial_config_async():
                logger.critical("Failed to generate initial configuration. Aborting.")
                return

            # 2. Prepare Output Directory
            if not self._prepare_output_directory():
                logger.critical("Failed to prepare output directory. Aborting.")
                return

            # 3. Prompt Refinement Cycles (if enabled)
            if self.prompt_refinement_cycles > 0:
                for i in range(self.prompt_refinement_cycles):
                    refinement_success = await self._run_prompt_refinement_cycle_async(
                        i
                    )
                    if not refinement_success:
                        logger.warning(
                            f"Prompt refinement cycle {i+1} failed or was skipped."
                        )
                        # Optionally break if refinement is critical, or continue with last good prompts
            else:
                logger.info("Skipping prompt refinement cycles as per configuration.")

            # 4. Generate Main Training Data
            training_samples = await self._generate_training_data_async()

            # 5. Generate Edge Case Data (if enabled)
            edge_case_samples = await self._generate_edge_cases_async()

            if model_type == "tensorflow":
                strategy = TensorFlowStrategy(input_dim=5000)
            elif model_type == "pytorch":
                strategy = PyTorchStrategy(input_dim=5000)
            elif model_type == "scikit":
                strategy = ScikitLearnStrategy()

            # 6. Train and save
            classifier = BinaryTextClassifier(strategy)
            metrics = classifier.train(self.dataset_path)
            classifier.strategy.export_to_onnx(f"{self.model_output_path}/model.onnx")
            classifier.save(f"{self.model_output_path}/model")

            self.run_predictions(model_type)

            # 7. Analyze Performance Data
            await self._analyze_performance_async()

            # 8. Save Final Configuration
            self._save_final_config()

        except (ValueError, OpenAIError) as e:
            logger.critical(
                f"Data generation process failed critically: {e}", exc_info=True
            )
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred during generation: {e}",
                exc_info=True,
            )
        finally:
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(
                f"=== Data Generation Process Finished. Duration: {duration} ==="
            )
            # Final summary logs
            logger.info(
                f"  Training samples generated: {training_samples if 'training_samples' in locals() else 'N/A'}"
            )
            logger.info(
                f"  Model performance feedback by LLM: {self.performance_analysis_result if self.performance_analysis_result else 'N/A'}"
            )
            if self.generate_edge_cases:
                logger.info(
                    f"  Edge case samples generated: {edge_case_samples if 'edge_case_samples' in locals() else 'N/A'}"
                )
            if self.final_config_path:
                logger.info(
                    f"  Find detailed configuration and results in: {self.final_config_path}"
                )
            else:
                logger.warning("Final configuration file path not set.")


# --- CLI Interface ---
async def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate Training Data, Edge Cases, and Analysis for a Binary Classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults
    )
    parser.add_argument(
        "-p",
        "--problem-description",
        required=True,
        help="Describe the classification problem.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=settings.DEFAULT_DATA_GEN_MODEL,
        help="OpenRouter model name for bulk data generation.",
    )
    parser.add_argument(
        "--config-model",
        default=settings.DEFAULT_CONFIG_MODEL,
        help="OpenRouter model for config, refinement, edge cases, analysis (should be capable).",
    )
    parser.add_argument(
        "-l",
        "--library",
        choices=["pytorch", "tensorflow", "scikit"],
        default="tensorflow",
        help="Specify the target ML library (optional, for documentation).",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=settings.DEFAULT_OUTPUT_PATH,
        help="Base directory for output.",
    )
    parser.add_argument(
        "--refinement-cycles",
        type=int,
        default=settings.DEFAULT_PROMPT_REFINEMENT_CYCLES,
        help="Number of prompt refinement cycles based on sample data.",
    )
    parser.add_argument(
        "--edge-cases",
        type=lambda x: (str(x).lower() == "true"),  # Handle boolean
        default=settings.DEFAULT_GENERATE_EDGE_CASES,
        help="Generate challenging edge case data (true/false).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.DATA_GEN_BATCH_SIZE,
        help="Number of parallel requests for bulk data generation.",
    )
    parser.add_argument(
        "--lang", default="english", help="Language for dataset (default: english)"
    )

    args = parser.parse_args()

    try:
        generator = BinaryClassifierDataGenerator(
            problem_description=args.problem_description,
            selected_data_gen_model=args.model,
            config_model=args.config_model,
            library=args.library,
            output_path=args.output_path,
            batch_size=args.batch_size,
            prompt_refinement_cycles=args.refinement_cycles,
            generate_edge_cases=args.edge_cases,
            language=args.lang,
        )
        await generator.generate_data_and_train_model_async(model_type=args.library)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during setup: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(cli_main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Application failed unexpectedly: {e}", exc_info=True)

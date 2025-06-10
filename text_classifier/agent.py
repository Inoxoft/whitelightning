import asyncio
import json
import logging
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np  # For X_sample in ONNX
from openai import AsyncOpenAI, OpenAIError

from text_classifier.train import TextClassifierRunner

try:
    import text_classifier.settings as settings
except ModuleNotFoundError:  # Handle if running script directly from its dir
    import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True


class MulticlassDataGenerator:  # Renamed
    def __init__(
        self,
        problem_description: str,
        selected_data_gen_model: str = settings.DEFAULT_DATA_GEN_MODEL,
        # library: Optional[Literal["pytorch", "tensorflow", "scikit"]] = None, # model_type will handle this
        output_path: str = settings.DEFAULT_OUTPUT_PATH,
        config_model: str = settings.DEFAULT_CONFIG_MODEL,
        api_key: Optional[str] = settings.OPEN_ROUTER_API_KEY,
        api_base_url: str = settings.OPEN_ROUTER_BASE_URL,
        batch_size: int = settings.DATA_GEN_BATCH_SIZE,
        prompt_refinement_cycles: int = settings.DEFAULT_PROMPT_REFINEMENT_CYCLES,
        generate_edge_cases: bool = settings.DEFAULT_GENERATE_EDGE_CASES,
        edge_case_volume_per_class: int = settings.DEFAULT_EDGE_CASE_VOLUME
        // 2,  # Adjust logic
        analyze_performance_data_path: Optional[str] = None,
        language: Optional[str] = None,
        max_features_tfidf: int = 5000,
        config_path: Optional[str] = None,
        skip_data_gen: bool = False,
        skip_model_training: bool = False,
    ):
        self.skip_data_gen = skip_data_gen
        self.skip_model_training = skip_model_training

        if not problem_description and not skip_data_gen:
            raise ValueError("Problem description cannot be empty.")
        if not api_key:  # From settings
            raise ValueError(
                "OpenRouter API key is not configured in .env or settings.py."
            )

        self.problem_description = problem_description
        self.selected_data_gen_model = selected_data_gen_model
        # self.library = library # Replaced by model_type in generate_and_train
        self.output_base_path = Path(output_path)
        self.config_model = config_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.batch_size = batch_size
        self.prompt_refinement_cycles = prompt_refinement_cycles
        self.generate_edge_cases = generate_edge_cases
        self.edge_case_volume_per_class = edge_case_volume_per_class  # Per class now
        self.analyze_performance_data_path = (
            Path(analyze_performance_data_path)
            if analyze_performance_data_path
            else None
        )
        self.language = language if language else "english"  # Default to english
        self.max_features_tfidf = max_features_tfidf

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                try:
                    self.resume_from_config = True
                    config_data = json.load(f)
                    self.initial_config = config_data
                    self.final_config = config_data.copy()
                    self.prompt_refinement_history = config_data.get(
                        "prompt_refinement_history", []
                    )
                    self.performance_analysis_result = None
                    self.classification_type = config_data.get("classification_type")
                    self.class_labels = sorted(
                        list(set(config_data.get("class_labels", [])))
                    )
                    self.num_classes = len(self.class_labels)

                    model_prefix = config_data.get("model_prefix")
                    paths = config_data.get("output_paths", {})
                    self.model_output_path = Path(paths.get("main_output_directory"))
                    self.raw_responses_path = Path(paths.get("raw_api_responses"))
                    self.dataset_path = Path(paths.get("training_data"))
                    self.edge_case_dataset_path = Path(paths.get("edge_case_data"))
                    self.final_config_path = Path(paths.get("final_config_file"))

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to load JSON from {config_path}: {e}")
        else:
            self.resume_from_config = False
            self.initial_config: Optional[Dict[str, Any]] = None
            self.final_config: Optional[Dict[str, Any]] = None
            self.prompt_refinement_history: List[Dict[str, Any]] = []
            self.performance_analysis_result: Optional[str] = None

            # Paths to be set after config is loaded
            self.model_output_path: Optional[Path] = None
            self.raw_responses_path: Optional[Path] = None
            self.dataset_path: Optional[Path] = None
            self.edge_case_dataset_path: Optional[Path] = None
            self.final_config_path: Optional[Path] = None
            self.classifier_metadata_path: Optional[Path] = None

            # Derived from config
            self.classification_type: Optional[str] = None
            self.class_labels: Optional[List[str]] = None
            self.num_classes: Optional[int] = None

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
            timeout=settings.API_TIMEOUT if hasattr(settings, "API_TIMEOUT") else 60.0,
            max_retries=(
                settings.API_MAX_RETRIES if hasattr(settings, "API_MAX_RETRIES") else 2
            ),
        )
        logger.info(
            f"DataGenerator initialized. Config Model: '{self.config_model}', Data Gen Model: '{self.selected_data_gen_model}'"
        )
        logger.info(f"Prompt Refinement Cycles: {self.prompt_refinement_cycles}")
        logger.info(
            f"Generate Edge Cases: {self.generate_edge_cases} (Target Volume per class: {self.edge_case_volume_per_class})"
        )

    async def _call_llm_async(  # (No changes needed here)
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
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
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling model {model}: {e}", exc_info=True)
            raise

    async def _generate_initial_config_async(self) -> bool:
        logger.info(
            f"Generating initial configuration using model: {self.config_model}"
        )
        user_prompt = settings.CONFIG_USER_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description
        )
        raw_response_content = ""
        try:
            raw_response_content = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
            )

            # Clean potential markdown code fences
            json_str = raw_response_content
            if "```json" in json_str:
                json_str = json_str.split("```json", 1)[1]
            if "```" in json_str:
                json_str = json_str.split("```", 1)[0]
            json_str = json_str.strip()

            config_data = json.loads(json_str)
            logger.info("Successfully generated initial configuration.")

            required_keys = [
                "summary",
                "classification_type",
                "class_labels",
                "prompts",
                "model_prefix",
                "training_data_volume",
            ]
            if not all(key in config_data for key in required_keys):
                raise ValueError(
                    f"Generated config missing keys. Expected: {required_keys}, Got: {list(config_data.keys())}"
                )
            if (
                not isinstance(config_data["class_labels"], list)
                or not config_data["class_labels"]
            ):
                raise ValueError(
                    "Generated config 'class_labels' must be a non-empty list."
                )
            if not isinstance(config_data["prompts"], dict) or set(
                config_data["prompts"].keys()
            ) != set(config_data["class_labels"]):
                raise ValueError(
                    "Generated config 'prompts' must be a dict with keys matching 'class_labels'."
                )

            self.initial_config = config_data
            self.final_config = config_data.copy()  # Start final config as a copy

            # Store initial params to final_config
            self.final_config["parameters"] = self._get_init_parameters()

            # Set class-related attributes
            self.classification_type = self.final_config["classification_type"]
            self.class_labels = sorted(
                list(set(self.final_config["class_labels"]))
            )  # Ensure unique & sorted
            self.num_classes = len(self.class_labels)
            self.final_config["class_labels"] = (
                self.class_labels
            )  # Update config with sorted unique list

            if self.num_classes < 2:
                raise ValueError(
                    f"At least two unique class labels are required. Got: {self.class_labels}"
                )

            logger.info(f"Classification type: {self.classification_type}")
            logger.info(
                f"Class labels: {self.class_labels} (Count: {self.num_classes})"
            )

            return True

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response from config model: {e}\nRaw response:\n{raw_response_content}",
                exc_info=True,
            )
            return False
        except (ValueError, OpenAIError) as e:  # Catch OpenAIError too
            logger.error(
                f"Failed to generate initial configuration: {e}", exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error in _generate_initial_config_async: {e}",
                exc_info=True,
            )
            return False

    def _get_init_parameters(self) -> Dict[str, Any]:  # Add max_features_tfidf
        return {
            "problem_description": self.problem_description,
            "selected_data_gen_model": self.selected_data_gen_model,
            "output_base_path": str(self.output_base_path),
            "config_model": self.config_model,
            "batch_size": self.batch_size,
            "prompt_refinement_cycles": self.prompt_refinement_cycles,
            "generate_edge_cases": self.generate_edge_cases,
            "edge_case_volume_per_class": self.edge_case_volume_per_class,
            "analyze_performance_data_path": (
                str(self.analyze_performance_data_path)
                if self.analyze_performance_data_path
                else None
            ),
            "language": self.language,
            "max_features_tfidf": self.max_features_tfidf,
        }

    def _prepare_output_directory(
        self,
    ):  # (No changes needed here, but verify settings paths)
        # ... (keep existing implementation, ensure it uses new settings.CONFIG_FILENAME etc.)
        if not self.final_config or "model_prefix" not in self.final_config:
            logger.error("Config unavailable, cannot prepare output directory.")
            return False  # Critical error

        model_prefix = self.final_config["model_prefix"]
        self.model_output_path = self.output_base_path / model_prefix
        self.raw_responses_path = self.model_output_path / settings.RAW_RESPONSES_DIR
        self.dataset_path = self.model_output_path / settings.TRAINING_DATASET_FILENAME
        self.edge_case_dataset_path = (
            self.model_output_path / settings.EDGE_CASE_DATASET_FILENAME
        )
        self.final_config_path = self.model_output_path / settings.CONFIG_FILENAME
        self.classifier_metadata_path = (
            self.model_output_path
            / f"{model_prefix}_{settings.CLASSIFIER_META_FILENAME}"
        )

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
        class_label_str: Optional[str],  # Changed from `label: int`
        model: str,
        status: str = "success",
    ):
        if not self.raw_responses_path:
            logger.warning("Raw responses path not set. Skipping save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label_file_part = f"_{class_label_str}" if class_label_str else ""
        safe_model = model.replace("/", "_").replace(
            ":", "_"
        )  # Sanitize model name for filename
        filename = (
            self.raw_responses_path
            / f"{prompt_type}{label_file_part}_{safe_model}_{status}_{ts}.json"
        )
        data = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model,
            "prompt_type": prompt_type,
            "class_label": class_label_str,  # Store string label
            "status": status,
            "prompt": prompt,
            "response": response,
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Log as warning, not error, as this is not critical path for data gen
            logger.warning(
                f"Failed to save raw response to {filename}: {e}", exc_info=False
            )

    def _append_to_dataset(
        self, text_data: str, class_label_str: str, target_path: Path
    ) -> int:
        if not class_label_str:  # Ensure class_label_str is provided
            logger.error("Class label string is required to append to dataset.")
            return 0
        if not target_path:
            logger.error("Target path not set, cannot append data.")
            return 0

        lines_added = 0
        try:
            # Create directory for target_path if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)

            write_header = not target_path.exists() or target_path.stat().st_size == 0
            with open(target_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                if write_header:
                    writer.writerow(["text", "label"])  # Label will be string here

                # Extract JSON content from text_data
                try:
                    json_start = text_data.find("{")
                    json_end = text_data.rfind("}") + 1
                    if json_start == -1 or json_end == 0:
                        raise ValueError("No valid JSON found in text_data.")

                    json_content = text_data[json_start:json_end]
                    parsed_data = json.loads(json_content)

                    if not isinstance(parsed_data, dict):
                        raise ValueError("Parsed JSON is not a dictionary.")

                    for value in parsed_data.values():
                        cleaned_line = str(value).strip().strip('"')
                        if len(cleaned_line) >= settings.MIN_DATA_LINE_LENGTH:
                            writer.writerow([f'"{cleaned_line}"', str(class_label_str)])
                            lines_added += 1

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse JSON from text_data: {e}", exc_info=False
                    )
                except ValueError as e:
                    logger.error(
                        f"Invalid JSON structure in text_data: {e}", exc_info=False
                    )

            return lines_added
        except IOError as e:
            logger.error(f"IOError appending to {target_path}: {e}", exc_info=False)
            return 0
        except Exception as e:  # Catch any other unexpected errors
            logger.error(
                f"Unexpected error appending to {target_path}: {e}", exc_info=True
            )
            return 0

    async def _generate_text_samples_batch_async(
        self,
        prompts_classlabels: List[
            Tuple[str, str, str]
        ],  # prompt, class_label_str, prompt_type
        model: str,
        system_prompt: str,
    ) -> List[Tuple[str, str, str, str]]:  # prompt, class_label_str, response, status
        tasks = []
        for prompt, class_label_str, prompt_type in prompts_classlabels:
            tasks.append(
                self._call_llm_async(
                    model=model, system_prompt=system_prompt, user_prompt=prompt
                )
            )

        # Use asyncio.gather with return_exceptions=True to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            original_prompt, class_label_str, prompt_type = prompts_classlabels[i]
            if isinstance(result, Exception):
                error_msg = f"LLM_API_ERROR: {type(result).__name__} - {str(result)}"
                logger.error(
                    f"Data generation failed for {prompt_type} (Class: {class_label_str}) using model {model} - {error_msg}"
                )
                self._save_raw_response(
                    prompt_type,
                    original_prompt,
                    error_msg,
                    class_label_str,
                    model,
                    status="failed",
                )
                processed_results.append(
                    (original_prompt, class_label_str, error_msg, "failed")
                )
            elif not result:  # Empty response check
                error_msg = "LLM_EMPTY_RESPONSE"
                logger.error(
                    f"Data generation returned empty response for {prompt_type} (Class: {class_label_str}) using model {model}"
                )
                self._save_raw_response(
                    prompt_type,
                    original_prompt,
                    error_msg,
                    class_label_str,
                    model,
                    status="failed_empty",
                )
                processed_results.append(
                    (original_prompt, class_label_str, error_msg, "failed_empty")
                )
            else:  # Success
                self._save_raw_response(
                    prompt_type,
                    original_prompt,
                    result,
                    class_label_str,
                    model,
                    status="success",
                )
                processed_results.append(
                    (original_prompt, class_label_str, result, "success")
                )
        return processed_results

    async def _run_prompt_refinement_cycle_async(self, cycle_num: int) -> bool:
        if not self.final_config or not self.class_labels or not self.num_classes:
            logger.error("Cannot run refinement: Final config or class info missing.")
            return False

        logger.info(
            f"--- Starting Prompt Refinement Cycle {cycle_num + 1}/{self.prompt_refinement_cycles} ---"
        )

        current_prompts_dict = self.final_config[
            "prompts"
        ]  # Dict of {class_label: prompt}
        data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
            language=self.language
        )

        # 1. Generate small sample batch with current prompts for each class
        logger.info("Generating sample data for refinement evaluation...")
        sample_prompts_classlabels = []
        # Ensure PROMPT_REFINEMENT_BATCH_SIZE is per class
        num_samples_per_class_for_refinement = settings.PROMPT_REFINEMENT_BATCH_SIZE

        for class_label in self.class_labels:
            prompt_for_class = current_prompts_dict.get(class_label)
            if not prompt_for_class:
                logger.warning(
                    f"No prompt found for class '{class_label}' in refinement cycle. Skipping."
                )
                continue
            for _ in range(num_samples_per_class_for_refinement):
                sample_prompts_classlabels.append(
                    (prompt_for_class, class_label, "refinement_sample")
                )

        if not sample_prompts_classlabels:
            logger.warning(
                f"Refinement Cycle {cycle_num + 1}: No valid prompts to generate samples. Skipping refinement."
            )
            return False

        sample_results = await self._generate_text_samples_batch_async(
            sample_prompts_classlabels,
            self.selected_data_gen_model,
            data_gen_sys_prompt,
        )

        samples_by_class = {label: [] for label in self.class_labels}
        for _, class_label, response_content, status in sample_results:
            if status == "success" and class_label in samples_by_class:
                # Split response content into individual samples if LLM returns multiple lines
                for line in response_content.split("\n"):
                    cleaned_line = line.strip()
                    if cleaned_line:  # Add if not empty
                        samples_by_class[class_label].append(cleaned_line)

        samples_by_class_str_parts = []
        any_samples_generated = False
        for class_label, samples in samples_by_class.items():
            if samples:
                any_samples_generated = True
                samples_preview = "\n".join(
                    [f"  - {s[:100]}..." for s in samples[:3]]
                )  # Show first 3 samples (truncated)
                samples_by_class_str_parts.append(
                    f"* Samples for Class '{class_label}':\n{samples_preview}"
                )
            else:
                samples_by_class_str_parts.append(
                    f"* Samples for Class '{class_label}': No samples generated."
                )

        if not any_samples_generated:
            logger.warning(
                f"Refinement Cycle {cycle_num + 1}: Failed to generate any sample data across all classes. Skipping refinement."
            )
            # Retain original prompts
            self.prompt_refinement_history.append(
                {
                    "cycle": cycle_num + 1,
                    "evaluation": "Skipped - No sample data generated.",
                    "previous_prompts": current_prompts_dict.copy(),
                    "refined_prompts": current_prompts_dict.copy(),  # No change
                }
            )
            return False  # Indicate no successful refinement occurred

        refinement_user_prompt = settings.PROMPT_REFINEMENT_TEMPLATE.format(
            problem_description=self.problem_description,
            classification_type=self.classification_type,
            class_labels_str=str(self.class_labels),
            current_prompts_json_str=json.dumps(current_prompts_dict, indent=2),
            samples_by_class_str="\n\n".join(samples_by_class_str_parts),
        )

        logger.info("Asking config model for prompt refinement suggestions...")
        raw_refinement_response = ""
        try:
            raw_refinement_response = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=refinement_user_prompt,
                temperature=0.4,  # Balance creativity and structure
            )

            json_ref_str = raw_refinement_response
            if "```json" in json_ref_str:
                json_ref_str = json_ref_str.split("```json", 1)[1]
            if "```" in json_ref_str:
                json_ref_str = json_ref_str.split("```", 1)[0]
            json_ref_str = json_ref_str.strip()

            refinement_data = json.loads(json_ref_str)

            if (
                "evaluation_summary" not in refinement_data
                or "refined_prompts" not in refinement_data
            ):
                raise ValueError(
                    "Refinement response missing required keys 'evaluation_summary' or 'refined_prompts'."
                )
            if not isinstance(refinement_data["refined_prompts"], dict):
                raise ValueError("'refined_prompts' must be a dictionary.")
            if set(refinement_data["refined_prompts"].keys()) != set(self.class_labels):
                raise ValueError(
                    f"Refined prompts keys {list(refinement_data['refined_prompts'].keys())} do not match class labels {self.class_labels}."
                )

            new_prompts_dict = refinement_data["refined_prompts"]
            logger.info(
                f"Refinement Cycle {cycle_num + 1} Evaluation: {refinement_data['evaluation_summary']}"
            )

            # Update prompts in final_config
            prompts_changed = False
            for class_label, new_prompt in new_prompts_dict.items():
                if self.final_config["prompts"].get(class_label) != new_prompt:
                    logger.info(f"Prompt for class '{class_label}' updated.")
                    self.final_config["prompts"][class_label] = new_prompt
                    prompts_changed = True

            if not prompts_changed:
                logger.info(
                    f"Refinement Cycle {cycle_num + 1}: No prompts were changed by the LLM."
                )

            self.prompt_refinement_history.append(
                {
                    "cycle": cycle_num + 1,
                    "evaluation": refinement_data["evaluation_summary"],
                    "previous_prompts": current_prompts_dict.copy(),  # Log old prompts
                    "refined_prompts": new_prompts_dict.copy(),  # Log new prompts
                }
            )
            logger.info(f"--- Prompt Refinement Cycle {cycle_num + 1} Completed ---")
            return True

        except json.JSONDecodeError as e:
            logger.error(
                f"Refinement Cycle {cycle_num + 1}: Failed to parse JSON response from config model: {e}\nRaw response:\n{raw_refinement_response}",
                exc_info=True,
            )
            return False
        except (ValueError, OpenAIError) as e:
            logger.error(
                f"Refinement Cycle {cycle_num + 1}: Failed: {e}", exc_info=True
            )
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error in refinement cycle {cycle_num + 1}: {e}",
                exc_info=True,
            )
            return False

    async def _generate_training_data_async(self) -> int:
        if (
            not self.final_config
            or not self.dataset_path
            or not self.class_labels
            or not self.num_classes
        ):
            logger.error(
                "Cannot generate training data: Configuration, dataset path, or class info missing."
            )
            return 0

        logger.info("--- Starting Bulk Training Data Generation ---")
        prompts_by_class = self.final_config["prompts"]
        target_total_volume = self.final_config.get(
            "training_data_volume", settings.DEFAULT_TRAINING_DATA_VOLUME
        )
        target_volume_per_class = max(2500, target_total_volume // self.num_classes)

        data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
            language=self.language
        )

        logger.info(
            f"Target total samples: ~{target_total_volume} (~{target_volume_per_class} per class). Batch size: {self.batch_size}"
        )
        logger.info(f"Using Model: {self.selected_data_gen_model}")

        total_samples_generated_overall = 0
        total_requests_made = 0
        total_failed_requests = 0

        required_requests_per_class = max(1, round(target_volume_per_class / 50))

        prompts_for_generation_round = []
        # Create a list of (prompt, class_label_str, 'training_data') tuples for all requests needed
        for class_idx, class_label in enumerate(self.class_labels):
            class_prompt = prompts_by_class.get(class_label)
            if not class_prompt:
                logger.warning(
                    f"Skipping data generation for class '{class_label}' as no prompt is defined."
                )
                continue
            for _ in range(required_requests_per_class):
                prompts_for_generation_round.append(
                    (class_prompt, class_label, "training_data")
                )

        # Shuffle to interleave if desired, or process class by class for more control
        # For now, process in order which groups by class, then by request for that class.
        # A more advanced approach might track generated counts per class and prioritize.

        num_batches = (
            len(prompts_for_generation_round) + self.batch_size - 1
        ) // self.batch_size

        samples_generated_per_class = {label: 0 for label in self.class_labels}

        for batch_num in range(num_batches):
            # Check if overall target or per-class targets are met (approximate)
            # This is a soft check; the loop might do one more batch.
            if all(
                samples_generated_per_class[lbl] >= target_volume_per_class
                for lbl in self.class_labels
                if prompts_by_class.get(lbl)
            ):
                logger.info(
                    "Approximate target volume per class reached. Stopping training data generation."
                )
                break

            start_index = batch_num * self.batch_size
            end_index = start_index + self.batch_size
            current_batch_prompts_labels = prompts_for_generation_round[
                start_index:end_index
            ]

            if (
                not current_batch_prompts_labels
            ):  # Should not happen if num_batches is correct
                break

            logger.info(
                f"Starting training data batch {batch_num + 1}/{num_batches} ({len(current_batch_prompts_labels)} API calls)"
            )
            total_requests_made += len(current_batch_prompts_labels)

            batch_results = await self._generate_text_samples_batch_async(
                current_batch_prompts_labels,
                self.selected_data_gen_model,
                data_gen_sys_prompt,
            )

            batch_samples_added_overall = 0
            num_failed_in_batch = 0
            for prompt_used, class_label_str, response_content, status in batch_results:
                if status == "success":
                    lines_added = self._append_to_dataset(
                        response_content, class_label_str, self.dataset_path
                    )
                    if lines_added > 0:
                        samples_generated_per_class[class_label_str] += lines_added
                        batch_samples_added_overall += lines_added
                else:
                    total_failed_requests += 1
                    num_failed_in_batch += 1

            total_samples_generated_overall += batch_samples_added_overall
            logger.info(
                f"Batch {batch_num + 1} completed. Added {batch_samples_added_overall} samples. "
                f"Total overall: {total_samples_generated_overall}. Failed in batch: {num_failed_in_batch}."
            )
            logger.info(f"Samples per class so far: {samples_generated_per_class}")

            await asyncio.sleep(getattr(settings, "API_DELAY_BETWEEN_BATCHES", 0.5))

        logger.info("--- Bulk Training Data Generation Finished ---")
        logger.info(
            f"Total training samples generated: {total_samples_generated_overall}"
        )
        logger.info(f"Final samples per class: {samples_generated_per_class}")
        logger.info(
            f"Total training API requests: {total_requests_made} (Failed: {total_failed_requests})"
        )
        if self.dataset_path and self.dataset_path.exists():
            logger.info(f"Training dataset saved to: {self.dataset_path}")
        else:
            logger.warning(
                f"Training dataset file not found at expected location: {self.dataset_path}"
            )

        return total_samples_generated_overall

    async def _generate_edge_cases_async(self) -> int:
        if not self.generate_edge_cases:
            logger.info("Skipping edge case generation as per configuration.")
            return 0
        if (
            not self.final_config
            or not self.edge_case_dataset_path
            or not self.class_labels
            or not self.num_classes
        ):
            logger.error(
                "Cannot generate edge cases: Config, path, or class info missing."
            )
            return 0

        logger.info("--- Starting Edge Case Generation ---")
        target_volume_per_class = self.edge_case_volume_per_class
        edge_case_model = self.config_model  # Use more capable model for edge cases
        edge_case_sys_prompt = (
            "You are a data generation assistant specializing in creating challenging test cases. "
            "Focus on subtlety and ambiguity as per instructions."
        )

        total_edge_cases_generated_overall = 0
        total_requests_made_edge = 0
        total_failed_requests_edge = 0

        lines_per_api_call_edge_estimate = getattr(
            settings, "EDGE_CASE_LINES_PER_API_CALL", 40
        )  # Fewer, more focused examples
        required_requests_per_class_edge = max(
            1, round(target_volume_per_class / lines_per_api_call_edge_estimate)
        )

        edge_prompts_for_generation = []

        for class_label_str in self.class_labels:
            # Use the generic EDGE_CASE_PROMPT_TEMPLATE, filling in the specific class_label
            edge_case_user_prompt = settings.EDGE_CASE_PROMPT_TEMPLATE.format(
                class_label=class_label_str,
                classification_type=self.classification_type,
                problem_description=self.problem_description,
                all_class_labels_str=str(
                    self.class_labels
                ),  # Provide context of all classes
                language=self.language,
            )
            # Store the specific prompt used for this class's edge cases in final_config
            if "edge_case_prompts" not in self.final_config:
                self.final_config["edge_case_prompts"] = {}
            self.final_config["edge_case_prompts"][
                class_label_str
            ] = edge_case_user_prompt

            for _ in range(required_requests_per_class_edge):
                edge_prompts_for_generation.append(
                    (edge_case_user_prompt, class_label_str, "edge_case")
                )

        edge_batch_size = max(
            1, self.batch_size // 2
        )  # Potentially smaller batches for config model
        num_batches_edge = (
            len(edge_prompts_for_generation) + edge_batch_size - 1
        ) // edge_batch_size

        edge_samples_generated_per_class = {label: 0 for label in self.class_labels}

        for batch_num in range(num_batches_edge):
            if all(
                edge_samples_generated_per_class[lbl] >= target_volume_per_class
                for lbl in self.class_labels
            ):
                logger.info(
                    "Approximate target edge case volume per class reached. Stopping generation."
                )
                break

            start_index = batch_num * edge_batch_size
            end_index = start_index + edge_batch_size
            current_batch_prompts_labels_edge = edge_prompts_for_generation[
                start_index:end_index
            ]

            if not current_batch_prompts_labels_edge:
                break

            logger.info(
                f"Starting edge case batch {batch_num + 1}/{num_batches_edge} ({len(current_batch_prompts_labels_edge)} API calls)"
            )
            total_requests_made_edge += len(current_batch_prompts_labels_edge)

            batch_results_edge = await self._generate_text_samples_batch_async(
                current_batch_prompts_labels_edge, edge_case_model, edge_case_sys_prompt
            )

            batch_samples_added_overall_edge = 0
            num_failed_in_batch_edge = 0
            for _, class_label_str, response_content, status in batch_results_edge:
                if status == "success":
                    lines_added = self._append_to_dataset(
                        response_content, class_label_str, self.edge_case_dataset_path
                    )
                    if lines_added > 0:
                        edge_samples_generated_per_class[class_label_str] += lines_added
                        batch_samples_added_overall_edge += lines_added
                else:
                    total_failed_requests_edge += 1
                    num_failed_in_batch_edge += 1

            total_edge_cases_generated_overall += batch_samples_added_overall_edge
            logger.info(
                f"Edge Batch {batch_num + 1} completed. Added {batch_samples_added_overall_edge} samples. "
                f"Total overall: {total_edge_cases_generated_overall}. Failed in batch: {num_failed_in_batch_edge}."
            )
            logger.info(
                f"Edge samples per class so far: {edge_samples_generated_per_class}"
            )

            await asyncio.sleep(
                getattr(settings, "API_DELAY_BETWEEN_EDGE_BATCHES", 1.0)
            )

        logger.info("--- Edge Case Generation Finished ---")
        logger.info(
            f"Total edge case samples generated: {total_edge_cases_generated_overall}"
        )
        logger.info(f"Final edge cases per class: {edge_samples_generated_per_class}")
        logger.info(
            f"Total edge case API requests: {total_requests_made_edge} (Failed: {total_failed_requests_edge})"
        )
        if self.edge_case_dataset_path and self.edge_case_dataset_path.exists():
            logger.info(f"Edge case dataset saved to: {self.edge_case_dataset_path}")
        else:
            logger.warning(
                f"Edge case dataset file not found at expected location: {self.edge_case_dataset_path}"
            )

        return total_edge_cases_generated_overall

    async def _analyze_performance_async(self):
        if not self.analyze_performance_data_path:
            logger.info("Skipping performance analysis: No results file provided.")
            return
        if not self.final_config or not self.class_labels:
            logger.error(
                "Cannot analyze performance: Configuration or class_labels missing."
            )
            return

        logger.info(
            f"--- Starting Performance Analysis using {self.analyze_performance_data_path} ---"
        )
        raw_analysis_response = ""
        try:
            if not self.analyze_performance_data_path.exists():
                self.performance_analysis_result = (
                    f"File not found: {self.analyze_performance_data_path}"
                )
                logger.error(self.performance_analysis_result)
                # Add this to final config as an error message
                self.final_config["performance_analysis"] = {
                    "input_file": str(self.analyze_performance_data_path),
                    "error": self.performance_analysis_result,
                    "llm_analysis": None,
                }
                return

            results_summary = ""
            try:
                # Assuming the CSV from run_predictions has 'text', 'label' (true), and one column per class_label for probabilities,
                # plus a 'predicted_label' column.
                df = pd.read_csv(self.analyze_performance_data_path)

                # Identify crucial columns
                text_col = "text"
                true_label_col = "label"  # Original true label (string)
                predicted_label_col = "predicted_label"  # Predicted label (string)

                if not all(
                    c in df.columns
                    for c in [text_col, true_label_col, predicted_label_col]
                ):
                    raise ValueError(
                        f"Performance data CSV missing one or more required columns: '{text_col}', '{true_label_col}', '{predicted_label_col}'. Found: {df.columns.tolist()}"
                    )

                df["is_correct"] = df[true_label_col] == df[predicted_label_col]
                accuracy = df["is_correct"].mean()
                logger.info(
                    f"Calculated Accuracy from '{self.analyze_performance_data_path}': {accuracy:.4f}"
                )

                # Prepare summary for LLM: focus on errors and some correct examples
                limit_samples = 20
                errors_df = df[~df["is_correct"]].head(limit_samples // 2)
                correct_df = df[df["is_correct"]].head(limit_samples - len(errors_df))
                sample_df = pd.concat([errors_df, correct_df]).reset_index(drop=True)

                sample_results_list = []
                for _, row in sample_df.iterrows():
                    text_snippet = str(row[text_col])[:150] + "..."
                    true_lbl = row[true_label_col]
                    pred_lbl = row[predicted_label_col]

                    # Try to find probability columns if they exist (e.g. class_label_prob)
                    prob_strs = []
                    for cl_lbl in self.class_labels:
                        prob_col_name = f"{cl_lbl}_prob"  # Assuming this naming convention in run_predictions
                        if prob_col_name in row:
                            prob_strs.append(f"{cl_lbl}: {row[prob_col_name]:.3f}")

                    prob_info = f" (Probs: {', '.join(prob_strs)})" if prob_strs else ""
                    sample_results_list.append(
                        f'- Text: "{text_snippet}", True: {true_lbl}, Predicted: {pred_lbl}{prob_info}'
                    )
                results_summary = "\n".join(sample_results_list)
                if not results_summary:
                    results_summary = "No sample misclassifications or results could be extracted for LLM summary."

            except Exception as e:
                logger.error(
                    f"Error reading or processing performance data file '{self.analyze_performance_data_path}': {e}",
                    exc_info=True,
                )
                results_summary = f"Error processing performance data file: {e}. Cannot provide detailed samples to LLM."

            analysis_prompt = settings.PERFORMANCE_ANALYSIS_PROMPT_TEMPLATE.format(
                problem_description=self.problem_description,
                classification_type=self.classification_type,
                class_labels_str=str(self.class_labels),
                final_prompts_json_str=json.dumps(
                    self.final_config["prompts"], indent=2
                ),
                test_results_summary=results_summary,
            )

            logger.info(
                "Asking config model for performance analysis and recommendations..."
            )
            raw_analysis_response = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=analysis_prompt,
                max_tokens=2000,  # Allow longer response for analysis
                temperature=0.4,
            )

            self.performance_analysis_result = raw_analysis_response
            self.final_config["performance_analysis"] = {
                "input_file": str(self.analyze_performance_data_path),
                "llm_analysis": self.performance_analysis_result,
                "accuracy_from_file": (
                    float(accuracy)
                    if "accuracy" in locals() and pd.notna(accuracy)
                    else None
                ),
            }
            logger.info("--- Performance Analysis Finished ---")
            logger.info(f"LLM Analysis:\n{self.performance_analysis_result[:500]}...")

        except FileNotFoundError:  # Already handled above by setting error message
            pass  # Error logged and recorded in config
        except (ValueError, OpenAIError) as e:
            logger.error(
                f"Performance analysis LLM call failed: {e}\nLLM response (if any) leading to error:\n{raw_analysis_response}",
                exc_info=True,
            )
            self.final_config["performance_analysis"] = {
                "input_file": (
                    str(self.analyze_performance_data_path)
                    if self.analyze_performance_data_path
                    else "N/A"
                ),
                "error": f"LLM call failed: {e}",
                "llm_analysis_raw_error_response": raw_analysis_response,
            }
        except Exception as e:
            logger.error(
                f"Performance analysis failed unexpectedly: {e}", exc_info=True
            )
            self.final_config["performance_analysis"] = {
                "input_file": (
                    str(self.analyze_performance_data_path)
                    if self.analyze_performance_data_path
                    else "N/A"
                ),
                "error": f"Unexpected error: {e}",
                "llm_analysis": None,
            }

    def _save_final_config(self):
        if (
            not self.final_config
            or not self.model_output_path
            or not self.final_config_path
        ):
            logger.error(
                "Cannot save final config: Configuration, model_output_path or final_config_path missing."
            )
            return False

        logger.info(f"Saving final configuration to: {self.final_config_path}")

        self.final_config["generation_timestamp"] = datetime.now().isoformat()
        # Ensure initial_config is what was first generated, not a reference that got modified
        # self.final_config["initial_config"] is already set if _generate_initial_config_async was successful

        # This will be populated by TextClassifier.save()
        # self.final_config["classifier_metadata_file"] = str(self.classifier_metadata_path) if self.classifier_metadata_path and self.classifier_metadata_path.exists() else None

        self.final_config["prompt_refinement_history"] = (
            self.prompt_refinement_history
        )  # Already a list of dicts

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
            "raw_api_responses": (
                str(self.raw_responses_path)
                if self.raw_responses_path and self.raw_responses_path.exists()
                else None
            ),
            "final_config_file": str(self.final_config_path),
            "trained_model_prefix": str(
                self.model_output_path / self.final_config.get("model_prefix", "model")
            ),
            "onnx_model_path": (
                str(
                    self.model_output_path
                    / f"{self.final_config.get('model_prefix', 'model')}.onnx"
                )
                if self.model_output_path
                else None
            ),
            "performance_predictions_csv": (
                str(self.analyze_performance_data_path)
                if self.analyze_performance_data_path
                and self.analyze_performance_data_path.exists()
                else None
            ),
        }

        # performance_analysis is added directly in _analyze_performance_async

        try:
            with open(self.final_config_path, "w", encoding="utf-8") as f:
                json.dump(self.final_config, f, indent=2, ensure_ascii=False)
            logger.info("Final configuration saved successfully.")
            return True
        except TypeError as e:
            # Try to find non-serializable parts for better debugging
            def get_non_serializable_paths(d, path=""):
                paths = []
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            json.dumps({k: v})
                        except TypeError:
                            paths.append(f"{path}.{k} (type: {type(v).__name__})")
                        paths.extend(get_non_serializable_paths(v, f"{path}.{k}"))
                elif isinstance(d, list):
                    for i, item in enumerate(d):
                        try:
                            json.dumps({str(i): item})
                        except TypeError:
                            paths.append(f"{path}[{i}] (type: {type(item).__name__})")
                        paths.extend(get_non_serializable_paths(item, f"{path}[{i}]"))
                return paths

            non_serializable = get_non_serializable_paths(self.final_config)
            logger.error(
                f"Failed to serialize final config to JSON: {e}. "
                f"Potential non-serializable parts: {non_serializable}",
                exc_info=True,
            )
            return False
        except IOError as e:
            logger.error(f"Failed to write final config file: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving final config: {e}", exc_info=True)
            return False

    def run_predictions_on_edge_cases(
        self, model_type_selection: str, classifier_to_test: TextClassifierRunner
    ):
        """Runs predictions on edge case data and saves them."""
        if not self.edge_case_dataset_path or not self.edge_case_dataset_path.exists():
            logger.warning(
                f"Edge case dataset not found at {self.edge_case_dataset_path}. Skipping predictions."
            )
            self.analyze_performance_data_path = None  # Ensure it's None if no file
            return

        logger.info(
            f"Running predictions on edge case data: {self.edge_case_dataset_path}"
        )

        try:
            df_edge = pd.read_csv(self.edge_case_dataset_path, encoding="utf-8")
            if df_edge.empty:
                logger.warning("Edge case dataset is empty. Skipping predictions.")
                self.analyze_performance_data_path = None
                return

            all_probas = classifier_to_test.predict(df_edge["text"].tolist())

            if self.classification_type == "binary_sigmoid":
                predictions = [1 if (p[0] if isinstance(p, list) else p) >= 0.5 else 0 for p in all_probas]
                df_edge["probability"] = [p[0] if isinstance(p, list) else p for p in all_probas]
                df_edge["predicted_label"] = predictions
            elif self.classification_type == "multiclass_softmax":
                df_edge["predicted_label"] = [classifier_to_test.labels[pred] for pred in all_probas]
            elif self.classification_type == "multilabel_sigmoid":
                for i, class_label in enumerate(classifier_to_test.labels):
                    df_edge[f"{class_label}_prob"] = all_probas[:, i]
                predictions = [[1 if p >= 0.5 else 0 for p in proba_vec] for proba_vec in all_probas]
                df_edge["predicted_label"] = [
                    ",".join([classifier_to_test.labels[i] for i, is_active in enumerate(pred) if is_active])
                    for pred in predictions
                ]

            # Output path for these predictions, used for analysis
            # Ensure model_output_path and model_prefix are available
            if (
                not self.model_output_path
                or not self.final_config
                or "model_prefix" not in self.final_config
            ):
                logger.error(
                    "Cannot determine output path for edge case predictions: model_output_path or model_prefix missing."
                )
                # Fallback path if needed, or raise error
                pred_output_path_str = f"edge_case_predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            else:
                pred_output_path_str = str(
                    self.model_output_path
                    / f"{self.final_config['model_prefix']}_edge_case_predictions.csv"
                )

            self.analyze_performance_data_path = Path(pred_output_path_str)
            df_edge.to_csv(
                self.analyze_performance_data_path, index=False, encoding="utf-8"
            )
            logger.info(
                f"Edge case predictions (with probabilities) saved to {self.analyze_performance_data_path}"
            )

        except FileNotFoundError:
            logger.warning(
                f"Edge case CSV file not found at {self.edge_case_dataset_path} during prediction run."
            )
            self.analyze_performance_data_path = None
        except pd.errors.EmptyDataError:
            logger.warning(
                f"Edge case CSV file at {self.edge_case_dataset_path} is empty."
            )
            self.analyze_performance_data_path = None
        except Exception as e:
            logger.error(f"Error running predictions on edge cases: {e}", exc_info=True)
            self.analyze_performance_data_path = None  # Nullify if error

    async def generate_data_and_train_model_async(self, model_type_selection: str):
        start_time = datetime.now()
        logger.info(
            f"=== Starting Data Generation & Model Training Process for: '{self.problem_description}' using {model_type_selection} ==="
        )

        training_samples_count = 0
        edge_case_samples_count = 0

        try:
            # 1. Generate Initial Configuration (sets self.class_labels, self.num_classes)
            if not self.resume_from_config:
                if not await self._generate_initial_config_async():
                    logger.critical(
                        "Failed to generate initial configuration. Aborting."
                    )
                    return
                # Config now contains class_labels, num_classes etc.

                # 2. Prepare Output Directory
                if (
                    not self._prepare_output_directory()
                ):  # Uses model_prefix from config
                    logger.critical("Failed to prepare output directory. Aborting.")
                    return

                # 3. Prompt Refinement Cycles
                if self.prompt_refinement_cycles > 0:
                    for i in range(self.prompt_refinement_cycles):
                        logger.info(
                            f"Starting prompt refinement cycle {i + 1} of {self.prompt_refinement_cycles}"
                        )
                        refinement_success = (
                            await self._run_prompt_refinement_cycle_async(i)
                        )
                        if not refinement_success:
                            logger.warning(
                                f"Prompt refinement cycle {i + 1} did not result in changes or failed."
                            )
                else:
                    logger.info(
                        "Skipping prompt refinement cycles as per configuration."
                    )

            # 4. Generate Main Training Data
            if (
                self.skip_data_gen
                and self.dataset_path.exists()
                and self.dataset_path.stat().st_size > 0
            ):
                logger.info(
                    f"Training data {self.dataset_path} already exists and is not empty. Skipping generation."
                )
                # Estimate count from existing file for logging, or load to get exact for sophisticated resume
                try:
                    df_train_existing = pd.read_csv(self.dataset_path)
                    training_samples_count = len(df_train_existing)
                    logger.info(
                        f"Found {training_samples_count} samples in existing training data."
                    )
                except Exception as e_read:
                    logger.warning(
                        f"Could not read existing training data file to get count: {e_read}"
                    )
                    training_samples_count = -1  # Indicate unknown
            else:
                training_samples_count = await self._generate_training_data_async()
                if training_samples_count == 0:
                    logger.critical(
                        "No training data was generated. Aborting model training."
                    )
                    self._save_final_config()  # Save what we have so far
                    return

            # 5. Generate Edge Case Data
            if self.generate_edge_cases:
                if (
                    self.edge_case_dataset_path
                    and self.edge_case_dataset_path.exists()
                    and self.edge_case_dataset_path.stat().st_size > 0
                ):
                    logger.info(
                        f"Edge case data {self.edge_case_dataset_path} already exists and is not empty. Skipping generation."
                    )
                    try:
                        df_edge_existing = pd.read_csv(self.edge_case_dataset_path)
                        edge_case_samples_count = len(df_edge_existing)
                        logger.info(
                            f"Found {edge_case_samples_count} samples in existing edge case data."
                        )
                    except Exception as e_read_edge:
                        logger.warning(
                            f"Could not read existing edge case data file to get count: {e_read_edge}"
                        )
                        edge_case_samples_count = -1
                else:
                    edge_case_samples_count = await self._generate_edge_cases_async()
            else:
                logger.info("Skipping edge case generation.")

            if self.skip_model_training:
                logger.info("Skipping model training as per configuration. Aborting.")
                self._save_final_config()
                return

            if not self.num_classes or not self.class_labels:
                logger.critical(
                    "Number of classes or class labels not determined from config. Aborting training."
                )
                self._save_final_config()
                return

            logger.info(
                f"Starting model training using {model_type_selection} strategy..."
            )
            if (
                not self.dataset_path
                or not self.dataset_path.exists()
                or self.dataset_path.stat().st_size == 0
            ):
                logger.critical(
                    f"Training data CSV {self.dataset_path} is missing or empty. Cannot train model."
                )
                self._save_final_config()
                return

            logger.debug(f"Training data path: {self.dataset_path}")

            model_save_prefix = str(
                self.model_output_path / self.final_config["model_prefix"]
            )

            # 6. Train the model using the strategy

            classifier_runner = TextClassifierRunner(
                train_path=self.dataset_path,
                test_path=self.edge_case_dataset_path
                or self.dataset_path,  # For simplicity, use same for test
                data_type=self.classification_type,
                labels=self.class_labels,
                library_type=model_type_selection,
                output_path=model_save_prefix,
            )
            logger.info(
                f"Classifier (model, preprocessors, metadata) saved with prefix: {model_save_prefix}"
            )

            classifier_runner.train_model()

            # 10. Run Predictions on Edge Cases (if generated) for performance analysis input
            if self.generate_edge_cases and edge_case_samples_count > 0:
                self.run_predictions_on_edge_cases(
                    model_type_selection, classifier_runner
                )
            else:
                logger.info(
                    "Skipping predictions on edge cases as they were not generated or dataset is empty."
                )
                self.analyze_performance_data_path = None  # Ensure it's cleared

            # 11. Analyze Performance Data (if predictions were made and path is set)
            if self.analyze_performance_data_path:
                await self._analyze_performance_async()
            else:
                logger.info(
                    "Skipping LLM performance analysis as no prediction data path is set."
                )
                if self.final_config:
                    self.final_config["performance_analysis"] = {
                        "status": "skipped",
                        "reason": "No edge case prediction data.",
                    }

            # 12. Save Final Configuration (includes all paths, metrics, analysis)
            self._save_final_config()

        except (ValueError, OpenAIError) as e:  # Catch init or early LLM errors
            logger.critical(
                f"Data generation process failed critically: {e}", exc_info=True
            )
            if self.final_config and self.model_output_path:  # Try to save what we have
                self.final_config["error_summary"] = f"CRITICAL_ERROR: {e}"
                self._save_final_config()
        except Exception as e:
            logger.critical(
                f"An unexpected critical error occurred: {e}", exc_info=True
            )
            if self.final_config and self.model_output_path:
                self.final_config["error_summary"] = f"UNEXPECTED_CRITICAL_ERROR: {e}"
                self._save_final_config()
        finally:
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(
                f"=== Data Generation & Model Training Process Finished. Duration: {duration} ==="
            )
            logger.info(
                f"  Total training samples generated/used: {training_samples_count if training_samples_count != -1 else 'Unknown (used existing)'}"
            )
            if self.generate_edge_cases:
                logger.info(
                    f"  Total edge case samples generated/used: {edge_case_samples_count if edge_case_samples_count != -1 else 'Unknown (used existing)'}"
                )

            if self.performance_analysis_result:
                logger.info(
                    f"  Model performance feedback by LLM (first 200 chars): {self.performance_analysis_result[:200]}..."
                )
            elif (
                self.final_config
                and "performance_analysis" in self.final_config
                and "error" in self.final_config["performance_analysis"]
            ):
                logger.warning(
                    f"  Performance analysis had an error: {self.final_config['performance_analysis']['error']}"
                )
            elif (
                self.final_config
                and "performance_analysis" in self.final_config
                and self.final_config["performance_analysis"].get("status") == "skipped"
            ):
                logger.info("  Performance analysis was skipped.")

            if self.final_config_path and self.final_config_path.exists():
                logger.info(
                    f"  Find detailed configuration, paths, and results in: {self.final_config_path}"
                )
            else:
                logger.warning(
                    "Final configuration file may not have been saved or path is incorrect."
                )


# --- CLI Interface ---
async def cli_main():
    parser = argparse.ArgumentParser(
        description="Generate Training Data, Edge Cases, Train a Multiclass Text Classifier, and Analyze Performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--problem-description",
        required=True,
        help="Describe the classification problem (e.g., 'Classify movie reviews as positive, negative, or neutral').",
    )
    parser.add_argument(
        "--model-type",
        choices=["torch", "tensorflow", "sklearn"],
        default="tensorflow",
        help="ML library to use for the classifier model.",
    )
    parser.add_argument(
        "--data-gen-model",
        default=settings.DEFAULT_DATA_GEN_MODEL,
        help="OpenRouter model name for bulk data generation.",
    )
    parser.add_argument(
        "--config-llm-model",
        default=settings.DEFAULT_CONFIG_MODEL,
        help="OpenRouter model for config, refinement, edge cases, analysis (should be powerful, e.g., GPT-4, Claude Opus).",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default=settings.DEFAULT_OUTPUT_PATH,
        help="Base directory for all outputs (model data, configs, etc.).",
    )
    parser.add_argument(
        "--refinement-cycles",
        type=int,
        default=settings.DEFAULT_PROMPT_REFINEMENT_CYCLES,
        help="Number of prompt refinement cycles (0 to disable).",
    )
    parser.add_argument(
        "--generate-edge-cases",
        type=lambda x: (str(x).lower() == "true"),  # Boolean arg
        default=settings.DEFAULT_GENERATE_EDGE_CASES,
        help="Generate challenging edge case data (true/false).",
    )
    parser.add_argument(
        "--edge-case-volume-per-class",
        type=int,
        default=settings.DEFAULT_EDGE_CASE_VOLUME
        // 2,  # Default based on old total / 2 classes
        help="Target number of edge case samples to generate per class.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.DATA_GEN_BATCH_SIZE,
        help="Number of parallel LLM API requests for bulk data generation batches.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="english",
        help="Primary language for the generated dataset (e.g., 'english', 'spanish').",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of features for TF-IDF Vectorizer.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="Continue from a previous configuration file (if any).",
    )
    parser.add_argument(
        "--skip-data-gen",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Skip data generation and only train the model from existing data.",
    ),
    parser.add_argument(
        "--skip-model-training",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Skip model training and only generate data.",
    )
    args = parser.parse_args()

    try:
        generator = MulticlassDataGenerator(  # Use renamed class
            problem_description=args.problem_description,
            selected_data_gen_model=args.data_gen_model,
            config_model=args.config_llm_model,  # Renamed arg
            output_path=args.output_path,
            batch_size=args.batch_size,
            prompt_refinement_cycles=args.refinement_cycles,
            generate_edge_cases=args.generate_edge_cases,
            edge_case_volume_per_class=args.edge_case_volume_per_class,
            language=args.lang,
            max_features_tfidf=args.max_features,
            config_path=args.config_path,
            skip_data_gen=args.skip_data_gen,
            skip_model_training=args.skip_model_training,
        )
        await generator.generate_data_and_train_model_async(
            model_type_selection=args.model_type
        )

    except ValueError as e:  # Catch config errors from __init__
        logger.error(
            f"Configuration Error during setup: {e}", exc_info=False
        )  # No need for full stack for config errors
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred at the CLI level: {e}", exc_info=True
        )


if __name__ == "__main__":
    try:
        asyncio.run(cli_main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e:  # Catch-all for truly unexpected top-level errors
        logger.critical(
            f"Application failed unexpectedly at the highest level: {e}", exc_info=True
        )

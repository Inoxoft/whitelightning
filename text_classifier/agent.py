
"""
Text Classification Agent with Smart Activation Detection
"""
import argparse
import asyncio
import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np  
from openai import AsyncOpenAI, OpenAIError
import random

from text_classifier.train import TextClassifierRunner
from text_classifier.prepare_dataset import DatasetPreparer

try:
    import text_classifier.settings as settings
except ModuleNotFoundError: 
    import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True


class MulticlassDataGenerator:  
    def __init__(
        self,
        problem_description: str,
        selected_data_gen_model: str = settings.DEFAULT_DATA_GEN_MODEL,
        output_path: str = settings.DEFAULT_OUTPUT_PATH,
        config_model: str = settings.DEFAULT_CONFIG_MODEL,
        api_key: Optional[str] = settings.OPEN_ROUTER_API_KEY,
        api_base_url: str = settings.OPEN_ROUTER_BASE_URL,
        batch_size: int = settings.DATA_GEN_BATCH_SIZE,
        prompt_refinement_cycles: int = settings.DEFAULT_PROMPT_REFINEMENT_CYCLES,
        generate_edge_cases: bool = settings.DEFAULT_GENERATE_EDGE_CASES,
        edge_case_volume_per_class: int = settings.DEFAULT_EDGE_CASE_VOLUME
        // 2,
        analyze_performance_data_path: Optional[str] = None,
        language: Optional[str] = None,
        max_features_tfidf: int = 5000,
        config_path: Optional[str] = None,
        skip_data_gen: bool = False,
        skip_model_training: bool = False,
        use_own_dataset: Optional[str] = None,
        activation: str = "auto",
    ):
        self.skip_data_gen = skip_data_gen
        self.skip_model_training = skip_model_training
        self.use_own_dataset = use_own_dataset

      
        if use_own_dataset:
            self.skip_data_gen = True
            logger.info(f"Using own dataset: {use_own_dataset}. Data generation will be skipped.")

        if not problem_description and not skip_data_gen and not use_own_dataset:
            raise ValueError("Problem description cannot be empty when not using own dataset.")
        if not api_key and not use_own_dataset:  # API key not needed when using own dataset
            raise ValueError(
                "OpenRouter API key is not configured in .env or settings.py."
            )

        self.problem_description = problem_description
        self.selected_data_gen_model = selected_data_gen_model
        self.output_base_path = Path(output_path)
        self.config_model = config_model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.batch_size = batch_size
        self.prompt_refinement_cycles = prompt_refinement_cycles
        self.generate_edge_cases = generate_edge_cases
        self.edge_case_volume_per_class = edge_case_volume_per_class
        self.analyze_performance_data_path = (
            Path(analyze_performance_data_path)
            if analyze_performance_data_path
            else None
        )
        self.language = language if language else "english"
        self.max_features_tfidf = max_features_tfidf
        self.activation = activation

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

           
            self.model_output_path: Optional[Path] = None
            self.raw_responses_path: Optional[Path] = None
            self.dataset_path: Optional[Path] = None
            self.edge_case_dataset_path: Optional[Path] = None
            self.final_config_path: Optional[Path] = None
            self.classifier_metadata_path: Optional[Path] = None

          
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

    async def _call_llm_async(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 30000,
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
        
       
        if self.activation == "sigmoid":
            activation_instruction = "**IMPORTANT: User specified sigmoid activation - generate MULTILABEL classification where multiple labels can apply to the same text.**"
            multilabel_prompt_instruction = "For multilabel classification, create prompts that will generate text samples that can naturally have multiple labels simultaneously. The generated data should contain texts where several categories apply to the same sample."
        elif self.activation == "softmax":
            activation_instruction = "**IMPORTANT: User specified softmax activation - generate SINGLE-LABEL classification where only one label applies per text.**"
            multilabel_prompt_instruction = "For single-label classification, create prompts that generate text samples that clearly belong to only one category."
        else:  
            activation_instruction = "**User selected automatic activation detection - choose the most appropriate classification type based on the problem description.**"
            multilabel_prompt_instruction = "Create prompts appropriate for the classification type you determine from the problem description."
        
        user_prompt = settings.CONFIG_USER_PROMPT_TEMPLATE.format(
            problem_description=self.problem_description,
            activation_instruction=activation_instruction,
            multilabel_prompt_instruction=multilabel_prompt_instruction
        )
        raw_response_content = ""
        try:
            raw_response_content = await self._call_llm_async(
                model=self.config_model,
                system_prompt=settings.CONFIG_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
            )

            
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
            self.final_config = config_data.copy()  

            
            self.final_config["parameters"] = self._get_init_parameters()

            
            self.classification_type = self.final_config["classification_type"]
            self.class_labels = sorted(
                list(set(self.final_config["class_labels"]))
            )  
            self.num_classes = len(self.class_labels)
            self.final_config["class_labels"] = (
                self.class_labels
            ) 

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
        except (ValueError, OpenAIError) as e:  
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

    def _get_init_parameters(self) -> Dict[str, Any]:  
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
    ): 
        if not self.final_config or "model_prefix" not in self.final_config:
            logger.error("Config unavailable, cannot prepare output directory.")
            return False  

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
        class_label_str: Optional[str],  
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
        ) 
        filename = (
            self.raw_responses_path
            / f"{prompt_type}{label_file_part}_{safe_model}_{status}_{ts}.json"
        )
        data = {
            "timestamp": datetime.now().isoformat(),
            "model_used": model,
            "prompt_type": prompt_type,
            "class_label": class_label_str, 
            "status": status,
            "prompt": prompt,
            "response": response,
        }
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
           
            logger.warning(
                f"Failed to save raw response to {filename}: {e}", exc_info=False
            )

    def _append_to_dataset(
        self, text_data: str, class_label_str: str, target_path: Path
    ) -> int:
        if not class_label_str: 
            logger.error("Class label string is required to append to dataset.")
            return 0
        if not target_path:
            logger.error("Target path not set, cannot append data.")
            return 0

        lines_added = 0
        try:
            
            target_path.parent.mkdir(parents=True, exist_ok=True)

            write_header = not target_path.exists() or target_path.stat().st_size == 0
            with open(target_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                if write_header:
                    writer.writerow(["text", "label"]) 

              
                try:
                    json_start = text_data.find("{")
                    json_end = text_data.rfind("}") + 1
                    if json_start == -1 or json_end == 0:
                        logger.info("JSON structure not found. Parsing as simple text format.")
                       
                        import re
                        pattern = r'"(\d+)"\s*:\s*"([^"]*)"'
                        matches = list(re.finditer(pattern, text_data.strip()))
                        
                        if matches:
                           
                            for match in matches:
                                index = int(match.group(1))
                                text = match.group(2)
                                if text and not text.endswith('\\'):
                                    writer.writerow([f'"{text}"', str(class_label_str)])
                                    lines_added += 1
                        else:
                           
                            lines = text_data.strip().split('\n')
                            for line in lines:
                                cleaned_line = line.strip().strip('"').strip()
                                if (cleaned_line and 
                                    len(cleaned_line) >= settings.MIN_DATA_LINE_LENGTH and
                                    not cleaned_line.startswith('{') and
                                    not cleaned_line.endswith('}')):
                                    writer.writerow([f'"{cleaned_line}"', str(class_label_str)])
                                    lines_added += 1

                        return lines_added

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
        except Exception as e:  
            logger.error(
                f"Unexpected error appending to {target_path}: {e}", exc_info=True
            )
            return 0

    def _check_dataset_duplicate_rate(self, dataset_path: Path) -> Dict[str, Any]:
        """
        Check for duplicate entries in the training dataset and calculate duplicate rate.
        
        Returns:
            dict: Contains duplicate statistics including rate, count, and whether it exceeds threshold
        """
        if not dataset_path or not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return {"error": "Dataset file not found"}
        
        try:
            import pandas as pd
            
           
            df = pd.read_csv(dataset_path, on_bad_lines="skip")
            
            if df.empty:
                logger.warning("Dataset is empty")
                return {"error": "Dataset is empty"}
            
            if 'text' not in df.columns:
                logger.error("Dataset missing 'text' column")
                return {"error": "Dataset missing 'text' column"}
            
          
            df['text_normalized'] = df['text'].astype(str).str.strip().str.lower()
            
           
            total_samples = len(df)
            unique_samples = df['text_normalized'].nunique()
            duplicate_count = total_samples - unique_samples
            duplicate_rate = (duplicate_count / total_samples) * 100 if total_samples > 0 else 0
            
           
            threshold = getattr(settings, 'DUPLICATE_RATE_THRESHOLD', 5.0)
            exceeds_threshold = duplicate_rate > threshold
            
           
            duplicate_examples = []
            if duplicate_count > 0:
                duplicated_texts = df[df.duplicated(subset=['text_normalized'], keep=False)]
                if not duplicated_texts.empty:
                   
                    duplicate_groups = duplicated_texts.groupby('text_normalized')['text'].apply(list).head(3)
                    for normalized_text, text_list in duplicate_groups.items():
                        duplicate_examples.append({
                            "text": text_list[0][:100] + "..." if len(text_list[0]) > 100 else text_list[0],
                            "count": len(text_list)
                        })
            
            return {
                "total_samples": total_samples,
                "unique_samples": unique_samples,
                "duplicate_count": duplicate_count,
                "duplicate_rate": round(duplicate_rate, 2),
                "exceeds_threshold": exceeds_threshold,
                "threshold": threshold,
                "examples": duplicate_examples[:3] 
            }
            
        except Exception as e:
            logger.error(f"Error checking dataset duplicates: {e}", exc_info=True)
            return {"error": f"Error checking duplicates: {str(e)}"}

    def _notify_duplicate_rate(self, duplicate_stats: Dict[str, Any], dataset_name: str = "training") -> None:
        """
        Notify user only when duplicate rate could harm model performance.
        
        Args:
            duplicate_stats: Dictionary containing duplicate statistics
            dataset_name: Name of the dataset (e.g., "training", "edge_case")
        """
        if "error" in duplicate_stats:
            return 
        
        duplicate_rate = duplicate_stats["duplicate_rate"]
        exceeds_threshold = duplicate_stats["exceeds_threshold"]
        threshold = duplicate_stats["threshold"]
        
       
        if exceeds_threshold:
            logger.warning(f"\n⚠️  DATA QUALITY WARNING - {dataset_name.upper()} DATASET")
            logger.warning(f"🚨 High duplicate rate detected: {duplicate_rate:.2f}% (threshold: {threshold}%)")
            logger.warning(f"💡 This may cause poor model performance due to:")
            logger.warning(f"   • Model overfitting on repeated examples")
            logger.warning(f"   • Reduced generalization ability")
            logger.warning(f"   • Biased training patterns")
            logger.warning(f"🔧 Consider regenerating data with more diverse prompts\n")

    async def _generate_text_samples_batch_async(
        self,
        prompts_classlabels: List[
            Tuple[str, str, str]
        ],  
        model: str,
        system_prompt: str,
    ) -> List[Tuple[str, str, str, str]]: 
        tasks = []
        for prompt, class_label_str, prompt_type in prompts_classlabels:
            tasks.append(
                self._call_llm_async(
                    model=model, system_prompt=system_prompt, user_prompt=prompt
                )
            )

        
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
            elif not result:  
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
            else: 
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
        ]  
        data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
            language=self.language
        )

       
        logger.info("Generating sample data for refinement evaluation...")
        sample_prompts_classlabels = []
        
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
              
                for line in response_content.split("\n"):
                    cleaned_line = line.strip()
                    if cleaned_line: 
                        samples_by_class[class_label].append(cleaned_line)

        samples_by_class_str_parts = []
        any_samples_generated = False
        for class_label, samples in samples_by_class.items():
            if samples:
                any_samples_generated = True
                samples_preview = "\n".join(
                    [f"  - {s[:100]}..." for s in samples[:3]]
                ) 
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
           
            self.prompt_refinement_history.append(
                {
                    "cycle": cycle_num + 1,
                    "evaluation": "Skipped - No sample data generated.",
                    "previous_prompts": current_prompts_dict.copy(),
                    "refined_prompts": current_prompts_dict.copy(),  
                }
            )
            return False  

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
                temperature=0.4,  
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
                    "previous_prompts": current_prompts_dict.copy(),  
                    "refined_prompts": new_prompts_dict.copy(), 
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

        
        if (self.classification_type and "multilabel" in self.classification_type) or self.activation == "sigmoid":
            data_gen_sys_prompt = settings.MULTILABEL_DATA_GEN_SYSTEM_PROMPT.format(
                language=self.language
            )
            logger.info("🏷️  Using MULTILABEL data generation system prompt")
        else:
            data_gen_sys_prompt = settings.DATA_GEN_SYSTEM_PROMPT.format(
                language=self.language
            )
            logger.info("🎯 Using standard data generation system prompt")

        logger.info(
            f"Target total samples: ~{target_total_volume} (~{target_volume_per_class} per class). Batch size: {self.batch_size}"
        )
        logger.info(f"Using Model: {self.selected_data_gen_model}")

        total_samples_generated_overall = 0
        total_requests_made = 0
        total_failed_requests = 0

        required_requests_per_class = max(1, round(target_volume_per_class / 50))

        prompts_for_generation_round = []
       
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


        num_batches = (
            len(prompts_for_generation_round) + self.batch_size - 1
        ) // self.batch_size

        samples_generated_per_class = {label: 0 for label in self.class_labels}

        for batch_num in range(num_batches):
           
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
            ):  
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
            
           
            duplicate_stats = self._check_dataset_duplicate_rate(self.dataset_path)
            self._notify_duplicate_rate(duplicate_stats, "training")
            
            
            if self.final_config:
                self.final_config["training_duplicate_stats"] = duplicate_stats
        else:
            logger.warning(
                f"Training dataset file not found at expected location: {self.dataset_path}"
            )

        return total_samples_generated_overall

    async def _generate_multilabel_training_data_async(self) -> int:
        """
        Generate multilabel training data using a reliable two-step approach:
        1. Generate high-quality single-label data for each class
        2. Programmatically combine samples to create multilabel format
        """
        if (
            not self.final_config
            or not self.dataset_path
            or not self.class_labels
            or not self.num_classes
        ):
            logger.error(
                "Cannot generate multilabel training data: Configuration, dataset path, or class info missing."
            )
            return 0

        logger.info("--- Starting Smart Multilabel Training Data Generation ---")
        logger.info("🎯 Step 1: Generate single-label data (proven method)")
        logger.info("🔄 Step 2: Convert to multilabel format programmatically")
        
       
        single_label_samples = await self._generate_training_data_async()
        
        if single_label_samples == 0:
            logger.error("Failed to generate single-label data. Cannot proceed with multilabel conversion.")
            return 0
            
        logger.info(f"✅ Generated {single_label_samples} single-label samples")
        
       
        logger.info("🔄 Converting single-label data to multilabel format...")
        multilabel_samples = await self._convert_to_multilabel_format()
        
        logger.info("--- Smart Multilabel Training Data Generation Finished ---")
        logger.info(f"📊 Total multilabel samples created: {multilabel_samples}")
        
        return multilabel_samples

    async def _convert_to_multilabel_format(self) -> int:
        """
        Convert single-label dataset to multilabel by intelligently combining samples.
        
        Strategy:
        - Keep some original single-label samples (50%)
        - Combine 2-3 samples from different classes (30% - 2 labels, 20% - 3 labels)
        - Maintain natural text flow and label relevance
        """
        if not self.dataset_path or not self.dataset_path.exists():
            logger.error("Single-label dataset not found for conversion")
            return 0
            
        try:
            import pandas as pd
            import random
            
            
            df = pd.read_csv(self.dataset_path)
            logger.info(f"📖 Reading {len(df)} single-label samples for conversion")
            
           
            samples_by_class = {}
            for _, row in df.iterrows():
                label = row['label']
                text = row['text'].strip().strip('"')
                
                if label not in samples_by_class:
                    samples_by_class[label] = []
                samples_by_class[label].append(text)
            
         
            for label, texts in samples_by_class.items():
                logger.info(f"📋 Class '{label}': {len(texts)} samples")
            
          
            multilabel_data = []
            
           
            single_label_count = int(len(df) * 0.5)
            logger.info(f"🔹 Keeping {single_label_count} samples as single-label")
            
            for _, row in df.sample(n=single_label_count).iterrows():
                text = row['text'].strip().strip('"')
                label = row['label']
                multilabel_data.append((text, label))
            
           
            two_label_count = int(len(df) * 0.3)
            logger.info(f"🔹 Creating {two_label_count} samples with 2 labels")
            
            for _ in range(two_label_count):
                
                if len(self.class_labels) >= 2:
                    selected_classes = random.sample(self.class_labels, 2)
                    
                   
                    text_parts = []
                    labels = []
                    
                    for class_label in selected_classes:
                        if class_label in samples_by_class and samples_by_class[class_label]:
                            text_parts.append(random.choice(samples_by_class[class_label]))
                            labels.append(class_label)
                    
                    if len(text_parts) >= 2:
                       
                        combined_text = f"{text_parts[0]} {text_parts[1]}"
                        combined_labels = ','.join(labels)
                        multilabel_data.append((combined_text, combined_labels))
            
          
            three_label_count = int(len(df) * 0.2)
            logger.info(f"🔹 Creating {three_label_count} samples with 3 labels")
            
            for _ in range(three_label_count):
                
                if len(self.class_labels) >= 3:
                    selected_classes = random.sample(self.class_labels, 3)
                    
                   
                    text_parts = []
                    labels = []
                    
                    for class_label in selected_classes:
                        if class_label in samples_by_class and samples_by_class[class_label]:
                            text_parts.append(random.choice(samples_by_class[class_label]))
                            labels.append(class_label)
                    
                    if len(text_parts) >= 3:
                       
                        combined_text = f"{text_parts[0]} {text_parts[1]} {text_parts[2]}"
                       
                        if len(combined_text) > 400:
                            combined_text = combined_text[:400] + "..."
                        combined_labels = ','.join(labels)
                        multilabel_data.append((combined_text, combined_labels))
            
            
            multilabel_path = self.dataset_path.parent / f"{self.dataset_path.stem}_multilabel.csv"
            
            logger.info(f"💾 Saving {len(multilabel_data)} multilabel samples to {multilabel_path}")
            
            with open(multilabel_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["text", "label"])
                
                for text, labels in multilabel_data:
                    writer.writerow([f'"{text}"', labels])
            
            
            logger.info(f"✅ Successfully converted to multilabel format")
            logger.info(f"📊 Distribution:")
            
           
            single_label_count = sum(1 for text, labels in multilabel_data if ',' not in labels)
            two_label_count = sum(1 for text, labels in multilabel_data if labels.count(',') == 1)
            three_label_count = sum(1 for text, labels in multilabel_data if labels.count(',') == 2)
            
            logger.info(f"   🔹 Single-label: {single_label_count}")
            logger.info(f"   🔹 Two-label: {two_label_count}")
            logger.info(f"   🔹 Three-label: {three_label_count}")
            
           
            import shutil
            shutil.copy2(str(multilabel_path), str(self.dataset_path))
            
            logger.info(f"📊 Final multilabel dataset saved to: {self.dataset_path}")
            
            return len(multilabel_data)
            
        except Exception as e:
            logger.error(f"Error converting to multilabel format: {e}", exc_info=True)
            return 0

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
        edge_case_model = self.config_model 
        edge_case_sys_prompt = (
            "You are a data generation assistant specializing in creating challenging test cases. "
            "Focus on subtlety and ambiguity as per instructions."
        )

        total_edge_cases_generated_overall = 0
        total_requests_made_edge = 0
        total_failed_requests_edge = 0

        lines_per_api_call_edge_estimate = getattr(
            settings, "EDGE_CASE_LINES_PER_API_CALL", 40
        )  
        required_requests_per_class_edge = max(
            1, round(target_volume_per_class / lines_per_api_call_edge_estimate)
        )

        edge_prompts_for_generation = []

        for class_label_str in self.class_labels:
          
            edge_case_user_prompt = settings.EDGE_CASE_PROMPT_TEMPLATE.format(
                class_label=class_label_str,
                classification_type=self.classification_type,
                problem_description=self.problem_description,
                all_class_labels_str=str(
                    self.class_labels
                ),  
                language=self.language,
            )
            
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
        )  
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
            
          
            edge_duplicate_stats = self._check_dataset_duplicate_rate(self.edge_case_dataset_path)
            self._notify_duplicate_rate(edge_duplicate_stats, "edge case")
            
           
            if self.final_config:
                self.final_config["edge_case_duplicate_stats"] = edge_duplicate_stats
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
               
                self.final_config["performance_analysis"] = {
                    "input_file": str(self.analyze_performance_data_path),
                    "error": self.performance_analysis_result,
                    "llm_analysis": None,
                }
                return

            results_summary = ""
            try:
               
                df = pd.read_csv(self.analyze_performance_data_path)

              
                text_col = "text"
                true_label_col = "label" 
                predicted_label_col = "predicted_label"  

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

                
                limit_samples = 20
                errors_df = df[~df["is_correct"]].head(limit_samples // 2)
                correct_df = df[df["is_correct"]].head(limit_samples - len(errors_df))
                sample_df = pd.concat([errors_df, correct_df]).reset_index(drop=True)

                sample_results_list = []
                for _, row in sample_df.iterrows():
                    text_snippet = str(row[text_col])[:150] + "..."
                    true_lbl = row[true_label_col]
                    pred_lbl = row[predicted_label_col]

                   
                    prob_strs = []
                    for cl_lbl in self.class_labels:
                        prob_col_name = f"{cl_lbl}_prob"  
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
                max_tokens=2000, 
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

        except FileNotFoundError:  
            pass 
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

    async def _process_own_dataset_async(self) -> Tuple[int, int]:
        """Process user's own dataset using DatasetPreparer"""
        logger.info(f"🔄 Processing user's own dataset: {self.use_own_dataset}")
        
        try:
           
            from .prepare_dataset import DatasetPreparer
            
           
            preparer = DatasetPreparer()
            
           
            if not self.dataset_path:
                raise ValueError("Dataset path not properly initialized")
            
           
            processed_path = preparer.process_dataset(
                input_path=self.use_own_dataset,
                output_path=str(self.dataset_path),
                clean_text=True,  
                max_samples=20000,  
                balance_classes=False,
                activation=self.activation  
            )
            
         
            if not processed_path or not Path(processed_path).exists():
                
                processed_path = str(self.dataset_path)
                if not Path(processed_path).exists():
                    raise ValueError(f"Processed dataset not found at {processed_path}")
            
            logger.info(f"📁 Loading processed dataset from: {processed_path}")
            
            
            df = pd.read_csv(processed_path, encoding='utf-8')
            
            
            report_path = processed_path.replace('.csv', '_analysis_report.json')
            
            logger.info(f"📄 Looking for analysis report at: {report_path}")
            
            if Path(report_path).exists():
                with open(report_path, 'r') as f:
                    analysis = json.load(f)
                
                logger.info(f"✅ Analysis report loaded successfully")
                
               
                self.classification_type = analysis.get('task_type', 'multiclass')
                
               
                if 'label' in df.columns:
                    unique_labels = df['label'].unique().tolist()
                    
                   
                    if analysis.get('task_type') == 'multilabel' and analysis.get('unique_individual_labels'):
                        self.class_labels = sorted(analysis['unique_individual_labels'])
                    else:
                        self.class_labels = sorted([str(label) for label in unique_labels])
                    
                    self.num_classes = len(self.class_labels)
                    
                    logger.info(f"✅ Dataset processed successfully using prepare_dataset.py")
                    logger.info(f"📊 Classification type: {self.classification_type}")
                    logger.info(f"🏷️ Found {self.num_classes} classes: {self.class_labels}")
                    logger.info(f"📈 Total samples: {len(df)}")
                    
                  
                    activation_decision = self._smart_activation_detection(df, self.classification_type)
                    final_classification_type = activation_decision["final_type"]
                    
                    logger.info(f"🎯 Activation Decision: {activation_decision['activation']}")
                    logger.info(f"📊 Final Classification Type: {final_classification_type}")
                    logger.info(f"🔍 Reasoning: {activation_decision['reasoning']}")
                    logger.info(f"✅ Confidence: {activation_decision['confidence']}")
                    
                    
                    if activation_decision["confidence"] != "user_specified":
                        alternative = "sigmoid" if activation_decision["activation"] == "softmax" else "softmax"
                        logger.info(f"💡 To use {alternative} instead, add: --activation {alternative}")
                    
                  
                    logger.info("💾 Saving processed dataset...")
                    df.to_csv(processed_path, index=False, encoding='utf-8')
                    logger.info(f"✅ Dataset saved to: {processed_path}")
                    
                    
                    logger.info(f"📊 Final dataset statistics:")
                    logger.info(f"   Shape: {df.shape}")
                    logger.info(f"   Columns: {list(df.columns)}")
                    logger.info(f"   Final classification type: {final_classification_type}")
                    logger.info(f"   Activation function: {activation_decision['activation']}")
                    
                   
                    if 'label' in df.columns:
                        label_counts = df['label'].value_counts()
                        logger.info(f"   Label distribution: {dict(label_counts.head(10))}")
                        
                       
                        if df['label'].astype(str).str.contains(',').any():
                            multilabel_count = df['label'].str.contains(',').sum()
                            logger.info(f"   Multilabel samples: {multilabel_count}/{len(df)} ({multilabel_count/len(df)*100:.1f}%)")
                    
                   
                    self.initial_config = {
                        "summary": f"User-provided dataset for {final_classification_type} classification",
                        "classification_type": final_classification_type,
                        "activation_function": activation_decision["activation"],
                        "activation_reasoning": activation_decision["reasoning"],
                        "activation_confidence": activation_decision["confidence"],
                        "class_labels": self.class_labels,
                        "model_prefix": f"user_dataset_{self.classification_type}_{activation_decision['activation']}",
                        "training_data_volume": len(df),
                        "data_source": "user_provided",
                        "original_dataset_path": self.use_own_dataset,
                        "processed_dataset_path": processed_path,
                        "analysis_report": analysis
                    }
                    self.final_config = self.initial_config.copy()
                    
                   
                    self.classification_type = final_classification_type
                    
                    return len(df), 0  
                else:
                    raise ValueError("Processed dataset does not contain 'label' column")
            else:
                logger.warning(f"Analysis report not found at {report_path}. Using fallback analysis.")
               
                unique_labels = df['label'].unique().tolist()
                
              
                has_comma_separated = df['label'].astype(str).str.contains(',').any()
                if has_comma_separated:
                    base_classification_type = "multilabel"
                 
                    all_individual_labels = set()
                    for label_str in df['label'].astype(str):
                        individual_labels = [l.strip() for l in label_str.split(',') if l.strip()]
                        all_individual_labels.update(individual_labels)
                    unique_individual_labels = sorted(list(all_individual_labels))
                else:
                    base_classification_type = "binary" if len(unique_labels) == 2 else "multiclass"
                    unique_individual_labels = None
                
               
                activation_decision = self._smart_activation_detection(df, base_classification_type)
                final_classification_type = activation_decision["final_type"]
                
                logger.info(f"🎯 Fallback Activation Decision: {activation_decision['activation']}")
                logger.info(f"📊 Final Classification Type: {final_classification_type}")
                logger.info(f"🔍 Reasoning: {activation_decision['reasoning']}")
                
                self.classification_type = final_classification_type
                
               
                if base_classification_type == 'multilabel' and unique_individual_labels:
                    self.class_labels = unique_individual_labels
                else:
                    self.class_labels = sorted([str(label) for label in unique_labels])
                self.num_classes = len(self.class_labels)
                
                analysis = {
                    'task_type': base_classification_type,
                    'text_column': 'text',
                    'label_column': 'label',
                    'confidence': 90,
                    'reasoning': 'Fallback analysis based on processed dataset',
                    'unique_individual_labels': unique_individual_labels
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing user dataset: {e}")
            raise ValueError(f"Failed to process user dataset: {e}")

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
       

        self.final_config["prompt_refinement_history"] = (
            self.prompt_refinement_history
        ) 

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

       

        try:
            with open(self.final_config_path, "w", encoding="utf-8") as f:
                json.dump(self.final_config, f, indent=2, ensure_ascii=False)
            logger.info("Final configuration saved successfully.")
            return True
        except TypeError as e:
           
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
            self.analyze_performance_data_path = None  
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

         
            if (
                not self.model_output_path
                or not self.final_config
                or "model_prefix" not in self.final_config
            ):
                logger.error(
                    "Cannot determine output path for edge case predictions: model_output_path or model_prefix missing."
                )
               
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
            self.analyze_performance_data_path = None  

    async def generate_data_and_train_model_async(self, model_type_selection: str):
        start_time = datetime.now()
        logger.info(
            f"=== Starting Data Generation & Model Training Process for: '{self.problem_description}' using {model_type_selection} ==="
        )

        training_samples_count = 0
        edge_case_samples_count = 0

        try:
          
            if not self.resume_from_config:
                if self.use_own_dataset:
                   
                    logger.info("Using user's own dataset - skipping LLM configuration generation")
                    
                   
                    def create_model_prefix(description: str) -> str:
                        """Create a clean model prefix from problem description"""
                        import re
                       
                        cleaned = re.sub(r'\b(classification|classify|analysis|analyze|of|for|the|a|an)\b', '', description.lower())
                       
                        cleaned = re.sub(r'[^\w\s]', '', cleaned)
                        cleaned = re.sub(r'\s+', '_', cleaned.strip())
                      
                        cleaned = re.sub(r'_+', '_', cleaned)
                        cleaned = cleaned.strip('_')[:30]  
                        return cleaned if cleaned else "user_dataset"
                    
                    model_prefix = create_model_prefix(self.problem_description)
                    
                    
                    self.final_config = {
                        "model_prefix": model_prefix,
                        "summary": f"User-provided dataset: {self.problem_description}",
                        "data_source": "user_provided"
                    }
                    
                   
                    if not self._prepare_output_directory():
                        logger.critical("Failed to prepare output directory. Aborting.")
                        return
                   
                    training_samples_count, edge_case_samples_count = await self._process_own_dataset_async()
                else:
                    
                    if not await self._generate_initial_config_async():
                        logger.critical(
                            "Failed to generate initial configuration. Aborting."
                        )
                        return
                  

                   
                    if (
                        not self._prepare_output_directory()
                    ):  
                        logger.critical("Failed to prepare output directory. Aborting.")
                        return

                  
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

           
            if self.use_own_dataset:
                logger.info("Using own dataset - skipping training data generation")
               
            elif (
                self.skip_data_gen
                and self.dataset_path.exists()
                and self.dataset_path.stat().st_size > 0
            ):
                logger.info("Skipping data generation as per configuration.")
                try:
                    import pandas as pd
                    df = pd.read_csv(self.dataset_path)
                    training_samples_count = len(df)
                    logger.info(
                        f"Found {training_samples_count} samples in existing training data."
                    )
                except Exception as e_read:
                    logger.warning(
                        f"Could not read existing training data file to get count: {e_read}"
                    )
                    training_samples_count = -1
            else:
               
                if (self.classification_type and "multilabel" in self.classification_type) or self.activation == "sigmoid":
                    logger.info("🏷️ Detected multilabel classification - using specialized multilabel data generation")
                    training_samples_count = await self._generate_multilabel_training_data_async()
                else:
                    logger.info("🎯 Using standard single-label data generation")
                    training_samples_count = await self._generate_training_data_async()
                
              
                if training_samples_count == 0:
                    logger.critical(
                        "No training data was generated. Aborting model training."
                    )
                    self._save_final_config() 
                    return

           
            if self.use_own_dataset:
                logger.info("Using own dataset - skipping edge case generation")
                edge_case_samples_count = 0
            elif self.generate_edge_cases:
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
                self.model_output_path
            )

          
            test_path = self.dataset_path  
            if (self.edge_case_dataset_path and 
                self.edge_case_dataset_path.exists() and 
                self.edge_case_dataset_path.stat().st_size > 0):
                test_path = self.edge_case_dataset_path
            
            classifier_runner = TextClassifierRunner(
                train_path=self.dataset_path,
                test_path=test_path,
                data_type=self.classification_type,
                labels=self.class_labels,
                library_type=model_type_selection,
                output_path=model_save_prefix,
            )
            logger.info(
                f"Classifier (model, preprocessors, metadata) saved with prefix: {model_save_prefix}"
            )

            classifier_runner.train_model()

          
            if self.generate_edge_cases and edge_case_samples_count > 0:
                self.run_predictions_on_edge_cases(
                    model_type_selection, classifier_runner
                )
            else:
                logger.info(
                    "Skipping predictions on edge cases as they were not generated or dataset is empty."
                )
                self.analyze_performance_data_path = None  

           
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

           
            self._save_final_config()

        except (ValueError, OpenAIError) as e:
            logger.critical(
                f"Data generation process failed critically: {e}", exc_info=True
            )
            if self.final_config and self.model_output_path: 
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
            
            # Add playground link for users to test their model
            logger.info("")
            logger.info("🎮 Try your trained model right here: https://whitelightning.ai/playground.html")
            logger.info("")

    def _smart_activation_detection(self, df: pd.DataFrame, classification_type: str) -> Dict[str, str]:
        """
        Smart detection of activation function based on dataset analysis and user preferences.
        
        Args:
            df: The dataset to analyze
            classification_type: The detected classification type (binary, multiclass, multilabel)
            
        Returns:
            Dictionary with activation decision and reasoning
        """
      
        if self.activation != "auto":
            return {
                "activation": self.activation,
                "final_type": f"{classification_type}_{self.activation}",
                "confidence": "user_specified",
                "reasoning": f"Activation set by user: --activation {self.activation}"
            }
        
      
        analysis = self._analyze_label_structure(df)
        
     
        if classification_type == "binary":
            return {
                "activation": "sigmoid",
                "final_type": "binary_sigmoid",
                "confidence": "high",
                "reasoning": "Binary classification always uses sigmoid activation"
            }
        
        elif analysis.get("has_comma_separated_labels", False):
            return {
                "activation": "sigmoid", 
                "final_type": "multilabel_sigmoid",
                "confidence": "high",
                "reasoning": "Found comma-separated labels → multi-label classification"
            }
        
        elif analysis.get("avg_labels_per_sample", 1.0) > 1.2:
            return {
                "activation": "sigmoid",
                "final_type": "multilabel_sigmoid", 
                "confidence": "medium",
                "reasoning": f"Average {analysis['avg_labels_per_sample']:.1f} labels per sample → likely multi-label"
            }
        
        else:
            return {
                "activation": "softmax",
                "final_type": f"{classification_type}_softmax",
                "confidence": "high" if classification_type == "multiclass" else "medium",
                "reasoning": f"Found {analysis.get('unique_labels', 'multiple')} classes without multi-label indicators → single-label classification"
            }
    
    def _analyze_label_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the structure of labels in the dataset to help determine the best activation function.
        
        Args:
            df: Dataset with 'label' column
            
        Returns:
            Dictionary with analysis results
        """
        if 'label' not in df.columns:
            return {}
        
        labels = df['label'].astype(str)
        
       
        has_comma_separated = labels.str.contains(',').any()
        
       
        has_pipe_separated = labels.str.contains('\\|').any()
        has_semicolon_separated = labels.str.contains(';').any()
        
       
        if has_comma_separated:
            avg_labels_per_sample = labels.str.split(',').str.len().mean()
        else:
            avg_labels_per_sample = 1.0
        
      
        if has_comma_separated:
          
            all_individual_labels = []
            for label_str in labels:
                individual_labels = [l.strip() for l in label_str.split(',')]
                all_individual_labels.extend(individual_labels)
            unique_labels = len(set(all_individual_labels))
        else:
            unique_labels = labels.nunique()
        
        return {
            "has_comma_separated_labels": has_comma_separated,
            "has_pipe_separated_labels": has_pipe_separated, 
            "has_semicolon_separated_labels": has_semicolon_separated,
            "avg_labels_per_sample": avg_labels_per_sample,
            "unique_labels": unique_labels,
            "total_samples": len(df),
            "has_multi_label_indicators": has_comma_separated or has_pipe_separated or has_semicolon_separated
        }





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
        type=lambda x: (str(x).lower() == "true"),  
        default=settings.DEFAULT_GENERATE_EDGE_CASES,
        help="Generate challenging edge case data (true/false).",
    )
    parser.add_argument(
        "--edge-case-volume-per-class",
        type=int,
        default=settings.DEFAULT_EDGE_CASE_VOLUME
        // 2,  
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
    parser.add_argument(
        "--use-own-dataset",
        type=str,
        default=None,
        help="Path to user's own dataset file (CSV, JSON, or TXT). When provided, skips LLM data generation and uses this dataset instead.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["sigmoid", "softmax", "auto"],
        default="auto",
        help="Activation function to use: 'sigmoid' for multi-label, 'softmax' for single-label, 'auto' for automatic detection.",
    )
    args = parser.parse_args()

    try:
        generator = MulticlassDataGenerator(  
            problem_description=args.problem_description,
            selected_data_gen_model=args.data_gen_model,
            config_model=args.config_llm_model,  
            output_path=args.output_path,
            batch_size=args.batch_size,
            prompt_refinement_cycles=args.refinement_cycles,
            generate_edge_cases=args.generate_edge_cases,
            edge_case_volume_per_class=args.edge_case_volume_per_class,
            language=args.lang,
            config_path=args.config_path,
            skip_data_gen=args.skip_data_gen,
            skip_model_training=args.skip_model_training,
            use_own_dataset=args.use_own_dataset,
            activation=args.activation,
        )
        await generator.generate_data_and_train_model_async(
            model_type_selection=args.model_type
        )

    except ValueError as e: 
        logger.error(
            f"Configuration Error during setup: {e}", exc_info=False
        ) 
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred at the CLI level: {e}", exc_info=True
        )


if __name__ == "__main__":
    try:
        asyncio.run(cli_main())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    except Exception as e: 
        logger.critical(
            f"Application failed unexpectedly at the highest level: {e}", exc_info=True
        )

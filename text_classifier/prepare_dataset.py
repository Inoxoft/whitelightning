#!/usr/bin/env python3
"""
Dataset Preparation Script for Text Classification
Automatically analyzes and converts user datasets to the correct format for model training.
Supports CSV, JSON, and TXT formats with automatic column detection using LLM.
"""

import os
import json
import pandas as pd
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
import openai
from openai import OpenAI
import re


load_dotenv()

class DatasetPreparer:
    def __init__(self):
        self.openrouter_api_key = os.getenv('OPEN_ROUTER_API_KEY')
        if not self.openrouter_api_key:
            raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables. Please set it in your .env file.")
        
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
        )
    
    def clean_text(self, text):
        """Clean text to ensure it's on a single line without wrapping"""
        if pd.isna(text):
            return ""
        
     
        text = str(text)
        
       
        text = re.sub(r'\s+', ' ', text)  
        text = text.replace('\n', ' ')    
        text = text.replace('\r', ' ')    
        text = text.replace('\t', ' ')    
        
       
        text = text.replace('""', '"')
        
       
        text = text.strip()
        
        return text
    
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, str]:
        """Load data from various formats (CSV, JSON, TXT)"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                separators = [',', '\t']
                df = None
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            sep_name = 'tab' if sep == '\t' else 'comma'
                            print(f"✅ Successfully loaded CSV with {encoding} encoding and {sep_name} separator")
                            break
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                    if df is not None:
                        break
                
                if df is None:
                    raise ValueError(f"Could not read CSV file with any supported encoding/separator combination")
                
                data_format = 'csv'
            elif file_extension == '.json':
               
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                   
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                       
                        if all(isinstance(v, list) for v in data.values()):
                            df = pd.DataFrame(data)
                        else:
                           
                            df = pd.DataFrame([data])
                    else:
                        raise ValueError("Unsupported JSON structure")
                    
                except json.JSONDecodeError:
                   
                    print("🔄 Trying JSONL format (JSON Lines)...")
                    data_list = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                try:
                                    data_list.append(json.loads(line))
                                except json.JSONDecodeError as e:
                                    print(f"⚠️ Skipping invalid JSON on line {line_num}: {e}")
                                    continue
                    
                    if not data_list:
                        raise ValueError("No valid JSON objects found in file")
                    
                    df = pd.DataFrame(data_list)
                    print(f"✅ Successfully loaded JSONL with {len(data_list)} records")
                
                data_format = 'json'
            elif file_extension == '.txt':
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                
                data_rows = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    
                    if '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            data_rows.append({'text': parts[0], 'label': parts[1]})
                    
                    elif ',' in line and not line.startswith('"'):
                        parts = line.rsplit(',', 1)  
                        if len(parts) == 2:
                            data_rows.append({'text': parts[0].strip(), 'label': parts[1].strip()})
                    
                    else:
                        data_rows.append({'text': line, 'label': None})
                
                df = pd.DataFrame(data_rows)
                data_format = 'txt'
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df, data_format
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")
    
    def analyze_data_with_llm(self, df: pd.DataFrame, sample_size: int = 10, activation_preference: str = 'auto') -> Dict[str, Any]:
        """Use LLM to analyze data structure with complete label information and activation compatibility"""
        
        
        columns_info = []
        potential_label_cols = []
        potential_text_cols = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'unique_count': unique_count,
                'null_count': df[col].isnull().sum(),
            }
            
           
            if unique_count <= 50 and unique_count >= 2:
               
                value_counts = df[col].value_counts()
                col_info['all_unique_values'] = value_counts.to_dict()
                potential_label_cols.append(col)
                col_info['sample_values'] = list(value_counts.head(10).index)
            else:
                col_info['sample_values'] = df[col].dropna().head(5).tolist()
            
            # For potential text columns
            if df[col].dtype == 'object':
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    avg_length = non_null_values.astype(str).str.len().mean()
                    col_info['avg_text_length'] = avg_length
                    if avg_length > 100:  
                        potential_text_cols.append(col)
                        
                        col_info['text_samples'] = non_null_values.astype(str).str[:200].head(3).tolist()
            
            columns_info.append(col_info)
        
        
        sample_data = []
        if potential_label_cols:
            main_label_col = potential_label_cols[0]
            unique_labels = df[main_label_col].unique()
            
           
            if len(unique_labels) > 1:
                samples_per_class = max(2, sample_size // min(len(unique_labels), 10))
                for label in unique_labels[:10]:  
                    class_samples = df[df[main_label_col] == label].head(samples_per_class)
                    sample_data.append(class_samples)
                sample_df = pd.concat(sample_data, ignore_index=True) if sample_data else df.head(sample_size)
            else:
                sample_df = df.head(sample_size)
        else:
            sample_df = df.head(sample_size)
        
        data_summary = {
            'shape': df.shape,
            'columns': columns_info,
            'potential_label_columns': potential_label_cols,
            'potential_text_columns': potential_text_cols,
            'sample_rows': sample_df.head(8).to_dict('records'),
            'activation_preference': activation_preference
        }
        
        prompt = f"""
Analyze this dataset for text classification with focus on activation function compatibility.

Dataset Summary:
{json.dumps(data_summary, indent=2, default=str)}

User's activation preference: {activation_preference}
- auto: Let you decide the best activation
- sigmoid: User wants multilabel classification (multiple labels per text)
- softmax: User wants multiclass classification (one label per text)

IMPORTANT: Look at the "all_unique_values" field for potential label columns - this shows ALL classes in the dataset.

Your task:
1. Identify text and label columns
2. Determine current data type (binary/multiclass/multilabel)
3. Analyze activation compatibility:
   - If user wants sigmoid: Can this data be converted to multilabel? Is it meaningful?
   - If user wants softmax: Can multilabel data be converted to single-label?
   - If auto: What's the best activation for this data?

For sigmoid/multilabel conversion, consider:
- Are there enough classes (3+) to create meaningful combinations?
- Is the text content rich enough to support multiple labels?
- Do the classes have semantic overlap (e.g., emotions can co-occur)?

Examples of good multilabel candidates:
- Emotions: "happy, excited" - can feel multiple emotions
- Movie genres: "action, comedy" - movies can be multiple genres
- Topic classification: "technology, business" - articles can cover multiple topics

Examples of poor multilabel candidates:
- Medical diagnosis: usually one primary diagnosis
- Binary sentiment: positive/negative are mutually exclusive
- Age groups: person can't be in multiple age brackets

Respond with JSON:
{{
    "text_column": "column_name",
    "label_column": "column_name", 
    "current_task_type": "binary|multiclass|multilabel",
    "activation_analysis": {{
        "sigmoid_feasible": true/false,
        "sigmoid_reasoning": "why sigmoid would/wouldn't work",
        "softmax_feasible": true/false,
        "softmax_reasoning": "why softmax would/wouldn't work",
        "recommended_activation": "sigmoid|softmax",
        "recommended_reasoning": "why this activation is best"
    }},
    "conversion_strategy": {{
        "needs_conversion": true/false,
        "conversion_type": "to_multilabel|to_single_label|none",
        "conversion_feasible": true/false,
        "conversion_reasoning": "explanation of conversion possibility"
    }},
    "label_mapping": {{}},
    "confidence": 0-100,
    "reasoning": "detailed explanation of analysis"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert specializing in text classification datasets and activation function selection. Provide detailed analysis of data structure and activation compatibility. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            
            return self._heuristic_analysis(df, activation_preference)
    
    def _heuristic_analysis(self, df: pd.DataFrame, activation_preference: str = 'auto') -> Dict[str, Any]:
        """Fallback heuristic analysis when LLM fails"""
        text_column = None
        label_column = None
        
        
        text_candidates = []
        label_candidates = []
        
        for col in df.columns:
            col_lower = col.lower()
            sample_values = df[col].dropna().head(10)
            
            
            if any(keyword in col_lower for keyword in ['text', 'content', 'message', 'review', 'comment', 'description']):
                text_candidates.append((col, 3))
            elif df[col].dtype == 'object' and sample_values.astype(str).str.len().mean() > 20:  # Longer strings likely text
                text_candidates.append((col, 2))
            elif df[col].dtype == 'object':
                text_candidates.append((col, 1))
            
           
            if any(keyword in col_lower for keyword in ['label', 'class', 'category', 'target', 'sentiment']):
                label_candidates.append((col, 3))
            elif df[col].nunique() <= 10 and df[col].nunique() >= 2:  # Reasonable number of classes
                label_candidates.append((col, 2))
        
        
        if text_candidates:
            text_column = max(text_candidates, key=lambda x: x[1])[0]
        if label_candidates:
            label_column = max(label_candidates, key=lambda x: x[1])[0]
        
       
        task_type = "binary"
        if label_column and df[label_column].nunique() > 2:
            task_type = "multiclass"
            
           
        sigmoid_feasible = label_column and df[label_column].nunique() >= 3
        softmax_feasible = True
        
        return {
            'text_column': text_column,
            'label_column': label_column,
            'current_task_type': task_type,
            'activation_analysis': {
                'sigmoid_feasible': sigmoid_feasible,
                'sigmoid_reasoning': 'Heuristic: enough classes for multilabel' if sigmoid_feasible else 'Heuristic: too few classes',
                'softmax_feasible': softmax_feasible,
                'softmax_reasoning': 'Heuristic: always possible',
                'recommended_activation': 'sigmoid' if sigmoid_feasible else 'softmax',
                'recommended_reasoning': 'Heuristic analysis based on class count'
            },
            'conversion_strategy': {
                'needs_conversion': False,
                'conversion_type': 'none',
                'conversion_feasible': False,
                'conversion_reasoning': 'Heuristic analysis - no conversion attempted'
            },
            'label_mapping': {},
            'confidence': 60,
            'reasoning': 'Heuristic analysis based on column names and data characteristics'
        }
    
    def _fast_convert_to_multilabel(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Fast programmatic conversion to multilabel format (like generated data).
        Uses the same logic as _convert_to_multilabel_format from agent.py.
        
        Strategy:
        - Keep 50% as single-label samples  
        - Create 30% as two-label combinations
        - Create 20% as three-label combinations
        """
        import random
        
        text_col = analysis['text_column']
        label_col = analysis['label_column']
        unique_labels = list(df[label_col].unique())
        
        print(f"⚡ Fast converting {len(df)} samples with {len(unique_labels)} classes")
        
        # Group samples by class
        samples_by_class = {}
        for _, row in df.iterrows():
            label = row[label_col]
            text = str(row[text_col]).strip().strip('"')
            
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append(text)
        
        # Log class distribution
        for label, texts in samples_by_class.items():
            print(f"📋 Class '{label}': {len(texts)} samples")
        
        multilabel_data = []
        
        # 50% single-label samples
        single_label_count = int(len(df) * 0.5)
        print(f"🔹 Keeping {single_label_count} samples as single-label")
        
        for _, row in df.sample(n=single_label_count, random_state=42).iterrows():
            text = str(row[text_col]).strip().strip('"')
            label = row[label_col]
            multilabel_data.append((text, label))
        
        # 30% two-label samples
        two_label_count = int(len(df) * 0.3)
        print(f"🔹 Creating {two_label_count} samples with 2 labels")
        
        for _ in range(two_label_count):
            if len(unique_labels) >= 2:
                selected_classes = random.sample(unique_labels, 2)
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
        
        # 20% three-label samples  
        three_label_count = int(len(df) * 0.2)
        print(f"🔹 Creating {three_label_count} samples with 3 labels")
        
        for _ in range(three_label_count):
            if len(unique_labels) >= 3:
                selected_classes = random.sample(unique_labels, 3)
                text_parts = []
                labels = []
                
                for class_label in selected_classes:
                    if class_label in samples_by_class and samples_by_class[class_label]:
                        text_parts.append(random.choice(samples_by_class[class_label]))
                        labels.append(class_label)
                
                if len(text_parts) >= 3:
                    combined_text = f"{text_parts[0]} {text_parts[1]} {text_parts[2]}"
                    # Truncate if too long
                    if len(combined_text) > 400:
                        combined_text = combined_text[:400] + "..."
                    combined_labels = ','.join(labels)
                    multilabel_data.append((combined_text, combined_labels))
        
        # Create new DataFrame
        import pandas as pd
        converted_df = pd.DataFrame(multilabel_data, columns=[text_col, label_col])
        
        # Log conversion statistics
        single_label_count_actual = sum(1 for _, labels in multilabel_data if ',' not in str(labels))
        two_label_count_actual = sum(1 for _, labels in multilabel_data if str(labels).count(',') == 1)
        three_label_count_actual = sum(1 for _, labels in multilabel_data if str(labels).count(',') == 2)
        
        print(f"📊 Fast conversion results:")
        print(f"   🔹 Single-label: {single_label_count_actual}")
        print(f"   🔹 Two-label: {two_label_count_actual}")  
        print(f"   🔹 Three-label: {three_label_count_actual}")
        print(f"   📈 Total: {len(multilabel_data)} samples")
        
        # Update analysis with individual labels
        analysis['unique_individual_labels'] = unique_labels
        
        return converted_df

    def convert_to_multilabel(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert single-label dataset to multilabel format using LLM analysis.
        
        Returns:
            DataFrame with multilabel format.
        """
        text_col = analysis['text_column']
        label_col = analysis['label_column']
        unique_labels = df[label_col].unique()
        
        print(f"🔄 Converting {len(df)} samples with {len(unique_labels)} classes to multilabel format")
        
        
        batch_size = 15
        converted_samples = []
        
        try:
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
                
                
                analysis_prompt = f"""Analyze each text and determine which labels from the available categories are most appropriate. Multiple labels can be assigned to the same text if they are relevant.

Available categories: {', '.join(unique_labels)}

For each text, consider:
1. What is the primary topic/category?
2. What secondary topics/categories might also apply?
3. Are there overlapping themes that justify multiple labels?

Be conservative - only assign multiple labels when they are clearly relevant.

Return results in JSON format:
{{
  "0": ["label1", "label2"],
  "1": ["label1"], 
  "2": ["label2", "label3"],
  ...
}}

Only use labels from this list: {', '.join(unique_labels)}

Texts to analyze:
{chr(10).join([f"{idx}: {row[text_col]}" for idx, (_, row) in enumerate(batch_df.iterrows())])}"""
                
                
                response = self.client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=[
                        {"role": "system", "content": "You are a text classification expert. Analyze texts and assign multiple relevant labels when appropriate. Be conservative - only assign multiple labels when they are clearly relevant."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.3  
                )
                
                
                response_content = response.choices[0].message.content
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    print(f"⚠️ No valid JSON in response for batch {i//batch_size + 1}")
                    
                    for _, row in batch_df.iterrows():
                        converted_samples.append({
                            text_col: row[text_col],
                            label_col: row[label_col]   
                        })
                    continue
                
                json_content = response_content[json_start:json_end]
                label_assignments = json.loads(json_content)
                
                
                for idx_str, assigned_labels in label_assignments.items():
                    try:
                        idx = int(idx_str)
                        if 0 <= idx < len(batch_df):
                            original_row = batch_df.iloc[idx]
                            
                           
                            valid_labels = [label for label in assigned_labels if label in unique_labels]
                            if not valid_labels:
                               
                                valid_labels = [original_row[label_col]]
                            
                           
                            multilabel_str = ','.join(valid_labels)
                            
                            converted_samples.append({
                                text_col: original_row[text_col],
                                label_col: multilabel_str
                            })
                        else:
                            print(f"⚠️ Invalid index {idx} in batch {i//batch_size + 1}")
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Error processing assignment {idx_str}: {e}")
                        continue
                
                
                time.sleep(0.5)
            
            
            if converted_samples:
                converted_df = pd.DataFrame(converted_samples)
                
                
                multilabel_count = converted_df[label_col].str.contains(',').sum()
                multilabel_percentage = (multilabel_count / len(converted_df)) * 100
                
                print(f"📊 Conversion results: {multilabel_count}/{len(converted_df)} ({multilabel_percentage:.1f}%) samples with multiple labels")
                
                return converted_df
            else:
                print("❌ No samples were converted - returning original data")
                return df
                
        except Exception as e:
            print(f"❌ Error during multilabel conversion: {e}")
            print("💡 Falling back to original data")
            return df
    
    def convert_to_standard_format(self, df: pd.DataFrame, analysis: Dict[str, Any], clean_text: bool = False, max_samples: int = None, balance_classes: bool = False) -> pd.DataFrame:
        """Convert dataset to standard format based on analysis"""
        
        text_col = analysis['text_column']
        label_col = analysis['label_column']
        task_type = analysis['current_task_type']
        
        if not text_col or text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in dataset")
        
        if not label_col or label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset")
        
        
        result_df = pd.DataFrame()
        result_df['text'] = df[text_col].astype(str)
        
        
        if clean_text:
            print("🧹 Cleaning text entries...")
            result_df['text'] = result_df['text'].apply(self.clean_text)
        
        if task_type == 'binary':
           
            unique_labels = df[label_col].unique()
            if len(unique_labels) > 2:
                print(f"Warning: Found {len(unique_labels)} unique labels for binary task, using first 2")
                unique_labels = unique_labels[:2]
            
            
            label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1} if len(unique_labels) == 2 else {unique_labels[0]: 0}
            result_df['label'] = df[label_col].map(lambda x: label_mapping.get(x, 0))
            
        elif task_type == 'multiclass':
            
            result_df['label'] = df[label_col].astype(str)
            
        elif task_type == 'multilabel':
                
            all_individual_labels = set()
            for label_str in df[label_col].astype(str):
                individual_labels = [l.strip() for l in label_str.split(',') if l.strip()]
                all_individual_labels.update(individual_labels)
            
           
            unique_labels = sorted(list(all_individual_labels))
            
            
            for label in unique_labels:
                result_df[f'label_{label}'] = 0
            
            
            for idx, label_str in enumerate(df[label_col].astype(str)):
                individual_labels = [l.strip() for l in label_str.split(',') if l.strip()]
                for label in individual_labels:
                    if label in unique_labels:
                        result_df.loc[idx, f'label_{label}'] = 1
            
           
            result_df['label'] = df[label_col].astype(str)
            
           
            analysis['unique_individual_labels'] = unique_labels
        
       
        result_df = result_df.dropna(subset=['text', 'label'])
        
       
        if clean_text:
            result_df = result_df[result_df['text'].str.len() > 0]
            print(f"📊 After cleaning text: {result_df.shape}")
        
        
        if balance_classes and task_type == 'binary' and len(result_df['label'].unique()) == 2:
            print("⚖️ Balancing classes...")
            
            
            class_counts = result_df['label'].value_counts()
            min_class_count = min(class_counts.values)
            samples_per_class = min(min_class_count, (max_samples or len(result_df)) // 2)
            
            
            balanced_dfs = []
            for label in result_df['label'].unique():
                class_df = result_df[result_df['label'] == label].sample(n=samples_per_class, random_state=42)
                balanced_dfs.append(class_df)
            
            result_df = pd.concat(balanced_dfs, ignore_index=True)
            result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print(f"⚖️ Balanced dataset shape: {result_df.shape}")
            print(f"📈 Class distribution:")
            print(result_df['label'].value_counts())
        
        
        if max_samples and len(result_df) > max_samples:
            result_df = result_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"📊 Limited to {max_samples} samples: {result_df.shape}")
        
        return result_df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str, analysis: Dict[str, Any], clean_text: bool = False):
        """Save the processed dataset and analysis report"""
        
        
        if clean_text:
            df.to_csv(output_path, index=False, escapechar='\\')
        else:
            df.to_csv(output_path, index=False)
            
        print(f"✅ Dataset saved to: {output_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Task type: {analysis['current_task_type']}")
        print(f"📝 Text column: {analysis['text_column']}")
        print(f"🏷️  Label column: {analysis['label_column']}")
        
        
        if clean_text:
            print("🔍 Validating text format...")
            max_text_length = df['text'].str.len().max()
            avg_text_length = df['text'].str.len().mean()
            
            print(f"📏 Text length stats:")
            print(f"   - Average: {avg_text_length:.0f} characters")
            print(f"   - Maximum: {max_text_length} characters")
            
            
            texts_with_newlines = df['text'].str.contains('\n').sum()
            if texts_with_newlines > 0:
                print(f"⚠️  Warning: {texts_with_newlines} texts still contain newlines")
            else:
                print("✅ All texts are single-line")
        
        
        print("\n📋 Sample data:")
        if clean_text:
            
            for i, row in df.head(3).iterrows():
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                print(f"   Row {i+1}: [{row['label']}] {text_preview}")
        else:
            print(df.head().to_string(index=False))
        
        
        if analysis['current_task_type'] in ['binary', 'multiclass']:
            print(f"\n📈 Label distribution:")
            print(df['label'].value_counts().to_string())
        
        
        report_path = output_path.replace('.csv', '_analysis_report.json')
        analysis['dataset_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'label_distribution': df['label'].value_counts().to_dict() if 'label' in df.columns else {}
        }
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📄 Analysis report saved to: {report_path}")
    
    def process_dataset(self, input_path: str, output_path: str = None, clean_text: bool = False, max_samples: int = None, balance_classes: bool = False, activation: str = 'auto') -> str:
        """Main method to process a dataset"""
        
        if not output_path:
            input_path_obj = Path(input_path)
            suffix = "_clean" if clean_text else "_prepared"
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}{suffix}.csv")
        
        print(f"🔄 Processing dataset: {input_path}")
        
        
        df, data_format = self.load_data(input_path)
        print(f"✅ Loaded {data_format.upper()} data: {df.shape}")
        
        
        print("🧠 Analyzing data structure with LLM...")
        analysis = self.analyze_data_with_llm(df, activation_preference=activation)
        print(f"🎯 Analysis confidence: {analysis['confidence']}%")
        print(f"💭 Reasoning: {analysis['reasoning']}")
        
        
        activation_analysis = analysis.get('activation_analysis', {})
        conversion_strategy = analysis.get('conversion_strategy', {})
        
        print(f"\n📊 LLM Analysis Results:")
        print(f"   📝 Text column: {analysis['text_column']}")
        print(f"   🏷️  Label column: {analysis['label_column']}")
        print(f"   🎯 Current task type: {analysis['current_task_type']}")
        print(f"   🎯 Sigmoid feasible: {activation_analysis.get('sigmoid_feasible', 'unknown')}")
        print(f"   🎯 Softmax feasible: {activation_analysis.get('softmax_feasible', 'unknown')}")
        print(f"   ⚡ Recommended activation: {activation_analysis.get('recommended_activation', 'unknown')}")
        print(f"   💡 Reasoning: {activation_analysis.get('recommended_reasoning', 'N/A')}")
        
        
        if activation == 'sigmoid':
            print("\n⚡ User requested sigmoid activation (multilabel)")
            if activation_analysis.get('sigmoid_feasible', False):
                print(f"✅ LLM confirms sigmoid is feasible: {activation_analysis.get('sigmoid_reasoning', 'N/A')}")
                analysis['final_task_type'] = 'multilabel'
                analysis['final_activation'] = 'sigmoid'
                    
                    # Convert to multilabel format if needed
                if conversion_strategy.get('needs_conversion', False) and conversion_strategy.get('conversion_feasible', False):
                    print("⚡ Converting to multilabel format (fast programmatic)...")
                    df = self._fast_convert_to_multilabel(df, analysis)
                    analysis['final_task_type'] = 'multilabel'
                    analysis['final_activation'] = 'sigmoid'
                    print("✅ Fast conversion completed")
                else:
                    print("ℹ️ No conversion needed - data is already suitable for sigmoid")
            else:
                print(f"⚠️ LLM advises against sigmoid: {activation_analysis.get('sigmoid_reasoning', 'N/A')}")
                print("🔧 However, user explicitly requested sigmoid - forcing multilabel conversion")
                # User explicitly requested sigmoid, so we force it with FAST programmatic conversion
                print("⚡ Using fast programmatic conversion (like generated data)...")
                df = self._fast_convert_to_multilabel(df, analysis)
                analysis['final_task_type'] = 'multilabel'
                analysis['final_activation'] = 'sigmoid'
                print("✅ Fast conversion completed")
        
        elif activation == 'softmax':
            print("\n⚡ User requested softmax activation (multiclass)")
            if activation_analysis.get('softmax_feasible', True):
                print(f"✅ LLM confirms softmax is feasible: {activation_analysis.get('softmax_reasoning', 'N/A')}")
                analysis['final_task_type'] = 'multiclass'
                analysis['final_activation'] = 'softmax'
                
                
                if analysis['current_task_type'] == 'multilabel':
                    print("⚠️ Converting multilabel data to single-label multiclass format")
                    
                    text_col = analysis['text_column']
                    label_col = analysis['label_column']
                    df[label_col] = df[label_col].astype(str).str.split(',').str[0]
                    analysis['final_task_type'] = 'multiclass'
            else:
                print(f"⚠️ LLM advises against softmax: {activation_analysis.get('softmax_reasoning', 'N/A')}")
                print("💡 This shouldn't happen - softmax should always be feasible")
                analysis['final_task_type'] = 'multiclass'
                analysis['final_activation'] = 'softmax'
        
        else:  # activation == 'auto'
            print("\n⚡ Auto-detecting optimal activation function")
            recommended = activation_analysis.get('recommended_activation', 'softmax')
            print(f"🎯 LLM recommends: {recommended}")
            print(f"💡 Reasoning: {activation_analysis.get('recommended_reasoning', 'N/A')}")
            
            
            if analysis['current_task_type'] == 'binary':
                print("🎯 Binary classification detected - using sigmoid activation")
                analysis['final_task_type'] = 'binary'
                analysis['final_activation'] = 'sigmoid'
            elif recommended == 'sigmoid':
                analysis['final_task_type'] = 'multilabel'
                analysis['final_activation'] = 'sigmoid'
            else:
                analysis['final_task_type'] = 'multiclass'
                analysis['final_activation'] = 'softmax'
            
            
            if (analysis['current_task_type'] != 'binary' and 
                recommended == 'sigmoid' and 
                conversion_strategy.get('needs_conversion', False) and 
                conversion_strategy.get('conversion_feasible', False)):
                print("⚡ Converting to multilabel format based on LLM recommendation (fast)...")
                df = self._fast_convert_to_multilabel(df, analysis)
                analysis['final_task_type'] = 'multilabel'
                analysis['final_activation'] = 'sigmoid'
                print("✅ Fast conversion completed")
        
        print(f"\n🎯 Final task type: {analysis.get('final_task_type', analysis['current_task_type'])}")
        print(f"⚡ Final activation: {analysis.get('final_activation', 'softmax')}")
        
        
        analysis['task_type'] = analysis.get('final_task_type', analysis['current_task_type'])
        analysis['activation'] = analysis.get('final_activation', 'softmax')
        
        
        print("\n🔄 Converting to standard format...")
        standard_df = self.convert_to_standard_format(df, analysis, clean_text, max_samples, balance_classes)
        
        
        self.save_dataset(standard_df, output_path, analysis, clean_text)
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for text classification training")
    parser.add_argument('input_path', help='Path to input dataset (CSV, JSON, or TXT)')
    parser.add_argument('-o', '--output', help='Output path for prepared dataset (optional)')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of samples to analyze with LLM')
    parser.add_argument('--no-clean-text', action='store_true', help='Disable text cleaning (cleaning is enabled by default)')
    parser.add_argument('--max-samples', type=int, default=20000, help='Maximum number of samples (default: 20000)')
    parser.add_argument('--balance-classes', action='store_true', help='Balance classes for binary classification')
    parser.add_argument('--activation', choices=['auto', 'sigmoid', 'softmax'], default='auto', 
                       help='Activation function preference: auto (default), sigmoid (multilabel), softmax (multiclass)')
    
    args = parser.parse_args()
    
    
    clean_text = not args.no_clean_text
    
    try:
        preparer = DatasetPreparer()
        output_path = preparer.process_dataset(
            args.input_path, 
            args.output,
            clean_text=clean_text,
            max_samples=args.max_samples,
            balance_classes=args.balance_classes,
            activation=args.activation
        )
        
        print(f"\n🎉 Successfully prepared dataset!")
        print(f"📁 Output: {output_path}")
        print(f"🧹 Text cleaning: {'Enabled' if clean_text else 'Disabled'}")
        print(f"📊 Max samples: {args.max_samples}")
        print(f"⚡ Activation: {args.activation}")
        if args.balance_classes:
            print(f"⚖️ Class balancing: Enabled")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
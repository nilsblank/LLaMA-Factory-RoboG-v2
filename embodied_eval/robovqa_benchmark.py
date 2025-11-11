# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboVQA benchmark implementation (dataset + evaluator merged)."""

import re
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


import tensorflow_datasets as tfds

import numpy as np
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# Hide any physical GPUs from TF
tf.config.set_visible_devices([], "GPU")
# Also ensure logical GPUs list is empty
if tf.config.list_physical_devices("GPU"):
    tf.config.set_visible_devices([], "GPU")


from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from base import BaseBenchmark, BaseModel, Sample
#import BaseBenchmark, BaseModel, Sample


class Task:
    """A class for handling tags and splits in a given task (adapted from RoboVQA)."""
    
    # Tags for default splitting, based on who is talking.
    PRED_STARTS = ['Robot:', 'Thought:', 'Action:']
    NOPRED_STARTS = ['User:', 'System:']
    
    # Tags surrounding all blocks needing to be predicted by the model.
    PRED_START = '<PRED>'
    PRED_END = '</PRED>'
    
    # Tags surrounding only binary answers, typically 'yes' and 'no'.
    PRED_ANSWER_BINARY_START = '<PRED:ANSWER:BINARY>'
    PRED_ANSWER_BINARY_END = '</PRED:ANSWER:BINARY>'
    
    # Tags surrounding all discrete answers coming from a limited set of classes.
    PRED_ANSWER_DISCRETE_START = '<PRED:ANSWER:DISCRETE>'
    PRED_ANSWER_DISCRETE_END = '</PRED:ANSWER:DISCRETE>'
    
    # Tags surrounding things that constitute an answer to a question.
    PRED_ANSWER_START = '<PRED:ANSWER'
    PRED_ANSWER_END = '</PRED:ANSWER'
    
    # Tags that have any sort of short-content value
    PRED_ALL_START = '<PRED:'
    PRED_ALL_END = '</PRED:'
    
    TAGS_RE = r'(</*\w[:\w]*>)'
    
    def __init__(self, text: str):
        """Initialize task with text."""
        self.text = text
    
    def get_splits(self, split_type: str = 'A:') -> List[tuple]:
        """
        Returns a list of (source, target) split pairs.
        
        Args:
            split_type: Type of split to perform
            
        Returns:
            List of (question, answer) tuples
        """
        if split_type == 'A:':
            return self.get_splits_from_tags(start_tags=['A:'], end_tags=[])
        elif split_type == 'answer':
            return self.get_splits_from_tags(
                start_tags=[self.PRED_ANSWER_START], 
                end_tags=[self.PRED_ANSWER_END]
            )
        else:
            raise ValueError(f'Unknown split type: {split_type}')
    
    def get_splits_from_tags(
        self, 
        start_tags: List[str], 
        end_tags: List[str]
    ) -> List[tuple]:
        """Returns a list of (source, target) split pairs given start/end tags."""
        split_positions = []
        position = 0
        
        while position < len(self.text):
            # Find the next start tag given current position.
            start_position = self.find_next_tag(position, start_tags)
            if start_position is None:
                break
            
            # Then find the first end tag after this start tag.
            end_position = self.find_next_tag(start_position, end_tags)
            if end_position is None:
                end_position = len(self.text)
            
            split_positions.append((start_position, end_position))
            position = end_position + 1
        
        # For every split point create a (source, target) pair.
        splits = []
        for start_pos, end_pos in split_positions:
            source = self.text[:start_pos]
            target = self.text[start_pos:end_pos]
            splits.append((source, target))
        
        return splits
    
    def find_next_tag(self, position: int, tags: List[str]) -> Optional[int]:
        """Finds the position of the next tag in the list after the given position."""
        if not tags:
            return None
        
        next_positions = []
        for tag in tags:
            pos = self.text.find(tag, position)
            if pos >= 0:
                next_positions.append(pos)
        
        return min(next_positions) if next_positions else None


class Tasks:
    """A class for parsing tasks from text (adapted from RoboVQA)."""
    
    @staticmethod
    def text_to_dict(text: str) -> Dict[str, Any]:
        """
        Convert task text into structured dict.
        
        Args:
            text: Raw task text from TFRecord
            
        Returns:
            Dict with 'question', 'answer', and 'task_type' keys
        """
        task = Task(text)
        splits = task.get_splits('A:')
        
        if not splits:
            # Fallback: treat entire text as question
            return {
                'question': text.strip(),
                'answer': '',
                'task_type': 'unknown'
            }
        
        # Take first split
        question, answer = splits[0]
        
        # Clean up question (remove tags)
        question = re.sub(Task.TAGS_RE, '', question).strip()
        
        # Extract answer from tags
        answer_match = re.search(
            f'{re.escape(Task.PRED_ANSWER_START)}.*?>(.*?){re.escape(Task.PRED_ANSWER_END)}',
            answer
        )
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = re.sub(Task.TAGS_RE, '', answer).strip()
        
        # Determine task type from tags
        task_type = 'open'
        if Task.PRED_ANSWER_BINARY_START in text:
            task_type = 'binary'
        elif Task.PRED_ANSWER_DISCRETE_START in text:
            task_type = 'discrete'
        
        return {
            'question': question,
            'answer': answer,
            'task_type': task_type
        }


class RoboVQABenchmark(BaseBenchmark):
    """
    RoboVQA benchmark implementation.
    
    Combines dataset loading and evaluation for the RoboVQA benchmark.
    Based on: https://github.com/google-deepmind/robovqa
    
    Features:
    - Loads data from TFRecord files
    - Parses task-based Q&A format
    - Evaluates using BLEU and ROUGE-L metrics
    - Supports task-type-specific metrics
    """
    
    def _validate_config(self) -> None:
        """Validate RoboVQA-specific config."""
        required = ['data_dir']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Set defaults
        if 'split' not in self.config:
            self.config.split = 'validation'
        if 'metrics' not in self.config:
            self.config.metrics = ['bleu', 'rouge-l']
    
    def _load_data(self) -> None:
        """Load RoboVQA dataset from TFRecord files."""
        split = self.config.split
        # tfrecord_pattern = str(self.data_dir / f"{split}*.tfrecord*")
        
        # # Find TFRecord files
        # tfrecord_files = tf.io.gfile.glob(tfrecord_pattern)
        # if not tfrecord_files:
        #     raise FileNotFoundError(
        #         f"No TFRecord files found matching pattern: {tfrecord_pattern}"
        #     )
        
        #print(f"Loading {len(tfrecord_files)} TFRecord file(s) from {split} split...")
        
        # Create dataset
        #raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        episodes = tfds.builder_from_directory(self.data_dir).as_dataset(split=split)
        
        # Parse examples
        max_samples = self.config.get('max_samples', None)
        count = 0
        
        for episode in tqdm(episodes, desc="Loading RoboVQA data"):



            if max_samples and count >= max_samples:
                break

            task_type = episode['task_type'].numpy().decode('utf-8')
            
            #example = tf.train.SequenceExample()
            #example.ParseFromString(raw_record.numpy())
            

            for step in episode['steps']:
                question = step["observation"]["raw_text_question"].numpy().decode('utf-8')
                answer = step["observation"]["raw_text_answer"].numpy().decode('utf-8')
                images = step["observation"]['images'].numpy()
                sample = Sample(
                    question=question,
                    answer=answer,
                    images=images,
                    metadata={'task_type': task_type}
                )
                self.samples.append(sample)
                count += 1
            continue
                


            # # Extract task text
            # task_text = example.features.feature['text'].bytes_list.value[0].decode('utf-8')
            
            # # Parse into Q&A
            # parsed = Tasks.text_to_dict(task_text)
            
            # # Filter by task if specified
            # if 'tasks' in self.config:
            #     # Extract task name from question (usually first few words)
            #     task_name = parsed.get('task_type', 'unknown')
            #     if task_name not in self.config.tasks:
            #         continue
            
            # # Create sample
            # sample = Sample(
            #     question=parsed['question'],
            #     answer=parsed['answer'],
            #     metadata={'task_type': parsed['task_type']}
            # )
            
            # self.samples.append(sample)
            # count += 1
        
        print(f"Loaded {len(self.samples)} samples.")
    
    def preprocess(self, sample: Sample) -> Sample:
        """Preprocess RoboVQA sample (minimal preprocessing needed)."""
        # RoboVQA is text-only, no image preprocessing needed
        return sample
    
    # def generate_prompt(self, sample: Sample, model: BaseModel) -> Union[str, Dict[str, str]]:
    #     """
    #     Generate prompt for RoboVQA question.
        
    #     Args:
    #         sample: Sample containing question
    #         model: Model instance (for model-specific formatting)
            
    #     Returns:
    #         Formatted prompt (string or chat format dict)
    #     """
    #     # Check if model expects chat format (OpenAI-style)
    #     if hasattr(model, 'use_chat_format') and model.use_chat_format:
    #         return {
    #             'role': 'user',
    #             'content': sample.question
    #         }
        
    #     # Default: simple string prompt
    #     return sample.question
    
    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RoboVQA predictions using BLEU and ROUGE-L.
        
        Args:
            predictions: List of model predictions (strings)
            ground_truths: List of ground truth answers (strings)
            metadata: List of metadata dicts (with 'task_type' key)
            
        Returns:
            Dict of evaluation metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) "
                "must have the same length"
            )
        
        # Initialize metrics
        results = {'overall': {}}
        
        # Convert to strings
        predictions = [str(p) for p in predictions]
        ground_truths = [str(gt) for gt in ground_truths]
        
        # Compute BLEU if requested
        if 'bleu' in self.config.metrics:
            bleu = BLEU()
            bleu_score = bleu.corpus_score(predictions, [ground_truths])
            results['overall']['bleu'] = bleu_score.score
        
        # Compute ROUGE-L if requested
        if 'rouge-l' in self.config.metrics or 'rouge_l' in self.config.metrics:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = []
            for pred, gt in zip(predictions, ground_truths):
                score = scorer.score(gt, pred)
                rouge_scores.append(score['rougeL'].fmeasure)
            results['overall']['rouge_l'] = np.mean(rouge_scores)
        
        # Compute task-type-specific metrics if metadata provided
        if metadata:
            task_types = defaultdict(lambda: {'predictions': [], 'ground_truths': []})
            
            for pred, gt, meta in zip(predictions, ground_truths, metadata):
                task_type = meta.get('task_type', 'unknown')
                task_types[task_type]['predictions'].append(pred)
                task_types[task_type]['ground_truths'].append(gt)
            
            # Compute metrics per task type
            for task_type, data in task_types.items():
                results[task_type] = {}
                
                if 'bleu' in self.config.metrics:
                    bleu = BLEU()
                    bleu_score = bleu.corpus_score(
                        data['predictions'], 
                        [data['ground_truths']]
                    )
                    results[task_type]['bleu'] = bleu_score.score
                
                if 'rouge-l' in self.config.metrics or 'rouge_l' in self.config.metrics:
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    rouge_scores = []
                    for pred, gt in zip(data['predictions'], data['ground_truths']):
                        score = scorer.score(gt, pred)
                        rouge_scores.append(score['rougeL'].fmeasure)
                    results[task_type]['rouge_l'] = np.mean(rouge_scores)
                
                results[task_type]['num_samples'] = len(data['predictions'])
        
        # Store results
        self.results = results
        
        return results

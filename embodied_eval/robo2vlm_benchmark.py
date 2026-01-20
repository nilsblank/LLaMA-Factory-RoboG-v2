"""Robo2VLM benchmark implementation.

This file provides an implementation of the evaluation of the VQA-style
dataset Robo2VLM. This benchmark asks questions about robotic images, and
evaluates predicted answers using letter extraction for multiple-choice
questions. The dataset can be doownloaded from
https://huggingface.co/datasets/keplerccc/Robo2VLM-1.
"""

import ast
from collections import defaultdict
from typing import Any, Dict, List, Optional

from datasets import load_dataset
import re
import numpy as np
from PIL import Image
import torch

from base import BaseBenchmark, Sample
from llamafactory.extras.packages import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams

CHOICE_LETTER_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

def choice_answer_clean(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred

def format_multiple_choice_question(question_text:str, choices:List[str]) -> str:
    """Format question with multiple choice options."""
    # Format choices as A, B, C, D, E
    formatted_choices = ""
    
    for i, choice in enumerate(choices):
        if i < len(CHOICE_LETTER_MAP):
            formatted_choices += f" {CHOICE_LETTER_MAP[i]}. {choice}"
    
    return f"{question_text}{formatted_choices}"

class Robo2VLMBenchmark(BaseBenchmark):
    """Robo2VLM benchmark adapater.

    Config keys (with defaults):
    - `data_dir`: path to dataset
    - `split`: dataset split to load (default: `test`)
    - `max_samples`: optional integer to limit loaded samples
    """

    def _validate_config(self) -> None:
        required = ['data_dir', 'split']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_data(self) -> None:
        # Load dataset
        dataset_path = self.config.get('data_dir')
        split = self.config.get('split', 'test')
        dataset = load_dataset(dataset_path, split=split, streaming=True)

        # Limit sample number for debugging
        max_samples = self.config.get('max_samples', None)

        for idx, sample in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break
            
            # Prepare question
            question = format_multiple_choice_question(sample["question"], ast.literal_eval(sample["choices"]))
            question = f"Answer the following multiple choice question by selecting the letter (A, B, C, D, or E). ONLY output the correct option letter, i.e., A, B, C, D, E. {question}"
            
            # Load image
            image = sample["image"]
            if isinstance(image, str):  # Path to image
                # Open with PIL, convert to RGB, and then to NumPy
                image = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, Image.Image):
                # Already a PIL image, just convert to NumPy
                image = np.array(image.convert('RGB'))
            elif not isinstance(image, np.ndarray):
                # If it's a tensor or other array-like, ensure it's a NumPy array
                image = np.array(image)
            
            # Prepare answer
            answer = CHOICE_LETTER_MAP[sample["correct_answer"]].upper()

            self.samples.append(Sample(
                question=question,
                answer=answer,
                images=[image],
                metadata={"id": sample["id"], "tag": sample.get("tag", "unknown")},
            ))

    def preprocess(self, sample: Sample) -> Sample:
        return sample

    def generate_prompt(self, sample: Sample, model: object) -> Any:
        """Generate messages in the same format used by VStar inference.

        Returns a list of messages (system + user). The user content uses a
        list containing a text dict and an image dict when an image (should
        be PIL.Image) is available.
        """

        image_prefix = "<image>" * (len(sample.images) if sample.images is not None else 0)
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": image_prefix + sample.question,
            }, {
                "role": "assistant",  # The models prepare_vllm_inputs_from_chat should remove this message
                "content": str(sample.answer),
            },
        ]
        return conversation

    def postprocess(self, prediction: Any, sample: Sample, model: object) -> Any:
        """
        Use Llama to extract the answer and then normalize into a
        single-letter choice when appropriate.
        """
        # Load Llama if it does not exist
        if not getattr(self, 'answer_extractor', None):
            self._init_answer_extractor()

        # Create prompt
        question = sample.question
        instruction_prompt = """
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

        Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
        Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

        Model response: The correct answer is (B) 8/11.

        Extracted answer: B
        """
        instruction_prompt = instruction_prompt.strip()
        extraction_prompt = f"{instruction_prompt}\n\n{question}\n\n{prediction}\n\nExtracted answer: "
        
        # Set sampling parameters for deterministic extraction
        sampling_params = SamplingParams(
            temperature=0,  # Use deterministic sampling
            max_tokens=5,   # We only need a single letter
        )
        
        # Generate the extraction
        outputs = self.answer_extractor.generate(
            prompts=[extraction_prompt],
            sampling_params=sampling_params,
        )
        
        # Process all outputs
        answer_text = outputs[0].outputs[0].text
        extracted_answer = choice_answer_clean(answer_text).upper()
        
        return extracted_answer

    def evaluate(self, predictions: List[Any], ground_truths: List[Any], metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Evaluate predictions and return a simple accuracy summary.

        - predictions: list of model outputs (strings)
        - ground_truths: list of ground truth answers (strings)
        - metadata: optional list with per-sample metadata (tags etc.)
        """
        n = len(predictions)
        if n == 0:
            return {}

        correct = [1 if (p and p == g) else 0 for p, g in zip(predictions, ground_truths)]
        accuracy = float(np.mean(correct)) if correct else 0.0

        # Per-tag breakdown if metadata provided
        per_tag = defaultdict(lambda: {"correct": 0, "total": 0})
        if metadata:
            for _, _, meta, c in zip(predictions, ground_truths, metadata, correct):
                tag = meta["tag"]
                per_tag[tag]['total'] += 1
                per_tag[tag]['correct'] += c
        per_tag_accuracy = {t: (v['correct'] / v['total'] if v['total'] > 0 else 0.0) for t, v in per_tag.items()}

        results = {
            'overall': {
                'accuracy': accuracy,
                'n': n,
            },
            'per_tag': per_tag_accuracy,
        }

        # Store results
        self.results = results

        return results
    
    def _init_answer_extractor(self, tensor_parallel_size=1, gpu_memory_utilization=0.3):
        torch.cuda.init()  # Apparently, this is needed to prevent an error when vllm tries to init CUDA
        extractor = LLM(
            model="meta-llama/Llama-3.2-3B-Instruct",  # Using smaller model for extraction
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=10240,  # Smaller context as we only need to extract answers
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.answer_extractor = extractor

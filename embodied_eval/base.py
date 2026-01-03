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

"""Base classes for the evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

import datasets


import numpy as np
from omegaconf import DictConfig, OmegaConf


@dataclass
class Sample:
    """Data sample containing multimodal inputs and targets."""
    
    question: str
    answer: str
    images: Optional[List[np.ndarray]] = None
    videos: Optional[List[np.ndarray]] = None
    audios: Optional[List[np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure metadata is a dictionary."""
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmark(ABC):
    """
    Base benchmark class combining dataset loading and evaluation.
    
    Each benchmark handles:
    - Data loading and preprocessing
    - Prompt generation
    - Evaluation metrics computation
    - Result saving
    
    Configured via Hydra/OmegaConf YAML files in configs/benchmarks/
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """
        Initialize benchmark from config.
        
        Args:
            cfg: Can be:
                - Path to YAML config file
                - DictConfig from Hydra
                - Plain dictionary
        """
        if isinstance(cfg, (str, Path)):
            self.config = OmegaConf.load(cfg)
        elif isinstance(cfg, DictConfig):
            self.config = cfg
        elif isinstance(cfg, dict):
            self.config = OmegaConf.create(cfg)
        else:
            raise ValueError("cfg must be a path, DictConfig, or dict")
        
        self.name = self.config.get("name", self.__class__.__name__)
        self.data_dir = Path(self.config.get("data_dir", "."))
        self.samples: List[Sample] = []
        self.results: Dict[str, Any] = {}
        
        self._validate_config()
        
        self._load_data()
        #self.share_gpt_dataset = self.get_sharegpt_format()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate that required config parameters are present."""
        pass
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load and parse the dataset."""
        pass
    
    @abstractmethod
    def preprocess(self, sample: Sample) -> Sample:
        """
        Preprocess a single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Preprocessed sample
        """
        pass
    
    def generate_prompt(self, sample: Sample, model: 'BaseModel') -> Union[str, Dict[str, str]]:
        """
        Generate a prompt for the given sample and model in sharegpt format.

        
        Args:
            sample: Input sample
            model: Model instance (for model-specific prompt formatting)
            
        Returns:
            Formatted prompt string or dict for chat format
        """
        

        image_tag = "<image>"
        
        conversation = []
        user_message = sample.question
        image_prefix = image_tag = "<image>" * (len(sample.images) if sample.images is not None else 0)

        user_message = image_prefix + sample.question
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": sample.answer})
        
        return conversation
    
    def postprocess(self, prediction: str, sample: Sample, model: 'BaseModel') -> Any:
        """
        Postprocess model prediction for a given sample.
        
        Args:
            prediction: Raw model prediction
            sample: Input sample
            model: Model instance (for model-specific postprocessing, such as denormalization)

        Returns:
            Postprocessed prediction
        """
        return prediction
        
    @abstractmethod
    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truths.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers (usually from samples)
            metadata: Optional list of metadata dicts (from samples)
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save evaluation results to file.
        
        Args:
            output_path: Path to save results
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"\n{'='*60}")
        print(f"{self.name} Evaluation Results")
        print(f"{'='*60}")
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        return self.samples[idx]
    
    def __iter__(self):
        """Iterate over samples."""
        return iter(self.samples)
    
    
    def get_sharegpt_path(self, output_path: Union[str, Path], model_name = None) -> Path:       
        """
        Get the path for ShareGPT formatted file.
        
        Args:
            output_path: Base path to save ShareGPT formatted data
            model_name: Optional model name to include in filename
            """
        if model_name is None:
            model_name = "generic_model"
        return Path(output_path) / self.name / f"{model_name}.jsonl"
    
    def share_gpt_exists(self, output_path: Union[str, Path], model_name = None) -> bool:
        """
        Check if ShareGPT formatted file exists.
        
        Args:
            output_path: Path to check for ShareGPT formatted data
        """
        sharegpt_path = self.get_sharegpt_path(output_path, model_name)
        return sharegpt_path.exists()
    
    def get_sharegpt_format(self) -> datasets.Dataset:
        """
        Get the benchmark samples in ShareGPT format as a HuggingFace Dataset.
        """
        
        sharegpt_data = []

        image_tag = "<image>"

        for index,sample in enumerate(self.samples):
            messages = []
            user_message = sample.question
            images = []
            if sample.images:
                for img_idx, img in enumerate(sample.images):
                    images.append(Image.fromarray(img))
                image_prefix  = image_tag * len(sample.images)
                user_message = image_prefix + user_message
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": sample.answer})


            conversation = {"messages": messages, "images": images, "metadata": sample.metadata}

            sharegpt_data.append(conversation)
        
        return datasets.Dataset.from_list(sharegpt_data)
        
        
    
    
    def save_sharegpt_format(self, dataset_info_path, output_path: Union[str, Path], image_output_dir, model_name = None,
                             force_regenerate = False) -> None:
        """
        Save the benchmark samples in ShareGPT format and updates LLamaFactory dataset_info.json.
        
        Args:
            output_path: Path to save the ShareGPT formatted data
            image_output_dir: Directory to save images
        """
        import json

        output_path = self.get_sharegpt_path(output_path, model_name)
        if output_path.exists():
            print(f"ShareGPT file {output_path} already exists. Skipping save.")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sharegpt_data = []

        image_tag = "<image>"
        
        
        
        image_output_dir = Path(image_output_dir) / self.name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        

        for index,sample in enumerate(self.samples):
            #image out dir for index is with 7 digit zero padded
            image_save_dir = image_output_dir / f"{index:07d}"
            image_save_dir.mkdir(parents=True, exist_ok=True)
            messages = []
            user_message = sample.question
            image_paths = []
            if sample.images:
                for img_idx, img in enumerate(sample.images):
                    image_path = image_save_dir / f"image_{img_idx}.png"
                    image_paths.append(str(image_path.relative_to(image_output_dir.parent)))
                    if os.path.exists(image_path):
                        continue
                    im = Image.fromarray(img)
                    im.save(image_path)
                
                image_prefix  = image_tag * len(sample.images)
                user_message = image_prefix + user_message
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": sample.answer})
            
            
            conversation = {"messages": messages, "images": image_paths, "metadata": sample.metadata}

            sharegpt_data.append(conversation)
        with open(output_path, 'w') as f:
            json.dump(sharegpt_data, f, indent=2)

        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        dataset_info[self.name] = {
            "file_name": str(output_path.resolve()),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        logging.info(f"Saved ShareGPT formatted data to {output_path} and updated dataset_info.json.")
        


class BaseModel(ABC):
    """
    Base model class for evaluation.
    
    Handles model-specific requirements like:
    - Bounding box resizing based on model output format
    - Image preprocessing
    - Generation with custom parameters
    - Output parsing
    
    Configured via Hydra/OmegaConf YAML files in configs/models/
    """
    
    def __init__(self, cfg: Union[str, Path, DictConfig, Dict] = None):
        """
        Initialize model from config.
        
        Args:
            cfg: Can be:
                - Path to YAML config file
                - DictConfig from Hydra
                - Plain dictionary
        """
        if isinstance(cfg, (str, Path)):
            self.config = OmegaConf.load(cfg)
        elif isinstance(cfg, DictConfig):
            self.config = cfg
        elif isinstance(cfg, dict):
            self.config = OmegaConf.create(cfg)
        else:
            raise ValueError("cfg must be a path, DictConfig, or dict")
        
        self.name = self.config.get("name", self.__class__.__name__)
        self.model_path = self.config.get("model_name_or_path", "")
        self.generation_kwargs = self.config.get("generation_kwargs", {})
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, Dict[str, str]],
        images: Optional[List[np.ndarray]] = None,
        videos: Optional[List[np.ndarray]] = None,
        audios: Optional[List[np.ndarray]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Text prompt or chat format dict
            images: Optional list of images
            videos: Optional list of videos
            audios: Optional list of audio
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    def process_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        target_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """
        Process bounding box coordinates based on model requirements.
        
        Args:
            bbox: Bounding box coordinates
            original_size: (width, height) of original image
            target_size: (width, height) of resized image
            format: Bbox format ("xyxy", "xywh", "cxcywh")
            
        Returns:
            Processed bounding box
        """
        # Default: no processing
        return bbox
    
    def denormalize_bbox(
        self,
        bbox: List[float],
        original_size: tuple,
        format: str = "xyxy"
    ) -> List[float]:
        """
        Denormalize bounding box coordinates to original image size.
        
        Args:
            bbox: Normalized bounding box coordinates
            original_size: (width, height) of original image
            format: Bbox format ("xyxy", "xywh", "cxcywh")
            
        Returns:
            Denormalized bounding box
        """
        # Default: no processing
        return bbox

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for the model.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Default: no processing
        return image
    
    def parse_output(self, output: str) -> Any:
        """
        Parse model output into structured format.
        
        Args:
            output: Raw model output string
            
        Returns:
            Parsed output (can be string, dict, list, etc.)
        """
        # Default: return as-is
        return output
    
    @abstractmethod
    def prepare_vllm_inputs_from_chat(
        self,
        chat_messages: List[Dict[str, str]],
        sample: 'Sample'
    ) -> Dict[str, Any]:
        """
        Prepare vLLM inputs from chat messages and sample data.
        
        This method integrates with the benchmark's generate_prompt output
        and prepares inputs in the format expected by vLLM.
        
        Args:
            chat_messages: Chat messages from benchmark.generate_prompt()
            sample: Sample object containing images, videos, metadata
            
        Returns:
            Dictionary with:
                - prompt_token_ids: Tokenized prompt (list of int)
                - multi_modal_data: Dict with image/video data (or None)
                - mm_processor_kwargs: Additional processor kwargs (or None)
        """
        pass

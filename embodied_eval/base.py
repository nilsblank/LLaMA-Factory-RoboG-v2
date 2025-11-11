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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
        if sample.images is not None:
            for _ in sample.images:
                user_message += f"\n{image_tag}"
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": sample.answer})
        
        return conversation
    
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

    def save_sharegpt_format(self, output_path: Union[str, Path]) -> None:
        """
        Save the benchmark samples in ShareGPT format and updates LLamaFactory dataset_info.json.
        
        Args:
            output_path: Path to save the ShareGPT formatted data
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sharegpt_data = []

        image_tag = "<image>"


        for sample in self.samples:
            conversation = []
            user_message = sample.question
            if sample.images:
                for img in sample.images:
                    #save the images 
                    user_message += f"\n{image_tag}"
            conversation.append({"role": "user", "content": user_message})
            conversation.append({"role": "assistant", "content": sample.answer})


            sharegpt_data.append({"conversation": conversation, "metadata": sample.metadata})


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

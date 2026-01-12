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

"""
Embodied AI Evaluation Framework

A modular, extensible evaluation framework for embodied AI datasets.
"""

from .base import BaseDataset, BaseEvaluator, BaseModel, Sample, BaseBenchmark
from .models import LlamaFactoryModel, MockModel
from .robovqa_dataset import RoboVQADataset, Task, Tasks
from .robovqa_evaluator import RoboVQAEvaluator
from .motionbench_benchmark import MotionBenchBenchmark
from .foundation_motion_benchmark import FoundationMotionBenchmark

__all__ = [
    # Base classes
    "BaseDataset",
    "BaseModel",
    "BaseEvaluator",
    "BaseBenchmark",
    "Sample",
    # Models
    "MockModel",
    "LlamaFactoryModel",
    # RoboVQA
    "RoboVQADataset",
    "RoboVQAEvaluator",
    "Task",
    "Tasks",
    # MotionBench
    "MotionBenchBenchmark",
    # FoundationMotion
    "FoundationMotionBenchmark",
]

__version__ = "0.1.0"

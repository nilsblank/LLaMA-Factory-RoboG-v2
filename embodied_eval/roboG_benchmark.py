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

import json
import logging
import re
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Optional, Union

from llamafactory.eval.evaluators import BoundingBoxEvaluator,LabelEvaluator, TaskInstructionEvaluator,TemporalAccuracyEvaluator

import numpy as np
import os

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from base import BaseBenchmark, BaseModel, Sample


TASK_OUTPUT_FORMATS = {
    # Tasks expect bbox and parse these formats:
    # 1. r'```json\s*([\s\S]*?)```' where "bbox_2d" is a list or list of lists of a bbox 
    # 2. r'bbox_2d"\s*:\s*\[(.*?)\]' contains the bbox
    # 3. r'<box>(.*?)</box>' contains the bbox
    # With a bbox being in the form [x1, y1, x2, y2]
    "object_detection": """Provide the output strictly as a JSON code block containing a list of bounding boxes under the key 'bbox_2d'. The boxes must be in the format [x1, y1, x2, y2], representing the top-left and bottom-right corners.

**Format Example:**
```json
[
  {
    "bbox_2d": [x1, y1, x2, y2]
  }
]
```""",
    "target_location": """Provide the output strictly as a JSON code block containing a list of bounding boxes under the key 'bbox_2d'. The boxes must be in the format [x1, y1, x2, y2], representing the top-left and bottom-right corners.

**Format Example:**
```json
[
  {
    "bbox_2d": [x1, y1, x2, y2]
  }
]
```""",

    # Task expects a bbox in the same format as above and an object label in these formats:
    # 1. r'```json\s*([\s\S]*?)```' with an object or a list of objects where "label" contains the label
    # 2. r'label"\s*:\s*"(.*?)"' contains the label
    # 3. r'<object>(.*?)</object>' contains the label
    "poc_grounding_only_video": """Provide the output strictly as a JSON code block containing a list of objects. Each object must include the key 'bbox_2d' for the bounding box in the format [x1, y1, x2, y2] (top-left and bottom-right corners) and the key 'label' for the object's name.
    
**Format Example:**
```json
[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "label": "object name"
  }
]
```""",

    # Task expects no specific output format, just text description in one of these forms:
    # 1. r'<task>(.*?)</task>'
    # 2. r'task: (.*)'
    # 3. Just take last line
    "task_detection": "",
    "roboG_reasoning": "",  # Does not evaluate reasoning steps, just final prediction

    # Task expects a bbox, object label, and interaction phases in these formats:
    # 1. r'```json\s*([\s\S]*?)```' with the keys "object", "bbox", and "interaction_phases"
    # 2. r'\{[\s\S]*\}' with the keys "object", "bbox", and "interaction_phases"
    # 3. Fallback: r'<object>(.*?)</object>' and r'<box>(.*?)</box>' to at add object label and bbox
    # Where "interaction_phases" can be a dict with time ranges as keys and interaction phase labels as values or a list of dicts with "phase", "start_time", and "end_time" keys.
    "action_localization": """Provide the output strictly as a JSON code block containing the keys 'object', 'bbox', and 'interaction_phases'. The 'bbox' must be in the format [x1, y1, x2, y2], representing the top-left and bottom-right corners. The 'interaction_phases' must be a dictionary where each key is a time range string (e.g., 'start - end') and the value is the interaction phase label. The phase labels must be strictly limited to the following: 'grasp', 'interact', or 'release'.

**Format Example:**
```json
    {
  "object": "object name",
  "bbox": [x1, y1, x2, y2],
  "interaction_phases": {
    "0.0 - 0.4": "grasp",
    "0.4 - 3.3": "interact",
    "3.3 - 4.0": "release"
  }
}
```""",
    "action_localization_normalized": """Provide the output strictly as a JSON code block containing the keys 'object', 'bbox', and 'interaction_phases'. The 'bbox' must be in the format [x1, y1, x2, y2], representing the top-left and bottom-right corners. The 'interaction_phases' must be a dictionary where each key is a time range string (e.g., 'start - end') and the value is the interaction phase label. All timestamps in the time range must be normalized between 0.0 (start of sequence) and 1.0 (end of sequence). The phase labels must be strictly limited to the following: 'grasp', 'interact', or 'release'.

**Format Example:**
```json
    {
  "object": "object name",
  "bbox": [x1, y1, x2, y2],
  "interaction_phases": {
    "0.0 - 0.25": "grasp",
    "0.25 - 0.8": "interact",
    "0.8 - 1.0": "release"
  }
}
```""",
}


class RoboGBenchmark(BaseBenchmark):
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
        pass
    
    def _load_data(self) -> None:
        
        """Load RoboVQA dataset from TFRecord files."""
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
        
        annotation_files = self.config.annotation_files
        
        all_annotations = []
        for annotation_file in annotation_files:
            #read jsonl lines
            with open(annotation_file, 'r') as f:
                for line in f:
                    all_annotations.append(json.loads(line))
        
        
        for annotation in tqdm(all_annotations, desc="Loading RoboG benchmark"):
            task_type = annotation["meta"]['task_type']
            
            question = annotation["messages"][0]["content"]
            answer = annotation["messages"][1]["content"]
            images = [Image.open(img_path) for img_path in annotation.get("images", [])]
            #convert to array
            images = [np.array(img) for img in images]
            videos = annotation.get("videos", [])
            if "roboG_reasoning" in task_type:
                s = 1
            # Exract video length if available
            timestamp_pattern = r"<(\d+\.\d+) seconds?>"
            matches = re.findall(timestamp_pattern, question)
            if not matches:
                video_length = None
            else:
                video_length = float(matches[-1])
            frame_count = question.count("<image>")
            
            sample = Sample(
                question=question,
                answer=answer,
                images=images,
                videos=videos,
                metadata={'task_type': task_type,
                "image_paths": [str(img_path) for img_path in annotation.get("images", [])],
                "video_paths": [str(vid_path) for vid_path in annotation.get("videos", [])],
                "video_length": video_length,
                "frame_count": frame_count,
                }
            )
            
            
            if task_type == "object_detection":
                
                #parse bbox and plot for sanity check
                #pass
                # boxes = BoundingBoxEvaluator.parse_bbox_from_text(sample.answer)
                    


                # img = Image.open(annotation.get("images", [])[0])

                # #unnorm box (from range 0 - 1000 to img width height)
                # img_width, img_height = img.size
                # boxes = [(int(x * img_width / 1000), int(y * img_height / 1000), int(w * img_width / 1000), int(h * img_height / 1000)) for (x, y, w, h) in boxes]

                # draw = ImageDraw.Draw(img)
                # for box in boxes:
                #     draw.rectangle(box, outline="red", width=2)
                # img.show()
                pass

            

            self.samples.append(sample)

        
        print(f"Loaded {len(self.samples)} samples.")
    
    def preprocess(self, sample: Sample) -> Sample:
        """Preprocess RoboVQA sample (minimal preprocessing needed)."""
        # RoboVQA is text-only, no image preprocessing needed
        return sample
    
    
    
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
        
        if "rynnbrain" in model.config.name.lower():
            frame_counter = 0

            def replacement_logic(match):
                nonlocal frame_counter
                # Format the replacement string as requested
                result = f"<frame {frame_counter}>: "
                frame_counter += 1
                return result

            # Matches tags like <1.23 seconds> or <45.6 seconds>
            timestamp_pattern = r"<\d+\.\d+ seconds?>"
            user_message = re.sub(timestamp_pattern, replacement_logic, user_message)
        elif "qwen3vl" not in model.config.name.lower():
            #remove all <digit.digit seconds> tags
            user_message = re.sub(r"<\d+\.\d+ seconds?>", "", user_message)
        
        if self.config.get("provide_output_format_in_prompt", False):
            task_type = sample.metadata.get("task_type", "unknown")
            if task_type == "action_localization" and self.config.get("normalize_time", False) and "rynnbrain" not in model.config.name.lower():
                output_format_prompt = TASK_OUTPUT_FORMATS.get("action_localization_normalized", "")
            else:
                output_format_prompt = TASK_OUTPUT_FORMATS.get(task_type, "")
            if len(output_format_prompt) > 0:
                conversation.append({"role": "system", "content": output_format_prompt})
                
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": sample.answer})
        
        return conversation

    
    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
        bbox_denormalizer: Optional[callable] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RoboVQA predictions using BLEU and ROUGE-L.
        
        Args:
            predictions: List of model predictions (strings)
            ground_truths: List of ground truth answers (strings)
            metadata: List of metadata dicts (with 'task_type' key)
            bbox_denormalizer: Optional function to denormalize bounding boxes
            
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

        
        # Compute task-type-specific metrics if metadata provided

        task_types = defaultdict(lambda: {'predictions': [], 'ground_truths': [], 'metadata': []})

        for pred, gt, meta in zip(predictions, ground_truths, metadata):
            task_type = meta.get('task_type', 'unknown')
            task_types[task_type]['predictions'].append(pred)
            task_types[task_type]['ground_truths'].append(gt)
            task_types[task_type]['metadata'].append(meta)

        # Compute metrics per task type
        all_task_types = []
        all_data = []
        for task_type, data in task_types.items():
            all_task_types.append(task_type)
            all_data.append(data)

        for task_type, data in task_types.items():
            results[task_type] = {}
            
            #use different eval strategy based on task type
            if "poc_grounding_only_video" in task_type:
                #two types: label and object name
                label_evaluator = LabelEvaluator(ground_truths=data['ground_truths'])
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'], bbox_denormalizer=bbox_denormalizer)
                label_results = label_evaluator.evaluate(data['predictions'])
                bbox_results = bbox_evaluator.evaluate(data['predictions'])
                
                results["Initial Location"] = bbox_results
                results["Interacted Object"] = label_results
            elif "object_detection" in task_type:
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'], bbox_denormalizer=bbox_denormalizer)
                bbox_results = bbox_evaluator.evaluate(data['predictions'])
                results[task_type] = bbox_results

                # #plot boxes on image
                # plt_idx = 81
                # sample = data["ground_truths"][plt_idx]
                # sample_image = data["metadata"][plt_idx]["image_paths"][0]
                # prediction = data["predictions"][plt_idx]

                # #parse bbox
                # boxes = BoundingBoxEvaluator.parse_bbox_from_text(sample)
                # boxes_pred = BoundingBoxEvaluator.parse_bbox_from_text(prediction)


                # img = Image.open(sample_image)

                # #unnorm box (from range 0 - 1000 to img width height)
                # img_width, img_height = img.size
                # boxes = [(int(x * img_width / 1000), int(y * img_height / 1000), int(w * img_width / 1000), int(h * img_height / 1000)) for (x, y, w, h) in boxes]
                # boxes_pred = [(int(x * img_width / 1000), int(y * img_height / 1000), int(w * img_width / 1000), int(h * img_height / 1000)) for (x, y, w, h) in boxes_pred]

                # draw = ImageDraw.Draw(img)
                # for box in boxes:
                #     draw.rectangle(box, outline="red", width=2)
                # #draw predicted boxes
                # for box in boxes_pred:
                #     draw.rectangle(box, outline="blue", width=2)
                # img.save("z.png")

            elif "target_location" in task_type:
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'], bbox_denormalizer=bbox_denormalizer)
                bbox_results = bbox_evaluator.evaluate(data['predictions'])
                results[task_type] = bbox_results
            elif "roboG_reasoning" in task_type or "task_detection" in task_type:
                #format: <task>Put the yellow object on the bottom left burner.</task>. extract task
                instruction_evaluator = TaskInstructionEvaluator(ground_truths=data['ground_truths'])
                #parse tasks from predictions
                instruction_results = instruction_evaluator.evaluate(data['predictions'])
                results[task_type] = instruction_results
            
            elif "action_localization" in task_type:
                time_style = None
                if self.config.get("normalize_time", False):
                    if "rynnbrain" in model_name.lower():
                        time_style = "frames"
                    else:
                        time_style = "normalized"
                action_evaluator = TemporalAccuracyEvaluator(ground_truths=data['ground_truths'], metadata=data['metadata'], time_denormalization_style=time_style)
                action_results = action_evaluator.evaluate(data['predictions'])
                results[task_type] = action_results

            else:
                logging.warning(f"Unknown task type: {task_type}")
                results[task_type] = {"error": "Unknown task type"}
            
            results[task_type]['num_samples'] = len(data['predictions'])


        # Store results
        self.results = results
        
        return results

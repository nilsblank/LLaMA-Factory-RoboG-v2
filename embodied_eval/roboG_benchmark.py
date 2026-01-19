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
#import BaseBenchmark, BaseModel, Sample


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
            sample = Sample(
                question=question,
                answer=answer,
                images=images,
                videos=videos,
                metadata={'task_type': task_type,
                "image_paths": [str(img_path) for img_path in annotation.get("images", [])],
                "video_paths": [str(vid_path) for vid_path in annotation.get("videos", [])]
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
        
        if "qwen3vl" not in model.config.name.lower():
            #remove all <digit.digit seconds> tags
            user_message = re.sub(r"<\d+\.\d+ seconds?>", "", user_message)
        
                
        conversation.append({"role": "user", "content": user_message})
        conversation.append({"role": "assistant", "content": sample.answer})
        
        return conversation

    
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
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'])
                label_results = label_evaluator.evaluate(data['predictions'])
                bbox_results = bbox_evaluator.evaluate(data['predictions'])
                
                results["Initial Location"] = bbox_results
                results["Interacted Object"] = label_results
            elif "object_detection" in task_type:
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'])
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
                bbox_evaluator = BoundingBoxEvaluator(ground_truths=data['ground_truths'])
                bbox_results = bbox_evaluator.evaluate(data['predictions'])
                results[task_type] = bbox_results
            elif "roboG_reasoning" in task_type or "task_detection" in task_type:
                
                if "task_detection" in task_type:
                    continue
                #format: <task>Put the yellow object on the bottom left burner.</task>. extract task
                instruction_evaluator = TaskInstructionEvaluator(ground_truths=data['ground_truths'])
                #parse tasks from predictions
                instruction_results = instruction_evaluator.evaluate(data['predictions'])
                results[task_type] = instruction_results
            
            
            elif "action_localization" in task_type:
                action_evaluator = TemporalAccuracyEvaluator(ground_truths=data['ground_truths'])
                action_results = action_evaluator.evaluate(data['predictions'])
                results[task_type] = action_results

            else:
                logging.warning(f"Unknown task type: {task_type}")
                results[task_type] = {"error": "Unknown task type"}

            
            results[task_type]['num_samples'] = len(data['predictions'])


        # Store results
        self.results = results
        
        return results

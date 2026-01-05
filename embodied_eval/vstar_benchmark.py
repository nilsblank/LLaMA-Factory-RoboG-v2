"""VStar benchmark implementation.

This module provides lightweight helpers for temporal and spatial IoU
calculations and a `VStarBenchmark` class that implements the
`BaseBenchmark` interface from `embodied_eval.base`. The data can be
downloaded from https://huggingface.co/datasets/V-STaR-Bench/V-STaR.

The implementation intentionally keeps evaluation simple and data-driven:
- VQA scoring is a token-overlap similarity mapped to integer score 0-3
- Temporal IoU and spatial IoU computations are provided
- `VStarBenchmark.evaluate` aggregates the benchmark's statistics and
  stores them in the dict `self.results`.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from base import BaseBenchmark, Sample


def calculate_temporal_iou(gt_range: Tuple[float, float], pred_range: Any) -> float:
    """Calculate temporal IoU between two ranges.

    `pred_range` may be a list/tuple of two numbers or a string representation.
    Returns 0.0 on parsing error or invalid inputs.
    """
    if not pred_range:
        return 0.0
    if isinstance(pred_range, str):
        try:
            pred_range = ast.literal_eval(pred_range)
        except (ValueError, SyntaxError):
            return 0.0
    if not isinstance(pred_range, (list, tuple)) or len(pred_range) != 2:
        return 0.0
    try:
        pred_start, pred_end = float(pred_range[0]), float(pred_range[1])
        gt_start, gt_end = float(gt_range[0]), float(gt_range[1])
    except Exception:
        return 0.0
    intersection = max(0.0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_iou(gt_bbox: Dict[str, float], pred_bbox: List[float]) -> float:
    """Compute 2D IoU between GT bbox dict and predicted [xmin,ymin,xmax,ymax]."""
    if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
        return 0.0
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = (
        gt_bbox["xmin"], gt_bbox["ymin"], gt_bbox["xmax"], gt_bbox["ymax"]
    )
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox
    x1 = max(gt_xmin, pred_xmin)
    y1 = max(gt_ymin, pred_ymin)
    x2 = min(gt_xmax, pred_xmax)
    y2 = min(gt_ymax, pred_ymax)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    gt_area = max(0.0, (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin))
    pred_area = max(0.0, (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin))
    union = gt_area + pred_area - intersection
    return intersection / union if union > 0 else 0.0


def calculate_bbox_iou(gt_bbox: Dict[str, float], pred_bboxes: Any) -> float:
    """Support single bbox or list of bboxes; return max IoU."""
    try:
        if not pred_bboxes:
            return 0.0
        if isinstance(pred_bboxes[0], (int, float)) and len(pred_bboxes) == 4:
            pred_bboxes = [pred_bboxes]
        return max(compute_iou(gt_bbox, pred) for pred in pred_bboxes)
    except Exception:
        return 0.0


def calculate_spatial_metrics(gt_bboxes: List[Dict[str, Any]], pred_bboxes: Dict[str, Any]) -> Tuple[List[float], float]:
    """Compute APs at thresholds and mean IoU.

    `gt_bboxes` is a list of dicts with keys xmin,ymin,xmax,ymax,timestamp.
    `pred_bboxes` is a mapping frame_id->bbox (or list of bboxes).
    """
    if not pred_bboxes:
        return [0.0] * 5, 0.0
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious: List[float] = []
    for gt in gt_bboxes:
        frame_id = str(gt["timestamp"])
        if frame_id in pred_bboxes:
            pred = pred_bboxes[frame_id]
            iou = calculate_bbox_iou(gt, pred)
            ious.append(iou)
        else:
            ious.append(0.0)
    mIoU = float(np.mean(ious)) if ious else 0.0
    aps = []
    for thr in iou_thresholds:
        scores = [1.0 if i >= thr else 0.0 for i in ious]
        aps.append(float(np.mean(scores)) if scores else 0.0)
    return aps, mIoU


def calculate_spatial_random(gt_bboxes: List[Dict[str, Any]], w: float, h: float) -> Tuple[List[float], float]:
    pred_bbox = [0.0, 0.0, float(w), float(h)]
    ious = [calculate_bbox_iou(gt, pred_bbox) for gt in gt_bboxes]
    mIoU = float(np.mean(ious)) if ious else 0.0
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    aps = []
    for thr in iou_thresholds:
        scores = [1.0 if i >= thr else 0.0 for i in ious]
        aps.append(float(np.mean(scores)) if scores else 0.0)
    return aps, mIoU


def calculate_benchmark_metrics(results_df):
    """
    Calculates metrics where VQA is a global metric and 
    Chain 1/2 contain specific temporal/spatial metrics.
    """
    if results_df.empty:
        return {}

    total_n = results_df['sample_id'].nunique()
    results_dict = {}

    # VQA Metrics
    vqa_df = results_df[results_df['task_type'] == "vqa"]
    acc_vqa = len(vqa_df['score'] >= 2) / total_n if not vqa_df.empty else 0
    vqa_scores = vqa_df[vqa_df['metric'] == 'qwen_score']['score']
    avg_all_vqa = vqa_scores.mean() if not vqa_scores.empty else 0
    valid_vqa_scores = vqa_df[(vqa_df['metric'] == 'qwen_score') & (vqa_df['valid'] == True)]['score']
    avg_valid_vqa = valid_vqa_scores.mean() if not valid_vqa_scores.empty else 0

    results_dict["vqa"] = {
        "avg_score": avg_all_vqa,
        "avg_valid_score": avg_valid_vqa,
        "accuracy": acc_vqa,
    }

    # Successful VQA sample IDs (used for Combined Success Ratios later)
    vqa_success_ids = set(vqa_df[(vqa_df['metric'] == 'qwen_score') & (vqa_df['score'] >= 2)]['sample_id'])

    # Metrics per chain
    def get_chain_metrics(suffix=""):
        t_task = f"Temporal{suffix}"
        s_task = f"Spatial{suffix}"
        
        # Temporal Metrics
        t_df = results_df[(results_df['task_type'] == t_task) & (results_df['metric'] == 'temporal_iou')]
        t_scores = t_df['score']
        mean_iou = t_scores.mean() if not t_scores.empty else 0
        
        # Spatial Metrics
        s_df = results_df[results_df['task_type'] == s_task]
        mean_miou = s_df[s_df['metric'] == 'spatial_mIoU']['score'].mean() or 0

        # Combined Success Ratios
        t_success_ids = set(t_df[t_df['score'] >= 0.3]['sample_id'])
        s_success_ids = set(s_df[(s_df['metric'] == 'spatial_mIoU') & (s_df['score'] >= 0.1)]['sample_id'])
        combined = {
            "vqa_temp": len(vqa_success_ids & t_success_ids) / total_n,
            "vqa_spat": len(vqa_success_ids & s_success_ids) / total_n,
            "temp_spat": len(t_success_ids & s_success_ids) / total_n,
            "vqa_temp_spat": len(vqa_success_ids & t_success_ids & s_success_ids) / total_n,
        }

        # AM and LGM
        am = (acc_vqa + mean_iou + mean_miou) / 3
        eps = 1e-6
        lgm_vals = [max(eps, 1 - acc_vqa), max(eps, 1 - mean_iou), max(eps, 1 - mean_miou)]
        lgm = -sum(math.log(v) for v in lgm_vals) / 3

        return {
            "temporal_mean_iou": mean_iou,
            "temporal_r1@iou=0.3": (t_scores >= 0.3).mean() if not t_scores.empty else 0,
            "temporal_r1@iou=0.5": (t_scores >= 0.5).mean() if not t_scores.empty else 0,
            "temporal_r1@iou=0.7": (t_scores >= 0.7).mean() if not t_scores.empty else 0,
            "spatial_mean_miou": mean_miou,
            "spatial_map@0.1": s_df[s_df['metric'] == f'spatial_AP@0.1']['score'].mean() or 0,
            "spatial_map@0.3": s_df[s_df['metric'] == f'spatial_AP@0.3']['score'].mean() or 0,
            "spatial_map@0.5": s_df[s_df['metric'] == f'spatial_AP@0.5']['score'].mean() or 0,
            "spatial_map@0.7": s_df[s_df['metric'] == f'spatial_AP@0.7']['score'].mean() or 0,
            "spatial_map@0.9": s_df[s_df['metric'] == f'spatial_AP@0.9']['score'].mean() or 0,
            "combined_scores": combined,
            "am": am,
            "lgm": lgm
        }

    # Assemble Final Dictionary
    results_dict["chain_1"] = get_chain_metrics(suffix="")
    results_dict["chain_2"] = get_chain_metrics(suffix="_2")
    results_dict["mAM"] = (results_dict["chain_1"]["am"] + results_dict["chain_2"]["am"]) / 2
    results_dict["mLGM"] = (results_dict["chain_1"]["lgm"] + results_dict["chain_2"]["lgm"]) / 2

    return results_dict


class VStarBenchmark(BaseBenchmark):
    """Simple VStar benchmark adapter.

    Loads a JSON annotation file and videos from a directory. Exposes a minimal `evaluate`
    implementation that aggregates VQA, temporal, and spatial metrics.
    """

    def _validate_config(self) -> None:
        """Validate V-STaR-specific config."""
        required = ['data_dir']
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_data(self) -> None:
        """Load JSON annotations and populate `self.samples`.
        The full item is stored in sample.metadata and videos are not loaded yet, only their path is added.
        """
        annotation_file = Path(self.config.get("data_dir")) / "V_STaR_test.json"
        if not annotation_file.exists():
            raise ValueError(f"The annotation file was not found: {annotation_file}")

        with open(annotation_file, "r") as f:
            items = json.load(f)

        # Limit sample number for debugging
        max_samples = self.config.get('max_samples', None)

        # Each item may contain multiple question types, therefore split into multiple Samples
        for original_id, item in enumerate(items):
            if max_samples is not None and original_id >= max_samples:
                break

            # Attach the original item as metadata base
            base_meta = dict(item)

            # Locate video
            vid = item.get("vid")
            target_filename = f"{vid}.mp4"
            video_path = None
            for root, _, files in (Path(self.config.get("data_dir")) / "vstar_videos").walk():
                if target_filename in files:
                    video_path = root / target_filename
            if not video_path:
                raise ValueError(f"Video file not found for vid {vid}")

            # Load bboxes
            bboxes = [[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]] for bbox in item["bboxes"]]

            # VQA
            if item.get("question") is not None:
                s = Sample(
                    question=f"Answer the question about the video: {item['question']} \n (If the answer is a person, you don't need to identify the person.)",
                    answer=item.get("answer", ""),
                    videos=[video_path],
                    metadata={**base_meta, "sample_id": original_id, "task_type": "vqa", question: item['question']},
                )
                self.samples.append(s)

            # Temporal (Chain 1)
            if item.get("temporal_question"):
                video_length = round(item['frame_count']/item['fps'], 2)
                temporal_question = item['temporal_question']
                question = f"This video is {video_length} seconds long. Answer the question about the video: {temporal_question} \n Output the start and end moment timestamps."
                s = Sample(
                    question=question,
                    answer=item["timestamps"],
                    videos=[video_path],
                    metadata={**base_meta, "sample_id": original_id, "task_type": "temporal"},
                )
                self.samples.append(s)

            # Spatial (Chain 1)
            if item.get("spatial_question"):
                st, et = math.ceil(item["timestamps"][0]), math.floor(item["timestamps"][1])
                time_range = list(range(st, et + 1))
                spatial_question = item["spatial_question"]
                question = f"""Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format. \n
                            For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format.
                            Example output: {{"{time_range[0]}": [x1, y1, x2, y2],...}}
                            """
                s = Sample(
                    question=question,
                    answer=item["bboxes"],
                    videos=[video_path],
                    metadata={**base_meta, "sample_id": original_id, "task_type": "spatial"},
                )
                self.samples.append(s)

            # Temporal (Chain 2)
            if item.get("temporal_question"):
                video_length = round(item['frame_count']/item['fps'], 2)
                temporal_question = item["temporal_question"]
                w, h = item['width'], item['height']
                question = f"This video is {video_length} seconds long with a resolution of {w}x{h} (width x height). Answer the question about the video: {temporal_question} \n There are {len(bboxes)} bounding boxes of the key object related to the question in the video without knowing the time, which are:{bboxes}. Output the start and end moment timestamps."  # Note tat the bboxes are not normalized but the original image size is given
                s = Sample(
                    question=question,
                    answer=item["timestamps"],
                    videos=[video_path],
                    metadata={**base_meta, "sample_id": original_id, "task_type": "temporal_2"},
                )

            # Spatial (Chain 2)
            if item.get("spatial_question_2"):
                spatial_question = item['spatial_question_2']
                question = f"""Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format. \n
                            For each whole second that may related to the question, output a series of bounding boxes of the object in JSON format. You only need to output {len(bboxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question.
                            The keys should be the whole seconds (as strings), and the values should be the bounding box in [x0,y0,x1,y1] format. 
                            \n Example output:
                            {{"0": [x0,y0,x1,y1], "1":..., ..., "{len(bboxes)}":...}} (if the frames at 0~{len(bboxes)} second are related to the questions)
                            """
                s = Sample(
                    question=question,
                    answer=item["bboxes"],
                    videos=[video_path],
                    metadata={**base_meta, "sample_id": original_id, "task_type": "spatial_2"},
                )
                self.samples.append(s)

    def preprocess(self, sample: Sample) -> Sample:
        return sample

    def generate_prompt(self, sample: Sample, model: object) -> List[Dict[str, Any]]:
        """Generate messages in the same format used by VStar inference.

        Returns a list of messages (system + user). The user content uses a
        list containing a text dict and a video dict when a video path is
        available.
        """
        
        video_prefix = "<video>" * (len(sample.videos) if sample.videos is not None else 0)
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": video_prefix + sample.question,
            }, {
                "role": "assistant",  # The models prepare_vllm_inputs_from_chat should remove this message
                "content": sample.answer,
            },
        ]
        return conversation

    def postprocess(self, prediction, sample, model):
        """Extract timestamps or denormalized bounding boxes from the model prediction."""
        # Extract timestamps
        if sample.metadata.get("task_type") == "temporal" or sample.metadata.get("task_type") == "temporal_2":
            match = re.findall(r"\b\d+(?:\.\d+)?\b", prediction)
            processed_pred = [float(match[0]), float(match[1])] if len(match) == 2 else []
        # Extact bounding boxes
        elif sample.metadata.get("task_type") == "spatial" or sample.metadata.get("task_type") == "spatial_2":
            # Match Markdown JSON
            match = re.search(r'```json\s*\n(\[.*?\]|\{.*?\})\s*\n```', prediction, re.DOTALL)
            # If there is no Markdown wrapper, then try to match the JSON format directly
            if not match:
                match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', prediction, re.DOTALL)
            # Return None if still no match found
            if not match:
                return None

            # Parse bounding boxes
            bounding_boxes_str = match.group(1).strip()
            # Replace single quotes with double quotes to conform to the JSON specification
            bounding_boxes_str = bounding_boxes_str.replace("'", '"')
            try:
                # Convert strings to dictionary or list format
                bounding_boxes = json.loads(bounding_boxes_str)
                # If it's a list and contains a dictionary inside, expand it to a single dictionary
                if isinstance(bounding_boxes, list) and all(isinstance(item, dict) for item in bounding_boxes):
                    combined_dict = {}
                    for item in bounding_boxes:
                        combined_dict.update(item)
                    bounding_boxes = combined_dict
                # Determine if the extracted JSON is a dictionary or a list.
                if isinstance(bounding_boxes, list):
                    processed_pred = {str(box[0]): box[1] for box in bounding_boxes}
                elif isinstance(bounding_boxes, dict):
                    processed_pred = {key: value for key, value in bounding_boxes.items()}
            except Exception as e:
                # Try to fix format issues
                # Counting left and right brackets
                open_square = bounding_boxes_str.count('[')
                close_square = bounding_boxes_str.count(']')
                open_curly = bounding_boxes_str.count('{')
                close_curly = bounding_boxes_str.count('}')

                # Complete the square brackets
                if open_square > close_square:
                    bounding_boxes_str += ']' * (open_square - close_square)
                elif close_square > open_square:
                    bounding_boxes_str = '[' * (close_square - open_square) + bounding_boxes_str

                # Complete the curly brackets
                if open_curly > close_curly:
                    bounding_boxes_str += '}' * (open_curly - close_curly)
                elif close_curly > open_curly:
                    bounding_boxes_str = '{' * (close_curly - open_curly) + bounding_boxes_str

                try:
                    bounding_boxes = json.loads(bounding_boxes_str)
                    if isinstance(bounding_boxes, list):
                        processed_pred = [box for box in bounding_boxes]
                    elif isinstance(bounding_boxes, dict):
                        processed_pred = {key: value for key, value in bounding_boxes.items()}
                except Exception as e:
                    print(f"Failed after fixing: {e}\nExtracted JSON: {bounding_boxes_str}")
                    return None

            # Denormalize bounding boxes
            original_size = sample.metadata['width'], sample.metadata['height']
            for key, bbox in processed_pred.items():
                if (isinstance(bbox, list) or isinstance(bbox, tuple)) and len(bbox) == 4:
                    processed_pred[key] = model.denormalize_bbox(bbox, original_size, format="xyxy")
        else:
            processed_pred = prediction
        return processed_pred

    def evaluate(
        self,
        predictions: List[Any],
        ground_truths: List[Any],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Aggregate metrics for provided predictions/ground truths.

        - `predictions` and `ground_truths` are lists of strings.
        - `metadata` (optional) may contain temporal/spatial GT and preds.
        """
        n = len(predictions)
        if n == 0:
            return {}
        columns = ["sample_id", "task_type", "domain", "duration", "valid", "metric", "score"]
        rows = []
        
        for pred, gt, meta in zip(predictions, ground_truths, metadata or []):
            video_length = round(meta['frame_count']/meta['fps'], 2)
            if video_length < 60:
                duration = "Short"
            elif 60 <= video_length < 180:
                duration = "Medium"
            else:
                duration = "Long"
            domain = meta.get("domain", "unknown")
            sample_id = meta["sample_id"]
        
            # VQA
            if meta.get("task_type") == "vqa":
                score = self._qwen2_5_score(meta["question"], gt, pred)
                if score == -1:
                    valid = False
                    score = 0
                else:
                    valid = True
                rows.append([sample_id, "VQA", domain, duration, valid, "qwen_score", score])

            # Temporal 1
            elif meta.get("task_type") == "temporal":
                score = calculate_temporal_iou(gt, pred)
                rows.append([sample_id, "Temporal", domain, duration, True, "temporal_iou", score])

            # Spatial 1
            elif meta.get("task_type") == "spatial":
                aps, mIoU = calculate_spatial_metrics(gt, pred)
                for ap, threshold in zip(aps, [0.1, 0.3, 0.5, 0.7, 0.9]):
                    rows.append(["Spatial", domain, duration, True, f"spatial_AP@{threshold}", ap])
                rows.append([sample_id, "Spatial", domain, duration, True, "spatial_mIoU", mIoU])
            
            # Temporal 2
            elif meta.get("task_type") == "temporal_2":
                score = calculate_temporal_iou(gt, pred)
                rows.append([sample_id, "Temporal_2", domain, duration, True, "temporal_iou", score])
            
            # Spatial 2
            elif meta.get("task_type") == "spatial_2":
                aps, mIoU = calculate_spatial_metrics(gt, pred)
                for ap, threshold in zip(aps, [0.1, 0.3, 0.5, 0.7, 0.9]):
                    rows.append(["Spatial_2", domain, duration, True, f"spatial_AP@{threshold}", ap])
                rows.append([sample_id, "Spatial_2", domain, duration, True, "spatial_mIoU", mIoU])

        # Aggregate results
        results_df = pd.DataFrame(rows, columns=columns)

        # Calculate overall, per-duration, and per-domain stats
        overall_stats = calculate_benchmark_metrics(results_df)
        duration_stats = {}
        for duration in results_df["duration"].unique():
            duration_stats[duration] = calculate_benchmark_metrics(results_df[results_df["duration"] == duration])
        domain_stats = {}
        for domain in results_df["domain"].unique():
            domain_stats[domain] = calculate_benchmark_metrics(results_df[results_df["domain"] == domain])
        results = {
            "overall": overall_stats,
            "per_duration": duration_stats,
            "per_domain": domain_stats,
        }

        # Store results
        self.results = results

        return results
    
    def _init_qwen2_5(self):
        model_name = "Qwen/Qwen2.5-72B-Instruct"
        self.eval_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.eval_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _qwen2_5_score(self, question, gt, candidate):
        # Load Qwen if it does not exist
        if not self.eval_model:
            self._init_qwen2_5()

        # Let Qwen evaluate the answer
        system_prompt = """
        As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
        The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
        Your assessment should range from 0 to 3, \
        based solely on the semantic similarity between the groundtruth and the candidate answer, \
        disregarding any grammatical differences.
        A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
        A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
        A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
        Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
        Your response should be a single integer from 0, 1, 2, or 3.
        """
        user_prompt=f"Question: {question}\nGroundtruth answer: {gt}\nCandidate answer: {candidate}\nYour response: "
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.eval_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.eval_tokenizer([text], return_tensors="pt").to(self.eval_model.device)

        generated_ids = self.eval_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        score = self.eval_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = -1
        return score

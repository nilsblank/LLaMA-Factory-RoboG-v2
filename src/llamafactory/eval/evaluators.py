import ast
import json
import os
from openai import OpenAI
import torch
import numpy as np
import re
from PIL import Image
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion
from typing import List, Dict, Any, Union, Optional, Tuple



from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from llamafactory.extras import logging

try:
    from sentence_transformers import SentenceTransformer, util
    use_sentence_transformers = True
except ImportError:
    use_sentence_transformers = False
    print("Warning: sentence-transformers not installed, falling back to simple string similarity for LabelEvaluator. Install with `pip install sentence-transformers` for better results.")
    
from difflib import SequenceMatcher

try:
    from joblib import Parallel, delayed
    use_joblib = True
except ImportError:
    use_joblib = False
    print("Warning: joblib not installed, falling back to sequential processing for TaskInstructionEvaluator. Install with `pip install joblib` for parallel processing.")



import os 
SECRET_KEY = os.getenv("OPENAI_API_KEY", "test")



def query_parallel_gpt(prompt, temperature=0.7, model="gpt-4o-mini"):


    base_url = "http://localhost:8000/v1"
    base_url = None
    #key = "test"

    
    if "system" in prompt:
        messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}]
    else:
        messages = [{"role": "user", "content": prompt["user"]}]
        
    if base_url is None:
        client = OpenAI(api_key=key)
    else:
        client = OpenAI(api_key=key,
                        base_url=base_url)

    n_retries = 2
    success = False
    while n_retries > 0 and not success:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                #extra_body={'repetition_penalty': 1.07},
            )
            success = True
        except Exception as e:
            logging.error(f"Error in LLM query: {e}")
            n_retries -= 1
            if n_retries == 0:
                return ["Error in response from LLM"]

    llm_response = chat_completion.choices[0].message.content

    return llm_response


class BaseEvaluator:
    """Base evaluator class that defines the common interface."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the evaluator with a name."""
        self.name = name
    
    def load_data(self):
        """Load necessary data for evaluation."""
        pass
    
    @staticmethod
    def extract_text(prediction: Union[str, Dict[str, Any]]) -> str:
        """
        Extract text from prediction, which can be a string or dict with 'text' key.
        
        Args:
            prediction: Prediction object or string
            
        Returns:
            Extracted text
        """
        if isinstance(prediction, dict) and "text" in prediction:
            return prediction["text"]
        return prediction  # Assume it's already text
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate the predictions.
        
        Args:
            predictions: List of prediction strings or dicts
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class BoundingBoxEvaluator(BaseEvaluator):
    """Evaluator for bounding box predictions."""
    
    def __init__(
        self, 
        ground_truth_file: Optional[str] = None, 
        ground_truths: Optional[List[Dict[str, Any]]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize the bounding box evaluator.
        
        Args:
            ground_truth_file: Path to ground truth file (one JSON per line)
            ground_truths: Ground truth data (alternative to file)
            name: Name for this evaluator
        """
        super().__init__(name=name or "BoundingBoxMAP")
        self.ground_truth_file = ground_truth_file
        self.ground_truths = ground_truths
        
        self.load_data()

    
    def load_data(self):
        """Load ground truth data if not provided directly."""
        if self.ground_truths is None and self.ground_truth_file:
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truths = [json.loads(line) for line in f]
        
        #Parse gt boxes from ground truths
        parsed = []
        for gt in self.ground_truths:
            gt_boxes = self.parse_bbox_from_text(gt)
            parsed.append(gt_boxes)
        self.ground_truths = parsed
        
            
    
    @staticmethod
    def parse_bbox_from_text(text: str) -> List[List[float]]:
        """
        Parse bounding box coordinates from prediction text Qwen Format.
        Expected format: "'```json\n[\n\t{"label": "orange", "bbox_2d": [68, 102, 97, 131]}\n]\n```'
        
        Args:
            text: String containing prediction
            
        Returns:
            List of bounding boxes as [x1, y1, x2, y2]
        """
        boxes = []
        try:
            #find everythin between ``` json and ```, allwoing newlines
            pattern = r'```json\s*([\s\S]*?)```'
            matches = re.findall(pattern, text)
            json_str = matches[0] if matches else None

            ## Remove the code block markers and parse JSON
            #json_str = text.strip().replace("```json", "").replace("```", "").strip()
            
            bbox_data = json.loads(json_str)
            
             # Initialize with empty box
            for item in bbox_data:
                if "bbox_2d" in item:
                    box = item["bbox_2d"]
                    if len(box) == 4:
                        boxes.append(box)
        except Exception as e:
            #fallback, try to parse from bbox_2d
            pattern = r'bbox_2d"\s*:\s*\[(.*?)\]'
            matches = re.findall(pattern, text)
            if matches:
                boxes = []
                for match in matches:
                    coords = match.split(',')
                    if len(coords) == 4:
                        try:
                            box = [int(coord.strip()) for coord in coords]
                            boxes.append(box)
                        except ValueError:
                            continue
                        
        if len(boxes) == 0:
            #try to parse from text
            
            if "<box>" in text and "</box>" in text:
                #find every thing between <box> and </box>
                
                pattern = r'<box>(.*?)</box>'
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    box_text = match.strip()
                    #bbox is format [x1, y1, x2, y2] as string
                    #convert to list of float
                    try:
                        bbox = ast.literal_eval(box_text)
                        if len(bbox) == 4:
                            boxes.append(bbox)
                    except Exception as e:
                        continue
                #bbox is format [x1, y1, x2, y2] as string
                #convert to list of float
        if len(boxes) == 0:
            boxes = [np.zeros((4), dtype=int)]  # Return empty box if no valid boxes found
            boxes = None
        
        return boxes
    

    def is_valid_box(self, box: List[float]) -> bool:
        #check if box in x1,y1,x2,y2 format is valid (e.g., x2 > x1 and y2 > y1)
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        return x2 > x1 and y2 > y1
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, float]:
        """
        Evaluate bounding box predictions against ground truth.
        
        Args:
            predictions: List of prediction strings or dicts with 'text' key
            
        Returns:
            Dictionary with MAP metrics
        """
        # Load ground truth data if needed
        
        if not self.ground_truths:
            raise ValueError("No ground truth data available. Provide either ground_truths or ground_truth_file.")
        
        # Ensure we have matching number of predictions and ground truths
        if len(predictions) != len(self.ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match ground truths ({len(self.ground_truths)})")
        
        # Prepare data for evaluation
        pred_list = []
        target_list = []
        dummy_label = 1  # Single class label


        
        for pred, target in zip(predictions, self.ground_truths):
            # Process prediction
            pred_text = self.extract_text(pred)
            
            # Parse predicted boxes
            box_list = self.parse_bbox_from_text(pred_text) if isinstance(pred_text, str) else []
            boxes = torch.tensor(box_list) if box_list else torch.empty((0, 4))
            # Process ground truth
            if boxes.numel() == 0 or not self.is_valid_box(boxes[0]):
                boxes = torch.empty((0, 4))



            gt_boxes = target

            if gt_boxes is None:
                gt_boxes = []

            gt_boxes = torch.tensor(gt_boxes) if len(gt_boxes) > 0 else torch.empty((0, 4))

            if gt_boxes.numel() == 0 or not self.is_valid_box(gt_boxes[0]):
                gt_boxes = torch.empty((0, 4))



            pred_dict = {
                "boxes": boxes,
                "scores": torch.ones(boxes.shape[0]),  # Constant score
                "labels": torch.full((boxes.shape[0],), dummy_label, dtype=torch.int)
            }
            pred_list.append(pred_dict)
            

            
            target_dict = {
                "boxes": gt_boxes,
                "labels": torch.full((gt_boxes.shape[0],), dummy_label, dtype=torch.int)
            }
            target_list.append(target_dict)


        #count boxes with all 0 in coordinates
        #n_zero_boxes = sum(1 for t in pred_list if t["boxes"].shape[0] == 1 and torch.all(t["boxes"] == 0))
        zero_indices = [i for i, t in enumerate(pred_list) if torch.all(t["boxes"] == 0)]
        #print the nzero boxes
        # print(f"Number of zero boxes: {len(zero_indices)}")
        # for i in zero_indices:
        #     print(f"Zero box found in prediction {i}: {pred_list[i]}")
        #     print({pred_list[i]["boxes"].numel()})

        # Calculate MAP
        metric = MeanAveragePrecision(
            
          compute_on_cpu=False, 
          sync_on_compute=False,
          dist_sync_on_step=True,)
        metric.update(pred_list, target_list)
        result = metric.compute()
        
        metric_iou = IntersectionOverUnion(
            compute_on_cpu=False,
            sync_on_compute=False,
            dist_sync_on_step=True,
        )
        metric_iou.update(pred_list, target_list)
        iou_result = metric_iou.compute()
        # Combine results
        result.update(iou_result)

        #fill 0 element tensors with 0.0
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.numel() == 0:
                result[k] = torch.tensor(0.0)
        
        return {k: v.item() for k, v in result.items()}




class LabelEvaluator(BaseEvaluator):
    """Evaluator for bounding box predictions."""
    
    def __init__(
        self, 
        ground_truth_file: Optional[str] = None, 
        ground_truths: Optional[List[Dict[str, Any]]] = None,
        name: Optional[str] = None,
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.7,
        grouping_threshold=0.7
    ):
        """
        Initialize the bounding box evaluator.
        
        Args:
            ground_truth_file: Path to ground truth file (one JSON per line)
            ground_truths: Ground truth data (alternative to file)
            name: Name for this evaluator
        """
        super().__init__(name=name or "LabelMAP")
        self.ground_truth_file = ground_truth_file
        self.ground_truths_labels = ground_truths

        if use_sentence_transformers:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        self.threshold = similarity_threshold
        
        self.grouping_threshold = grouping_threshold

        self.load_data()

    
    def load_data(self):
        """Load ground truth data if not provided directly."""
        if self.ground_truths_labels is None and self.ground_truth_file:
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truths_labels = [json.loads(line) for line in f]
        
        #Parse gt boxes from ground truths
        parsed = []
        for gt in self.ground_truths_labels:
            labels = self.parse_labels_from_text(gt)
            
            parsed.append(labels)
        self.ground_truths_labels = parsed
        
            
    def group_objects(self, all_objects: List[str]) -> Dict[str, List[str]]:
        """
        Group similar object names using string similarity.
        
        Args:
            all_objects: List of all unique object names
            
        Returns:
            Dictionary mapping canonical name to list of similar names
        """
        if not all_objects:
            return {}
        
        groups = {}
        processed = set()
        
        for obj in all_objects:
            if obj in processed:
                continue
            
            # Start new group with this object as canonical name
            group = [obj]
            processed.add(obj)
            
            # Find similar objects
            for other_obj in all_objects:
                if other_obj in processed:
                    continue
                
                # Calculate similarity
                if use_sentence_transformers and self.model:
                    emb1 = self.model.encode(obj, convert_to_tensor=True, normalize_embeddings=True)
                    emb2 = self.model.encode(other_obj, convert_to_tensor=True, normalize_embeddings=True)
                    sim = float(util.cos_sim(emb1, emb2))
                else:
                    sim = SequenceMatcher(None, obj, other_obj).ratio()
                
                if sim >= self.grouping_threshold:
                    group.append(other_obj)
                    processed.add(other_obj)
            
            # Use shortest name as canonical
            canonical = min(group, key=len)
            groups[canonical] = group
        
        return groups
    @classmethod
    def parse_labels_from_text(self, text: str) -> List[List[float]]:
        """
        Parse bounding box coordinates from prediction text Qwen Format.
        Expected format: "'```json\n[\n\t{"label": "orange", "bbox_2d": [68, 102, 97, 131]}\n]\n```'
        
        Args:
            text: String containing prediction
            
        Returns:
            List of bounding boxes as [x1, y1, x2, y2]
        """
        labels = []
    
        #find everythin between ``` json and ```, allwoing newlines
        pattern = r'```json\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)
        json_str = matches[0] if matches else None

        if not json_str:
            #fallback, try to parse from label
            pattern = r'label"\s*:\s*"(.*?)"'
            matches = re.findall(pattern, text)
            if matches:
                labels = [self.process_label(m) for m in matches]
            else:
                #try to parse from <object>...</object>
                pattern = r'<object>(.*?)</object>'
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    labels = [self.process_label(m.strip()) for m in matches]
            return labels
        try:
            json_str = json_str.strip()
            if not (json_str.startswith('[') and json_str.endswith(']')):
                json_str = f'[{json_str}]'  # Ensure it's a list
            data = json.loads(json_str)
        except Exception as e:
            print(text)
            print(f"Failed to parse JSON: {e}")
            return labels
        
        
            # Initialize with empty box
        for item in data:
            if "label" in item:
                label = item["label"]
                label = LabelEvaluator.process_label(label)

                label = self.process_label(label)
                labels.append(label)
 
        return labels
    
    @classmethod
    def process_label(self, label : str) -> str:        
        label = label.lower().strip()
        label = re.sub(r'[^\w\s]', '', label)  # remove punctuation/emojis
        label = re.sub(r'\s+', ' ', label)     # normalize whitespace
        return label

    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, float]:
        """
        Evaluate bounding box predictions against ground truth.
        
        Args:
            predictions: List of prediction strings or dicts with 'text' key
            
        Returns:
            Dictionary with MAP metrics
        """
        # Load ground truth data if needed
        
        if not self.ground_truths_labels:
            raise ValueError("No ground truth data available. Provide either ground_truths or ground_truth_file.")
        
        # Ensure we have matching number of predictions and ground truths
        if len(predictions) != len(self.ground_truths_labels):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match ground truths ({len(self.ground_truths_labels)})")
        


        results = list()

        object_results = {}  # Track results per object
        all_gt_objects = set()


        for pred, target in zip(predictions, self.ground_truths_labels):

            pred_text = self.extract_text(pred)            
            pred_labels = self.parse_labels_from_text(pred_text) if isinstance(pred_text, str) else ""
            pred_labels = [self.process_label(label) for label in pred_labels]
            
            all_gt_objects.update(target)

            for gt_label in target:
                if gt_label not in object_results:
                    object_results[gt_label] = {'correct': 0, 'total': 0}
                object_results[gt_label]['total'] += 1

            if not use_sentence_transformers:
                #fallback to simple string similarity
                #match all with all
                mat = []
                for pred_label in pred_labels:
                    for target_label in target:
                        sim = SequenceMatcher(None, target_label, pred_label).ratio()
                        correct = sim >= self.threshold
                        mat.append(sim)
                sim_mat = np.resize(mat, (len(target), len(pred_labels))) if len(mat) > 0 else np.zeros((len(target), len(pred_labels)))
                #take maximum match for each target
                if sim_mat.shape[0] > 0 and sim_mat.shape[1] > 0:
                    max_sims = np.max(sim_mat, axis=1)
                    for i, (gt_label, max_sim) in enumerate(zip(target, max_sims)):
                        if max_sim >= self.threshold:
                            object_results[gt_label]['correct'] += 1
                    
                    correct = np.sum(max_sims >= self.threshold)
                    results.append(int(correct == len(target)))
                else:
                    results.append(0)
                
                continue
            true_emb = self.model.encode(target, convert_to_tensor=True, normalize_embeddings=True)
            pred_emb = self.model.encode(pred_label, convert_to_tensor=True, normalize_embeddings=True)



            # Compute cosine similarity
            sim = float(util.cos_sim(true_emb, pred_emb))

            correct = sim >= self.threshold
            results.append(int(correct))
            
        accuracy = np.mean(results)
        
        y_true = np.ones(len(results))
        y_pred = np.array(results)  

        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        object_groups = self.group_objects(list(all_gt_objects))
        grouped_results = {}


        for canonical_name, group_members in object_groups.items():
            total_correct = 0
            total_count = 0
            
            for member in group_members:
                if member in object_results:
                    total_correct += object_results[member]['correct']
                    total_count += object_results[member]['total']
            
            if total_count > 0:
                grouped_results[canonical_name] = {
                    'accuracy': total_correct / total_count,
                    'correct': total_correct,
                    'total': total_count,
                    'grouped_objects': group_members
                }
        #sort grouped results by total count descending
        grouped_results = dict(sorted(grouped_results.items(), key=lambda item: item[1]['total'], reverse=True))

        return {
            'accuracy': accuracy,
            'precision': p,
            'recall': r,
            'f1_score': f,
            'per_object_results': grouped_results
        }



class PointEvaluator(BaseEvaluator):
    """Evaluator for point predictions."""
    
    def __init__(
        self, 
        mask_dir: Optional[str] = None,
        mask_paths: Optional[List[str]] = None,
        name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the point evaluator.
        
        Args:
            mask_dir: Directory containing masks
            mask_paths: Specific mask paths (alternative to mask_dir)
            name: Name for this evaluator
            verbose: Whether to print processing details
        """
        super().__init__(name=name or "PointAccuracy")
        self.mask_dir = mask_dir
        self.mask_paths = mask_paths
        self.verbose = verbose
    
    def load_data(self, num_samples: Optional[int] = None):
        """Prepare mask paths if not provided directly."""
        if self.mask_paths is None and self.mask_dir:
            # Create default mask paths based on indices
            if num_samples is not None:
                self.mask_paths = [os.path.join(self.mask_dir, f"{i:02d}.jpg") for i in range(num_samples)]
    
    @staticmethod
    def text2pts(text: str) -> np.ndarray:
        """
        Extract points from model output text.
        Expected format: points like (x,y) or x,y
        
        Args:
            text: String containing point coordinates
            
        Returns:
            Array of [x, y] points
        """
        # Match (x,y) or x,y patterns
        pattern = r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?'
        matches = re.findall(pattern, text)
        
        if not matches:
            return np.zeros((0, 2))
        
        points = np.array([[int(x), int(y)] for x, y in matches])
        return points
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate point predictions against ground truth masks.
        
        Args:
            predictions: List of prediction strings or dicts with 'text' key
            
        Returns:
            Dictionary with accuracy metrics
        """
        # Prepare mask paths if needed
        self.load_data(num_samples=len(predictions))
        
        if not self.mask_paths:
            raise ValueError("No mask paths available. Provide either mask_paths or mask_dir.")
        
        # Ensure we have matching number of predictions and masks
        if len(predictions) != len(self.mask_paths):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match masks ({len(self.mask_paths)})")
        
        # Evaluate predictions
        accuracies = []
        
        for idx, (pred, mask_path) in enumerate(zip(predictions, self.mask_paths)):
            # Extract text from prediction
            pred_text = self.extract_text(pred)
            
            # Parse points
            try:
                points = self.text2pts(pred_text)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to parse points for sample {idx}: {e}")
                points = np.zeros((0, 2))
            
            # Load mask
            try:
                mask = np.array(Image.open(mask_path)) / 255.0
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load mask for sample {idx}: {e}")
                accuracies.append(0.0)
                continue
            
            # Calculate accuracy
            acc = 0.0
            if len(points) > 0:
                # Check which points are within the mask boundaries
                in_range = ((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) &
                           (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0]))
                
                # Get mask values at valid point locations
                valid_points = points[in_range]
                valid_values = np.zeros(len(points))
                
                if len(valid_points) > 0:
                    # Extract mask values for valid points
                    valid_values[:len(valid_points)] = mask[valid_points[:, 1], valid_points[:, 0]]
                
                # Average mask values (points in mask = 1.0, outside = 0.0)
                acc = valid_values.mean()
            
            accuracies.append(acc)
        
        # Calculate overall accuracy
        mean_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        return {
            "accuracy": mean_accuracy,
            "individual_accuracies": accuracies
        }


class TaskInstructionEvaluator(BaseEvaluator):
    """Evaluator for comparing task instructions using LLM-based semantic similarity."""
    
    def __init__(
        self,
        ground_truth_file: Optional[str] = None,
        ground_truths: Optional[List[str]] = None,
        name: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        n_jobs: int = 32
    ):
        """
        Initialize the task instruction evaluator.
        
        Args:
            ground_truth_file: Path to ground truth file (one instruction per line)
            ground_truths: Ground truth task instructions (alternative to file)
            name: Name for this evaluator
            model: Model to use for comparison
            temperature: Temperature for LLM queries
            n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        """
        super().__init__(name=name or "TaskInstructionSimilarity")
        self.ground_truth_file = ground_truth_file
        self.ground_truths = ground_truths
        self.model = model
        self.temperature = temperature
        self.n_jobs = n_jobs if use_joblib else 1
        
        self.load_data()
    

    def parse_task_instruction(self, text: str) -> str:
        """
        Parse task instruction from text.
        May be enclosed in <task>...</task> tags or last line after newline.
        """

        # Check for <task>...</task> tags
        pattern = r'<task>(.*?)</task>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Fallback: take last non-empty line
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return text.strip()

    def load_data(self):
        """Load ground truth data if not provided directly."""
        if self.ground_truths is None and self.ground_truth_file:
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truths = [line.strip() for line in f if line.strip()]

        
        parsed = []

        for gt in self.ground_truths:
            instruction = self.parse_task_instruction(gt)
            parsed.append(instruction)
        self.ground_truths = parsed
    
    
    @staticmethod
    def create_comparison_prompt(instruction1: str, instruction2: str) -> Dict[str, str]:
        """
        Create a prompt to compare two task instructions.
        
        Args:
            instruction1: First task instruction (predicted)
            instruction2: Second task instruction (ground truth)
            
        Returns:
            Dictionary with system and user prompts
        """
        system_prompt = """You are an expert at comparing task instructions for semantic equivalence.
Your job is to determine if two task instructions describe the same task, even if they use different wording.

Consider instructions equivalent if they:
- Describe the same actions and goals
- Involve the same objects or entities
- Have the same intended outcome
- May differ in phrasing, word order, or level of detail but convey the same meaning

Respond with ONLY a JSON object in this exact format:
{"equivalent": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}"""

        user_prompt = f"""Compare these two task instructions:

Instruction 1 (Predicted): "{instruction1}"

Instruction 2 (Ground Truth): "{instruction2}"

Are these instructions semantically equivalent? Respond in JSON format."""

        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _compare_single_instruction(
        self, 
        pred_instruction: str, 
        gt_instruction: str,
        index: int
    ) -> Dict[str, Any]:
        """
        Compare a single predicted instruction with ground truth.
        
        Args:
            pred_instruction: Predicted instruction
            gt_instruction: Ground truth instruction
            index: Index of the comparison
            
        Returns:
            Dictionary with comparison results
        """
        prompt = self.create_comparison_prompt(pred_instruction, gt_instruction)
        
        try:
            response = query_parallel_gpt(
                prompt=prompt,
                temperature=self.temperature,
                model=self.model
            )
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response:
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            elif "```" in response:
                json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
            
            result = json.loads(response.strip())
            
            return {
                "index": index,
                "equivalent": result.get("equivalent", False),
                "confidence": result.get("confidence", 0.0),
                "reason": result.get("reason", ""),
                "pred_instruction": pred_instruction,
                "gt_instruction": gt_instruction,
                "error": None
            }
            
        except Exception as e:
            logging.error(f"Error comparing instruction {index}: {e}")
            return {
                "index": index,
                "equivalent": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "pred_instruction": pred_instruction,
                "gt_instruction": gt_instruction,
                "error": str(e)
            }
    
    def evaluate(self, predictions: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate task instruction predictions against ground truth.
        
        Args:
            predictions: List of predicted task instructions (strings or dicts with 'text' key)
            
        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        if not self.ground_truths:
            raise ValueError("No ground truth data available. Provide either ground_truths or ground_truth_file.")
        
        if len(predictions) != len(self.ground_truths):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) doesn't match "
                f"ground truths ({len(self.ground_truths)})"
            )
        
        # Extract text from predictions
        pred_instructions = [self.extract_text(pred) for pred in predictions]
        parsed_instructions = []
        gt_to_compare = []
        for idx, pred in enumerate(pred_instructions):
            parsed = self.parse_task_instruction(pred)
            parsed_instructions.append(parsed)
            gt_to_compare.append(self.ground_truths[idx])
        pred_instructions = parsed_instructions

        
        # Compare instructions in parallel or sequentially
        if use_joblib and self.n_jobs != 1:
            logging.info(f"Comparing {len(predictions)} instructions in parallel with {self.n_jobs} jobs...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compare_single_instruction)(pred, gt, i)
                for i, (pred, gt) in enumerate(zip(pred_instructions, gt_to_compare))
            )
        else:
            logging.info(f"Comparing {len(predictions)} instructions sequentially...")
            results = [
                self._compare_single_instruction(pred, gt, i)
                for i, (pred, gt) in enumerate(zip(pred_instructions, gt_to_compare))
            ]

        

        # Calculate metrics
        equivalences = [r["equivalent"] for r in results]
        confidences = [r["confidence"] for r in results]
        errors = [r for r in results if r["error"] is not None]
        
        accuracy = np.mean(equivalences) if equivalences else 0.0
        mean_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate confidence-weighted accuracy
        weighted_scores = [
            float(r["equivalent"]) * r["confidence"] 
            for r in results
        ]
        weighted_accuracy = np.mean(weighted_scores) if weighted_scores else 0.0
        
        return {
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "weighted_accuracy": weighted_accuracy,
            "num_equivalent": sum(equivalences),
            "num_total": len(equivalences),
            "num_errors": len(errors),
            "detailed_results": results
        }

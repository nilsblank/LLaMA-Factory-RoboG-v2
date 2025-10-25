import torch
import wandb
from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from typing import Optional, List, Dict, Any
import logging


def decode_predictions(tokenizer, predictions):
    """Decode model predictions to text"""
    prediction_text = tokenizer.batch_decode(
        predictions.predictions.argmax(axis=-1),
        skip_special_tokens=True
    )
    return prediction_text


class EvaluatorCallback(TrainerCallback):
    """
    Adapter that converts any BaseEvaluator into a TrainerCallback.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        evaluator
    ):
        """
        Initialize the adapter.
        
        Args:
            trainer: The trainer instance
            tokenizer: The tokenizer for decoding predictions
            val_dataset: Validation dataset
            evaluator: The evaluator to use for evaluation
        """
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.evaluator = evaluator
        self.name = evaluator.name
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Run evaluation during training."""
        # Get model predictions and decode them
        
        if self.trainer.latest_predictions is not None:
            predictions = self.trainer.latest_predictions
            predictions[predictions == -100] = self.tokenizer.pad_token_id
            predictions = predictions[:len(self.val_dataset)]
            logging.info("Predictions shape:", predictions.shape)
            pred_texts = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
        else:
            predictions = self.trainer.predict(self.val_dataset)
            predictions = predictions[:len(self.val_dataset)]
            logging.info("Predictions shape:", predictions.shape)
            pred_texts = decode_predictions(self.tokenizer, predictions)
        
        # Run evaluation
        result = self.evaluator.evaluate(pred_texts)
        log_dict = {}
        
        for key in result:
            
            value = result[key]
            
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                value_tensor = torch.tensor([value], device="cuda")
                gathered = [torch.zeros_like(value_tensor) for _ in range(torch.distributed.get_world_size())]
                #print(gathered)
                torch.distributed.all_gather(gathered, value_tensor)
                #print("Gathered values:", gathered)
                value = torch.stack(gathered).float().mean().item()
            log_dict[f"eval/{key}"] = value
        
        # Log results
        if self.trainer.is_world_process_zero():
            for key, value in log_dict.items():
                log_name = f"{self.name}_{key}"
                self.trainer.log_metrics("eval",{log_name: value})
            
            #get wandb instance from trainer
            self.trainer.callback_handler.callbacks[1]._wandb.log(log_dict, step=state.global_step)
            wandb.log(log_dict)
        


# Import evaluators here rather than at the top to avoid circular imports
from llamafactory.eval.evaluators import BoundingBoxEvaluator, LabelEvaluator, PointEvaluator


class BoundingBoxEvaluatorCallback(EvaluatorCallback):
    """
    Callback wrapper specifically for BoundingBoxEvaluator.
    This maintains backward compatibility with the existing callback interface.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        name: Optional[str] = None
    ):
        # Create BoundingBoxEvaluator with the validation dataset
        evaluator = BoundingBoxEvaluator(
            ground_truths=val_dataset,  # Pass dataset directly as ground truth
            name=name or "BoundingBoxMAP"
        )
        
        super().__init__(trainer, tokenizer, val_dataset, evaluator)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Specialized evaluation for bounding boxes."""
        result = super().on_evaluate(args, state, control, **kwargs)
        
        # Specifically log MAP score for bounding box evaluation
        
        # if "map" in result:
        #     if self.trainer.is_world_process_zero():
        #         self.trainer.log_metrics({self.name: result["map"]})
        #         wandb.log({f"validate/{self.name}": result["map"]}, step=state.global_step)
        

class LabelEvaluatorCallback(EvaluatorCallback):
    """
    Callback wrapper specifically for LabelEvaluator.
    This maintains backward compatibility with the existing callback interface.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        name: Optional[str] = None
    ):
        # Create LabelEvaluator with the validation dataset
        evaluator = LabelEvaluator(
            ground_truths=val_dataset,  # Pass dataset directly as ground truth
            name=name or "LabelAccuracy"
        )
        
        super().__init__(trainer, tokenizer, val_dataset, evaluator)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Specialized evaluation for labels."""
        result = super().on_evaluate(args, state, control, **kwargs)
        


class PointEvaluatorCallback(EvaluatorCallback):
    """
    Callback wrapper specifically for PointEvaluator.
    This maintains backward compatibility with the existing callback interface.
    """
    
    def __init__(
        self, 
        trainer, 
        tokenizer, 
        val_dataset, 
        mask_dir: str,
        name: Optional[str] = None
    ):
        # Create mask paths from validation dataset items
        mask_paths = []
        for idx, sample in enumerate(val_dataset):
            mask_path = sample.get("mask_path", f"{mask_dir}/{idx:02d}.jpg")
            mask_paths.append(mask_path)
        
        # Create PointEvaluator with mask paths
        evaluator = PointEvaluator(
            mask_paths=mask_paths,
            name=name or "PointAccuracy"
        )
        
        super().__init__(trainer, tokenizer, val_dataset, evaluator)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Specialized evaluation for points."""
        result = super().on_evaluate(args, state, control, **kwargs)
        
        # Specifically log accuracy for point evaluation
        if "accuracy" in result:
            self.trainer.log_metrics({self.name: result["accuracy"]})
            wandb.log({f"validate/{self.name}": result["accuracy"]}, step=state.global_step)
        

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

import ast
import glob
import json
import logging
import math
import time
import os
from tqdm import tqdm
import av
from PIL import Image, ImageDraw, ImageFont
from langchain_core.output_parsers import JsonOutputParser

import pandas as pd

from datasets import load_dataset

#add llamafactory to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llamafactory.eval.evaluators import BoundingBoxEvaluator,LabelEvaluator

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output



def parse_box_and_label(text):
    bbox = None
    #first try to find json
    parse_json = False
    if "bbox_2d" in text and "label" in text:
        parse_json = True
    try:
        lines = text.splitlines()
        for i, line in enumerate(lines):    
            if line == "```json":
                text = "\n".join(lines[i+1:])  # Remove everything before "```json"
                text = text.split("```")[0]  # Remove everything after the closing "```"
                parse_json = True
                break  # Exit the loop once "```json" is found
    except Exception as e:
        pass
    if parse_json:
        try:
            parsed = JsonOutputParser().parse(text)
            parsed = parsed[0] if isinstance(parsed, list) else parsed
            bbox = parsed["bbox_2d"]
            label = parsed["label"]
            return bbox, label
        except Exception as e:
            pass
    else:
        if "<box>" in text and "</box>" in text:
            start_idx = text.index("<box>") + len("<box>")
            end_idx = text.index("</box>")
            box_text = text[start_idx:end_idx].strip()
            #bbox is format [x1, y1, x2, y2] as string
            #convert to list of float
            try:
                bbox = ast.literal_eval(box_text)
            except Exception as e:
                bbox = None

        else:
            bbox = None
        if "<object>" in text and "</object>" in text:
            start_idx = text.index("<object>") + len("<object>")
            end_idx = text.index("</object>")
            label = text[start_idx:end_idx].strip()
        else:
            label = None
        return bbox, label
    
            
        
        
    

def plot_bounding_boxes(im, bounding_box,label = None, input_width = 100, input_height = 100, is_gt = False):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im.copy()
    width, height = img.size
    #print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] 

    # Parsing out the markdown fencing
    
    if bounding_box is not None:
        

        font = ImageFont.load_default()


        

        #bounding_box = {"bbox_2d": bounding_box}
        # Select a color from the list
        id = 1 if is_gt else 0
        color = colors[id % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box[1]/input_height * height)
        abs_x1 = int(bounding_box[0]/input_width * width)
        abs_y2 = int(bounding_box[3]/input_height * height)
        abs_x2 = int(bounding_box[2]/input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
    if label is not None:
        try:
            font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
        except OSError:
            font = ImageFont.load_default()
        text_bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=4)
        text_height = text_bbox[3] - text_bbox[1]
        bar_padding = 8
        bar_height = text_height + 2 * bar_padding
        draw.rectangle(
            [(0, height - bar_height), (width, height)],
            fill="white"
        )
        draw.multiline_text(
            (10, height - bar_height + bar_padding),
            label,
            fill="black",
            font=font,
            spacing=4
        )       

    # Display the image
    return img


# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def main(filenames: str, save_images = False):


    loaded_datasets = [load_dataset("json", data_files=filename, split="train") for filename in filenames]
    
    
    
    eval_results = {}
    
    evaluators = [BoundingBoxEvaluator,LabelEvaluator]
    
    for i,dataset in enumerate((loaded_datasets)):
        
        model_name = "_".join(filenames[i].split("/")[-5:-3])
        exp_id = filenames[i].split("/")[-2]
        
        exp_id = model_name + "_" + exp_id

        ds_results = {}
        for evaluator in evaluators:
            evaluator = evaluator(
                ground_truths=[sample["label"] for sample in dataset]
            )
            results = evaluator.evaluate([sample["predict"] for sample in dataset])
            
            #only iou for BoundingBoxEvaluator
            if evaluator.__class__.__name__ == "BoundingBoxEvaluator":
                ds_results[evaluator.__class__.__name__] = results["iou"]
            elif evaluator.__class__.__name__ == "LabelEvaluator":
                ds_results[evaluator.__class__.__name__] = results["accuracy"]
                #print top 10 per object results
                per_object_results = results["per_object_results"]
                print(f"Per Object Results for {exp_id}:")
                for object_name, object_results in list(per_object_results.items())[:10]:
                    print(f"Total: {object_results['total']}, Object: {object_name}, Accuracy: {object_results['correct']/object_results['total']:.4f}")
                #print lowest with count > 10:
                print(f"Lowest Accuracy Objects for {exp_id} (count > 10):")
                low_count_objects = {k: v for k, v in per_object_results.items() if v['total'] > 10}
                low_count_objects = dict(sorted(low_count_objects.items(), key=lambda item: item[1]['correct']/item[1]['total']))
                for object_name, object_results in list(low_count_objects.items())[:10]:
                    print(f"Total: {object_results['total']}, Object: {object_name}, Accuracy: {object_results['correct']/object_results['total']:.4f}")
            eval_results[exp_id] = ds_results
    #print as table
    df = pd.DataFrame.from_dict(eval_results, orient='index')
    print(df)

    
    print(f"Out Path: {os.path.dirname(filenames[0])}")
    

    if save_images: 
        assert all(len(loaded_datasets[0]) == len(ds) for ds in loaded_datasets), "All datasets must have the same length"

        #iterate dataset and plot bboxes and detected objects next to each other
        for i in tqdm(range(len(loaded_datasets[0]))):
            sample = loaded_datasets[0][i]
            path = "/".join(sample["image"][0].split("/")[8:-2])
            orig_image = Image.open(sample["image"][0])
            new_height, new_width = smart_resize(
                    orig_image.height, orig_image.width
                )
            
            gt_box, gt_label = parse_box_and_label(sample["label"])
            pred_box, pred_label = parse_box_and_label(sample["predict"])
            
            if pred_label is None:
                pred_label = "N/A"

            label_string = "GT: " + (gt_label if gt_label is not None else "N/A") + "\nPred: " + pred_label
            
            img_1 = plot_bounding_boxes(
                    orig_image,
                    gt_box,
                    label_string,
                    new_width, new_height,
                    is_gt=  True
                )
            img_2 = plot_bounding_boxes(
                    img_1,
                    pred_box,
                    label = None,
                    input_width = new_width,
                    input_height = new_height
                )
            
            #write the name (second last part of the path) of the model on the top left corner and the path below
            exp_id = filenames[0].split("/")[-2]
            draw = ImageDraw.Draw(img_2)
            font = ImageFont.load_default()
            text = filenames[0].split("/")[-2]
            draw.text((5, 5), text, fill="white", font=font)
            draw.text((5, 20), path, fill="white", font=font)
            
            images_to_cat = [img_2]
            

            
            for j,ds in enumerate(loaded_datasets[1:]):
                path = "/".join(sample["image"][0].split("/")[8:-2])
                other_sample = ds[i]
                assert sample["image"] == other_sample["image"], "All datasets must have the same image"
                gt_box, gt_label = parse_box_and_label(other_sample["label"])

            
                other_pred_box, other_pred_label = parse_box_and_label(other_sample["predict"])
                if other_pred_label is None:
                    other_pred_label = "N/A"
                label_string = "GT: " + (gt_label if gt_label is not None else
                    "N/A") + "\nPred: " + other_pred_label
                img_1 = plot_bounding_boxes(
                    orig_image,
                    gt_box,
                    label_string,
                    new_width, new_height,
                    is_gt=  True
                )
                img_2 = plot_bounding_boxes( 
                    img_1,
                    other_pred_box,
                    label = None,
                    input_width = new_width,
                    input_height = new_height
                )
                
                draw = ImageDraw.Draw(img_2)
                font = ImageFont.load_default()
                text = filenames[j+1].split("/")[-2]
                draw.text((5, 5), text, fill="white", font=font)
                draw.text((5, 20), path, fill="white", font=font)

                images_to_cat.append(img_2)
                
                
            
            total_width = sum(img.width for img in images_to_cat)
            max_height = max(img.height for img in images_to_cat)
            new_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images_to_cat:
                new_img.paste(img, (x_offset,0))
                x_offset += img.width
            
            out_path = os.path.dirname(filenames[0])
            out_path = os.path.join(out_path, "results")
            os.makedirs(out_path, exist_ok=True)
            new_img.save(os.path.join(out_path, f"{i}_comparison.png"))
            



if __name__ == "__main__":
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/$/generated_predictions.jsonl"
    
    #Video + First frame
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train/generated_predictions.jsonl"
    
    # Video + First frame + Detection Cotrain
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train, hd-epic-detection_train/generated_predictions.jsonl"
    #filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train, hd-epic-detection_train/checkpoint-800/generated_predictions.jsonl"
    
    
    filenames  = glob.glob("/home/hk-project-sustainebot/bm3844/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/*poc*/generated_predictions.jsonl")
    filenames_new  = glob.glob("/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/saves/qwen2_5vl-3b*/*/sft/*poc*/generated_predictions.jsonl")
    
    
    filenames_qwen_3 = glob.glob("/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/saves/qwen3_5vl-4b/full/sft/*/generated_predictions.jsonl")
    
    filenames.extend(filenames_qwen_3)
    filenames.extend(filenames_new)
    filenames = [f for f in filenames if "train" in f]
    filenames = ["/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/saves/qwen2_5vl-3b/full/sft/roboG_stagepoc_ablation_two_frames_train_traineval/generated_predictions.jsonl"]
    #filenames = [f for f in filenames if "roboG_stagepoc_temporal_grounding_no_box_train" in f]
    
    main(filenames,save_images=False)




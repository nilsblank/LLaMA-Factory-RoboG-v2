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
import json
import logging
import math
import time
import os
from tqdm import tqdm
import av
from PIL import Image, ImageDraw, ImageFont

from datasets import load_dataset


from llamafactory.eval.evaluators import BoundingBoxEvaluator

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


def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
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
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      

      #bounding_box = {"bbox_2d": bounding_box}
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

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


def main(filename: str, save_images = False):
    start_time = time.time()

    dataset = load_dataset("json", data_files=filename, split="train")
    
    ground_truths = [sample["label"] for sample in dataset]
    predictions = [sample["predict"] for sample in dataset]


    evaluator = BoundingBoxEvaluator(
        ground_truths=ground_truths)
    
    res = evaluator.evaluate(predictions)
    print(res)
    
    ground_truths_vid = [sample["label"] for sample in dataset if sample["video"] is not None]
    predictions_vid = [sample["predict"] for sample in dataset if sample["video"] is not None]
    
    evaluator_vid = BoundingBoxEvaluator(
        ground_truths=ground_truths_vid)
    res_vid = evaluator_vid.evaluate(predictions_vid)
    print(res_vid)
    


    out_path = os.path.dirname(filename)
    out_path = os.path.join(out_path, "results")
    for sample in tqdm(dataset):
        
        if sample["video"] is None:
            continue
        
        if "image" in sample and sample["image"]:
            
            frame = Image.open(sample["image"][0])
        else:
            video = av.open(sample["video"][0])
            #get first frame of video
            first_frame = None
            for frame in video.decode(video.streams.video[0]):
                first_frame = frame
                break


            #to pil
            frame = Image.fromarray(first_frame.to_ndarray(format="rgb24"))
        new_height, new_width = smart_resize(
                frame.height, frame.width
            )
        
        if save_images:
            img = plot_bounding_boxes(
                frame,
                sample["predict"],
                new_width,
                new_height
            )

            
            img_2 = plot_bounding_boxes(
                frame,
                sample["label"],
                new_width,
                new_height
            )

            #save img
            cur_save_dir = os.path.join(out_path, f"{sample['video'][0].split('/')[-1].split('.')[0]}_bbox.png")
            os.makedirs(out_path, exist_ok=True)
            img_2.save(cur_save_dir)



if __name__ == "__main__":
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/$/generated_predictions.jsonl"
    
    #Video + First frame
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train/generated_predictions.jsonl"
    
    # Video + First frame + Detection Cotrain
    filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train, hd-epic-detection_train/generated_predictions.jsonl"
    #filename = "/home/blank/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/hd-epic-bbox_first_frame_train, hd-epic-detection_train/checkpoint-800/generated_predictions.jsonl"
    main(filename,save_images=False)




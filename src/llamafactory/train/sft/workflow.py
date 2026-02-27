# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

import copy
import json
import math
import os
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from accelerate import Accelerator
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from ...custom_models.temporal_resamplers.hybrid_slot_query.hybrid_slot_query_losses import HybridLoss
from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
from ...eval.callback_adapters import BoundingBoxEvaluatorCallback,LabelEvaluatorCallback
from dataclasses import asdict


import json


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)



def _unwrap_model_for_hsq(model):
    candidates = [model]
    seen_ids = {id(model)}

    for _ in range(6):
        new_candidates = []
        for cand in candidates:
            for attr in ("model", "module", "base_model"):
                if hasattr(cand, attr):
                    nxt = getattr(cand, attr)
                    if id(nxt) not in seen_ids:
                        seen_ids.add(id(nxt))
                        new_candidates.append(nxt)

        candidates.extend(new_candidates)

    for cand in candidates:
        if hasattr(cand, "hybrid_slot_query_model") and hasattr(cand, "visual"):
            return cand

    raise ValueError("Cannot locate Qwen3VL model wrapper containing `hybrid_slot_query_model` and `visual`.")


def _build_qwen_patch_pos_emb_batched(
    qwen3vlmodelinstance,
    video_grid_thw: torch.Tensor,
    max_t: int,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
    use_pooled_output: Optional[bool] = None,
) -> torch.Tensor:
    """Build per-frame patch positional embeddings using Qwen `fast_pos_embed_interpolate`.

    Returns tensor of shape (B, max_t, lmax, Cpos).
    """
    visual = qwen3vlmodelinstance.visual
    if use_pooled_output is None:
        use_pooled_output = bool(getattr(qwen3vlmodelinstance, "slot_query_use_pooled_output", True))
    merge = visual.spatial_merge_size
    cpos = visual.pos_embed.embedding_dim
    bsz = video_grid_thw.shape[0]

    pos_batched = torch.zeros((bsz, max_t, lmax, cpos), device=device, dtype=dtype)

    for vid_idx in range(bsz):
        t = int(video_grid_thw[vid_idx, 0].item())
        h = int(video_grid_thw[vid_idx, 1].item())
        w = int(video_grid_thw[vid_idx, 2].item())
        h_m = max(1, h // merge)
        w_m = max(1, w // merge)
        per_frame = h_m * w_m if use_pooled_output else h * w

        # Primary path: use Qwen's own interpolation routine.
        local_grid = video_grid_thw[vid_idx : vid_idx + 1]
        pos_flat = visual.fast_pos_embed_interpolate(local_grid).to(device=device, dtype=dtype)  # (Lpos, Cpos)

        expected_len = t * per_frame
        if pos_flat.shape[0] == expected_len:
            pos_flat = pos_flat.view(t, per_frame, cpos)
        elif expected_len > 0 and pos_flat.shape[0] % expected_len == 0:
            # If fast_pos returns a finer sequence than merged tokens, average each local group.
            ratio = pos_flat.shape[0] // expected_len
            pos_flat = pos_flat.view(t, per_frame, ratio, cpos).mean(dim=2)
        else:
            # Fallback safety: manual interpolate from learned pos table to merged patch grid.
            pos_table = visual.pos_embed.weight
            side = visual.num_grid_per_side
            pos_2d = pos_table.view(side, side, cpos).permute(2, 0, 1).unsqueeze(0).float()
            interp_h, interp_w = (h_m, w_m) if use_pooled_output else (h, w)
            pos_interp = F.interpolate(pos_2d, size=(interp_h, interp_w), mode="bilinear", align_corners=False)
            pos_flat = pos_interp.squeeze(0).permute(1, 2, 0).reshape(1, per_frame, cpos).expand(t, -1, -1)

        pos_batched[vid_idx, :t, :per_frame] = pos_flat

    return pos_batched


def _extract_visual_tokens_for_hsq(qwen3vlmodelinstance, batch: dict[str, torch.Tensor]):
    if "pixel_values_videos" not in batch or "video_grid_thw" not in batch:
        return None, None

    pixel_values_videos = batch["pixel_values_videos"]
    video_grid_thw = batch["video_grid_thw"]

    if pixel_values_videos is None or video_grid_thw is None:
        return None, None

    if video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
        return None, None

    with torch.no_grad():
        pixel_values_videos = pixel_values_videos.type(qwen3vlmodelinstance.visual.dtype)
        vision_output = qwen3vlmodelinstance.visual(pixel_values_videos, grid_thw=video_grid_thw)

    use_pooled_output = bool(getattr(qwen3vlmodelinstance, "slot_query_use_pooled_output", True))
    spatial_merge = qwen3vlmodelinstance.visual.spatial_merge_size
    if use_pooled_output:
        split_sizes = (video_grid_thw.prod(-1) // (spatial_merge**2)).tolist()
        token_source = vision_output.pooler_output
    else:
        split_sizes = video_grid_thw.prod(-1).tolist()
        token_source = vision_output.last_hidden_state
        slot_query_input_proj = getattr(qwen3vlmodelinstance, "slot_query_input_proj", None)
        if slot_query_input_proj is not None:
           token_source = slot_query_input_proj(token_source)

    video_tokens_list = torch.split(token_source, split_sizes)

    bsz = len(video_tokens_list)
    per_frame_list = []
    t_list = []
    d_in = video_tokens_list[0].shape[-1]
    tokens_tf_list = []

    for vid_idx in range(bsz):
        t = int(video_grid_thw[vid_idx, 0].item())
        h = int(video_grid_thw[vid_idx, 1].item())
        w = int(video_grid_thw[vid_idx, 2].item())
        per_frame = (h * w) // (spatial_merge**2) if use_pooled_output else (h * w)

        vt = video_tokens_list[vid_idx]
        if vt.shape[0] != t * per_frame:
            raise ValueError(
                f"Video token length mismatch at idx={vid_idx}: got {vt.shape[0]}, expected {t * per_frame}."
            )

        tokens_tf_list.append(vt.view(t, per_frame, d_in))
        per_frame_list.append(per_frame)
        t_list.append(t)

    max_t = max(t_list)
    lmax = max(per_frame_list)
    device = video_tokens_list[0].device
    dtype = video_tokens_list[0].dtype

    x_batched = torch.zeros((bsz, max_t, lmax, d_in), device=device, dtype=dtype)
    attn_mask = torch.zeros((bsz, max_t, lmax), device=device, dtype=torch.bool)

    for vid_idx, (tokens_tf, t, l) in enumerate(zip(tokens_tf_list, t_list, per_frame_list)):
        x_batched[vid_idx, :t, :l] = tokens_tf
        attn_mask[vid_idx, :t, :l] = True

    return x_batched, attn_mask


def _get_first_available(batch: dict[str, torch.Tensor], candidates: list[str]):
    for key in candidates:
        if key in batch and isinstance(batch[key], torch.Tensor):
            return batch[key]
    return None


def _build_slot_color_palette(num_slots: int) -> torch.Tensor:
    if num_slots <= 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    h = torch.arange(num_slots, dtype=torch.float32) / float(max(1, num_slots))
    r = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * (h + 0.00))
    g = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * (h + 0.33))
    b = 0.5 + 0.5 * torch.sin(2.0 * torch.pi * (h + 0.67))
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)






def _save_slot_mask_visualizations(
    masks: torch.Tensor,
    video_grid_thw: torch.Tensor,
    pixel_values_videos: Optional[torch.Tensor],
    output_dir: str,
    split: str,
    global_step: int,
    epoch: int,
    patch_size: int,
    temporal_patch_size: int,
    in_channels: int,
    spatial_merge_size: int,
    use_pooled_output: bool,
    max_samples: int,
    upsample_factor: int,
    overlay_alpha: float,
    rng: random.Random,
) -> None:
    if masks.ndim != 5 or video_grid_thw.ndim != 2 or video_grid_thw.shape[-1] != 3:
        return

    bsz = masks.shape[0]
    if bsz == 0:
        return

    # Fixed visualization layout: 8 videos x 8 frames.
    # Use deterministic video indices to keep rows stable across visualizations.
    fixed_videos = 8
    fixed_frames = 8
    sample_count = min(fixed_videos, bsz)
    sample_indices = list(range(sample_count))

    base_dir = os.path.join(output_dir, "slot_mask_vis", split, f"step_{global_step:07d}")
    frames_grid_dir = os.path.join(base_dir, "frames_grid")
    decoder_masks_grid_dir = os.path.join(base_dir, "decoder_masks_grid")
    overlay_grid_dir = os.path.join(base_dir, "overlay_grid")
    slot_by_frame_grid_dir = os.path.join(base_dir, "slot_by_frame_grid")
    os.makedirs(frames_grid_dir, exist_ok=True)
    os.makedirs(decoder_masks_grid_dir, exist_ok=True)
    os.makedirs(overlay_grid_dir, exist_ok=True)
    os.makedirs(slot_by_frame_grid_dir, exist_ok=True)

    split_sizes = (video_grid_thw.prod(-1)).tolist()
    video_patch_list = []
    if pixel_values_videos is not None and pixel_values_videos.ndim == 2:
        try:
            video_patch_list = list(torch.split(pixel_values_videos, split_sizes, dim=0))
        except RuntimeError:
            video_patch_list = []

    num_slots = masks.shape[2]
    palette = _build_slot_color_palette(num_slots)
    if num_slots <= 0:
        return

    def _reconstruct_base_frame(vpatch: torch.Tensor, t_idx: int, t: int, h: int, w: int) -> Optional[torch.Tensor]:
        d_patch = in_channels * temporal_patch_size * patch_size * patch_size
        if vpatch.ndim != 2 or vpatch.shape[-1] != d_patch:
            return None

        frame_patch_count = h * w
        if frame_patch_count <= 0:
            return None

        token_offset = t_idx * frame_patch_count
        if token_offset < 0 or token_offset + frame_patch_count > vpatch.shape[0]:
            return None

        frame_tokens = vpatch[token_offset : token_offset + frame_patch_count]

        m = spatial_merge_size
        if h % m == 0 and w % m == 0:
            frame_tokens = frame_tokens.view(h // m, w // m, m, m, d_patch)
            frame_tokens = frame_tokens.permute(0, 2, 1, 3, 4).contiguous().view(h, w, d_patch)
        else:
            frame_tokens = frame_tokens.view(h, w, d_patch)

        frame_tokens = frame_tokens.view(h, w, in_channels, temporal_patch_size, patch_size, patch_size)
        frame_tokens = frame_tokens[:, :, :, 0, :, :]  # (h, w, C, ps, ps)
        base_image = frame_tokens.permute(2, 0, 3, 1, 4).reshape(in_channels, h * patch_size, w * patch_size)
        base_image = base_image.float()

        if torch.isfinite(base_image).all():
            if base_image.min() < -0.1 and base_image.max() <= 1.5:
                base_image = (base_image + 1.0) / 2.0
            else:
                img_min = base_image.amin(dim=(1, 2), keepdim=True)
                img_max = base_image.amax(dim=(1, 2), keepdim=True)
                base_image = (base_image - img_min) / (img_max - img_min + 1e-6)

        return base_image.clamp(0.0, 1.0)

    def _to_rgb(image: torch.Tensor) -> torch.Tensor:
        if image.shape[0] == 3:
            return image
        if image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        if image.shape[0] > 3:
            return image[:3]
        pad = torch.zeros((3 - image.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        return torch.cat([image, pad], dim=0)

    def _frame_indices(t: int, n: int) -> list[int]:
        if t <= 0:
            return [0] * n
        if t >= n:
            return torch.linspace(0, t - 1, steps=n).round().long().tolist()
        idx = list(range(t))
        idx.extend([idx[-1]] * (n - t))
        return idx

    def _resize_video_stack(stack: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        if stack.shape[-2:] == size_hw:
            return stack
        return F.interpolate(stack, size=size_hw, mode="bilinear", align_corners=False)

    videos_frames: list[torch.Tensor] = []
    videos_decoder_masks: list[torch.Tensor] = []
    videos_overlay: list[torch.Tensor] = []
    target_hw: Optional[tuple[int, int]] = None

    for sample_rank, b_idx in enumerate(sample_indices):
        t = int(video_grid_thw[b_idx, 0].item())
        h = int(video_grid_thw[b_idx, 1].item())
        w = int(video_grid_thw[b_idx, 2].item())
        if t <= 0 or h <= 0 or w <= 0:
            continue
            
        if use_pooled_output:
            h_m = max(1, h // spatial_merge_size)
            w_m = max(1, w // spatial_merge_size)
            n_tokens = h_m * w_m
        else:
            h_m = h
            w_m = w
            n_tokens = h * w
        if n_tokens <= 0:
            continue

        sample_masks = masks[b_idx, :t, :, :n_tokens, 0].float()  # (T, K, N)
        if sample_masks.numel() == 0:
            continue

        palette_local = palette.to(sample_masks.device)
        frame_indices = _frame_indices(t, fixed_frames)

        vpatch = video_patch_list[b_idx] if b_idx < len(video_patch_list) else None
        cur_frames = []
        cur_decoder_masks = []
        cur_overlay = []

        for t_idx in frame_indices:
            slot_flat = sample_masks[t_idx]  # (K, N)
            if slot_flat.numel() == 0:
                continue

            slot_maps = slot_flat.view(num_slots, h_m, w_m).clamp(0.0, 1.0)  # (K, Hm, Wm)

            seg_idx = slot_maps.argmax(dim=0)  # (Hm, Wm)
            seg_rgb = palette_local[seg_idx].permute(2, 0, 1).contiguous()  # (3, Hm, Wm)
            seg_conf = slot_maps.max(dim=0).values.unsqueeze(0)  # (1, Hm, Wm)

            base_image = None
            if vpatch is not None:
                base_image = _reconstruct_base_frame(vpatch, t_idx=t_idx, t=t, h=h, w=w)

            if base_image is not None:
                base_image = _to_rgb(base_image)
                base_h, base_w = base_image.shape[-2], base_image.shape[-1]
                seg_rgb_up = F.interpolate(seg_rgb.unsqueeze(0), size=(base_h, base_w), mode="bilinear", align_corners=False).squeeze(0)
                seg_conf_up = F.interpolate(seg_conf.unsqueeze(0), size=(base_h, base_w), mode="nearest").squeeze(0)
                seg_conf_up = seg_conf_up.clamp(0.0, 1.0)

                alpha_map = (overlay_alpha * seg_conf_up).clamp(0.0, 1.0)
                overlay = ((1.0 - alpha_map) * base_image + alpha_map * seg_rgb_up).clamp(0.0, 1.0)
                seg_rgb_vis = seg_rgb_up
                frame_vis = base_image
            else:
                seg_rgb_vis = seg_rgb
                if upsample_factor > 1:
                    seg_rgb_vis = F.interpolate(
                        seg_rgb.unsqueeze(0),
                        scale_factor=upsample_factor,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                overlay = seg_rgb_vis
                frame_vis = torch.zeros_like(seg_rgb_vis)

            # Decoder soft masks from softmax(alpha, dim=1) rendered as color mixture.
            # slot_maps shape is (K, Hm, Wm) and sums to 1 over K.
            decoder_soft = (slot_maps.unsqueeze(1) * palette_local[:, :, None, None]).sum(dim=0).clamp(0.0, 1.0)
            if decoder_soft.shape[-2:] != seg_rgb_vis.shape[-2:]:
                decoder_soft = F.interpolate(
                    decoder_soft.unsqueeze(0),
                    size=seg_rgb_vis.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            cur_frames.append(frame_vis)
            cur_decoder_masks.append(decoder_soft)
            cur_overlay.append(overlay)

        if len(cur_frames) == 0:
            continue

        if len(cur_frames) < fixed_frames:
            pad_n = fixed_frames - len(cur_frames)
            cur_frames.extend([cur_frames[-1].clone() for _ in range(pad_n)])
            cur_decoder_masks.extend([cur_decoder_masks[-1].clone() for _ in range(pad_n)])
            cur_overlay.extend([cur_overlay[-1].clone() for _ in range(pad_n)])

        frames_stack = torch.stack(cur_frames[:fixed_frames], dim=0)  # (8, 3, H, W)
        decoder_stack = torch.stack(cur_decoder_masks[:fixed_frames], dim=0)  # (8, 3, H, W)
        overlay_stack = torch.stack(cur_overlay[:fixed_frames], dim=0)  # (8, 3, H, W)

        if target_hw is None:
            target_hw = (frames_stack.shape[-2], frames_stack.shape[-1])

        frames_stack = _resize_video_stack(frames_stack, target_hw)
        decoder_stack = _resize_video_stack(decoder_stack, target_hw)
        overlay_stack = _resize_video_stack(overlay_stack, target_hw)

        videos_frames.append(frames_stack)
        videos_decoder_masks.append(decoder_stack)
        videos_overlay.append(overlay_stack)

    if len(videos_frames) == 0:
        return

    while len(videos_frames) < fixed_videos:
        videos_frames.append(torch.zeros_like(videos_frames[0]))
        videos_decoder_masks.append(torch.zeros_like(videos_decoder_masks[0]))
        videos_overlay.append(torch.zeros_like(videos_overlay[0]))

    frames_grid_tensor = torch.cat(videos_frames[:fixed_videos], dim=0)  # (64, 3, H, W)
    decoder_grid_tensor = torch.cat(videos_decoder_masks[:fixed_videos], dim=0)  # (64, 3, H, W)
    overlay_grid_tensor = torch.cat(videos_overlay[:fixed_videos], dim=0)  # (64, 3, H, W)

    frames_grid = make_grid(frames_grid_tensor, nrow=fixed_frames, padding=2)
    decoder_grid = make_grid(decoder_grid_tensor, nrow=fixed_frames, padding=2)
    overlay_grid = make_grid(overlay_grid_tensor, nrow=fixed_frames, padding=2)

    name = f"ep{epoch:03d}_s{global_step:07d}.png"
    save_image(frames_grid, os.path.join(frames_grid_dir, name))
    save_image(decoder_grid, os.path.join(decoder_masks_grid_dir, name))
    save_image(overlay_grid, os.path.join(overlay_grid_dir, name))

    # Extra visualization: first batch item only, grid with rows=slots and cols=frames.
    # Uses decoder soft masks (softmax over slots) as requested.
    first_b = 0
    t0 = int(video_grid_thw[first_b, 0].item())
    h0 = int(video_grid_thw[first_b, 1].item())
    w0 = int(video_grid_thw[first_b, 2].item())
    if t0 <= 0 or h0 <= 0 or w0 <= 0:
        return

    if use_pooled_output:
        h0_m = max(1, h0 // spatial_merge_size)
        w0_m = max(1, w0 // spatial_merge_size)
        n0_tokens = h0_m * w0_m
    else:
        h0_m = h0
        w0_m = w0
        n0_tokens = h0 * w0

    if n0_tokens <= 0:
        return

    sample0_masks = masks[first_b, :t0, :, :n0_tokens, 0].float()  # (T, K, N)
    if sample0_masks.numel() == 0:
        return

    frame_indices0 = _frame_indices(t0, fixed_frames)
    palette0 = palette.to(sample0_masks.device)
    first_slot_frames: list[torch.Tensor] = []

    # Align this grid to the same final size as other visualizations when possible.
    slot_target_hw = target_hw
    if slot_target_hw is None:
        slot_target_hw = (max(1, h0_m * max(1, upsample_factor)), max(1, w0_m * max(1, upsample_factor)))


    
    #sample0_masks = sample0_masks.view(sample0_masks.shape[0]* sample0_masks.shape[1], sample0_masks.shape[2])  # (T*K, N)
    sample0_masks = sample0_masks.view(sample0_masks.shape[0], sample0_masks.shape[1], h0_m, w0_m)  # (T*K, Hm, Wm)
    sample0_masks = F.interpolate(sample0_masks, size=slot_target_hw, mode="bilinear") # (T*K, H, W)
    sample0_masks = sample0_masks.view(sample0_masks.shape[0]*sample0_masks.shape[1], sample0_masks.shape[2], sample0_masks.shape[3])  # (T*K, H*W)

    sample0_masks = 1- sample0_masks
    
    grid = make_grid(sample0_masks.unsqueeze(1), nrow=frame_indices0.__len__())
    grid = (grid.cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(grid).save(os.path.join(slot_by_frame_grid_dir, name))



def _run_slot_query_pretrain(
    model,
    dataset_module,
    data_collator,
    training_args,
    custom_args: Optional[dict],
):
    custom_args = custom_args or {}

    if not training_args.do_train:
        logger.warning_rank0_once("`slot_query_pretrain` is enabled but `do_train` is False. Nothing to run.")
        return
    


    qwen3vlmodelinstance = _unwrap_model_for_hsq(model)
    slot_query_model = qwen3vlmodelinstance.hybrid_slot_query_model

    vision_dim = qwen3vlmodelinstance.vision_dim
    slot_query_model.slot_discovery.initialize_pretrain_modules(
        patch_pos_dim=qwen3vlmodelinstance.visual.pos_embed.embedding_dim,
        vision_dim=vision_dim,
        dtype=model.dtype,
    )






    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_query_model.parameters():
        p.requires_grad_(True)

    slot_query_use_pooled_output = bool(getattr(qwen3vlmodelinstance, "slot_query_use_pooled_output", True))
    slot_query_input_proj = getattr(qwen3vlmodelinstance, "slot_query_input_proj", None)
    train_slot_input_proj = (not slot_query_use_pooled_output) and (slot_query_input_proj is not None)
    if train_slot_input_proj:
        for p in slot_query_input_proj.parameters():
            p.requires_grad_(True)
        logger.info_rank0("Token-level HSQ enabled: unfreezing `slot_query_input_proj` for stable pretraining.")

    report_to = "wandb"
    if isinstance(report_to, str):
        report_to = [report_to]
    report_to = [x for x in (report_to or []) if x and x.lower() != "none"]

    accelerator = Accelerator(
        gradient_accumulation_steps=custom_args["slot_pretrain_gradient_accumulation_steps"],
        log_with=report_to if len(report_to) > 0 else None,
        
    )

    if accelerator.is_main_process and len(report_to) > 0:
        tracker_name = "slot_query_pretrain"
        accelerator.init_trackers(tracker_name, config=custom_args)

    device = accelerator.device

    params = list(slot_query_model.parameters())
    if train_slot_input_proj:
        params += list(slot_query_input_proj.parameters())
    optimizer = AdamW(
        params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    train_dataset = dataset_module.get("train_dataset")
    if train_dataset is None:
        raise ValueError("No `train_dataset` found for slot-query pretraining.")

    _num_workers = training_args.dataloader_num_workers
    _prefetch = custom_args.get("slot_pretrain_prefetch_factor", 4) if _num_workers > 0 else None
    from llamafactory.data.lerobot_bridge import lerobot_worker_init_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=custom_args.get("slot_pretrain_batch_size", 32),
        shuffle=True,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=_num_workers,
        persistent_workers=_num_workers > 0,  # keep workers + video decoders alive across epochs
        pin_memory=True,
        prefetch_factor=_prefetch,
        worker_init_fn=lerobot_worker_init_fn,
    )

    eval_dataset = dataset_module.get("eval_dataset")
    eval_loader = None
    if training_args.do_eval and eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=_num_workers,
            persistent_workers=_num_workers > 0,
            pin_memory=True,
            prefetch_factor=_prefetch,
            worker_init_fn=lerobot_worker_init_fn,
        )

    total_steps_per_epoch = max(1, (len(train_loader) + custom_args["slot_pretrain_gradient_accumulation_steps"] - 1) // custom_args["slot_pretrain_gradient_accumulation_steps"])
    if training_args.max_steps > 0:
        max_steps = training_args.max_steps
        num_epochs = 10**9
    else:
        num_epochs = int(custom_args["slot_pretrain_num_train_epochs"])
        max_steps = total_steps_per_epoch * max(1, num_epochs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=max_steps,
    )




    # import functools
    # from torch.profiler import record_function, profile, ProfilerActivity
    # from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

    # def profile_module(module, module_name):
    #     """Wraps a module's forward pass in a profiler context to avoid hook dict issues."""
    #     original_forward = module.forward
        
    #     @functools.wraps(original_forward)
    #     def forward_with_profiling(*args, **kwargs):
    #         # The context manager wraps the forward pass. 
    #         # Autograd will automatically link the corresponding backward C++ ops to this name!
    #         with record_function(f"Module: {module_name}"):
    #             return original_forward(*args, **kwargs)
                
    #     # Override the original forward method with our wrapped one
    #     module.forward = forward_with_profiling

    # def auto_tag_modules(model, prefix="", current_depth=1, target_depth=2):
    #     """Recursively iterates through submodules and patches them for profiling."""
    #     for name, submodule in model.named_children():
    #         full_name = f"{prefix}.{name}" if prefix else name
    #         has_children = len(list(submodule.named_children())) > 0
            
    #         if current_depth >= target_depth or not has_children:
    #             profile_module(submodule, full_name)
    #             # Optional: print(f"Patched: {full_name}")
    #         else:
    #             auto_tag_modules(
    #                 submodule, 
    #                 prefix=full_name, 
    #                 current_depth=current_depth + 1, 
    #                 target_depth=target_depth
    #             )

    # prof = profile(
    # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # schedule=schedule(wait=1, warmup=1, active=3, repeat=1), # Profiles steps 3, 4, and 5
    # #on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
    # record_shapes=True,
    # with_stack=True # Captures the exact line of code causing the bottleneck
    # )   

    # auto_tag_modules(slot_query_model)


    slot_query_model, optimizer, train_loader, scheduler = accelerator.prepare(
        slot_query_model,
        optimizer,
        train_loader,
        scheduler,
    )
    slot_discovery = accelerator.unwrap_model(slot_query_model).slot_discovery
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    hybrid_loss = HybridLoss(
        slot_box_weight=float(custom_args.get("slot_box_weight", 2.0)),
        query_box_weight=float(custom_args.get("query_box_weight", 2.0)),
        slot_class_weight=float(custom_args.get("slot_class_weight", 1.0)),
        temporal_weight=float(custom_args.get("slot_temporal_weight", 0.5)),
        contrastive_weight=float(custom_args.get("slot_contrastive_weight", 1.0)),
        contrastive_temperature=float(custom_args.get("slot_contrastive_temperature", 0.1)),
        contrastive_queue_size=int(custom_args.get("slot_contrastive_queue_size", 0)),
    ).to(device)

    patch_recon_weight = float(custom_args.get("slot_patch_recon_weight", 1.0))
    patch_recon_mask_ratio = float(custom_args.get("slot_patch_recon_mask_ratio", 0))
    use_supervised_slot_loss = bool(custom_args.get("slot_pretrain_use_supervised_losses", True))
    vis_masks = bool(custom_args.get("slot_vis_masks", False))
    vis_steps = int(custom_args.get("slot_vis_steps", max(1, training_args.logging_steps or 100)))
    vis_max_samples = int(custom_args.get("slot_vis_max_samples", 2))
    vis_upsample_factor = int(custom_args.get("slot_vis_upsample_factor", 8))
    vis_eval = bool(custom_args.get("slot_vis_eval", False))
    vis_overlay_alpha = float(custom_args.get("slot_vis_overlay_alpha", 0.55))
    vis_seed = int(custom_args.get("slot_vis_seed", 42))
    vis_rng = random.Random(vis_seed + accelerator.process_index)
    slot_query_use_pooled_output = bool(getattr(qwen3vlmodelinstance, "slot_query_use_pooled_output", True))

    gt_key_map = custom_args.get("slot_loss_keys", {})
    moved_key_candidates = gt_key_map.get(
        "moved",
        ["gt_moved_object_boxes", "moved_object_boxes", "moved_boxes", "bbox_track"],
    )
    start_key_candidates = gt_key_map.get("start", ["gt_start_location", "start_location", "start_box"])
    end_key_candidates = gt_key_map.get("end", ["gt_end_location", "end_location", "end_box"])
    target_key_candidates = gt_key_map.get("target", ["gt_target_box", "target_box", "target_container_box"])
    robot_key_candidates = gt_key_map.get("robot", ["gt_robot_box", "robot_box", "gripper_box"])

    logger.info_rank0("Starting Hybrid Slot Query pretraining inside SFT workflow.")
    logger.info_rank0(f"Patch reconstruction config: weight={patch_recon_weight}, mask_ratio={patch_recon_mask_ratio}.")
    if vis_masks:
        logger.info_rank0(
            f"Slot mask visualization enabled: steps={vis_steps}, max_samples={vis_max_samples}, upsample={vis_upsample_factor}, eval={vis_eval}."
        )

    model.train()
    slot_query_model.train()

    eval_steps = int(custom_args.get("slot_pretrain_eval_steps", max(1, training_args.logging_steps or 100)))
    max_eval_batches = int(custom_args.get("slot_pretrain_max_eval_batches", 0))

    def _compute_batch_losses(cur_batch: dict[str, torch.Tensor], return_visuals: bool = False):
        # Keep existing batch-to-device behavior exactly as requested.
        cur_batch = cur_batch.to(device)
        x_batched, attn_mask = _extract_visual_tokens_for_hsq(qwen3vlmodelinstance, cur_batch)
        if x_batched is None:
            return None

        hybrid_output = slot_query_model(x_batched, attention_mask=attn_mask)

        supervised_loss = torch.tensor(0.0, device=device)
        loss_dict = {
            "total": supervised_loss,
            "temporal": torch.tensor(0.0, device=device),
            "contrastive": torch.tensor(0.0, device=device),
        }
        if use_supervised_slot_loss:
            moved_gt = _get_first_available(cur_batch, moved_key_candidates)
            start_gt = _get_first_available(cur_batch, start_key_candidates)
            end_gt = _get_first_available(cur_batch, end_key_candidates)
            target_gt = _get_first_available(cur_batch, target_key_candidates)
            robot_gt = _get_first_available(cur_batch, robot_key_candidates)

            #if moved_gt is not None and start_gt is not None and end_gt is not None:
            hybrid_loss_inputs = {
                "slot_boxes": hybrid_output["slot_boxes_per_frame"],
                "query_boxes": hybrid_output["query_boxes"],
                "slot_per_frame": hybrid_output["slots_internal_per_frame"],
            }
            loss_dict = hybrid_loss(
                output=hybrid_loss_inputs,
                gt_moved_object_boxes=moved_gt,
                gt_start_location=start_gt,
                gt_end_location=end_gt,
                gt_target_box=target_gt,
                gt_robot_box=robot_gt,
            )
            supervised_loss = loss_dict["total"]
            # else:
            #     logger.warning_rank0_once(
            #         "Supervised slot/query losses requested, but GT fields are missing in batch. "
            #         "Only reconstruction loss will be used for those batches."
            #     )

        patch_pos_qwen = _build_qwen_patch_pos_emb_batched(
            qwen3vlmodelinstance=qwen3vlmodelinstance,
            video_grid_thw=cur_batch["video_grid_thw"],
            max_t=x_batched.shape[1],
            lmax=x_batched.shape[2],
            device=x_batched.device,
            dtype=x_batched.dtype,
            use_pooled_output=slot_query_use_pooled_output,
        )

        recon_out = slot_discovery.compute_batch_losses(
            visual_features=x_batched,
            attention_mask=attn_mask,
            patch_pos_qwen=patch_pos_qwen,
            supervised_loss=supervised_loss,
            patch_recon_weight=patch_recon_weight,
            patch_recon_mask_ratio=patch_recon_mask_ratio,
            hybrid_output=hybrid_output,
            return_visuals=return_visuals,
            video_grid_thw=cur_batch.get("video_grid_thw", None),
            pixel_values_videos=cur_batch.get("pixel_values_videos", None),
        )

        return {
            "total_loss": recon_out["total_loss"],
            "supervised_loss": loss_dict,
            "recon_loss": recon_out["recon_loss"],
            "vis_payload": recon_out["vis_payload"],
        }

    @torch.no_grad()
    def _run_eval_loss(cur_step: int, cur_epoch: int) -> Optional[float]:
        if eval_loader is None:
            return None

        slot_query_model.eval()

        eval_sum = torch.tensor(0.0, device=device)
        eval_count = torch.tensor(0, device=device, dtype=torch.long)

        saved_eval_vis = False
        for e_idx, e_batch in enumerate(eval_loader):
            if max_eval_batches > 0 and e_idx >= max_eval_batches:
                break

            need_eval_vis = vis_masks and vis_eval and accelerator.is_main_process and (not saved_eval_vis)
            batch_loss_out = _compute_batch_losses(e_batch, return_visuals=need_eval_vis)
            if batch_loss_out is None:
                continue

            if need_eval_vis and batch_loss_out["vis_payload"] is not None:
                _save_slot_mask_visualizations(
                    masks=batch_loss_out["vis_payload"]["masks"],
                    video_grid_thw=batch_loss_out["vis_payload"]["video_grid_thw"],
                    pixel_values_videos=batch_loss_out["vis_payload"].get("pixel_values_videos"),
                    output_dir=training_args.output_dir,
                    split="eval",
                    global_step=cur_step,
                    epoch=cur_epoch,
                    patch_size=qwen3vlmodelinstance.visual.patch_size,
                    temporal_patch_size=2,
                    in_channels=qwen3vlmodelinstance.visual.patch_embed.in_channels,
                    spatial_merge_size=qwen3vlmodelinstance.visual.spatial_merge_size,
                    use_pooled_output=slot_query_use_pooled_output,
                    max_samples=vis_max_samples,
                    upsample_factor=vis_upsample_factor,
                    overlay_alpha=vis_overlay_alpha,
                    rng=vis_rng,
                )
                saved_eval_vis = True
                accelerator.print(f"Saved slot mask visualizations for eval at step {cur_step}, epoch {cur_epoch} to {training_args.output_dir}.")

            e_total = batch_loss_out["total_loss"]
            eval_sum = eval_sum + e_total.detach()
            eval_count = eval_count + 1

        if accelerator.num_processes > 1:
            eval_sum = accelerator.reduce(eval_sum, reduction="sum")
            eval_count = accelerator.reduce(eval_count, reduction="sum")

        slot_query_model.train()

        if eval_count.item() == 0:
            return None

        return (eval_sum / eval_count).item()

    save_steps = 4000
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_main_process, desc="slot_pretrain")
    temp_patch_size = 2
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(slot_query_model):
                if global_step >= max_steps:
                    break

                should_save_train_vis = (
                    vis_masks and accelerator.is_main_process and vis_steps > 0 and ((global_step + 1) % vis_steps == 0 or 
                                                                                     global_step == 0)
                )
                batch_loss_out = _compute_batch_losses(batch, return_visuals=should_save_train_vis)
                if batch_loss_out is None:
                    logger.warning_rank0_once(
                        "Skipping slot pretraining batch because required video fields are missing: "
                        "`pixel_values_videos` and `video_grid_thw`."
                    )
                    continue

                if should_save_train_vis and batch_loss_out["vis_payload"] is not None:
                    _save_slot_mask_visualizations(
                        masks=batch_loss_out["vis_payload"]["masks"],
                        video_grid_thw=batch_loss_out["vis_payload"]["video_grid_thw"],
                        pixel_values_videos=batch_loss_out["vis_payload"].get("pixel_values_videos"),
                        output_dir=training_args.output_dir,
                        split="train",
                        global_step=global_step + 1,
                        epoch=epoch,
                        patch_size=qwen3vlmodelinstance.visual.patch_size,
                        temporal_patch_size=temp_patch_size,
                        in_channels=qwen3vlmodelinstance.visual.patch_embed.in_channels,
                        spatial_merge_size=qwen3vlmodelinstance.visual.spatial_merge_size,
                        use_pooled_output=slot_query_use_pooled_output,
                        max_samples=vis_max_samples,
                        upsample_factor=vis_upsample_factor,
                        overlay_alpha=vis_overlay_alpha,
                        rng=vis_rng,
                    )
                    accelerator.print(f"Saved slot mask visualizations for train at step {global_step}, epoch {epoch} to {training_args.output_dir}.")


                total_loss = batch_loss_out["total_loss"]
                supervised_loss = batch_loss_out["supervised_loss"]["total"]
                temporal_loss = batch_loss_out["supervised_loss"]["temporal"] if "temporal" in batch_loss_out["supervised_loss"] else torch.tensor(0.0, device=device)
                constrastive_loss = batch_loss_out["supervised_loss"]["contrastive"] if "contrastive" in batch_loss_out["supervised_loss"] else torch.tensor(0.0, device=device)
                recon_loss = batch_loss_out["recon_loss"]
                loss = total_loss




                accelerator.backward(constrastive_loss)

                log_payload = {}
                #get gradient norms for logging. seperately for slot_query_model, patch_pos_proj, and patch_decoder, pos_embed_simple
                log_payload["slot_pretrain/gradient_norm_slot_query"] = 0.0
                log_payload["slot_pretrain/gradient_norm_patch_pos_proj"] = 0.0
                log_payload["slot_pretrain/gradient_norm_patch_decoder"] = 0.0
                log_payload["slot_pretrain/gradient_norm_pos_embed_simple"] = 0.0



                #prefix is always hybrid_slot_query_model

                print("\n--- Suspicious Weight Diagnostics ---")
                for name, p in slot_query_model.named_parameters():
                    # Filter for the layers we know are behaving wildly
                    if "slot_discovery" in name and any(x in name for x in ["gru", "mlp", "predictor", "to_q", "to_k", "to_v", "patch_pos_proj", "patch_decoder"]):
                        if p.data is not None:
                            norm = p.data.norm().item()
                            max_val = p.data.max().item()
                            min_val = p.data.min().item()
                            has_nan = torch.isnan(p.data).any().item()
                            #grad norm
                            grad_norm = p.grad.norm().item() if p.grad is not None else 0.0

                            # Format nicely for easy reading
                            print(f"{name:.<50} Norm: {norm:>10.6f} | Max: {max_val:>8.6f} | Min: {min_val:>8.6f} | NaN: {has_nan} | Grad Norm: {grad_norm:>8.6f}")
                print("-------------------------------------\n")

                for name, param in slot_query_model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "patch_pos_proj" in name:
                        if param.grad is not None:
                            log_payload["slot_pretrain/gradient_norm_patch_pos_proj"] += param.grad.detach().data.norm(2).item() ** 2
                    elif "patch_decoder" in name:
                        if param.grad is not None:
                            log_payload["slot_pretrain/gradient_norm_patch_decoder"] += param.grad.detach().data.norm(2).item() ** 2
                    elif "slot_discovery" in name:
                        if param.grad is not None:
                            log_payload["slot_pretrain/gradient_norm_slot_query"] += param.grad.detach().data.norm(2).item() ** 2
                    elif train_slot_input_proj and "slot_query_input_proj" in name:
                        if param.grad is not None:
                            log_payload["slot_pretrain/gradient_norm_slot_query"] += param.grad.detach().data.norm(2).item() ** 2
                    elif "pos_embed_simple" in name:
                        if param.grad is not None:
                            log_payload["slot_pretrain/gradient_norm_pos_embed_simple"] += param.grad.detach().data.norm(2).item() ** 2

                log_payload["slot_pretrain/gradient_norm_slot_query"] = math.sqrt(log_payload["slot_pretrain/gradient_norm_slot_query"])
                log_payload["slot_pretrain/gradient_norm_patch_pos_proj"] = math.sqrt(log_payload["slot_pretrain/gradient_norm_patch_pos_proj"])
                log_payload["slot_pretrain/gradient_norm_patch_decoder"] = math.sqrt(log_payload["slot_pretrain/gradient_norm_patch_decoder"])
                log_payload["slot_pretrain/gradient_norm_pos_embed_simple"] = math.sqrt(log_payload["slot_pretrain/gradient_norm_pos_embed_simple"])

                #if (step + 1) % training_args.gradient_accumulation_steps == 0:
                if training_args.max_grad_norm and training_args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(params, training_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress_bar.update(1)

                log_payload_losses = {
                    "slot_pretrain/total_loss": total_loss.detach().item(),
                    "slot_pretrain/supervised_loss": supervised_loss.detach().item(),
                    "slot_pretrain/temporal_loss": temporal_loss.detach().item(),
                    "slot_pretrain/contrastive_loss": constrastive_loss.detach().item(),
                    "slot_pretrain/recon_loss": recon_loss.detach().item(),
                    "slot_pretrain/lr": scheduler.get_last_lr()[0],
                }

                log_payload.update(log_payload_losses)

                accelerator.log(log_payload, step=global_step)

                if accelerator.is_main_process:
                    progress_bar.set_postfix(
                        total=f"{log_payload['slot_pretrain/total_loss']:.4f}",
                        sup=f"{log_payload['slot_pretrain/supervised_loss']:.4f}",
                        rec=f"{log_payload['slot_pretrain/recon_loss']:.4f}",
                        temporal=f"{log_payload['slot_pretrain/temporal_loss']:.4f}",
                        contrastive=f"{log_payload['slot_pretrain/contrastive_loss']:.4f}",
                    )
                    #set epoch in prefix
                    epoch_frac = (step + 1) / total_steps_per_epoch
                    progress_bar.set_description(f"slot_pretrain Epoch {epoch + epoch_frac:.2f}")

                # if training_args.logging_steps > 0 and global_step % training_args.logging_steps == 0:
                #     logger.info_rank0(
                #         "slot_pretrain step=%d total_loss=%.6f supervised=%.6f recon=%.6f"
                #         % (global_step, total_loss.item(), supervised_loss.item(), recon_loss.item())
                #     )

                if eval_loader is not None and eval_steps > 0 and global_step % eval_steps == 0:
                    eval_loss = _run_eval_loss(global_step, epoch)
                    if eval_loss is not None:
                        accelerator.log({"slot_pretrain/eval_loss": eval_loss}, step=global_step)
                        logger.info_rank0("slot_pretrain eval step=%d eval_loss=%.6f" % (global_step, eval_loss))

                if save_steps > 0 and global_step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        os.makedirs(training_args.output_dir, exist_ok=True)
                        ckpt_path = os.path.join(training_args.output_dir, f"slot_pretrain_step_{global_step}.pt")
                        torch.save(
                            {
                                "step": global_step,
                                "hybrid_slot_query_model": accelerator.get_state_dict(slot_query_model),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                            },
                            ckpt_path,
                        )
                        logger.info_rank0(f"Saved slot pretraining checkpoint: {ckpt_path}")

        

        if global_step >= max_steps:
            break

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        final_path = os.path.join(training_args.output_dir, "slot_pretrain_final.pt")
        torch.save(
            {
                "step": global_step,
                "hybrid_slot_query_model": accelerator.get_state_dict(slot_query_model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            final_path,
        )
        logger.info_rank0(f"Finished slot pretraining. Final checkpoint saved at: {final_path}")

    if len(report_to) > 0:
        accelerator.end_training()


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    custom_args = None,
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)


    
    #set env var LEROBOT_DATASETS  to json.dumps({"droid": "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success", "bridge": "/e/scratch/m3/jnogga/bridge_data_v2_teleop",})

    os.environ["LEROBOT_DATASETS"] = json.dumps({
        "droid": "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success",
        "bridge": "/e/scratch/m3/jnogga/bridge_data_v2_teleop",
    })



    #add custom model args for later use (everything after custom_model_architecture if not None)

    if "custom_model_architecture" in model_args.__dict__ and model_args.custom_model_architecture is not None:
        model_arg_dict = asdict(model_args)
        custom_start_index = list(model_arg_dict.keys()).index("custom_model_architecture") + 1
        custom_model_args = {key: value for key, value in model_arg_dict.items() if list(model_arg_dict.keys()).index(key) >= custom_start_index}

        #also add custom_args if not none
        if custom_args is not None:
            custom_model_args = {**custom_model_args, **custom_args}
            
        tokenizer_module["processor"].custom_model_args = custom_model_args 

    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    ref_model = None
    if finetuning_args.use_asft_loss:
        ref_model = create_ref_model(model_args, finetuning_args)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Optional: slot-query pretraining mode integrated into SFT workflow.
    if custom_args is not None and custom_args.get("slot_query_pretrain", False):
        _run_slot_query_pretrain(
            model=model,
            dataset_module=dataset_module,
            data_collator=data_collator,
            training_args=training_args,
            custom_args=custom_args,
        )
        return

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            ref_model=ref_model,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
    
    # Add evaluation callback for bounding box tasks
    eval_tokenizer = copy.deepcopy(tokenizer)
    eval_tokenizer.padding_side = "left"  # use right-padding in evaluation
    
    
    #decode val dataset
    if training_args.predict_with_generate:
        ground_truths = []
        for example in dataset_module["eval_dataset"]:
            ground_truths.append(example["labels"])
        
        ground_truths_decoded = eval_tokenizer.batch_decode(
            ground_truths, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        if custom_args is not None:
            for evaluator in custom_args.get("evaluators", []):
                if evaluator == "bbox_evaluator":
                    trainer.add_callback(
                        BoundingBoxEvaluatorCallback(
                            trainer=trainer,
                            tokenizer=eval_tokenizer,
                            val_dataset=ground_truths_decoded,
                        )
                    )
                if evaluator == "label_evaluator":
                    trainer.add_callback(
                        LabelEvaluatorCallback(
                            trainer=trainer,
                            tokenizer=eval_tokenizer,
                            val_dataset=ground_truths_decoded,
                        )
                    )
                    # Training
                    # if training_args.do_train:
                    # metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                    # trainer.log_metrics("eval", metrics)
                    # trainer.save_metrics("eval", metrics)

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)
    
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

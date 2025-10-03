import argparse
import gc
import math
import os
from typing import Tuple

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import utils

from datasets import load_dataset
from omegaconf import OmegaConf

from PIL import Image

from showme_utils.siglip_vae.modelling_siglip_vae import SiglipModel
from torchvision.transforms import InterpolationMode

from tqdm import trange
from transformers import AutoModel


def resize_center_crop(
    img: Image.Image,
    size: Tuple[int, int],
    *,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Image.Image:
    """
    Resize a PIL image with torchvision, then center-crop to (h, w) with F.crop — no padding.

    Let (h0, w0) be original size and (h, w) be target:
      • If h/w > h0/w0 (original is wider relative to target), resize height to h, then center-crop width.
      • If h/w < h0/w0 (original is taller/narrower relative to target), resize width to w, then center-crop height.
      • If ratios equal, direct resize to (h, w).

    Args:
        img: PIL image.
        size: (h, w) target size.
        fill: Unused (kept for API parity). No padding performed.
        interpolation: torchvision InterpolationMode for resizing.

    Returns:
        PIL Image of shape (h, w).
    """
    target_h, target_w = int(size[0]), int(size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("size must be positive integers (h, w)")

    orig_w, orig_h = img.size  # PIL order: (width, height)
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Input image has non-positive dimensions")

    target_ratio = target_h / target_w
    orig_ratio = orig_h / orig_w

    # If ratios match, avoid rounding surprises by resizing directly to the target.
    if math.isclose(target_ratio, orig_ratio, rel_tol=0.0, abs_tol=1e-12):
        # pyre-ignore
        return F.resize(
            img,  # pyre-ignore
            [target_h, target_w],
            interpolation=interpolation,
            antialias=True,
        )

    # Determine scale so that the post-resize image fully contains the target crop.
    if target_ratio > orig_ratio:
        # Original is wider relative to target -> match height
        scale = target_h / orig_h
    else:
        # Original is taller/narrower relative to target -> match width
        scale = target_w / orig_w

    new_w = int(math.ceil(orig_w * scale))
    new_h = int(math.ceil(orig_h * scale))

    # Ensure resized dims are at least target to prevent out-of-bounds crops (ceil usually guarantees this)
    new_w = max(new_w, target_w)
    new_h = max(new_h, target_h)

    # pyre-ignore
    resized = F.resize(img, [new_h, new_w], interpolation=interpolation, antialias=True)

    # Center crop with torchvision F.crop(top, left, height, width)
    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    # Clamp in case of off-by-one from ceil/ints
    left = min(left, new_w - target_w)
    top = min(top, new_h - target_h)

    # pyre-ignore
    return F.crop(resized, top=top, left=left, height=target_h, width=target_w)


def image_transform(
    image,
    resolution=(512, 512),
    normalize=True,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    centercrop=False,
):
    if centercrop:
        image = resize_center_crop(image, resolution)
    else:
        image = F.resize(
            image,
            [resolution[0], resolution[1]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    tensor = transforms.ToTensor()(image)
    if normalize:
        tensor = transforms.Normalize(mean=mean, std=std, inplace=True)(tensor)

    return tensor


def create_showme_model(config_file, ckpt_path):
    model = SiglipModel.from_pretrained("google/siglip2-base-patch16-512")
    vision_model = model.vision_model
    return vision_model, model


def extract_lvm_features(config_file, lvm_model_name, dataset, args):

    save_path = utils.to_feature_filename(
        args.output_dir,
        args.dataset,
        args.subset,
        "siglip_patch_emb",
        pool=None,
        prompt=None,
        caption_idx=None,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\ndataset: \t{args.dataset}")
    print(f"subset:    \t{args.subset}")
    print(f"processing:\t{lvm_model_name}")
    print(f"save_path: \t{save_path}")

    vision_model, model = create_showme_model(
        config_file=config_file, ckpt_path=lvm_model_name
    )
    model.eval().cuda()
    vision_model.eval().cuda()
    lvm_param_count = sum([p.numel() for p in vision_model.parameters()])
    lvm_feats = []

    for i in trange(0, len(dataset), args.batch_size):
        with torch.no_grad():
            ims = torch.stack(
                [
                    image_transform(dataset[j]["image"])
                    for j in range(i, i + args.batch_size)
                ]
            ).cuda()
            vision_model_outputs = vision_model(ims)
            image_embeds = vision_model_outputs["last_hidden_state"]

            feats = image_embeds.mean(dim=1)
            feats = feats.unsqueeze(1)

            lvm_feats.append(feats.cpu())

    torch.save(
        {"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path
    )

    del vision_model, lvm_feats
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--output_dir", type=str, default="./results/features")
    args = parser.parse_args()

    # load dataset once outside
    dataset = load_dataset(args.dataset, revision=args.subset, split="train")
    extract_lvm_features(args.config_file, args.ckpt_path, dataset, args)

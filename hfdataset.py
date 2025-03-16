from transformers import CLIPTokenizer
from torch.utils.data import IterableDataset
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random
import io
import h5py
import numpy as np

MODEL_ID = "stabilityai/stable-diffusion-2-1-base"

tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_ID, subfolder="tokenizer", revision=None
)

def getEncodedPrompt(prompt):
    return tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

class CustomDataset(IterableDataset):
    def __init__(self, dataset, transform_image=None, transform_image_seg = None):
        self.dataset = dataset
        self.transform_image = transform_image
        self.transform_image_seg = transform_image_seg

    def __len__(self):
        return self.dataset.info.splits["train"].num_examples

    @property
    def num_shards(self):
        return self.dataset.num_shards

    def __tone_map_color(self, item):
        with h5py.File(io.BytesIO(item['color_data']), 'r') as f:
            rgb = f["dataset"][:].astype(np.float32)

        brightness = 0.3 * rgb[:, :, 0] + 0.59 * rgb[:, :, 1] + 0.11 * rgb[:, :, 2]

        p90 = np.percentile(brightness, 90)

        brightness_target = 0.8
        eps = 1e-4
        scale = np.power(brightness_target, 2.2) / p90 if p90 > eps else 1.0

        rgb_tone = np.power(np.maximum(scale * rgb, 0), 1/2.2)
        rgb_tone = np.clip(rgb_tone, 0, 1)
        return rgb_tone

    def __normalize_depth(self, item):
        with h5py.File(io.BytesIO(item['depth_data']), 'r') as f:
            depth = f["dataset"][:].astype(np.float32)

        min_depth = np.nanmin(depth)
        max_depth = np.nanmax(depth)
        eps = 1e-4

        if (max_depth - min_depth) < eps:
            normalized_depth = np.zeros_like(depth)
            normalized_depth[np.isnan(depth)] = np.nan
        else:
            normalized_depth = (depth - min_depth) / (max_depth - min_depth)

        # # Directly map [0,1] -> [-1,1]
        # normalized_depth = normalized_depth * 2 - 1
        return normalized_depth


    def __iter__(self):
        for item in self.dataset:
            image = self.__tone_map_color(item)
            image_depth = self.__normalize_depth(item)

            # --- Random crop of 512x512 ---
            crop_h, crop_w = 512, 512
            H, W, _ = image.shape  # Assuming image shape is (height, width, channels)
            if H >= crop_h and W >= crop_w:
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                image = image[top:top+crop_h, left:left+crop_w, :]
                image_depth = image_depth[top:top+crop_h, left:left+crop_w]
            else:
                raise ValueError("Image size is smaller than the crop size.")

            # For the image, determine invalid pixels (if any channel is NaN or Inf)
            invalid_image = np.isnan(image).any(axis=-1) | np.isinf(image).any(axis=-1)
            # Replace invalid image values with -1 (this sets all channels at that pixel to -1)
            image[invalid_image] = -1
            combined_mask_image = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
            combined_mask_image[invalid_image] = 0

            # For the depth, find invalid pixels
            invalid_depth = np.isnan(image_depth) | np.isinf(image_depth)
            image_depth[invalid_depth] = -1
            combined_mask_image[invalid_depth] = 0
         
            # Convert numpy arrays to torch tensors
            # For image: Convert from H x W x C to C x H x W
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            # For depth: Add a channel dimension to convert from H x W to 1 x H x W
            depth_tensor = torch.from_numpy(image_depth).unsqueeze(0).repeat(3, 1, 1)
            mask_tensor = torch.from_numpy(combined_mask_image)

            # Apply random horizontal flip
            if torch.rand(1) < 0.5:
                flipped_image = F.hflip(image_tensor)
                flipped_depth = F.hflip(depth_tensor)
                flipped_mask_tensor = F.hflip(mask_tensor)
            else:
                flipped_image = image_tensor
                flipped_depth = depth_tensor
                flipped_mask_tensor = mask_tensor

            # Normalize the tensors; adjust the mean and std for image channels
            normalised_image = F.normalize(flipped_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            normalised_depth = F.normalize(flipped_depth, [0.5], [0.5])

            yield {
                "image": normalised_image,
                "image_depth": normalised_depth,
                "input_ids": getEncodedPrompt(""),
                "mask": flipped_mask_tensor,
            }


def load_hf_dataset(num_processes, process_index):
    ds_shard = load_dataset("alexnasa/hypersim-depth", split="train", streaming=True).shard(num_processes, process_index)
    train_dataset = CustomDataset(ds_shard.with_format("torch"))
    
    return train_dataset

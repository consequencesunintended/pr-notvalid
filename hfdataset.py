from transformers import CLIPTokenizer
from torch.utils.data import IterableDataset
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random
from PIL import Image
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


    def __iter__(self):
        for item in self.dataset:
            # Open images and convert to RGB so that both have 3 channels.
            image = Image.open(io.BytesIO(item['color'])).convert("RGB")
            image_depth = Image.open(io.BytesIO(item['depth'])).convert("RGB")
            
            # Convert PIL images to NumPy arrays.
            image_np = np.array(image)
            depth_np = np.array(image_depth)

            # --- Random crop of 512x512 ---
            crop_h, crop_w = 512, 512
            H, W, _ = image_np.shape  # image_np has shape (height, width, channels)
            if H >= crop_h and W >= crop_w:
                top = random.randint(0, H - crop_h)
                left = random.randint(0, W - crop_w)
                image_np = image_np[top:top+crop_h, left:left+crop_w, :]
                depth_np = depth_np[top:top+crop_h, left:left+crop_w, :]
            else:
                raise ValueError("Image size is smaller than the crop size.")

            # Convert numpy arrays to torch tensors and scale to [0, 1].
            image_tensor = torch.from_numpy(image_np).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # Convert to (C, H, W)

            depth_tensor = torch.from_numpy(depth_np).float() / 255.0
            depth_tensor = depth_tensor.permute(2, 0, 1)  # Convert to (C, H, W)

            # Apply random horizontal flip.
            if torch.rand(1) < 0.5:
                image_tensor = F.hflip(image_tensor)
                depth_tensor = F.hflip(depth_tensor)

            # Normalize the tensors using mean and std of 0.5 for each of the 3 channels.
            normalized_image = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            normalized_depth = F.normalize(depth_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

            yield {
                "image": normalized_image,
                "image_depth": normalized_depth,
                "input_ids": getEncodedPrompt("")
            }



def load_hf_dataset(num_processes, process_index):
    ds_shard = load_dataset("alexnasa/hypersim-depth", split="train", streaming=True).shard(num_processes, process_index)
    train_dataset = CustomDataset(ds_shard.with_format("torch"))
    
    return train_dataset

from torch.utils.data import IterableDataset
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random
from PIL import Image
import io
import numpy as np


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

    def state_dict(self):
        return self.data.state_dict()

    def load_state_dict(self, state):
        self.data.load_state_dict(state)

    def __iter__(self):
        for item in self.dataset:
            # Open images and convert to RGB so that both have 3 channels.
            image = item['color'].convert("RGB")
            image_depth = item['depth'].convert("RGB")
            
            # Resize images so that the height is 512 while maintaining the aspect ratio.
            orig_width, orig_height = image.size  # PIL: (width, height)
            new_height = 512
            scale = new_height / orig_height
            new_width = int(orig_width * scale)
            new_width = 512
            image = image.resize((512, 512), resample=Image.BILINEAR)
            image_depth = image_depth.resize((new_width, new_height), resample=Image.BILINEAR)
            
            # Convert PIL images to NumPy arrays.
            image_np = np.array(image)
            depth_np = np.array(image_depth)
            
            # --- Random horizontal crop to 512x512 ---
            crop_h, crop_w = 512, 512
            # Since height is exactly 512, we only need to crop width.
            if new_width >= crop_w:
                left = random.randint(0, new_width - crop_w)
                image_np = image_np[:, left:left+crop_w, :]
                depth_np = depth_np[:, left:left+crop_w, :]
            else:
                raise ValueError("Resized width is smaller than the crop width.")
            
            # Convert numpy arrays to torch tensors and scale to [0, 1].
            image_tensor = torch.from_numpy(image_np).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            depth_tensor = torch.from_numpy(depth_np).float() / 255.0
            depth_tensor = depth_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Apply random horizontal flip.
            if torch.rand(1) < 0.5:
                image_tensor = F.hflip(image_tensor)
                depth_tensor = F.hflip(depth_tensor)
            
            # Normalize the tensors using mean and std of 0.5 for each channel.
            normalized_image = F.normalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            normalized_depth = F.normalize(depth_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            
            yield {
                "image": normalized_image,
                "image_depth": normalized_depth,
            }




def load_hf_dataset(num_processes, process_index):
    ds_shard = load_dataset("alexnasa/hypersim-depth", split="train", streaming=True).shard(num_processes, process_index)
    train_dataset = CustomDataset(ds_shard)
    
    return train_dataset

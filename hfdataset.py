from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import IterableDataset
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F1
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
import gc
import random
import torch.nn as nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR

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

    def __iter__(self):
        for item in self.dataset:
            image = item["image"]
            image_depth = item['conditioning_image']

            # Random horizontal flip (manually deciding based on probability)
            float_image = image.float().div(255.0)
            float_depth = image_depth.float().div(255.0)

            if torch.rand(1) < 0.5:
                flipped_image = F.hflip(float_image)
                flipped_depth = F.hflip(float_depth)
            else:
                flipped_image = float_image
                flipped_depth = float_depth

            # Normalize
            normalised_image = F.normalize(flipped_image, [0.5], [0.5])
            normalised_depth = F.normalize(flipped_depth, [0.5], [0.5])

            yield {
                "image": normalised_image,
                "image_depth" : normalised_depth,
                "input_ids": getEncodedPrompt(""),
            }



def load_hf_dataset():
    train_dataset = CustomDataset(load_dataset("wangherr/coco2017_caption_depth", split="train", streaming=True).with_format("torch"))

    return train_dataset

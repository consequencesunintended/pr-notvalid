from transformers import CLIPTokenizer
from torch.utils.data import IterableDataset
import torch
from datasets import load_dataset
import torchvision.transforms.functional as F
import random

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



def load_hf_dataset(num_processes, process_index):
    ds_shard = load_dataset("wangherr/coco2017_caption_depth", split="train", streaming=True).shard(num_processes, process_index)
    shuffled_ds_shard = ds_shard.shuffle(buffer_size=10_000, seed=random.randint(0, 1_000_000))
    train_dataset = CustomDataset(shuffled_ds_shard.with_format("torch"))
    
    return train_dataset

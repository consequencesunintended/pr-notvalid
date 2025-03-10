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
import gc
import random
import torch.nn as nn
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR
from trainer import Trainer


def main():

    trainer = Trainer()

    trainer.train()

if __name__ == "__main__":
    main()
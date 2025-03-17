from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import torch.nn.functional as F1
from torch.utils.data import DataLoader
import gc
import os
from hfdataset import load_hf_dataset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
import multiprocessing
from PIL import Image
from transformers import get_cosine_schedule_with_warmup

MODEL_ID = "stabilityai/stable-diffusion-2-1-base"

class Trainer:

    def train(self):
        multiprocessing.set_start_method("spawn", force=True)

        ddp_kwargs = DistributedDataParallelKwargs()

        # Initialize the Accelerator for distributed training
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False,
            split_batches=False,
        )

        # Initialize Accelerator
        self.accelerator = Accelerator()

        # Number of GPUs or processes
        num_gpus = self.accelerator.num_processes

        # Desired effective batch size
        desired_effective_batch_size = 128

        # Assuming each GPU gets one item per step
        local_batch_size = 1

        # Compute gradient accumulation steps dynamically
        gradient_accumulation_steps = desired_effective_batch_size // (num_gpus * local_batch_size)

        # Now, initialize your Accelerator with the computed gradient_accumulation_steps
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        print(f"Using {num_gpus} GPUs with gradient_accumulation_steps set to {gradient_accumulation_steps}")


        noise_scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID, subfolder="tokenizer", revision=None
        )
        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", revision=None
        ).to("cuda")
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID, subfolder="vae", revision=None
        ).to("cuda")
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", revision=None
        ).to("cuda")
        
        train_dataset = load_hf_dataset(self.accelerator.num_processes, self.accelerator.process_index)

        data_loader = DataLoader(train_dataset, 
                                batch_size=local_batch_size,
                                num_workers=min(train_dataset.num_shards,2),
                                pin_memory=True,
                                persistent_workers=True,)

        prompt = ""
        uncond_tokens = ""

        text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to("cuda")
        prompt_embeds = text_encoder(text_inputs, attention_mask=None)[0]

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt").input_ids.to("cuda")
        negative_prompt_embeds = text_encoder(uncond_input, attention_mask=None)[0]

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()

        if self.accelerator.is_local_main_process:
            print("Data Prepared!")

        optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-5)

        num_training_steps = 40
        num_warmup_steps = 5

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        unet, vae, optimizer, data_loader, scheduler = self.accelerator.prepare(unet, vae, optimizer, data_loader, scheduler)

        stop_flag = {"force_stop": False}

        def cycle(loader, stop_flag):
            while True:
                for batch in loader:
                    if stop_flag["force_stop"]:
                        break
                    yield batch
                if stop_flag["force_stop"]:
                    break

        self.model = unet

        for i, batch in enumerate(cycle(data_loader, stop_flag)):

            with self.accelerator.accumulate(self.model):

                with torch.no_grad():
                    image = batch["image"].to("cuda")
                    depth_image = batch["image_depth"].to("cuda")


                    x_0 = vae.encode(image).latent_dist.sample()
                    x_0 = x_0 * vae.config.scaling_factor

                    bsz = x_0.shape[0]

                    # Generate a batch of random probabilities, each in [0, 1)
                    random_prob = torch.randint(0, 2, (bsz, 1)).float().to("cuda")
                    task_emb = torch.cat([random_prob, 1 - random_prob], dim=1).float().to("cuda")
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=1)

                    # Sample a random timestep for each image
                    fixed_timestep = noise_scheduler.config.num_train_timesteps - 1  # assuming 0-indexing
                    timesteps = torch.full((bsz,), fixed_timestep, device=x_0.device, dtype=torch.long)

                    rgb_latents = x_0

                latent_model_input = rgb_latents

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                noise_pred = self.model(latent_model_input, timesteps, encoder_hidden_states, class_labels=task_emb, return_dict=False)[0]

                scalar_timestep = timesteps[0].item()
                noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device="cuda")

                predicted_annotation_latents = noise_pred
                image_reconstructed_latents = noise_scheduler.step(noise_pred, scalar_timestep, rgb_latents, return_dict=False)[0]

                predicted_annotation = vae.decode(predicted_annotation_latents / vae.config.scaling_factor, return_dict=False)[0]

                image_reconstructed = vae.decode(image_reconstructed_latents / vae.config.scaling_factor, return_dict=False)[0]

                # Compute per-pixel MSE losses without reduction
                loss_pred_per_sample = F1.mse_loss(predicted_annotation, depth_image, reduction="mean")
                loss_recon_per_sample = F1.mse_loss(image_reconstructed, image, reduction="mean")

                # Combine the losses with your weights
                weights = random_prob.squeeze()
                weighted_loss = weights * loss_pred_per_sample + (1 - weights) * loss_recon_per_sample
                loss = weighted_loss.mean()

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if self.accelerator.sync_gradients:                
                reduced_loss = self.accelerator.reduce(loss, reduction="mean")

                if self.accelerator.is_main_process:
                    print(f'loss:{reduced_loss}', flush=True)

                    predicted_np = (predicted_annotation / 2 + 0.5).clamp(0, 1)
                    image_np = predicted_np[0].float().permute(1, 2, 0).detach().cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    im = Image.fromarray(image_np)
                    output_dir = "/root/output/images"
                    os.makedirs(output_dir, exist_ok=True)
                    im.save(f'{output_dir}/my_image_{i}.png')

                    if i >= num_training_steps:  
                        checkpoint_path = "/root/output/checkpoint"
                        os.makedirs(checkpoint_path, exist_ok=True)
                        # Save the trained UNet checkpoint
                        model_to_save = self.accelerator.unwrap_model(self.model)
                        model_to_save.save_pretrained(checkpoint_path)
                        print(f"Saved UNet checkpoint to {checkpoint_path}")  
                        stop_flag["force_stop"] = True  

        print("training ended")
        raise EndOfTraining
        self.accelerator.end_training()
        

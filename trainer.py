from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import torch.nn.functional as F1
from torch.utils.data import DataLoader
import os
from hfdataset import load_hf_dataset
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
import multiprocessing
from PIL import Image
from transformers import get_cosine_schedule_with_warmup
from torch.distributed import all_gather_object, barrier

MODEL_ID = "stabilityai/stable-diffusion-2-1-base"

class Trainer:

    def __init__(self):
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
        self.local_batch_size = 1

        # Compute gradient accumulation steps dynamically
        self.gradient_accumulation_steps = desired_effective_batch_size // (num_gpus * self.local_batch_size)

        # Now, initialize your Accelerator with the computed self.gradient_accumulation_steps
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        print(f"Using {num_gpus} GPUs with self.gradient_accumulation_steps set to {self.gradient_accumulation_steps}")


        self.noise_scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", revision=None
        ).to("cuda")
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_ID, subfolder="vae", revision=None
        ).to("cuda")
        self.unet = UNet2DConditionModel .from_pretrained(
            MODEL_ID, subfolder="unet", revision=None
        ).to("cuda")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID, subfolder="tokenizer", revision=None
        )

        self.checkpoint_path = "/root/output/checkpoint"
        os.makedirs(self.checkpoint_path, exist_ok=True)

        train_dataset = load_hf_dataset(self.accelerator.num_processes, self.accelerator.process_index)

        data_loader = DataLoader(train_dataset, 
                                batch_size=self.local_batch_size,
                                num_workers=min(train_dataset.num_shards,2),
                                pin_memory=True,
                                persistent_workers=True,)

        if self.accelerator.is_local_main_process:
            print("Data Prepared!")

        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=3e-5)

        self.num_training_steps = 10000
        self.num_warmup_steps = 500

        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        self.model, self.optimizer, self.current_dataloader, self.scheduler = self.accelerator.prepare(self.unet, self.optimizer, data_loader, self.scheduler)
        self.save_per_updates = 100


    def getEncodedPrompt(self, prompt, batch_size=1):
        prompts = [prompt] * batch_size
        # now tokenizer will return a tensor of shape (batch_size, seq_len)
        return self.tokenizer(
            prompts,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _unwrap_dataset(self, ds):
        """
        Walk through Accelerate’s wrappers until we hit the real HF dataset
        that actually owns the iterator state.
        """
        while hasattr(ds, "dataset"):
            ds = ds.dataset          # peel off DataLoaderShard → IterableDatasetShard
        return ds

    def _get_dataset_state(self):
        ds = self._unwrap_dataset(self.current_dataloader.dataset)

        return ds.state_dict() if hasattr(ds, "state_dict") else None
    

    def evaluate(self, image):
        with torch.no_grad():
            
            image = image.convert("RGB")
            image = image.resize((512, 512), resample=Image.BILINEAR)

            # 1) convert and normalize
            image_np     = np.array(image)                  # (H, W, 3)
            image_tensor = torch.from_numpy(image_np).float() / 255.0

            # 2) move channels first & add batch dim
            #    from (H, W, 3) → (3, H, W) → (1, 3, H, W)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to("cuda")

            rgb_latents = self.vae.encode(image_tensor).latent_dist.sample()
            rgb_latents = rgb_latents * self.vae.config.scaling_factor

            bsz = rgb_latents.shape[0]

            # Generate a batch of random probabilities, each in [0, 1)
            task_one  = torch.tensor([[1.0, 0.0]], device="cuda")  # annotate‑depth
            task_one_emb = torch.cat([torch.sin(task_one), torch.cos(task_one)], dim=1)  # shape (1,4)

            # Sample a random timestep for each image
            fixed_timestep = self.noise_scheduler.config.num_train_timesteps - 1  # assuming 0-indexing
            timesteps = torch.full((bsz,), fixed_timestep, device=rgb_latents.device, dtype=torch.long)

            # 1) Tokenize:
            input_ids = self.getEncodedPrompt("", bsz).to(self.text_encoder.device)

            # 2) Get embeddings from the text encoder:
            text_outputs = self.text_encoder(input_ids)
            encoder_hidden_states = text_outputs.last_hidden_state  # shape (bsz, seq_len, hidden_size)

            noise_pred = self.model(rgb_latents, timesteps, encoder_hidden_states, class_labels=task_one_emb, return_dict=False)[0]

            predicted_annotation_latents = noise_pred
            predicted_annotation = self.vae.decode(predicted_annotation_latents / self.vae.config.scaling_factor, return_dict=False)[0]
            predicted_np = (predicted_annotation / 2 + 0.5).clamp(0, 1)
            image_np = predicted_np[0].float().permute(1, 2, 0).detach().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            im = Image.fromarray(image_np)

            return im

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()

        # each rank gathers its dataset state
        my_ds_state = self._get_dataset_state()
        gathered_states = [None] * self.accelerator.num_processes
        all_gather_object(gathered_states, my_ds_state)
        barrier()  # ensure gather is complete

        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                all_dataset_states=gathered_states,   # list indexed by rank
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")

    def load_checkpoint(self):
        # ————————————————————————————————————————————————————————————
        # 1) Look for the latest checkpoint file, just as before
        # ————————————————————————————————————————————————————————————
        if (
            not os.path.exists(self.checkpoint_path)
            or not any(f.endswith(".pt") for f in os.listdir(self.checkpoint_path))
        ):
            return 0

        # ensure everyone waits here so filesystem is consistent
        self.accelerator.wait_for_everyone()

        # pick “last” or highest‐numbered
        files = os.listdir(self.checkpoint_path)
        if "model_last.pt" in files:
            latest = "model_last.pt"
        else:
            cks = [f for f in files if f.startswith("model_") and f.endswith(".pt")]
            latest = sorted(cks, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]

        path = os.path.join(self.checkpoint_path, latest)

        # ————————————————————————————————————————————————————————————
        # 2) Load the merged checkpoint on every rank
        # ————————————————————————————————————————————————————————————
        checkpoint = torch.load(path, map_location="cpu")

        # ————————————————————————————————————————————————————————————
        # 3) Load model / optimizer / scheduler / EMA
        # ————————————————————————————————————————————————————————————
        # (these are identical on every rank)
        model_sd = checkpoint["model_state_dict"]
        opt_sd   = checkpoint["optimizer_state_dict"]
        sched_sd = checkpoint["scheduler_state_dict"]

        self.accelerator.unwrap_model(self.model).load_state_dict(model_sd)
        self.accelerator.unwrap_model(self.optimizer).load_state_dict(opt_sd)
        self.scheduler.load_state_dict(sched_sd)

        # ————————————————————————————————————————————————————————————
        # 4) Restore each rank’s dataset state
        # ————————————————————————————————————————————————————————————
        # we saved a list indexed by rank
        all_states = checkpoint["all_dataset_states"]
        my_state  = all_states[self.accelerator.process_index]

        ds = self._unwrap_dataset(self.current_dataloader.dataset)
        if hasattr(ds, "load_state_dict") and my_state is not None:
            ds.load_state_dict(my_state)

        # ————————————————————————————————————————————————————————————
        # 5) Return the update number for training loop
        # ————————————————————————————————————————————————————————————
        return checkpoint.get("update", 0)

    def train(self):

        start_update = self.load_checkpoint()
        global_update = start_update

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.train()

        stop_flag = {"force_stop": False}

        def cycle(loader, stop_flag):
            while True:
                for batch in loader:
                    if stop_flag["force_stop"]:
                        break
                    yield batch
                if stop_flag["force_stop"]:
                    break

        for i, batch in enumerate(cycle(self.current_dataloader, stop_flag)):

            with self.accelerator.accumulate(self.model):

                with torch.no_grad():
                    image = batch["image"].to("cuda")
                    depth_image = batch["image_depth"].to("cuda")


                    rgb_latents = self.vae.encode(image).latent_dist.sample()
                    rgb_latents = rgb_latents * self.vae.config.scaling_factor

                    bsz = rgb_latents.shape[0]

                    # Generate a batch of random probabilities, each in [0, 1)
                    random_prob = torch.randint(0, 2, (bsz, 1)).float().to("cuda")
                    task_emb = torch.cat([random_prob, 1 - random_prob], dim=1).float().to("cuda")
                    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=1)

                    # Sample a random timestep for each image
                    fixed_timestep = self.noise_scheduler.config.num_train_timesteps - 1  # assuming 0-indexing
                    timesteps = torch.full((bsz,), fixed_timestep, device=rgb_latents.device, dtype=torch.long)

                # 1) Tokenize:
                input_ids = self.getEncodedPrompt("", bsz).to(self.text_encoder.device)

                # 2) Get embeddings from the text encoder:
                text_outputs = self.text_encoder(input_ids)
                encoder_hidden_states = text_outputs.last_hidden_state  # shape (bsz, seq_len, hidden_size)
                noise_pred = self.model(rgb_latents, timesteps, encoder_hidden_states, class_labels=task_emb, return_dict=False)[0]

                scalar_timestep = timesteps[0].item()
                self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps, device="cuda")

                predicted_annotation_latents = noise_pred
                image_reconstructed_latents = self.noise_scheduler.step(noise_pred, scalar_timestep, rgb_latents, return_dict=False)[0]

                predicted_annotation = self.vae.decode(predicted_annotation_latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image_reconstructed = self.vae.decode(image_reconstructed_latents / self.vae.config.scaling_factor, return_dict=False)[0]

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
                    
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                global_update += 1

            if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                self.save_checkpoint(global_update)
                self.save_checkpoint(global_update, last=True)

            if self.accelerator.sync_gradients:                

                if self.accelerator.is_main_process:
                    print(f'loss:{loss.item()} learning_rate:{self.scheduler.get_last_lr()[0]} step:{global_update}', flush=True)

                    predicted_np = (predicted_annotation / 2 + 0.5).clamp(0, 1)
                    image_np = predicted_np[0].float().permute(1, 2, 0).detach().cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    im = Image.fromarray(image_np)
                    output_dir = "/root/output/images"
                    os.makedirs(output_dir, exist_ok=True)
                    im.save(f'{output_dir}/lotus_depth_prediction_{global_update}.png')

                    depth_image_np = (depth_image / 2 + 0.5).clamp(0, 1)
                    image_np = depth_image_np[0].float().permute(1, 2, 0).detach().cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    im = Image.fromarray(image_np)
                    output_dir = "/root/output/images"
                    os.makedirs(output_dir, exist_ok=True)
                    im.save(f'{output_dir}/lotus_depth_target_{global_update}.png')

                    if global_update >= self.num_training_steps:  
                                               
                        stop_flag["force_stop"] = True  

        print("training ended")
        self.accelerator.end_training()

        # accelerate as a subprocess currently doesn't terminate some of the process correctly
        # hence the force termination by calling an exception, it won't terminate gracefully but
        # the modal container will be stopped
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        raise Exception("Training complete")   


from modal import App, Image, method, gpu
import modal
import subprocess
import sys


# 1. Create a Modal Stub
app = App("multi-gpu-accelerator-app")
volume = modal.Volume.from_name("my-persisted-volume", create_if_missing=True)

REPOSITORY_NAME = "pr-notvalid"

# 2. Define a base image with CUDA + Python 3.10 and install dependencies
image = (
    Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "clang", "ffmpeg")
    .pip_install(
        "datasets",
        "diffusers",
        "torchvision",
        "transformers",
        "accelerate",
        "packaging",
        "ninja",
        "wheel",     
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .run_commands(
        f"git clone https://github.com/consequencesunintended/{REPOSITORY_NAME} /root/{REPOSITORY_NAME}",
        force_build=True
    )
    
)

NUM_GPUS = 8

@app.cls(
    image=image,
    gpu=gpu.A100(count=NUM_GPUS),
    cpu=2 * NUM_GPUS,
    volumes={"/root/output": volume},
    timeout=3000,
    allow_concurrent_inputs=True,
)
class TTSModel:
    def __enter__(self):
        sys.path.append("/root/{REPOSITORY_NAME}")
        return self

    @method()
    def train(self):

        # Run training script with accelerate
        command = [
            "accelerate", "launch",
            "--num_processes", f"{NUM_GPUS}",
            "--multi_gpu",
            f"/root/{REPOSITORY_NAME}/train_cli.py",
        ]

        print("Executing command:", " ".join(command))

        subprocess.run(
            command,
            stdout=sys.stdout, stderr=sys.stderr,
            check=True
        )


if __name__ == "__main__":
    with app.run():
        result = TTSModel().train()
        print(result)
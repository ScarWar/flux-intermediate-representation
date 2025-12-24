import json
import os
import time
from datetime import datetime

import torch
from fire import Fire
from transformers import pipeline

from flux.sampling import get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)

NSFW_THRESHOLD = 0.85


class HookManager:
    """Manages forward hooks to capture intermediate representations from Flux blocks."""

    def __init__(self):
        self.double_blocks_outputs = {}
        self.single_blocks_outputs = {}
        self.hooks = []
        self.current_timestep = 0

    def register_hooks(self, model):
        """Register forward hooks on all double and single blocks."""
        # Register hooks on double blocks
        for i, block in enumerate(model.double_blocks):
            block_name = f"block_{i}"

            def make_double_hook(block_idx, block_name):
                def hook(module, input, output):
                    # output is a tuple (img, txt)
                    if block_name not in self.double_blocks_outputs:
                        self.double_blocks_outputs[block_name] = []
                    # Store both img and txt tensors as tuple, move to CPU to save memory
                    img_out, txt_out = output
                    self.double_blocks_outputs[block_name].append((
                        img_out.detach().cpu(),
                        txt_out.detach().cpu(),
                    ))
                return hook

            hook_handle = block.register_forward_hook(make_double_hook(i, block_name))
            self.hooks.append(hook_handle)

        # Register hooks on single blocks
        for i, block in enumerate(model.single_blocks):
            block_name = f"block_{i}"

            def make_single_hook(block_idx, block_name):
                def hook(module, input, output):
                    # output is a single tensor
                    if block_name not in self.single_blocks_outputs:
                        self.single_blocks_outputs[block_name] = []
                    # Store tensor, move to CPU to save memory
                    self.single_blocks_outputs[block_name].append(output.detach().cpu())
                return hook

            hook_handle = block.register_forward_hook(make_single_hook(i, block_name))
            self.hooks.append(hook_handle)

    def set_timestep(self, timestep: int):
        """Set the current timestep for tracking."""
        self.current_timestep = timestep

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_captured_data(self):
        """Get all captured data organized by block type."""
        return {
            "double_blocks": self.double_blocks_outputs,
            "single_blocks": self.single_blocks_outputs,
        }


def denoise_with_capture(
    model,
    hook_manager: HookManager,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: torch.Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
):
    """Modified denoise function that captures intermediate representations at each timestep."""
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for timestep_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # Set current timestep in hook manager
        hook_manager.set_timestep(timestep_idx)
        
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        
        # Call model - hooks will capture intermediate representations
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


@torch.inference_mode()
def main(
    model: str = "flux-schnell",
    width: int = 256,
    height: int = 256,
    prompt: str = "a beautiful landscape",
    offload: bool = False,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    guidance: float = 2.5,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Generate an image using the Flux model while capturing intermediate representations
    from all double and single blocks.

    Args:
        model: Name of the model to load (default: "flux-schnell")
        width: Width of the sample in pixels (should be a multiple of 16, default: 256)
        height: Height of the sample in pixels (should be a multiple of 16, default: 256)
        prompt: Prompt used for sampling
        offload: Whether to offload models to CPU when not in use
        seed: Set a seed for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        guidance: guidance value used for guidance distillation
        output_dir: Directory to save output image and intermediate representations
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    if model not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {model}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if model == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    intermediates_dir = os.path.join(output_dir, "intermediates")
    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)

    # Load models
    print(f"Loading model: {model}")
    t5 = load_t5(torch_device, max_length=256 if model == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    flux_model = load_flow_model(model, device="cpu" if offload else torch_device)
    ae = load_ae(model, device="cpu" if offload else torch_device)

    # Initialize hook manager and register hooks
    print("Registering hooks on all blocks...")
    hook_manager = HookManager()
    hook_manager.register_hooks(flux_model)
    print(f"Registered hooks on {len(flux_model.double_blocks)} double blocks and {len(flux_model.single_blocks)} single blocks")

    # Initialize NSFW classifier
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    # Generate random seed if not provided
    rng = torch.Generator(device="cpu")
    if seed is None:
        seed = rng.seed()

    print(f"Generating with seed {seed}:\n{prompt}")
    t0 = time.perf_counter()

    # Prepare input
    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    
    if offload:
        ae = ae.cpu()
        torch.cuda.empty_cache()
        t5, clip = t5.to(torch_device), clip.to(torch_device)
    
    inp = prepare(t5, clip, x, prompt=prompt)
    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(model != "flux-schnell"))

    # Offload TEs to CPU, load model to GPU
    if offload:
        t5, clip = t5.cpu(), clip.cpu()
        torch.cuda.empty_cache()
        flux_model = flux_model.to(torch_device)

    # Denoise with capture
    print(f"Starting denoising with {num_steps} steps...")
    x = denoise_with_capture(
        flux_model,
        hook_manager,
        **inp,
        timesteps=timesteps,
        guidance=guidance,
    )

    # Remove hooks after denoising
    print("Removing hooks...")
    hook_manager.remove_hooks()

    # Offload model, load autoencoder to GPU
    if offload:
        flux_model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    # Decode latents to pixel space
    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Save image
    output_name = os.path.join(output_dir, "img_0.jpg")
    print(f"Done in {t1 - t0:.1f}s. Saving {output_name}")
    save_image(
        nsfw_classifier, model, output_name, 0, x, add_sampling_metadata, prompt, track_usage=False
    )

    # Get captured data and prepare for saving
    captured_data = hook_manager.get_captured_data()
    
    # Prepare metadata
    metadata = {
        "model": model,
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "timesteps": timesteps,
        "seed": seed,
        "guidance": guidance,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add metadata to captured data
    captured_data["metadata"] = metadata

    # Save captured intermediate representations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_filename = f"capture_{timestamp}_{model}.pt"
    capture_path = os.path.join(intermediates_dir, capture_filename)
    
    print(f"Saving captured intermediate representations to {capture_path}...")
    torch.save(captured_data, capture_path)
    print(f"Saved {len(captured_data['double_blocks'])} double blocks and {len(captured_data['single_blocks'])} single blocks")
    
    # Also save metadata as JSON for easy inspection
    metadata_filename = f"metadata_{timestamp}_{model}.json"
    metadata_path = os.path.join(intermediates_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("Done!")


if __name__ == "__main__":
    Fire(main)


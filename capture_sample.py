import hashlib
import json
import os
import time
from datetime import datetime

import torch
from fire import Fire
from PIL import Image
from tqdm import tqdm

from flux.modules.layers import timestep_embedding
from flux.sampling import get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)


def generate_short_id(seed: int | None = None, prompt: str = "") -> str:
    """
    Generate a short unique ID for each generation.
    
    Args:
        seed: Random seed used for generation
        prompt: Prompt text
        
    Returns:
        Short 8-character hexadecimal ID
    """
    # Combine timestamp, seed, and prompt for uniqueness
    timestamp = datetime.now().isoformat()
    combined = f"{timestamp}_{seed}_{prompt}"
    # Generate hash and take first 8 characters
    hash_obj = hashlib.md5(combined.encode())
    short_id = hash_obj.hexdigest()[:8]
    return short_id


def get_multi_gpu_device_map(num_gpus: int = 8) -> dict:
    """
    Get the device mapping for multi-GPU distribution.
    
    Strategy (for 8 GPUs):
    - GPU 0: T5 encoder
    - GPU 1: CLIP encoder  
    - GPU 2: Flux model (main model, largest)
    - GPU 5: Autoencoder encoder
    - GPU 6: Autoencoder decoder
    - GPU 7: Reserved for intermediate operations
    
    Args:
        num_gpus: Number of available GPUs
        
    Returns:
        Dictionary mapping model components to their target devices
    """
    device_map = {}
    
    # T5 on GPU 0
    device_map["t5"] = torch.device("cuda:0")
    
    # CLIP on GPU 1
    device_map["clip"] = torch.device("cuda:1")
    
    # Flux model on GPU 2
    device_map["flux"] = torch.device("cuda:2")
    
    # Autoencoder: encoder and decoder on separate GPUs if available
    if num_gpus >= 6:
        device_map["ae_encoder"] = torch.device("cuda:5")
        device_map["ae_decoder"] = torch.device("cuda:6")
    else:
        # Fewer GPUs: put AE on last available GPU
        device_map["ae"] = torch.device(f"cuda:{min(3, num_gpus - 1)}")
    
    return device_map


def distribute_models_multi_gpu(
    t5, clip, flux_model, ae, num_gpus: int = 8
) -> dict:
    """
    Distribute models across multiple GPUs for better utilization.
    
    Strategy (for 8 GPUs):
    - GPU 0: T5 encoder
    - GPU 1: CLIP encoder  
    - GPU 2-4: Flux model (main model, largest)
    - GPU 5: Autoencoder encoder
    - GPU 6: Autoencoder decoder
    - GPU 7: Reserved for intermediate operations
    
    Args:
        t5: T5 model
        clip: CLIP model
        flux_model: Flux model
        ae: Autoencoder model
        num_gpus: Number of available GPUs
        
    Returns:
        Dictionary mapping model components to their devices
    """
    if num_gpus < 2:
        raise ValueError("Need at least 2 GPUs for multi-GPU distribution")
    
    device_map = {}
    
    # T5 on GPU 0
    t5_device = torch.device("cuda:0")
    t5 = t5.to(t5_device)
    device_map["t5"] = t5_device
    print(f"T5 → {t5_device}")
    
    # CLIP on GPU 1
    clip_device = torch.device("cuda:1")
    clip = clip.to(clip_device)
    device_map["clip"] = clip_device
    print(f"CLIP → {clip_device}")
    
    # Flux model on GPU 2 (largest model, keep on one GPU for simplicity)
    # Note: For true model parallelism, you'd need to split blocks across GPUs
    # which requires custom forward pass handling
    flux_device = torch.device("cuda:2")
    flux_model = flux_model.to(flux_device)
    device_map["flux"] = flux_device
    print(f"Flux Model → {flux_device}")
    
    # Autoencoder: encoder and decoder on separate GPUs if available
    if num_gpus >= 6:
        ae_encoder_gpu = 5
        ae_decoder_gpu = 6
        if hasattr(ae, "encoder"):
            ae.encoder.to(torch.device(f"cuda:{ae_encoder_gpu}"))
            device_map["ae_encoder"] = torch.device(f"cuda:{ae_encoder_gpu}")
            print(f"AE Encoder → cuda:{ae_encoder_gpu}")
        
        if hasattr(ae, "decoder"):
            ae.decoder.to(torch.device(f"cuda:{ae_decoder_gpu}"))
            device_map["ae_decoder"] = torch.device(f"cuda:{ae_decoder_gpu}")
            print(f"AE Decoder → cuda:{ae_decoder_gpu}")
        else:
            # If decoder is not a separate attribute, move entire AE
            ae.to(torch.device(f"cuda:{ae_decoder_gpu}"))
            device_map["ae"] = torch.device(f"cuda:{ae_decoder_gpu}")
            print(f"AE → cuda:{ae_decoder_gpu}")
    else:
        # Fewer GPUs: put AE on last available GPU
        ae_device = torch.device(f"cuda:{min(3, num_gpus - 1)}")
        ae = ae.to(ae_device)
        device_map["ae"] = ae_device
        print(f"AE → {ae_device}")
    
    return device_map


class HookManager:
    """Manages forward hooks to capture intermediate representations from Flux blocks."""

    def __init__(self):
        self.double_blocks_outputs = {}
        self.single_blocks_outputs = {}
        self.hooks = []
        self.current_timestep = 0
        # Denoising context needed for decode_intermediates
        self.denoising_context = {
            "y": [],              # CLIP embedding (y) per timestep
            "txt_seq_len": [],    # txt sequence length per timestep
            "img_state": [],      # img tensor before model call per timestep
            "timestep_pairs": [], # (t_curr, t_prev) pairs
            "guidance": None,     # guidance value (same for all timesteps)
        }

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

    def store_context(
        self,
        y: torch.Tensor,
        txt_seq_len: int,
        img_state: torch.Tensor,
        t_curr: float,
        t_prev: float,
        guidance: float,
    ):
        """Store denoising context for the current timestep."""
        self.denoising_context["y"].append(y.detach().cpu())
        self.denoising_context["txt_seq_len"].append(txt_seq_len)
        self.denoising_context["img_state"].append(img_state.detach().cpu())
        self.denoising_context["timestep_pairs"].append((t_curr, t_prev))
        self.denoising_context["guidance"] = guidance  # Same for all timesteps

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
            "denoising_context": self.denoising_context,
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
        
        # Store denoising context for decode_intermediates
        # We store the CLIP embedding (y), txt sequence length, and img state
        hook_manager.store_context(
            y=vec,  # vec parameter is the CLIP embedding (y)
            txt_seq_len=txt.shape[1],
            img_state=img,
            t_curr=t_curr,
            t_prev=t_prev,
            guidance=guidance,
        )
        
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


def decode_intermediates(
    captured_data: dict,
    flux_model,
    ae,
    height: int,
    width: int,
    output_dir: str,
    device: torch.device,
    device_map: dict | None = None,
):
    """
    Decode intermediate representations by treating each block output as the final prediction.
    
    For each block at each timestep:
    1. Pass the block output through final_layer to get a prediction
    2. Apply the denoising step: result = img_state + (t_prev - t_curr) * pred
    3. Unpack and decode with autoencoder
    4. Save as separate image files
    
    Args:
        captured_data: Dictionary containing captured block outputs and denoising context
        flux_model: The Flux model (needed for final_layer)
        ae: The autoencoder for decoding
        height: Image height in pixels
        width: Image width in pixels
        output_dir: Directory to save decoded images
        device: Torch device to use
    """
    double_blocks = captured_data["double_blocks"]
    single_blocks = captured_data["single_blocks"]
    context = captured_data["denoising_context"]
    
    # Determine devices for multi-GPU mode
    flux_device = device_map["flux"] if device_map else device
    ae_device = device_map.get("ae_decoder") or device_map.get("ae") if device_map else device
    
    # Create output directory for decoded images
    decoded_dir = os.path.join(output_dir, "intermediates", "decoded")
    os.makedirs(decoded_dir, exist_ok=True)
    
    num_timesteps = len(context["timestep_pairs"])
    num_double_blocks = len(double_blocks)
    num_single_blocks = len(single_blocks)
    
    print(f"Decoding intermediates: {num_double_blocks} double blocks, {num_single_blocks} single blocks, {num_timesteps} timesteps")
    
    # Get guidance value
    guidance = context["guidance"]
    
    # Process each timestep
    for timestep_idx in range(num_timesteps):
        # Create step directory
        step_dir = os.path.join(decoded_dir, f"step_{timestep_idx}")
        os.makedirs(step_dir, exist_ok=True)
        t_curr, t_prev = context["timestep_pairs"][timestep_idx]
        y = context["y"][timestep_idx].to(flux_device)
        txt_seq_len = context["txt_seq_len"][timestep_idx]
        img_state = context["img_state"][timestep_idx].to(flux_device)
        
        # Compute the internal vec (conditioning) that final_layer expects
        # vec = time_in(timestep_embedding) + guidance_in(guidance_embedding) + vector_in(y)
        t_vec = torch.full((1,), t_curr, dtype=torch.bfloat16, device=flux_device)
        guidance_vec = torch.full((1,), guidance, dtype=torch.bfloat16, device=flux_device)
        
        vec = flux_model.time_in(timestep_embedding(t_vec, 256))
        if flux_model.params.guidance_embed:
            vec = vec + flux_model.guidance_in(timestep_embedding(guidance_vec, 256))
        vec = vec + flux_model.vector_in(y)
        
        # Process double blocks
        for block_idx in range(num_double_blocks):
            block_name = f"block_{block_idx}"
            # Get img tensor from captured tuple (ignore txt)
            img_out, _ = double_blocks[block_name][timestep_idx]
            img_out = img_out.to(flux_device)
            
            # Pass through final_layer to get prediction
            pred = flux_model.final_layer(img_out, vec)
            
            # Apply denoising step
            result = img_state + (t_prev - t_curr) * pred
            
            # Unpack and decode - move to AE device
            result_unpacked = unpack(result.float(), height, width)
            result_unpacked = result_unpacked.to(ae_device)
            with torch.autocast(device_type=ae_device.type, dtype=torch.bfloat16):
                decoded = ae.decode(result_unpacked)
            
            # Save image
            decoded = decoded.clamp(-1, 1)
            decoded_img = (decoded[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().byte().numpy()
            img_pil = Image.fromarray(decoded_img)
            save_path = os.path.join(step_dir, f"double_block_{block_idx}.jpg")
            img_pil.save(save_path, quality=95)
        
        # Process single blocks
        for block_idx in range(num_single_blocks):
            block_name = f"block_{block_idx}"
            # Get output tensor
            output = single_blocks[block_name][timestep_idx].to(flux_device)
            
            # Slice off txt portion (like model.py line 116)
            img_out = output[:, txt_seq_len:, ...]
            
            # Pass through final_layer to get prediction
            pred = flux_model.final_layer(img_out, vec)
            
            # Apply denoising step
            result = img_state + (t_prev - t_curr) * pred
            
            # Unpack and decode - move to AE device
            result_unpacked = unpack(result.float(), height, width)
            result_unpacked = result_unpacked.to(ae_device)
            with torch.autocast(device_type=ae_device.type, dtype=torch.bfloat16):
                decoded = ae.decode(result_unpacked)
            
            # Save image
            decoded = decoded.clamp(-1, 1)
            decoded_img = (decoded[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().byte().numpy()
            img_pil = Image.fromarray(decoded_img)
            save_path = os.path.join(step_dir, f"single_block_{block_idx}.jpg")
            img_pil.save(save_path, quality=95)
    
    total_images = num_timesteps * (num_double_blocks + num_single_blocks)
    print(f"Saved {total_images} decoded intermediate images to {decoded_dir}")


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
    should_decode_intermediates: bool = True,
    multi_gpu: bool = False,
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
        should_decode_intermediates: Whether to decode intermediate representations to images (default: True)
        multi_gpu: If True, distribute models across multiple GPUs for better utilization (default: False)
    """
    if model not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {model}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if model == "flux-schnell" else 50
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if multi_gpu and num_gpus < 2:
        print(f"Warning: multi_gpu=True but only {num_gpus} GPU(s) available. Disabling multi-GPU mode.")
        multi_gpu = False

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    # Generate unique ID for this generation
    # We'll generate seed first if not provided, then create ID
    rng = torch.Generator(device="cpu")
    if seed is None:
        seed = rng.seed()
    
    gen_id = generate_short_id(seed=seed, prompt=prompt)
    print(f"Generation ID: {gen_id}")
    
    # Create output directories with unique ID
    gen_output_dir = os.path.join(output_dir, gen_id)
    if not os.path.exists(gen_output_dir):
        os.makedirs(gen_output_dir)
    intermediates_dir = os.path.join(gen_output_dir, "intermediates")
    if not os.path.exists(intermediates_dir):
        os.makedirs(intermediates_dir)

    # Determine device mapping for multi-GPU mode
    device_map = None
    if multi_gpu and num_gpus >= 2:
        device_map = get_multi_gpu_device_map(num_gpus)
        print(f"\nMulti-GPU mode: Distributing models across {num_gpus} GPUs")
        print(f"  T5 → {device_map['t5']}")
        print(f"  CLIP → {device_map['clip']}")
        print(f"  Flux Model → {device_map['flux']}")
        if "ae_encoder" in device_map:
            print(f"  AE Encoder → {device_map['ae_encoder']}")
            print(f"  AE Decoder → {device_map['ae_decoder']}")
        else:
            print(f"  AE → {device_map['ae']}")
        print()
    
    # Load models with progress bars
    # In multi-GPU mode, load directly to target GPUs
    # In single-GPU mode, load to target device (or CPU if offload)
    print(f"Loading model: {model}")
    
    if multi_gpu and num_gpus >= 2:
        # Load directly to target GPUs
        # For AE, load to decoder device (or main AE device), then we'll split encoder/decoder
        ae_target_device = device_map.get("ae_decoder") or device_map.get("ae") or device_map["flux"]
        models_to_load = [
            ("T5", lambda: load_t5(device_map["t5"], max_length=256 if model == "flux-schnell" else 512)),
            ("CLIP", lambda: load_clip(device_map["clip"])),
            ("Flux Model", lambda: load_flow_model(model, device=device_map["flux"])),
            ("Autoencoder", lambda: load_ae(model, device=ae_target_device)),
        ]
    else:
        # Single GPU mode: load to CPU if offload, otherwise to target device
        load_device = "cpu" if offload else torch_device
        models_to_load = [
            ("T5", lambda: load_t5(load_device, max_length=256 if model == "flux-schnell" else 512)),
            ("CLIP", lambda: load_clip(load_device)),
            ("Flux Model", lambda: load_flow_model(model, device=load_device)),
            ("Autoencoder", lambda: load_ae(model, device=load_device)),
        ]
    
    t5, clip, flux_model, ae = None, None, None, None
    with tqdm(total=len(models_to_load), desc="Loading models", unit="model") as pbar:
        for model_name, load_func in models_to_load:
            pbar.set_description(f"Loading {model_name}")
            if model_name == "T5":
                t5 = load_func()
            elif model_name == "CLIP":
                clip = load_func()
            elif model_name == "Flux Model":
                flux_model = load_func()
            elif model_name == "Autoencoder":
                ae = load_func()
            pbar.update(1)
    
    # For multi-GPU mode, distribute AE encoder/decoder if needed
    if multi_gpu and num_gpus >= 2 and "ae_encoder" in device_map:
        print("Distributing Autoencoder components...")
        if hasattr(ae, "encoder"):
            ae.encoder.to(device_map["ae_encoder"])
            print(f"  AE Encoder → {device_map['ae_encoder']}")
        if hasattr(ae, "decoder"):
            ae.decoder.to(device_map["ae_decoder"])
            print(f"  AE Decoder → {device_map['ae_decoder']}")
        print()
    elif not multi_gpu and not offload:
        # Single GPU mode: move all models to the specified device
        t5 = t5.to(torch_device)
        clip = clip.to(torch_device)
        flux_model = flux_model.to(torch_device)
        ae = ae.to(torch_device)

    # Initialize hook manager and register hooks
    print("Registering hooks on all blocks...")
    hook_manager = HookManager()
    hook_manager.register_hooks(flux_model)
    print(f"Registered hooks on {len(flux_model.double_blocks)} double blocks and {len(flux_model.single_blocks)} single blocks")

    print(f"Generating with seed {seed}:\n{prompt}")
    t0 = time.perf_counter()

    # Prepare input
    # In multi-GPU mode, use the flux device for initial tensors
    input_device = device_map["flux"] if device_map else torch_device
    x = get_noise(
        1,
        height,
        width,
        device=input_device,
        dtype=torch.bfloat16,
        seed=seed,
    )
    
    if offload and not multi_gpu:
        with tqdm(total=2, desc="Managing model memory", unit="step") as pbar:
            pbar.set_description("Moving AE to CPU")
            ae = ae.cpu()
            torch.cuda.empty_cache()
            pbar.update(1)
            
            pbar.set_description("Moving T5 and CLIP to VRAM")
            t5_device = device_map["t5"] if device_map else torch_device
            clip_device = device_map["clip"] if device_map else torch_device
            t5, clip = t5.to(t5_device), clip.to(clip_device)
            pbar.update(1)
    
    # Prepare input - use appropriate devices for T5 and CLIP
    # In multi-GPU mode, prepare will handle device placement automatically
    # but we need to ensure x is on the right device for the flux model
    if device_map:
        # Move x to flux device if needed
        flux_device = device_map["flux"]
        if x.device != flux_device:
            x = x.to(flux_device)
    
    inp = prepare(t5, clip, x, prompt=prompt)
    
    # In multi-GPU mode, move prepared inputs to flux device
    if device_map:
        flux_device = device_map["flux"]
        inp["img"] = inp["img"].to(flux_device)
        inp["img_ids"] = inp["img_ids"].to(flux_device)
        inp["txt"] = inp["txt"].to(flux_device)
        inp["txt_ids"] = inp["txt_ids"].to(flux_device)
        inp["vec"] = inp["vec"].to(flux_device)
    timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(model != "flux-schnell"))

    # Offload TEs to CPU, load model to GPU
    if offload and not multi_gpu:
        with tqdm(total=2, desc="Managing model memory", unit="step") as pbar:
            pbar.set_description("Moving T5 and CLIP to CPU")
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            pbar.update(1)
            
            pbar.set_description("Moving Flux Model to VRAM")
            flux_device = device_map["flux"] if device_map else torch_device
            # Note: In multi-GPU mode, model is already on its device
            if not multi_gpu:
                flux_model = flux_model.to(torch_device)
            pbar.update(1)

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
    if offload and not multi_gpu:
        with tqdm(total=2, desc="Managing model memory", unit="step") as pbar:
            pbar.set_description("Moving Flux Model to CPU")
            flux_model.cpu()
            torch.cuda.empty_cache()
            pbar.update(1)
            
            pbar.set_description("Moving AE decoder to VRAM")
            ae_device = device_map["ae_decoder"] if device_map and "ae_decoder" in device_map else (device_map["ae"] if device_map else x.device)
            if hasattr(ae, "decoder"):
                ae.decoder.to(ae_device)
            else:
                ae.to(ae_device)
            pbar.update(1)

    # Decode latents to pixel space
    # In multi-GPU mode, move x to AE device
    if device_map:
        ae_device = device_map.get("ae_decoder") or device_map.get("ae")
        if ae_device:
            x = x.to(ae_device)
    
    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
        x = ae.decode(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Save image
    output_name = os.path.join(gen_output_dir, "img_0.jpg")
    print(f"Done in {t1 - t0:.1f}s. Saving {output_name}")
    save_image(
        None, model, output_name, 0, x, add_sampling_metadata, prompt, track_usage=False
    )

    # Get captured data and prepare for saving
    captured_data = hook_manager.get_captured_data()
    
    # Prepare metadata
    metadata = {
        "generation_id": gen_id,
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

    # Decode intermediate representations to images
    if should_decode_intermediates:
        print("\nDecoding intermediate representations...")
        # Ensure model components are on the right device
        if offload and not multi_gpu:
            with tqdm(total=1, desc="Moving Flux Model to VRAM", unit="model") as pbar:
                # Move flux_model back to GPU for final_layer access
                flux_model = flux_model.to(torch_device)
                pbar.update(1)
            # ae should already have decoder on GPU from above
        elif multi_gpu and device_map:
            # In multi-GPU mode, ensure models are on their assigned devices
            # Model should already be on the right device from loading, but verify
            flux_device = device_map["flux"]
            # Check device of first parameter to verify model location
            first_param_device = next(flux_model.parameters()).device
            if first_param_device != flux_device:
                flux_model = flux_model.to(flux_device)
        decode_intermediates(
            captured_data=captured_data,
            flux_model=flux_model,
            ae=ae,
            height=height,
            width=width,
            output_dir=gen_output_dir,
            device=torch_device,
            device_map=device_map,
        )
        if offload:
            flux_model.cpu()
            torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    Fire(main)


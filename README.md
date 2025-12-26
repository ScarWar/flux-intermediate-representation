# FLUX Intermediate Representation Research

Research project for capturing and analyzing intermediate representations from FLUX diffusion models.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Initialize the project

```bash
# Navigate to the project directory
cd research/flux-intermediate-representation

# Sync dependencies (creates .venv and installs all packages)
uv sync

# For analysis tools (matplotlib, pandas, etc.)
uv sync --extra analysis
```

## Usage

### Capture Intermediate Representations

```bash
# Run with default settings (flux-schnell, 256x256)
uv run python capture_sample.py --prompt "a beautiful landscape"

# Run with custom settings
uv run python capture_sample.py \
    --model flux-schnell \
    --width 512 \
    --height 512 \
    --prompt "a photo of a cat" \
    --seed 42

# With memory offloading (for lower VRAM)
uv run python capture_sample.py --prompt "your prompt" --offload

# Multi-GPU mode (distributes models across multiple GPUs for better utilization)
uv run python capture_sample.py \
    --prompt "a beautiful landscape" \
    --multi_gpu True
```

### Command Line Options

- `--model`: Model name (default: `flux-schnell`)
- `--width`, `--height`: Image dimensions (must be multiples of 16, default: 256)
- `--prompt`: Text prompt for image generation
- `--seed`: Random seed for reproducibility (optional)
- `--offload`: Enable CPU offloading to save VRAM (default: `False`)
- `--multi_gpu`: Distribute models across multiple GPUs (default: `False`)
- `--num_steps`: Number of denoising steps (default: 4 for schnell, 50 for dev)
- `--guidance`: Guidance value for guidance-distilled models (default: 2.5)
- `--output_dir`: Output directory (default: `output`)
- `--should_decode_intermediates`: Decode intermediate representations to images (default: `True`)
- `--add_sampling_metadata`: Add prompt to image EXIF metadata (default: `True`)

### Available Models

- `flux-schnell` (default) - Fast model, 4 steps
- `flux-dev` - Development model, 50 steps
- `flux-dev-krea` - KREA fine-tuned model

### Multi-GPU Support

When running on nodes with multiple GPUs, you can enable multi-GPU mode to distribute models across different GPUs:

**Distribution Strategy (for 8 GPUs):**
- GPU 0: T5 encoder
- GPU 1: CLIP encoder
- GPU 2: Flux model (main model)
- GPU 5: Autoencoder encoder
- GPU 6: Autoencoder decoder

This allows better GPU utilization by running different model components in parallel. Models are loaded directly to their target GPUs, avoiding CPU intermediate steps.

**Note:** Multi-GPU mode and offload mode are mutually exclusive. Use `--multi_gpu True` for multi-GPU setups and `--offload` for single-GPU systems with limited VRAM.

### Output

Each generation gets a unique 8-character ID based on timestamp, seed, and prompt. All outputs are organized in a directory named with this ID:

```
output/
  └── {generation_id}/          # e.g., a1b2c3d4
      ├── img_0.jpg             # Generated image
      └── intermediates/
          ├── capture_{timestamp}_{model}.pt      # Raw intermediate tensors
          ├── metadata_{timestamp}_{model}.json   # Generation metadata
          └── decoded/                            # Decoded intermediate images
              ├── step_0/
              │   ├── double_block_0.jpg
              │   ├── double_block_1.jpg
              │   ├── ...
              │   ├── single_block_0.jpg
              │   └── ...
              ├── step_1/
              └── ...
```

The generation ID is printed at the start of each run, making it easy to identify and organize multiple generations.

### Decoding Intermediate Representations

By default, the script decodes each block's output as if it were the final prediction. This allows you to visualize what information is present at different depths of the network.

```bash
# With intermediate decoding (default)
uv run python capture_sample.py --prompt "a beautiful landscape"

# Disable intermediate decoding (faster, saves only raw tensors)
uv run python capture_sample.py --prompt "a beautiful landscape" --should_decode_intermediates=False
```

The decoded images are organized by timestep:
- `step_{timestep}/double_block_{block_idx}.jpg` - Output from double stream blocks (19 blocks)
- `step_{timestep}/single_block_{block_idx}.jpg` - Output from single stream blocks (38 blocks)

Each image shows what the model would generate if that block's output was treated as the final prediction, enabling research into which layers capture different aspects of the image. Images are grouped by timestep for easier comparison across the denoising process.

### Creating Visualizations

The `create_visualization.py` utility can generate videos or GIFs from decoded intermediate images, showing the denoising process with text overlays indicating step ID, block type, and block ID.

```bash
# Create a GIF with all blocks (default)
uv run python create_visualization.py --generation_id a1b2c3d4

# Create an MP4 video
uv run python create_visualization.py \
    --generation_id a1b2c3d4 \
    --output_format mp4

# Only show double blocks
uv run python create_visualization.py \
    --generation_id a1b2c3d4 \
    --block_type double

# Only show single blocks
uv run python create_visualization.py \
    --generation_id a1b2c3d4 \
    --block_type single

# Limit to first 5 blocks per step
uv run python create_visualization.py \
    --generation_id a1b2c3d4 \
    --max_blocks 5

# Custom FPS and font size
uv run python create_visualization.py \
    --generation_id a1b2c3d4 \
    --fps 4.0 \
    --font_size 32
```

**Visualization Options:**
- `--generation_id`: Unique generation ID (required)
- `--output_dir`: Base output directory (default: `"output"`)
- `--output_format`: `"gif"` or `"mp4"` (default: `"gif"`)
- `--fps`: Frames per second (default: `2.0`)
- `--block_type`: Filter by `"double"`, `"single"`, or `None` for all
- `--max_blocks`: Maximum number of blocks per step to include
- `--font_size`: Font size for text overlay (default: `24`)

The visualization will be saved as `visualization_{generation_id}.gif` or `.mp4` in the generation directory. Each frame displays:
- **Step ID**: Current denoising step
- **Block Type**: `double` or `single`
- **Block ID**: Index of the block within its type

This makes it easy to visualize how the image evolves through different layers and timesteps of the denoising process.

### Loading Captured Data

```python
import torch

# Load captured intermediate representations
# Note: Files are now organized by generation ID
data = torch.load("output/{generation_id}/intermediates/capture_YYYYMMDD_HHMMSS_flux-schnell.pt")

# Access double block outputs (19 blocks for standard FLUX)
# Each block has a list of (img, txt) tuples, one per timestep
double_blocks = data["double_blocks"]
block_0_outputs = double_blocks["block_0"]  # List of tuples

# Access single block outputs (38 blocks for standard FLUX)
# Each block has a list of tensors, one per timestep
single_blocks = data["single_blocks"]
block_0_outputs = single_blocks["block_0"]  # List of tensors

# Access denoising context (needed for custom decoding)
context = data["denoising_context"]
# context["y"] - CLIP embeddings per timestep
# context["img_state"] - Image state before each model call
# context["timestep_pairs"] - (t_curr, t_prev) for each step
# context["txt_seq_len"] - Text sequence length per timestep
# context["guidance"] - Guidance value used

# Access metadata
metadata = data["metadata"]
print(f"Generation ID: {metadata['generation_id']}")
print(f"Model: {metadata['model']}")
print(f"Prompt: {metadata['prompt']}")
print(f"Steps: {metadata['num_steps']}")
```

### Features

- **Progress Bars**: Visual progress indicators during model loading and VRAM transfers
- **Multi-GPU Support**: Automatic distribution of models across multiple GPUs for better utilization
- **Unique Generation IDs**: Each generation gets a unique ID for easy organization
- **Organized Output**: Files organized by generation ID and timestep for easy navigation
- **Memory Management**: CPU offloading support for systems with limited VRAM
- **Visualization Tools**: Create videos/GIFs from decoded intermediate representations with text overlays

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linting
uv run ruff check .

# Format code
uv run ruff format .
```

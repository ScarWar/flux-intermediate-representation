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
```

### Available Models

- `flux-schnell` (default) - Fast model, 4 steps
- `flux-dev` - Development model, 50 steps
- `flux-dev-krea` - KREA fine-tuned model

### Output

The script saves:
- Generated image: `output/img_0.jpg`
- Intermediate representations: `output/intermediates/capture_{timestamp}_{model}.pt`
- Metadata JSON: `output/intermediates/metadata_{timestamp}_{model}.json`

### Loading Captured Data

```python
import torch

# Load captured intermediate representations
data = torch.load("output/intermediates/capture_YYYYMMDD_HHMMSS_flux-schnell.pt")

# Access double block outputs (19 blocks for standard FLUX)
# Each block has a list of (img, txt) tuples, one per timestep
double_blocks = data["double_blocks"]
block_0_outputs = double_blocks["block_0"]  # List of tuples

# Access single block outputs (38 blocks for standard FLUX)
# Each block has a list of tensors, one per timestep
single_blocks = data["single_blocks"]
block_0_outputs = single_blocks["block_0"]  # List of tensors

# Access metadata
metadata = data["metadata"]
print(f"Model: {metadata['model']}")
print(f"Prompt: {metadata['prompt']}")
print(f"Steps: {metadata['num_steps']}")
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linting
uv run ruff check .

# Format code
uv run ruff format .
```

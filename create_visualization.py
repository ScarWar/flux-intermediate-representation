"""
Utility script to create video or GIF visualizations from decoded intermediate representations.

Given a generation ID, this script will:
1. Load all decoded intermediate images
2. Add text overlays showing step ID, block type, and block ID
3. Create a video (MP4) or GIF showing the denoising process
"""

import os
import re
from pathlib import Path
from typing import Literal

import imageio
from fire import Fire
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Constants
DEFAULT_FONT_SIZE = 24
DEFAULT_FPS = 2.0
FONT_SIZE_RATIO = 0.03  # 3% of image size
MIN_FONT_SIZE = 12
MIN_FONT_SIZE_FALLBACK = 10
MARGIN_RATIO = 0.01  # 1% of image size for margin
PADDING_RATIO = 0.2  # 20% of font size for padding
BG_ALPHA = 200

# Common font paths
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

OutputFormat = Literal["gif", "mp4"]
BlockType = Literal["double", "single"]
ImageInfo = tuple[int, str, int, Path]  # (step_id, block_type, block_id, file_path)


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to find a suitable font for text overlay.
    
    Args:
        size: Font size in pixels
        
    Returns:
        PIL Font object
    """
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
    
    # Fallback to default font
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def calculate_font_size(img_width: int, img_height: int, base_size: int | None = None) -> int:
    """Calculate appropriate font size based on image dimensions.
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        base_size: Optional base font size (if None, calculates from image size)
        
    Returns:
        Font size in pixels
    """
    if base_size is not None:
        return base_size
    
    # Use ~3% of the smaller dimension as base font size
    base_size = min(img_width, img_height) * FONT_SIZE_RATIO
    return max(MIN_FONT_SIZE, int(base_size))


def calculate_text_position(img_width: int, img_height: int) -> tuple[int, int]:
    """Calculate text position in top-left corner with margin.
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        (x, y) position tuple
    """
    margin = max(5, int(min(img_width, img_height) * MARGIN_RATIO))
    return (margin, margin)


def ensure_text_fits(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    img_width: int,
    img_height: int,
    position: tuple[int, int],
    font_size: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int, int]:
    """Ensure text fits within image bounds, adjusting font size if needed.
    
    Args:
        text: Text to measure
        font: Current font object
        img_width: Image width
        img_height: Image height
        position: Text position (x, y)
        font_size: Current font size
        
    Returns:
        Tuple of (font, text_width, text_height, padding)
    """
    # Create temporary draw object to measure text
    temp_img = Image.new("RGBA", (img_width, img_height))
    draw = ImageDraw.Draw(temp_img)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    padding = max(3, int(font_size * PADDING_RATIO))
    max_text_width = img_width - position[0] - padding * 2
    max_text_height = img_height - position[1] - padding * 2
    
    # If text doesn't fit, reduce font size
    if text_width > max_text_width or text_height > max_text_height:
        scale_factor = min(max_text_width / text_width, max_text_height / text_height)
        font_size = max(MIN_FONT_SIZE_FALLBACK, int(font_size * scale_factor * 0.9))
        font = find_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = max(3, int(font_size * PADDING_RATIO))
    
    return font, text_width, text_height, padding


def add_text_overlay(
    img: Image.Image,
    text: str,
    position: tuple[int, int] | None = None,
    font_size: int | None = None,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_alpha: int = BG_ALPHA,
) -> Image.Image:
    """Add text overlay to an image with semi-transparent background.
    
    Font size is automatically calculated relative to image size if not provided.
    
    Args:
        img: Input image
        text: Text to overlay
        position: Text position (x, y), or None for auto-calculated top-left
        font_size: Font size in pixels, or None for auto-calculated
        bg_color: Background color RGB tuple
        text_color: Text color RGB tuple
        bg_alpha: Background alpha (0-255)
        
    Returns:
        Image with text overlay
    """
    img_with_text = img.copy()
    
    # Convert to RGBA if needed
    if img_with_text.mode != "RGBA":
        img_with_text = img_with_text.convert("RGBA")
    
    img_width, img_height = img_with_text.size
    
    # Calculate font size and position
    font_size = calculate_font_size(img_width, img_height, font_size)
    if position is None:
        position = calculate_text_position(img_width, img_height)
    
    # Load font and ensure text fits
    font = find_font(font_size)
    font, text_width, text_height, padding = ensure_text_fits(
        text, font, img_width, img_height, position, font_size
    )
    
    # Calculate background rectangle
    bg_rect = [
        position[0] - padding,
        position[1] - padding,
        position[0] + text_width + padding,
        position[1] + text_height + padding,
    ]
    
    # Clamp to image bounds
    bg_rect[0] = max(0, bg_rect[0])
    bg_rect[1] = max(0, bg_rect[1])
    bg_rect[2] = min(img_width, bg_rect[2])
    bg_rect[3] = min(img_height, bg_rect[3])
    
    # Create overlay for semi-transparent background
    overlay = Image.new("RGBA", img_with_text.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(bg_rect, fill=(*bg_color, bg_alpha))
    img_with_text = Image.alpha_composite(img_with_text, overlay)
    
    # Draw text
    draw = ImageDraw.Draw(img_with_text)
    draw.text(position, text, fill=text_color, font=font)
    
    return img_with_text


def parse_image_filename(filename: str) -> tuple[str, int] | None:
    """Parse block type and ID from filename.
    
    Args:
        filename: Filename without extension (e.g., "double_block_0")
        
    Returns:
        Tuple of (block_type, block_id) or None if parsing fails
    """
    match = re.match(r"(double|single)_block_(\d+)", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None


def get_image_files(decoded_dir: Path) -> list[ImageInfo]:
    """Get all image files sorted by step, block type, and block ID.
    
    Args:
        decoded_dir: Directory containing step_X subdirectories
        
    Returns:
        List of tuples: (step_id, block_type, block_id, file_path)
    """
    images: list[ImageInfo] = []
    
    # Find all step directories and sort by step number
    step_dirs = sorted(
        decoded_dir.glob("step_*"),
        key=lambda x: int(x.name.split("_")[1])
    )
    
    for step_dir in step_dirs:
        step_id = int(step_dir.name.split("_")[1])
        
        # Find all image files in this step
        for img_file in sorted(step_dir.glob("*.jpg")):
            parsed = parse_image_filename(img_file.stem)
            if parsed:
                block_type, block_id = parsed
                images.append((step_id, block_type, block_id, img_file))
    
    return images


def filter_images(
    images: list[ImageInfo],
    block_type: BlockType | None = None,
    max_blocks: int | None = None,
) -> list[ImageInfo]:
    """Filter images by block type and limit blocks per step.
    
    Args:
        images: List of image info tuples
        block_type: Filter by block type ("double", "single", or None)
        max_blocks: Maximum blocks per step to include
        
    Returns:
        Filtered list of image info tuples
    """
    # Filter by block type
    if block_type:
        images = [(s, bt, bi, p) for s, bt, bi, p in images if bt == block_type]
    
    # Limit blocks per step
    if max_blocks:
        step_groups: dict[int, list[tuple[str, int, Path]]] = {}
        for step_id, block_type, block_id, path in images:
            if step_id not in step_groups:
                step_groups[step_id] = []
            step_groups[step_id].append((block_type, block_id, path))
        
        # Sort and limit each step
        filtered_images: list[ImageInfo] = []
        for step_id in sorted(step_groups.keys()):
            step_images = sorted(step_groups[step_id], key=lambda x: (x[0], x[1]))
            step_images = step_images[:max_blocks]
            for block_type, block_id, path in step_images:
                filtered_images.append((step_id, block_type, block_id, path))
        images = filtered_images
    
    return images


def load_and_process_images(
    images: list[ImageInfo],
    font_size: int | None = None,
) -> list[Image.Image]:
    """Load images and add text overlays.
    
    Args:
        images: List of image info tuples
        font_size: Optional font size (None for auto-calculate)
        
    Returns:
        List of PIL Images with text overlays
    """
    frames: list[Image.Image] = []
    
    for step_id, block_type, block_id, img_path in tqdm(
        images, desc="Processing images", unit="image"
    ):
        try:
            img = Image.open(img_path)
            
            # Ensure image is in RGB mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Create text overlay
            text = f"Step {step_id} | {block_type} block {block_id}"
            img_with_text = add_text_overlay(img, text, font_size=font_size)
            frames.append(img_with_text)
            
        except Exception as e:
            print(f"\nWarning: Failed to load {img_path}: {e}")
            continue
    
    return frames


def convert_frames_to_numpy(frames: list[Image.Image]) -> list:
    """Convert PIL images to numpy arrays for video/GIF creation.
    
    Args:
        frames: List of PIL Images
        
    Returns:
        List of numpy arrays
    """
    frames_np = []
    for frame in tqdm(frames, desc="Converting frames", unit="frame"):
        frames_np.append(frame.convert("RGB"))
    return frames_np


def save_visualization(
    frames_np: list,
    output_path: Path,
    output_format: OutputFormat,
    fps: float,
) -> None:
    """Save frames as GIF or MP4.
    
    Args:
        frames_np: List of numpy array frames
        output_path: Output file path
        output_format: "gif" or "mp4"
        fps: Frames per second
    """
    if output_format == "gif":
        imageio.mimsave(
            str(output_path),
            frames_np,
            fps=fps,
            loop=0,  # Infinite loop
        )
    elif output_format == "mp4":
        imageio.mimsave(
            str(output_path),
            frames_np,
            fps=fps,
            codec="libx264",
            quality=8,
        )
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'gif' or 'mp4'")


def format_file_size(file_size: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        file_size: File size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB" or "0.50 GB")
    """
    file_size_mb = file_size / (1024 * 1024)
    file_size_gb = file_size / (1024 * 1024 * 1024)
    
    if file_size_gb >= 1:
        return f"{file_size_gb:.2f} GB"
    return f"{file_size_mb:.2f} MB"


def create_visualization(
    generation_id: str,
    output_dir: str = "output",
    output_format: OutputFormat = "gif",
    fps: float = DEFAULT_FPS,
    block_type: BlockType | None = None,
    max_blocks: int | None = None,
    font_size: int | None = DEFAULT_FONT_SIZE,
):
    """Create a video or GIF visualization from decoded intermediate images.
    
    Args:
        generation_id: Unique generation ID
        output_dir: Base output directory (default: "output")
        output_format: "gif" or "mp4" (default: "gif")
        fps: Frames per second for video/GIF (default: 2.0)
        block_type: Filter by block type ("double", "single", or None for all)
        max_blocks: Maximum number of blocks per step to include (default: None for all)
        font_size: Font size for text overlay, or None for auto-calculate (default: 24)
    """
    # Validate output format
    if output_format.lower() not in ("gif", "mp4"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'gif' or 'mp4'")
    output_format = output_format.lower()  # type: ignore
    
    # Find decoded images directory
    decoded_dir = Path(output_dir) / generation_id / "intermediates" / "decoded"
    
    if not decoded_dir.exists():
        raise ValueError(f"Decoded images directory not found: {decoded_dir}")
    
    print(f"Loading images from {decoded_dir}...")
    
    # Get and filter images
    all_images = get_image_files(decoded_dir)
    
    if not all_images:
        raise ValueError(f"No images found in {decoded_dir}")
    
    # Filter images
    filtered_images = filter_images(all_images, block_type, max_blocks)
    
    if not filtered_images:
        block_type_msg = f" {block_type}" if block_type else ""
        raise ValueError(f"No{block_type_msg} block images found")
    
    print(f"Found {len(filtered_images)} images")
    
    # Load and process images
    print("Processing images and adding text overlays...")
    frames = load_and_process_images(filtered_images, font_size)
    
    if not frames:
        raise ValueError("No valid images could be loaded")
    
    # Create output path
    block_filter = f"_{block_type}" if block_type else ""
    output_filename = f"visualization_{generation_id}{block_filter}.{output_format}"
    output_path = Path(output_dir) / generation_id / output_filename
    
    print(f"Creating {output_format.upper()} with {len(frames)} frames at {fps} fps...")
    print(f"Saving to {output_path}")
    
    # Convert frames
    print("Converting images...")
    frames_np = convert_frames_to_numpy(frames)
    
    # Save visualization
    print(f"Writing {output_format.upper()} file...")
    save_visualization(frames_np, output_path, output_format, fps)
    
    # Display results
    file_size = output_path.stat().st_size
    size_str = format_file_size(file_size)
    
    print(f"✓ Saved visualization to {output_path}")
    print(f"  Format: {output_format.upper()}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  File size: {size_str} ({file_size:,} bytes)")


if __name__ == "__main__":
    Fire(create_visualization)

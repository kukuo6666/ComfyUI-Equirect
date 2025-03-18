# ComfyUI-Equirect

A ComfyUI extension for converting between equirectangular panoramas and cubemaps. This tool provides essential functionality for 360° image processing and VR/AR content creation.

## Features

- **EquirectToCubemapNode**: Convert equirectangular panoramas into 6 cubemap faces
- **CubemapToEquirectNode**: Convert 6 cubemap faces back to equirectangular panorama

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Equirect.git
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-Equirect
   pip install -r requirements.txt
   ```

## Usage

### EquirectToCubemapNode

Converts a 360° panorama into six individual faces of a cubemap.

Inputs:
- `equirect_image`: Equirectangular panorama input (2:1 aspect ratio)
- `face_size`: Output size for each cubemap face
- `fov`: Field of view angle (default: 90 degrees)

Outputs:
- `front`, `right`, `back`, `left`, `top`, `bottom`: 6 cubemap faces

### CubemapToEquirectNode

Reconstructs a 360° panorama from six cubemap faces.

Inputs:
- `front`, `right`, `back`, `left`, `top`, `bottom`: 6 cubemap faces
- `output_height`: Height of the output panorama (width will be 2x height)

Outputs:
- `equirect_image`: Converted equirectangular panorama

## Technical Details

- High-quality conversion using `py360convert` library
- Automatic fallback to custom implementation if needed
- Batch processing support
- Format preservation throughout conversion

## Dependencies

- torch
- numpy
- pillow
- opencv-python
- py360convert

## Parameters

- **Input Image**: Equirectangular panorama (2:1 aspect ratio)
- **Face Size**: Cubemap face edge length (default: 512, range: 64-4096)
- **Field of View**: View angle per face (default: 90°, range: 60-120°)

## System Requirements

- ComfyUI latest version
- Python 3.7+
- CUDA-compatible GPU recommended
- Sufficient GPU memory for large panoramas

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version

Current version: 1.1.0 
# Installation Guide

## Requirements
- Python 3.8+
- Latest version of ComfyUI
- Sufficient GPU memory for processing large panoramic images

## Basic Installation

1. Clone this repository to ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Equirect.git
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-Equirect
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Manual Installation of py360convert

If `pip install py360convert` fails, you can try installing from source:

```bash
git clone https://github.com/sunset1995/py360convert.git
cd py360convert
pip install -e .
```

## Common Issues

### Import Errors
If you encounter import errors, ensure all dependencies are correctly installed:
```
ImportError: No module named 'py360convert'
```

Solution: Manually install the py360convert library.

### Memory Errors
Processing large panoramic images may require substantial memory. If you encounter memory errors, try:
1. Reduce the input image size
2. Lower the cubemap face size
3. Use a device with more GPU memory

### Conversion Quality

For best conversion quality:
1. Use high-resolution input images (at least 4K)
2. Set cubemap face size to 512 or higher
3. Keep default FOV (90 degrees)

## Performance Optimization

To speed up processing:
1. Ensure PyTorch with CUDA support is correctly installed
2. Use reasonable image sizes (unnecessarily large sizes will increase processing time)
3. Process images individually if batch processing is not required 
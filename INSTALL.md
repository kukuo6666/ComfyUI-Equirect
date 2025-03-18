# Installation Guide

## Requirements

- Python 3.8 or higher
- Latest version of ComfyUI
- Sufficient GPU memory for processing large panoramic images
- CUDA-compatible GPU recommended

## Basic Installation

1. Clone this repository to your ComfyUI's `custom_nodes` directory:
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

If `pip install py360convert` fails, you can install from source:

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
Solution: Manually install the py360convert library as described above.

### Memory Errors
Processing large panoramic images requires substantial memory. If you encounter memory errors:
1. Reduce input image size
2. Lower cubemap face size
3. Use a device with more GPU memory

### Conversion Quality

For optimal conversion quality:
1. Use high-resolution input images (4K or higher recommended)
2. Set cubemap face size to 512 or higher
3. Maintain default FOV (90 degrees)

## Performance Optimization

To improve processing speed:
1. Ensure PyTorch with CUDA support is properly installed
2. Use appropriate image sizes (larger sizes increase processing time)
3. Process images individually when batch processing is not required

## Version Compatibility

- Tested with Python 3.8-3.12
- Compatible with latest ComfyUI releases
- Requires CUDA-capable GPU for optimal performance

## Support

For issues and questions:
1. Check the GitHub issues page
2. Ensure all dependencies are correctly installed
3. Verify input image formats and sizes 
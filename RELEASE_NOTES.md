# ComfyUI-Equirect Release Notes

## Version 1.1.0

### Features
- ✨ Bidirectional Conversion between Equirectangular Panorama and Cubemap (等距環景圖與立方體貼圖的雙向轉換)
  - 🔄 Equirectangular to Cubemap conversion
  - 🔄 Cubemap to Equirectangular conversion
- 📊 Real-time Progress Display
- 🚀 Performance Optimization
- 🌐 Complete English Documentation

### Improvements
- ⚡️ Optimized Coordinate Conversion Algorithm
- 💾 Improved Memory Usage Efficiency
- 🛠️ Enhanced Error Handling
- 📦 Added Batch Processing Support
- 🔧 Improved Code Structure and Naming

### Technical Details
- 🔍 Support for Various Input Formats
- 📐 Adjustable Field of View
- 🎯 Precise Pixel Mapping
- 🧮 Smart Boundary Handling

### System Requirements
- Python 3.8+
- ComfyUI (latest version)
- CUDA-capable GPU
- Sufficient GPU memory for large panoramic images

### Installation
See INSTALL.md for detailed instructions

### Usage
1. Load nodes in ComfyUI
2. Select desired conversion node (Equirect to Cubemap or vice versa)
3. Connect inputs and outputs
4. Execute conversion

### Notes
- High-quality input images recommended for best results
- Ensure sufficient GPU memory is available
- Large images may require longer processing time

### Key Features Explained
- **Equirectangular to Cubemap**: Converts 360° panoramic images into six square faces that form a cube
- **Cubemap to Equirectangular**: Transforms six cube faces back into a single 360° panoramic image
- **Real-time Progress**: Shows conversion progress with estimated time remaining
- **Memory Efficient**: Optimized algorithms to handle large panoramic images
- **Batch Processing**: Process multiple images in sequence 
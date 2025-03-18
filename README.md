# ComfyUI-Equirect

ComfyUI节点用于全景图（等距矩形图）和立方体贴图之间的转换。

## 功能

- **EquirectToCubemapNode**: 将全景图转换为6个立方体贴图面
- **CubemapToEquirectNode**: 将6个立方体贴图面转换回全景图

## 安装

1. 将此仓库克隆到ComfyUI的`custom_nodes`目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Equirect.git
   ```

2. 安装依赖：
   ```bash
   cd ComfyUI-Equirect
   pip install -r requirements.txt
   ```

## 使用方法

### EquirectToCubemapNode（全景图转立方体贴图）

输入：
- `equirect_image`: 全景图输入（等距矩形格式，2:1宽高比）
- `face_size`: 输出立方体贴图面的尺寸
- `fov`: 视场角度（默认为90度）

输出：
- `front`, `right`, `back`, `left`, `top`, `bottom`: 6个立方体贴图面

### CubemapToEquirectNode（立方体贴图转全景图）

输入：
- `front`, `right`, `back`, `left`, `top`, `bottom`: 6个立方体贴图面
- `output_height`: 输出全景图的高度（宽度将自动设为高度的2倍）

输出：
- `equirect_image`: 转换后的全景图像

## 技术说明

- 优先使用`py360convert`库进行高质量转换
- 如果库不可用，会自动回退到自定义实现
- 支持批处理和保持正确的图像格式

## 依赖项

- torch
- numpy
- pillow
- opencv-python
- py360convert

## Parameters

- **Input Image**: Equirectangular panorama image with 2:1 aspect ratio
- **Face Size**: Edge length for each cubemap face (default: 512, range: 64-4096)
- **Field of View**: FOV angle for each face (default: 90, range: 60-120)

## System Requirements

- ComfyUI
- Python 3.7+
- PyTorch
- Pillow (PIL)
- NumPy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details 
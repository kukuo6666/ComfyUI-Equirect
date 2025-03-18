"""
全景圖處理節點集合，用於 ComfyUI
"""

from .equirect_to_cubemap import EquirectToCubemapNode

NODE_CLASS_MAPPINGS = {
    "EquirectToCubemap": EquirectToCubemapNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectToCubemap": "全景圖轉 Cubemap"
}

__version__ = "1.0.0"

"""
ComfyUI nodes for equirectangular image processing
"""

from .equirect_to_cubemap import EquirectToCubemapNode
from .cubemap_to_equirect import CubemapToEquirectNode

NODE_CLASS_MAPPINGS = {
    "EquirectToCubemap": EquirectToCubemapNode,
    "CubemapToEquirect": CubemapToEquirectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectToCubemap": "Equirectangular to Cubemap",
    "CubemapToEquirect": "Cubemap to Equirectangular",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

__version__ = "1.1.0"

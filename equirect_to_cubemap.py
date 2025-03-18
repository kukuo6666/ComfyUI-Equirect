"""
Equirectangular to Cubemap Conversion Node for ComfyUI
Version: 1.1.0

This module provides functionality to convert equirectangular (360°) panoramic images
into six cubemap faces. It supports both high-quality conversion using py360convert
and a custom fallback implementation for maximum compatibility.

Author: kukuo6666
License: MIT
"""

import torch
import numpy as np
from PIL import Image
import cv2
import py360convert
from tqdm import tqdm
import time

class EquirectToCubemapNode:
    """
    A ComfyUI node that converts equirectangular panoramic images into six cubemap faces.
    
    Features:
    - High-quality conversion using py360convert library
    - Custom fallback implementation with FOV control
    - Progress tracking during conversion
    - Batch processing support
    - Automatic format handling
    
    Input format: BHWC (Batch, Height, Width, Channels)
    Output faces: front, right, back, left, top, bottom
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.batch_size = 1
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "equirect_image": ("IMAGE",),
                "face_size": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "fov": ("FLOAT", {"default": 90, "min": 60, "max": 120}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "right", "back", "left", "top", "bottom")
    FUNCTION = "convert_to_cubemap"
    CATEGORY = "equirect"
    OUTPUT_NODE = True

    def ensure_bhwc(self, x):
        """
        Ensure input tensor is in BHWC format.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Tensor in BHWC format
        """
        if len(x.shape) == 3:  # HWC format
            x = x.unsqueeze(0)  # Add batch dimension -> BHWC
        return x

    def convert_to_cubemap(self, equirect_image, face_size, fov):
        """
        Convert equirectangular image to six cubemap faces with progress tracking.
        
        Args:
            equirect_image (torch.Tensor): Input equirectangular image
            face_size (int): Size of each cubemap face
            fov (float): Field of view angle in degrees
            
        Returns:
            tuple: Six tensors representing cubemap faces (front, right, back, left, top, bottom)
        """
        # Input validation and preprocessing
        if not isinstance(equirect_image, torch.Tensor):
            equirect_image = torch.from_numpy(equirect_image)
        
        # Ensure float32 type and 0-1 range
        if equirect_image.dtype != torch.float32:
            equirect_image = equirect_image.float()
        if equirect_image.max() > 1.0:
            equirect_image = equirect_image / 255.0
            
        # Ensure BHWC format and validate dimensions
        equirect_image = self.ensure_bhwc(equirect_image)
        B, H, W, C = equirect_image.shape
        
        # Validate aspect ratio
        ratio = W / H
        if not (1.99 <= ratio <= 2.01):
            error_msg = (
                f"Input image must have a 2:1 aspect ratio (equirectangular format).\n"
                f"Current dimensions (BHWC): Batch={B}, Height={H}, Width={W}, Channels={C}\n"
                f"Current ratio: {ratio:.2f} (expected: 2.00)"
            )
            raise ValueError(error_msg)
            
        # Process each batch with progress tracking
        all_face_tensors = []
        for b in tqdm(range(B), desc="Processing batches"):
            # Convert tensor to numpy array
            img = equirect_image[b].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            try:
                # Attempt high-quality conversion
                cube_faces = py360convert.e2c(img, face_w=face_size, mode='bicubic', cube_format='list')
            except Exception as e:
                print(f"Warning: py360convert failed ({str(e)}), using fallback method...")
                cube_faces = self.equirect_to_cubemap_fallback(img, face_size, fov, C == 1)
            
            # Convert faces to tensors
            batch_tensors = []
            for face in cube_faces:
                if C == 1 and len(face.shape) == 2:
                    face = face[..., None]
                face_tensor = torch.from_numpy(face).float() / 255.0
                face_tensor = face_tensor.unsqueeze(0)
                batch_tensors.append(face_tensor)
            
            all_face_tensors.append(batch_tensors)
        
        # Combine results from all batches
        final_tensors = [
            torch.cat([batch[i] for batch in all_face_tensors], dim=0)
            for i in range(6)
        ]
        
        return tuple(final_tensors)

    def equirect_to_cubemap_fallback(self, img, face_size, fov, is_grayscale):
        """
        Custom implementation for equirectangular to cubemap conversion.
        
        Args:
            img (np.ndarray): Input equirectangular image
            face_size (int): Size of each cubemap face
            fov (float): Field of view angle in degrees
            is_grayscale (bool): Whether the image is grayscale
            
        Returns:
            list: Six numpy arrays representing cubemap faces
        """
        if is_grayscale and len(img.shape) == 3:
            img = img[:, :, 0]
        
        pil_img = Image.fromarray(img, mode='L' if is_grayscale else 'RGB')
        width, height = pil_img.size
        faces = []

        # Pre-calculate transformation parameters
        half_fov_rad = np.radians(fov / 2)
        u, v = np.meshgrid(
            np.linspace(-1, 1, face_size),
            np.linspace(-1, 1, face_size)
        )
        
        img_array = np.array(pil_img)

        # Face configurations
        face_configs = [
            ("front",  (0,0,-1), (0,1,0), (1,0,0)),
            ("right",  (1,0,0),  (0,1,0), (0,0,1)),
            ("back",   (0,0,1),  (0,1,0), (-1,0,0)),
            ("left",   (-1,0,0), (0,1,0), (0,0,-1)),
            ("top",    (0,1,0),  (0,0,1), (1,0,0)),
            ("bottom", (0,-1,0), (0,0,-1), (1,0,0))
        ]

        for face_name, direction, up, right in face_configs:
            # Calculate face vectors
            fx, fy, fz = direction
            ux, uy, uz = up
            rx, ry, rz = right
            
            # Calculate ray directions
            x = fx + rx * u * np.tan(half_fov_rad) + ux * v * np.tan(half_fov_rad)
            y = fy + ry * u * np.tan(half_fov_rad) + uy * v * np.tan(half_fov_rad)
            z = fz + rz * u * np.tan(half_fov_rad) + uz * v * np.tan(half_fov_rad)

            # Normalize vectors
            norm = np.sqrt(x*x + y*y + z*z)
            x, y, z = x/norm, y/norm, z/norm

            # Convert to spherical coordinates
            theta = np.arctan2(x, z)
            phi = np.arcsin(np.clip(y, -1, 1))

            # Map to image coordinates
            sample_x = (theta / (2 * np.pi) + 0.5) * width
            sample_y = (phi / np.pi + 0.5) * height

            # Sample with bounds checking
            sample_x = np.clip(sample_x, 0, width - 1).astype(np.int32)
            sample_y = np.clip(sample_y, 0, height - 1).astype(np.int32)

            # Extract face pixels
            face_pixels = img_array[sample_y, sample_x]
            
            # Format output array
            if is_grayscale:
                face_array = face_pixels
                if len(face_array.shape) == 3 and face_array.shape[2] == 1:
                    face_array = face_array[:, :, 0]
            else:
                face_array = face_pixels
                if len(face_array.shape) == 2:
                    face_array = np.stack([face_array] * 3, axis=2)
            
            faces.append(face_array)

        return faces

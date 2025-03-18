"""
Cubemap to Equirectangular Conversion Node for ComfyUI
Version: 1.1.0

This module provides functionality to convert six cubemap faces back into
an equirectangular (360°) panoramic image. It supports both high-quality
conversion using py360convert and a custom fallback implementation.

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

class CubemapToEquirectNode:
    """
    A ComfyUI node that converts six cubemap faces into an equirectangular panorama.
    
    Features:
    - High-quality conversion using py360convert library
    - Custom fallback implementation
    - Progress tracking during conversion
    - Batch processing support
    - Automatic format handling
    
    Input format: Six BHWC tensors (Batch, Height, Width, Channels)
    Output: Single BHWC tensor representing the equirectangular panorama
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.batch_size = 1
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE",),
                "right": ("IMAGE",),
                "back": ("IMAGE",),
                "left": ("IMAGE",),
                "top": ("IMAGE",),
                "bottom": ("IMAGE",),
                "output_height": ("INT", {"default": 1024, "min": 32, "max": 8192}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("equirect_image",)
    FUNCTION = "cubemap_to_equirect"
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

    def validate_faces(self, faces):
        """
        Validate that all cubemap faces have consistent dimensions.
        
        Args:
            faces (list): List of six cubemap face tensors
            
        Returns:
            tuple: Batch size, face size, and number of channels
        """
        shapes = [face.shape for face in faces]
        if not all(len(s) == 4 for s in shapes):
            raise ValueError("All faces must be 4D tensors (BHWC format)")
            
        B = shapes[0][0]
        H = shapes[0][1]
        C = shapes[0][3]
        
        if not all(s[0] == B for s in shapes):
            raise ValueError("All faces must have the same batch size")
        if not all(s[1] == s[2] == H for s in shapes):
            raise ValueError("All faces must be square and have the same size")
        if not all(s[3] == C for s in shapes):
            raise ValueError("All faces must have the same number of channels")
            
        return B, H, C

    def cubemap_to_equirect(self, front, right, back, left, top, bottom, output_height):
        """
        Convert six cubemap faces to an equirectangular panorama with progress tracking.
        
        Args:
            front, right, back, left, top, bottom (torch.Tensor): Cubemap face tensors
            output_height (int): Height of the output panorama
            
        Returns:
            torch.Tensor: Equirectangular panorama in BHWC format
        """
        # Ensure all inputs are tensors in BHWC format
        faces = [front, right, back, left, top, bottom]
        faces = [self.ensure_bhwc(face) for face in faces]
        
        # Validate face dimensions
        B, face_size, C = self.validate_faces(faces)
        output_width = output_height * 2
        
        # Process each batch with progress tracking
        all_equirect_tensors = []
        for b in tqdm(range(B), desc="Processing batches"):
            # Convert tensors to numpy arrays
            face_arrays = []
            for face in faces:
                img = face[b].cpu().numpy()
                img = (img * 255).astype(np.uint8)
                face_arrays.append(img)
            
            try:
                # Attempt high-quality conversion
                equirect = py360convert.c2e(face_arrays, h=output_height, w=output_width, mode='bicubic')
            except Exception as e:
                print(f"Warning: py360convert failed ({str(e)}), using fallback method...")
                equirect = self.cubemap_to_equirect_fallback(face_arrays, output_height)
            
            # Convert back to tensor
            equirect_tensor = torch.from_numpy(equirect).float() / 255.0
            equirect_tensor = equirect_tensor.unsqueeze(0)
            all_equirect_tensors.append(equirect_tensor)
        
        # Combine results from all batches
        final_tensor = torch.cat(all_equirect_tensors, dim=0)
        return (final_tensor,)

    def cubemap_to_equirect_fallback(self, faces, output_height):
        """
        Custom implementation for cubemap to equirectangular conversion.
        
        Args:
            faces (list): List of six numpy arrays representing cubemap faces
            output_height (int): Height of the output panorama
            
        Returns:
            np.ndarray: Equirectangular panorama
        """
        output_width = output_height * 2
        is_grayscale = len(faces[0].shape) == 2 or faces[0].shape[2] == 1
        
        # Create output array
        if is_grayscale:
            equirect = np.zeros((output_height, output_width), dtype=np.uint8)
        else:
            equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Generate spherical coordinates
        phi, theta = np.meshgrid(
            np.linspace(-np.pi/2, np.pi/2, output_height),
            np.linspace(-np.pi, np.pi, output_width)
        )
        
        # Convert to Cartesian coordinates
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)
        
        # Find dominant axis for each pixel
        ax = np.abs(x)
        ay = np.abs(y)
        az = np.abs(z)
        
        # Create masks for each face
        front_mask = (az >= ax) & (az >= ay) & (z < 0)
        right_mask = (ax >= ay) & (ax >= az) & (x > 0)
        back_mask = (az >= ax) & (az >= ay) & (z > 0)
        left_mask = (ax >= ay) & (ax >= az) & (x < 0)
        top_mask = (ay >= ax) & (ay >= az) & (y > 0)
        bottom_mask = (ay >= ax) & (ay >= az) & (y < 0)
        
        # Map coordinates to each face
        face_size = faces[0].shape[0]
        
        def sample_face(face, mask, u, v):
            u = ((u[mask] + 1) * (face_size - 1) / 2).astype(np.int32)
            v = ((v[mask] + 1) * (face_size - 1) / 2).astype(np.int32)
            equirect[mask] = face[v, u]
        
        # Front face (negative z)
        sample_face(faces[0], front_mask, x/(-z), -y/(-z))
        
        # Right face (positive x)
        sample_face(faces[1], right_mask, z/x, -y/x)
        
        # Back face (positive z)
        sample_face(faces[2], back_mask, -x/z, -y/z)
        
        # Left face (negative x)
        sample_face(faces[3], left_mask, -z/(-x), -y/(-x))
        
        # Top face (positive y)
        sample_face(faces[4], top_mask, x/y, z/y)
        
        # Bottom face (negative y)
        sample_face(faces[5], bottom_mask, x/(-y), -z/(-y))
        
        return equirect 
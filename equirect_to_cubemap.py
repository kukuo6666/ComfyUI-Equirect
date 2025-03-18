import torch
import numpy as np
from PIL import Image
import cv2
import py360convert

class EquirectToCubemapNode:
    """
    ComfyUI node to convert equirectangular images to six cubemap faces.
    Input/Output format: BHWC (Batch, Height, Width, Channels)
    Output face order: front, right, back, left, top, bottom
    """

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

    def ensure_bhwc(self, x):
        """Ensure input tensor is in BHWC format"""
        if len(x.shape) == 3:  # HWC format
            x = x.unsqueeze(0)  # Add batch dimension -> BHWC
        return x

    def convert_to_cubemap(self, equirect_image, face_size, fov):
        """
        Convert equirectangular image to six cubemap faces.
        Input/Output format: BHWC (Batch, Height, Width, Channels)
        """
        # Ensure input image is in correct format
        if not isinstance(equirect_image, torch.Tensor):
            equirect_image = torch.from_numpy(equirect_image)
        
        # Ensure float32 type and 0-1 range
        if equirect_image.dtype != torch.float32:
            equirect_image = equirect_image.float()
        if equirect_image.max() > 1.0:
            equirect_image = equirect_image / 255.0
            
        # Ensure BHWC format
        equirect_image = self.ensure_bhwc(equirect_image)
        B, H, W, C = equirect_image.shape
        
        # Check aspect ratio (allow small margin of error)
        ratio = W / H
        if not (1.99 <= ratio <= 2.01):  # Allow 1% error
            error_msg = (
                f"Input image must have a 2:1 aspect ratio (equirectangular format).\n"
                f"Current dimensions (BHWC): Batch={B}, Height={H}, Width={W}, Channels={C}\n"
                f"Aspect ratio: {ratio:.2f}\n"
                f"Expected: Width should be approximately twice the height."
            )
            raise ValueError(error_msg)
            
        # Convert tensor to numpy array for processing
        img = equirect_image[0].cpu().numpy()  # Take first batch
        img = (img * 255).astype(np.uint8)  # Convert to 0-255 range uint8
        
        # Use py360convert to convert equirectangular to cubemap
        try:
            # Use py360convert's e2c function
            cube_faces = py360convert.e2c(img, face_w=face_size, mode='bicubic', cube_format='list')
            print(f"Converted {len(cube_faces)} cube faces")
        except Exception as e:
            print(f"Error during conversion: {e}")
            # Fall back to our own implementation using FOV
            print(f"Using custom conversion method, FOV={fov}...")
            cube_faces = self.equirect_to_cubemap_fallback(img, face_size, fov, C == 1)
        
        # Convert numpy arrays to BHWC format Tensors
        face_tensors = []
        for face in cube_faces:
            # Ensure correct dimensions
            if C == 1 and len(face.shape) == 2:
                face = face[..., None]  # Add channel dimension -> HWC
                
            # Convert to torch tensor, ensure float32 type and 0-1 range
            face_tensor = torch.from_numpy(face).float() / 255.0  # HWC
            face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
            face_tensors.append(face_tensor)

        return tuple(face_tensors)

    def equirect_to_cubemap_fallback(self, img, face_size, fov, is_grayscale):
        """
        Convert equirectangular image to cubemap faces using spherical projection, considering FOV.
        This is a fallback method when external libraries are not available.
        Output order: front, right, back, left, top, bottom
        
        Coordinate system:
        - front: looking at negative z-axis, x right, y up
        - right: looking at positive x-axis, z right, y up
        - back:  looking at positive z-axis, x left, y up
        - left:  looking at negative x-axis, z left, y up
        - top:   looking at positive y-axis, x right, z down
        - bottom: looking at negative y-axis, x right, z up
        """
        if is_grayscale and len(img.shape) == 3:
            img = img[:, :, 0]
        
        if is_grayscale:
            pil_img = Image.fromarray(img, mode='L')
        else:
            pil_img = Image.fromarray(img)
            
        width, height = pil_img.size
        faces = []

        # Pre-calculate grid points and FOV
        half_fov_rad = np.radians(fov / 2)
        u, v = np.meshgrid(
            np.linspace(-1, 1, face_size),
            np.linspace(-1, 1, face_size)
        )
        
        # Convert to numpy array for faster processing
        img_array = np.array(pil_img)

        # Define basic direction and up vectors for each face
        face_configs = [
            # name     direction   up vector  right vector
            ("front",  (0,0,-1),  (0,1,0), (1,0,0)),
            ("right",  (1,0,0),   (0,1,0), (0,0,1)),
            ("back",   (0,0,1),   (0,1,0), (-1,0,0)),
            ("left",   (-1,0,0),  (0,1,0), (0,0,-1)),
            ("top",    (0,1,0),   (0,0,1), (1,0,0)),
            ("bottom", (0,-1,0),  (0,0,-1), (1,0,0))
        ]

        for face_name, direction, up, right in face_configs:
            # Calculate base vectors for the face
            fx, fy, fz = direction
            ux, uy, uz = up
            rx, ry, rz = right
            
            # Calculate 3D direction vector for each pixel, using FOV scaling
            x = fx + rx * u * np.tan(half_fov_rad) + ux * v * np.tan(half_fov_rad)
            y = fy + ry * u * np.tan(half_fov_rad) + uy * v * np.tan(half_fov_rad)
            z = fz + rz * u * np.tan(half_fov_rad) + uz * v * np.tan(half_fov_rad)

            # Normalize vectors
            norm = np.sqrt(x*x + y*y + z*z)
            x = x / norm
            y = y / norm
            z = z / norm

            # Calculate spherical angles
            theta = np.arctan2(x, z)  # longitude (-π to π)
            phi = np.arcsin(np.clip(y, -1, 1))  # latitude (-π/2 to π/2)


            # Convert to image coordinates (Note: in equirectangular format, left is -π, right is π)
            sample_x = (theta / (2 * np.pi) + 0.5) * width
            sample_y = (phi / np.pi + 0.5) * height

            # Clip to valid range
            sample_x = np.clip(sample_x, 0, width - 1).astype(np.int32)
            sample_y = np.clip(sample_y, 0, height - 1).astype(np.int32)

            # Sample the image
            face_pixels = img_array[sample_y, sample_x]
            
            # Convert to appropriate array format
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

import torch
import numpy as np
from PIL import Image
import time
import cv2
import py360convert

class CubemapToEquirectNode:
    """
    ComfyUI node to convert cubemap faces back to equirectangular image.
    Input/Output format: BHWC (Batch, Height, Width, Channels)
    Input face order: front, right, back, left, top, bottom
    """

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
                "output_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("equirect_image",)
    FUNCTION = "convert_to_equirect"
    CATEGORY = "equirect"
    OUTPUT_NODE = True

    def ensure_bhwc(self, x):
        """Ensure input tensor is in BHWC format"""
        if len(x.shape) == 3:  # HWC format
            x = x.unsqueeze(0)  # Add batch dimension -> BHWC
        return x

    def convert_to_equirect(self, front, right, back, left, top, bottom, output_height):
        """
        Convert six cubemap faces to equirectangular image.
        Input/Output format: BHWC (Batch, Height, Width, Channels)
        """
        # Ensure all inputs are torch.Tensor
        faces = [front, right, back, left, top, bottom]
        faces = [self.ensure_bhwc(f) if not isinstance(f, torch.Tensor) else f for f in faces]
        
        # Check if all faces have the same size
        face_size = faces[0].shape[1]
        for i, face in enumerate(faces):
            if face.shape[1:3] != faces[0].shape[1:3]:
                raise ValueError(f"All cubemap faces must have the same size. Face {i} has different size from face 0")

        # Ensure all faces are float32 type and in 0-1 range
        faces = [f.float() if f.dtype != torch.float32 else f for f in faces]
        faces = [f / 255.0 if f.max() > 1.0 else f for f in faces]

        # Convert to numpy arrays for processing
        face_arrays = []
        for face in faces:
            img = face[0].cpu().numpy()  # Take first batch
            img = (img * 255).astype(np.uint8)
            face_arrays.append(img)

        # Calculate output width (maintain 2:1 aspect ratio)
        output_width = output_height * 2
        
        # Try to use py360convert for conversion
        try:
            # First arrange faces into a dice format for py360convert
            print("Using py360convert for cubemap to equirect conversion...")
            dice = self.list_to_dice(face_arrays, face_size)
            
            # Convert dice to equirectangular
            equirect_array = py360convert.c2e(
                dice,
                h=output_height,
                w=output_width,
                mode='bicubic'
            )
            
            # Convert to 0-1 range float tensor
            equirect_tensor = torch.from_numpy(equirect_array).float() / 255.0
            
            # Ensure tensor has batch dimension and correct channel dimension
            if len(equirect_tensor.shape) == 2:  # Grayscale without channel
                equirect_tensor = equirect_tensor.unsqueeze(2)  # Add channel dimension
            equirect_tensor = equirect_tensor.unsqueeze(0)  # Add batch dimension
            
            return (equirect_tensor,)
            
        except Exception as e:
            print(f"Error using py360convert: {e}")
            print("Falling back to custom implementation...")
            
            # Convert to PIL Image for the fallback method
            face_images = []
            for face_array in face_arrays:
                if face_array.shape[2] == 1:  # Grayscale
                    img = Image.fromarray(face_array.squeeze(), mode='L')
                else:  # RGB
                    img = Image.fromarray(face_array)
                face_images.append(img)
            
            # Use fallback method
            equirect = self.cubemap_to_equirect_fallback(
                face_images,
                output_width,
                output_height,
                faces[0].shape[3] == 1
            )
            
            # Convert back to BHWC format Tensor
            equirect_np = np.array(equirect)
            if faces[0].shape[3] == 1 and len(equirect_np.shape) == 2:
                equirect_np = equirect_np[..., None]
    
            equirect_tensor = torch.from_numpy(equirect_np).float() / 255.0
            equirect_tensor = equirect_tensor.unsqueeze(0)  # Add batch dimension
    
            return (equirect_tensor,)

    def list_to_dice(self, cube_faces, face_w):
        """
        Arrange 6 cube faces (order [front, right, back, left, top, bottom])
        into a dice format:
             ┌─────┐
             │  top  │
        ┌─────┬─────┬─────┬─────┐
        │ left│front│right│ back│
        └─────┴─────┴─────┴─────┘
             ┌─────┐
             │bottom│
             └─────┘
        """
        front  = cube_faces[0]
        right  = cube_faces[1]
        back   = cube_faces[2]
        left   = cube_faces[3]
        top    = cube_faces[4]
        bottom = cube_faces[5]
        
        # Determine face channel count
        if len(front.shape) == 2:  # Grayscale without channel
            channels = 1
            # Add channel dimension to all faces
            front = front[..., np.newaxis]
            right = right[..., np.newaxis]
            back = back[..., np.newaxis]
            left = left[..., np.newaxis]
            top = top[..., np.newaxis]
            bottom = bottom[..., np.newaxis]
        else:
            channels = front.shape[2]
        
        # Create output dice
        dice = np.zeros((3 * face_w, 4 * face_w, channels), dtype=np.uint8)
        
        # First row: top in middle (second block)
        dice[0:face_w, face_w:2*face_w] = top
        
        # Second row: left, front, right, back
        dice[face_w:2*face_w, 0:face_w]          = left
        dice[face_w:2*face_w, face_w:2*face_w]   = front
        dice[face_w:2*face_w, 2*face_w:3*face_w] = right
        dice[face_w:2*face_w, 3*face_w:4*face_w] = back
        
        # Third row: bottom in middle (second block)
        dice[2*face_w:3*face_w, face_w:2*face_w] = bottom
        
        return dice

    def cubemap_to_equirect_fallback(self, faces, width, height, is_grayscale):
        """
        Convert cubemap faces to equirectangular image.
        This is a fallback method when py360convert is not available.
        
        Coordinate system:
        - front: looking at negative z-axis, x right, y up
        - right: looking at positive x-axis, z right, y up
        - back:  looking at positive z-axis, x left, y up
        - left:  looking at negative x-axis, z left, y up
        - top:   looking at positive y-axis, x right, z down
        - bottom: looking at negative y-axis, x right, z up
        """
        # Create output image
        if is_grayscale:
            equirect = Image.new('L', (width, height))
        else:
            equirect = Image.new('RGB', (width, height))

        # Convert faces to numpy arrays for faster processing
        face_arrays = [np.array(face) for face in faces]
        face_size = face_arrays[0].shape[0]

        # Create output array
        if is_grayscale:
            output = np.zeros((height, width), dtype=np.uint8)
        else:
            output = np.zeros((height, width, 3), dtype=np.uint8)

        # Process each pixel with progress updates
        total_pixels = height * width
        processed_pixels = 0
        last_update_time = time.time()
        update_interval = 0.1  # Update every 0.1 seconds

        for y in range(height):
            for x in range(width):
                # Convert equirectangular coordinates to 3D vector
                theta = (x / width - 0.5) * 2 * np.pi  # longitude (-π to π)
                phi = (y / height - 0.5) * np.pi       # latitude (-π/2 to π/2)

                # Convert spherical coordinates to Cartesian
                x3d = np.cos(phi) * np.sin(theta)
                y3d = np.sin(phi)
                z3d = np.cos(phi) * np.cos(theta)

                # Find the dominant axis
                abs_x, abs_y, abs_z = abs(x3d), abs(y3d), abs(z3d)
                max_axis = max(abs_x, abs_y, abs_z)

                # Determine which face to use and calculate UV coordinates
                if max_axis == abs_x:
                    if x3d > 0:  # right face
                        u = z3d / x3d
                        v = y3d / x3d
                        face_idx = 1
                    else:  # left face
                        u = -z3d / -x3d
                        v = y3d / -x3d
                        face_idx = 3
                elif max_axis == abs_y:
                    if y3d > 0:  # top face
                        # Rotate coordinates for top face
                        u = x3d / y3d
                        v = -z3d / y3d
                        face_idx = 4
                    else:  # bottom face
                        # Rotate coordinates for bottom face
                        u = x3d / -y3d
                        v = z3d / -y3d
                        face_idx = 5
                else:  # abs_z is max
                    if z3d > 0:  # back face
                        u = -x3d / z3d
                        v = y3d / z3d
                        face_idx = 2
                    else:  # front face
                        u = x3d / -z3d
                        v = y3d / -z3d
                        face_idx = 0

                # Convert UV coordinates to pixel coordinates
                # For top and bottom faces, rotate the coordinates
                if face_idx in [4, 5]:  # top or bottom face
                    # Rotate UV coordinates by 90 degrees
                    u, v = v, -u
                
                px = int((u + 1) * face_size / 2)
                py = int((v + 1) * face_size / 2)

                # Clamp coordinates to valid range
                px = max(0, min(face_size - 1, px))
                py = max(0, min(face_size - 1, py))

                # Sample the face
                output[y, x] = face_arrays[face_idx][py, px]

                # Update progress
                processed_pixels += 1
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    progress = processed_pixels / total_pixels
                    print(f"Converting cubemap to equirectangular: {progress*100:.1f}%")
                    last_update_time = current_time

        return Image.fromarray(output) 
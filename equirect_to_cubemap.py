import torch
import numpy as np
from PIL import Image

class EquirectToCubemapNode:
    """
    將全景圖（equirectangular）轉換為六個面的 cubemap 的 ComfyUI 節點。
    輸入/輸出格式：BHWC（批次、高度、寬度、通道）
    輸出面的順序：前、右、後、左、上、下
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
        """確保輸入張量是 BHWC 格式"""
        if len(x.shape) == 3:  # HWC 格式
            x = x.unsqueeze(0)  # 加入批次維度 -> BHWC
        return x

    def convert_to_cubemap(self, equirect_image, face_size, fov):
        """
        將全景圖轉換為六個 cubemap 面。
        輸入/輸出格式：BHWC（批次、高度、寬度、通道）
        """
        # 輸出輸入圖像的原始資訊
        original_shape = equirect_image.shape
        print(f"輸入圖像形狀: {original_shape}, 資料類型: {equirect_image.dtype}")
        
        # 確保輸入圖像是正確的格式
        if not isinstance(equirect_image, torch.Tensor):
            equirect_image = torch.from_numpy(equirect_image)
        
        # 確保是 float32 類型和 0-1 範圍
        if equirect_image.dtype != torch.float32:
            equirect_image = equirect_image.float()
        if equirect_image.max() > 1.0:
            equirect_image = equirect_image / 255.0
            
        # 確保是 BHWC 格式
        equirect_image = self.ensure_bhwc(equirect_image)
        B, H, W, C = equirect_image.shape
        print(f"BHWC 形狀: {equirect_image.shape}")
        
        # 檢查寬高比（允許一定的誤差）
        ratio = W / H
        if not (1.99 <= ratio <= 2.01):  # 允許 1% 的誤差
            error_msg = (
                f"輸入圖像必須具有 2:1 的寬高比（全景圖格式）。\n"
                f"目前維度 (BHWC): 批次={B}, 高度={H}, 寬度={W}, 通道={C}\n"
                f"寬高比: {ratio:.2f}\n"
                f"預期：寬度應該約為高度的兩倍。"
            )
            raise ValueError(error_msg)
            
        # 轉換為 PIL 圖像進行處理
        img = equirect_image[0]  # 取第一個批次
        img = (img * 255).byte().cpu().numpy()  # 保持 HWC 格式
        
        if C == 1:  # 灰階圖
            img = Image.fromarray(img.squeeze(), mode='L')
        else:  # RGB 圖
            img = Image.fromarray(img)

        # 轉換全景圖到 cubemap
        cubemap_faces = self.equirect_to_cubemap(img, face_size, fov, C == 1)

        # 轉換 PIL Image 為 BHWC 格式的 Tensor
        face_tensors = []
        for face in cubemap_faces:
            # 轉換為 numpy array
            face_np = np.array(face)
            
            # 確保正確的維度
            if C == 1 and len(face_np.shape) == 2:
                face_np = face_np[..., None]  # 加入通道維度 -> HWC
                
            # 轉換為 torch tensor，確保 float32 類型和 0-1 範圍
            face_tensor = torch.from_numpy(face_np).float() / 255.0  # HWC
            face_tensor = face_tensor.unsqueeze(0)  # 加入批次維度 -> BHWC
            face_tensors.append(face_tensor)

        return tuple(face_tensors)

    def equirect_to_cubemap(self, img, face_size, fov, is_grayscale):
        """
        透過球面投影將全景圖轉換為 cubemap faces，考慮 FOV。
        輸出順序：前、右、後、左、上、下
        
        座標系統說明：
        - front（前）: 看向 z 軸負方向，x 向右，y 向上
        - right（右）: 看向 x 軸正方向，z 向右，y 向上
        - back（後）:  看向 z 軸正方向，x 向左，y 向上
        - left（左）:  看向 x 軸負方向，z 向左，y 向上
        - top（上）:   看向 y 軸正方向，x 向右，z 向下
        - bottom（下）: 看向 y 軸負方向，x 向右，z 向上
        """
        width, height = img.size
        faces = []

        # 預先計算網格點
        half_fov_rad = np.radians(fov / 2)
        u, v = np.meshgrid(
            np.linspace(-1, 1, face_size),
            np.linspace(-1, 1, face_size)
        )
        
        # 轉換為 numpy array 以加快處理速度
        img_array = np.array(img)

        # 定義每個面的基本方向和上向量
        face_configs = [
            # 名稱     方向      上方向    右方向
            ("front",  (0,0,-1),  (0,1,0), (1,0,0)),
            ("right",  (1,0,0),   (0,1,0), (0,0,1)),
            ("back",   (0,0,1),   (0,1,0), (-1,0,0)),
            ("left",   (-1,0,0),  (0,1,0), (0,0,-1)),
            ("top",    (0,1,0),   (0,0,1), (1,0,0)),
            ("bottom", (0,-1,0),  (0,0,-1), (1,0,0))
        ]

        for face_name, direction, up, right in face_configs:
            # 計算面的基向量
            fx, fy, fz = direction
            ux, uy, uz = up
            rx, ry, rz = right
            
            # 計算每個像素的 3D 方向向量
            x = fx + rx * u * np.tan(half_fov_rad) + ux * v * np.tan(half_fov_rad)
            y = fy + ry * u * np.tan(half_fov_rad) + uy * v * np.tan(half_fov_rad)
            z = fz + rz * u * np.tan(half_fov_rad) + uz * v * np.tan(half_fov_rad)

            # 正規化向量
            norm = np.sqrt(x*x + y*y + z*z)
            x = x / norm
            y = y / norm
            z = z / norm

            # 計算球面角度
            theta = np.arctan2(x, z)  # 經度 (-π到π)
            phi = np.arcsin(np.clip(y, -1, 1))  # 緯度 (-π/2到π/2)

            # 轉換為圖像座標（注意：全景圖格式中，左邊是-π，右邊是π）
            sample_x = (theta / (2 * np.pi) + 0.5) * width
            sample_y = (phi / np.pi + 0.5) * height

            # 裁剪到有效範圍
            sample_x = np.clip(sample_x, 0, width - 1).astype(np.int32)
            sample_y = np.clip(sample_y, 0, height - 1).astype(np.int32)

            # 取樣圖像
            face_pixels = img_array[sample_y, sample_x]
            
            # 建立 PIL Image
            if is_grayscale:
                face = Image.fromarray(face_pixels.astype(np.uint8), mode='L')
            else:
                face = Image.fromarray(face_pixels.astype(np.uint8), mode='RGB')
            
            faces.append(face)

            # 輸出除錯資訊
            print(f"處理 {face_name} 面:")
            print(f"方向向量: {direction}")
            print(f"上方向量: {up}")
            print(f"右方向量: {right}")
            print(f"角度範圍 - 經度: [{np.min(theta):.2f}, {np.max(theta):.2f}], 緯度: [{np.min(phi):.2f}, {np.max(phi):.2f}]")
            print("---")

        return faces

# ComfyUI 全景圖工具

這個 ComfyUI 自訂節點提供了全景圖（equirectangular）影像處理功能。目前支援將全景圖轉換為 cubemap 格式。

## 功能

- 將全景圖（2:1 寬高比）轉換為六個 cubemap 面（前、右、後、左、上、下）
- 支援自訂輸出面的大小
- 支援自訂視場角度（FOV）
- 支援 RGB 和灰階影像

## 安裝

1. 將此儲存庫複製到 ComfyUI 的 `custom_nodes` 目錄：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kukuo6666/ComfyUI-Equirect.git
```

2. 重新啟動 ComfyUI

## 使用方法

1. 在 ComfyUI 工作區中，搜尋「全景圖轉 Cubemap」節點
2. 將全景圖影像連接到節點的輸入端
3. 設定所需的輸出面大小和視場角度
4. 節點將輸出六個 cubemap 面

## 參數說明

- **輸入影像**：全景圖影像，必須是 2:1 寬高比
- **輸出面大小**：每個 cubemap 面的邊長（預設：512，範圍：64-4096）
- **視場角度**：每個面的視場角度（預設：90，範圍：60-120）

## 系統需求

- ComfyUI
- Python 3.7+
- PyTorch
- Pillow (PIL)
- NumPy

## 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案 
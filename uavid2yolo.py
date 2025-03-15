import os
import cv2
import numpy as np

# 設定路徑
image_dir = "./train/seq1/images"  # UAVid 原始影像
mask_dir = "./train/seq1/TrainId"  # UAVid Mask（單通道標籤）
output_dir = "./train/seq1/yolo_labels"  # YOLO 標註輸出

# 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# 類別對應表（只保留 "人" 和 "車"）
class_mapping = {
    6: 0,  # human -> YOLO class 0
    3: 1,  # static car -> YOLO class 1
    7: 1   # moving car -> YOLO class 1
}

# 轉換 UAVid Mask 為 YOLO 格式
def convert_mask_to_yolo(mask_path, output_txt, image_shape):
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 讀取 mask（灰階）
    h, w = image_shape[:2]  # 取得影像尺寸

    with open(output_txt, "w") as f:
        for pixel_value, class_id in class_mapping.items():
            class_mask = (mask_img == pixel_value).astype(np.uint8)  # 提取該類別的 mask
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 5:  # 避免太小的標註
                    continue

                # 轉換座標至 0~1（歸一化）
                polygon = contour.flatten().tolist()
                polygon = [str(p / w) if i % 2 == 0 else str(p / h) for i, p in enumerate(polygon)]

                # 儲存標註
                f.write(f"{class_id} " + " ".join(polygon) + "\n")

# 轉換所有標註
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith(".png"):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, mask_file)  # 假設影像與 mask 同名
        output_txt = os.path.join(output_dir, mask_file.replace(".png", ".txt"))

        if not os.path.exists(image_path):  # 確保影像存在
            print(f"影像 {mask_file} 不存在，跳過...")
            continue

        # 讀取影像尺寸
        img = cv2.imread(image_path)
        convert_mask_to_yolo(mask_path, output_txt, img.shape)

print("✅ UAVid 標註已轉換為 YOLO segmentation 格式（僅包含人和車類別）！")

import json
import os
import glob

# 只保留「人」與「車輛」兩類
class_mapping = {
    'person': 0, 'rider': 0,  # 人類類別
    'car': 1, 'truck': 1, 'bus': 1, 'train': 1, 'motorcycle': 1, 'bicycle': 1  # 有輪子的車輛類別
}

# 原始標註資料夾
input_rootdir = 'gtFine_trainvaltest/gtFine/val'
# 目標 YOLO 標註的目錄
output_rootdir = '4yolo/labels/val'
os.makedirs(output_rootdir, exist_ok=True)

# 將多邊形座標歸一化 (0~1)
def normalize_polygon(size, polygon):
    img_w, img_h = size
    norm_poly = []
    for point in polygon:
        norm_x = point[0] / img_w
        norm_y = point[1] / img_h
        norm_poly.append(f"{norm_x:.6f} {norm_y:.6f}")
    return " ".join(norm_poly)

# 處理單個 JSON 標註
def convert_annotation(json_file, output_dir):
    if not os.path.getsize(json_file):  # 檢查文件是否為空
        print(f"Warning: {json_file} is empty. Skipping.")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode {json_file}. Skipping.")
            return

    img_w, img_h = data["imgWidth"], data["imgHeight"]
    objects = data.get("objects", [])  # 避免 KeyError
    
    if not objects:
        return  # 如果沒有物件就跳過
    
    json_id = os.path.splitext(os.path.basename(json_file))[0]
    output_file_path = os.path.join(output_dir, f"{json_id}.txt")
    
    with open(output_file_path, 'w') as out_file:
        for obj in objects:
            label = obj.get("label", "")
            if label in class_mapping:
                class_id = class_mapping[label]
                polygon = obj.get("polygon", [])
                if not polygon:
                    continue  # 沒有多邊形就跳過
                norm_polygon = normalize_polygon((img_w, img_h), polygon)
                out_file.write(f"{class_id} {norm_polygon}\n")

# 遍歷所有城市目錄
for city_name in os.listdir(input_rootdir):
    city_path = os.path.join(input_rootdir, city_name)
    if os.path.isdir(city_path):
        output_dir = os.path.join(output_rootdir, city_name)
        os.makedirs(output_dir, exist_ok=True)
        
        json_files = glob.glob(os.path.join(city_path, "*.json"))
        for json_file in json_files:
            convert_annotation(json_file, output_dir)

print("YOLO Segmentation 格式標註檔案已成功轉換！")

# feature_extractor.py

import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from real_feature_extractor import FeatureExtractor

# --- CẤU HÌNH ---
IMAGE_DIR = './data/coco_images/' 
OUTPUT_FEATURE_DIR = './data/coco_features_2048d/' 
# Danh sách các file JSON nguồn để lấy danh sách ảnh
CAPTION_JSON_FILES = [
    './data/train_captions.json',
    './data/val_captions.json',
    './data/test_captions.json'
]

D_MODEL = 2048 
N_REGIONS = 36 

def get_all_image_names(json_files):
    """Lấy danh sách tất cả các tên ảnh (image_name) duy nhất từ các file JSON."""
    all_names = set()
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                for item in annotations:
                    if 'image_name' in item:
                        all_names.add(item['image_name'])
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file {file_path}. Bỏ qua.")
    return list(all_names)

def extract_and_save_features():
    print("Bắt đầu quá trình trích xuất đặc trưng...")
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    
    image_names = get_all_image_names(CAPTION_JSON_FILES)
    if not image_names:
        print("Lỗi: Không tìm thấy ảnh nào. Vui lòng kiểm tra đường dẫn CAPTION_JSON_FILES.")
        return
        
    print(f"Tìm thấy {len(image_names)} ảnh duy nhất cần xử lý.")
    
    try:
        extractor = FeatureExtractor(d_model=D_MODEL, d_region=N_REGIONS)
    except Exception as e:
        print(f"Lỗi khởi tạo FeatureExtractor: {e}.")
        return
        
    for img_name in tqdm(image_names, desc="Trích xuất đặc trưng"):
        img_file = os.path.join(IMAGE_DIR, img_name)
        output_name = os.path.splitext(img_name)[0]
        output_file = os.path.join(OUTPUT_FEATURE_DIR, f"{output_name}.npz")

        if os.path.exists(output_file):
            continue
            
        try:
            V_features, g_raw = extractor.extract(img_file)
            np.savez_compressed(
                output_file, 
                V_features=V_features, 
                g_raw=g_raw            
            )
        except Exception as e:
            print(f"\nLỗi xử lý ảnh {img_name}: {e}. Bỏ qua.")
            continue
        
    print(f"Hoàn thành trích xuất đặc trưng. Đã lưu tại: {OUTPUT_FEATURE_DIR}")

if __name__ == '__main__':
    extract_and_save_features()
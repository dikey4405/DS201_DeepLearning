import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from real_feature_extractor import FeatureExtractor 

DATASET_ROOT = '/kaggle/input/data-dl'
OUTPUT_FEATURE_DIR = '/kaggle/working/coco_features_2048d/'

SPLITS = [
    {
        'name': 'train',
        'json_path': os.path.join(DATASET_ROOT, 'Captions', 'train.json'),
        'image_folder': os.path.join(DATASET_ROOT, 'Images', 'Images', 'train')
    },
    {
        'name': 'dev',
        'json_path': os.path.join(DATASET_ROOT, 'Captions', 'dev.json'),
        'image_folder': os.path.join(DATASET_ROOT, 'Images', 'Images', 'dev')
    },
    {
        'name': 'test',
        'json_path': os.path.join(DATASET_ROOT, 'Captions', 'test.json'),
        'image_folder': os.path.join(DATASET_ROOT, 'Images', 'Images', 'test')
    }
]

D_MODEL = 2048 
N_REGIONS = 36 

def get_image_list_from_split(split_info):
    """Đọc file JSON và trả về danh sách đường dẫn ảnh đầy đủ."""
    json_path = split_info['json_path']
    img_folder = split_info['image_folder']
    
    image_names = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            for item in annotations:
                if 'image_name' in item:
                    image_names.add(item['image_name'])
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy file {json_path}. Bỏ qua.")
        return []
        
    full_image_paths = []
    for name in image_names:
        full_path = os.path.join(img_folder, name)
        full_image_paths.append((name, full_path))
        
    return full_image_paths

def extract_and_save_features():
    print("Bắt đầu quá trình trích xuất đặc trưng...")
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    
    try:
        extractor = FeatureExtractor(d_model=D_MODEL, d_region=N_REGIONS)
        print("Đã khởi tạo model ResNet-101 thành công.")
    except Exception as e:
        print(f"Lỗi khởi tạo FeatureExtractor: {e}")
        return

    for split in SPLITS:
        print(f"\n--- Đang xử lý tập: {split['name']} ---")
        image_list = get_image_list_from_split(split)
        
        if not image_list:
            print(f"Không tìm thấy ảnh nào cho tập {split['name']}.")
            continue
            
        print(f"Số lượng ảnh cần xử lý: {len(image_list)}")
        
        for img_name, img_full_path in tqdm(image_list, desc=f"Extracting {split['name']}"):
            output_name = os.path.splitext(img_name)[0]
            output_file = os.path.join(OUTPUT_FEATURE_DIR, f"{output_name}.npz")

            if os.path.exists(output_file):
                continue
                
            try:
                V_features, g_raw = extractor.extract(img_full_path)
                
                if V_features is None:
                    print(f"Bỏ qua ảnh lỗi hoặc không đọc được: {img_name}")
                    continue
                
                np.savez_compressed(
                    output_file, 
                    V_features=V_features, 
                    g_raw=g_raw            
                )
            except Exception as e:
                print(f"Lỗi ngoại lệ khi xử lý ảnh {img_name}: {e}")
                continue
        
    print(f"\nHoàn thành! File lưu tại: {OUTPUT_FEATURE_DIR}")

if __name__ == '__main__':
    extract_and_save_features()

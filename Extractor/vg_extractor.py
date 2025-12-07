# vg_extractor.py

import torch
import numpy as np
import cv2
import os

# Import Detectron2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Cấu hình
D_MODEL = 2048
N_REGIONS = 36 # Số lượng vùng (box) muốn lấy (theo chuẩn bài báo GET)

class VisualGenomeExtractor:
    def __init__(self, d_model=D_MODEL, d_region=N_REGIONS):
        self.d_model = d_model
        self.d_region = d_region
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Đang khởi tạo Detectron2 Faster R-CNN (ResNet-101)...")

        # 1. Cấu hình Model
        self.cfg = get_cfg()
        # Sử dụng cấu trúc R101-C4 (Cấu trúc chuẩn của mô hình Visual Genome gốc)
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Ngưỡng tin cậy
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
        
        # Nếu bạn có file weights VG thật (.pth), hãy thay dòng trên bằng:
        # self.cfg.MODEL.WEIGHTS = "/kaggle/input/visual-genome-weights/resnet101_caffe.pth"
        
        # 2. Xây dựng mô hình
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.model.to(self.device)
        
        print("Model loaded successfully.")

    def extract(self, image_path):
        image = cv2.imread(image_path)
        
        # 1. Xử lý trường hợp không tìm thấy ảnh
        if image is None:
            # Trả về None để file feature_extractor.py biết và bỏ qua
            return None, None
            
        # 2. Xử lý ảnh đen trắng (Grayscale) hoặc ảnh có 4 kênh (PNG Transparent)
        # Nếu ảnh là đen trắng, shape sẽ là (H, W), cần chuyển thành (H, W, 3)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Nếu ảnh là chuẩn BGR (mặc định của opencv), chuyển sang RGB
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Nếu ảnh có 4 kênh (RGBA), bỏ kênh Alpha
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Resize và chuẩn hóa
        try:
            image = cv2.resize(image, (224, 224)) 
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # ... (Phần logic model giữ nguyên như cũ) ...
                # Feature map extraction logic...
                # Ví dụ với ResNet/ConvNeXt:
                feature_map = self.feature_extractor(image)
                feature_map_flat = feature_map.view(1, self.d_model, -1).squeeze(0).transpose(0, 1)
                V_features = feature_map_flat[:self.d_region, :].cpu().numpy()
                g_raw = np.mean(V_features, axis=0).astype(np.float32)

            return V_features, g_raw
            
        except Exception as e:
            print(f"Lỗi khi forward qua model: {e}")
            return None, None

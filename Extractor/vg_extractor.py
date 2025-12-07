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
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        height, width = img.shape[:2]
        
        # Preprocessing chuẩn của Detectron2
        from detectron2.data.detection_utils import read_image
        from detectron2.data.transforms import ResizeShortestEdge
        
        # Resize ảnh sao cho cạnh ngắn nhất >= 800 (Chuẩn SOTA)
        aug = ResizeShortestEdge(short_edge_length=800, max_size=1333)
        input_image = aug.get_transform(img).apply_image(img)
        
        # Chuyển sang Tensor (C, H, W)
        image_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))
        
        inputs = [{"image": image_tensor, "height": height, "width": width}]

        with torch.no_grad():
            images = self.model.preprocess_image(inputs)
            
            # 1. Trích xuất Features từ Backbone (ResNet-101)
            features = self.model.backbone(images.tensor)
            
            # 2. Lấy Region Proposals (Hộp đề xuất)
            proposals, _ = self.model.proposal_generator(images, features, None)
            
            # 3. Lấy Box Features (Đặc trưng vùng) thông qua RoI Pooling
            # Lấy top N_REGIONS hộp có điểm cao nhất
            instances, _ = self.model.roi_heads(images, features, proposals, None)
            
            # Trích xuất feature vector trước lớp classification cuối cùng (2048 chiều)
            # Detectron2 lưu feature này ở box_features (sau AvgPool của Res5Head)
            box_features = self.model.roi_heads.box_pooler(
                [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES], 
                [x.proposal_boxes for x in instances]
            )
            
            # Qua lớp Res5 head (lớp conv cuối)
            box_features = self.model.roi_heads.res5(box_features)
            
            # Global Average Pooling để ra vector 2048
            box_features = box_features.mean(dim=[2, 3]) # (Num_Proposals, 2048)
            
            # Chọn top K features
            if box_features.size(0) < self.d_region:
                # Nếu không đủ 36 vùng, pad thêm số 0
                padding = torch.zeros((self.d_region - box_features.size(0), self.d_model), device=self.device)
                V_features = torch.cat([box_features, padding], dim=0)
            else:
                V_features = box_features[:self.d_region]

            # Chuyển về Numpy
            V_features = V_features.cpu().numpy() # (36, 2048)
            
            # Tính g_raw (Trung bình cộng)
            g_raw = np.mean(V_features, axis=0)

        return V_features, g_raw

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
from detectron2.data.transforms import ResizeShortestEdge

# Cấu hình
D_MODEL = 2048
N_REGIONS = 36 

class VisualGenomeExtractor:
    def __init__(self, d_model=D_MODEL, d_region=N_REGIONS):
        self.d_model = d_model
        self.d_region = d_region
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Đang khởi tạo Detectron2 Faster R-CNN (R101-C4)...")

        # 1. Cấu hình Model
        self.cfg = get_cfg()
        # Load cấu hình R101-C4 chuẩn
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
        
        # 2. Xây dựng mô hình
        self.model = build_model(self.cfg)
        self.model.eval()
        self.checkpointer = DetectionCheckpointer(self.model)
        self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.model.to(self.device)
        
        print("Model loaded successfully.")

    def extract(self, image_path):
        # 1. Đọc ảnh
        img = cv2.imread(image_path)
        
        if img is None:
            return None, None
        
        if len(img.shape) == 2: # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4: # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        height, width = img.shape[:2]

        try:
            # 2. Preprocessing
            aug = ResizeShortestEdge(short_edge_length=800, max_size=1333)
            input_image = aug.get_transform(img).apply_image(img)
            image_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))
            
            inputs = [{"image": image_tensor, "height": height, "width": width}]

            with torch.no_grad():
                images = self.model.preprocess_image(inputs)
                
                # Forward Pass qua Backbone
                features = self.model.backbone(images.tensor)
                
                # Lấy Region Proposals
                proposals, _ = self.model.proposal_generator(images, features, None)
                
                # 1. Lấy boxes từ proposals
                instances, _ = self.model.roi_heads(images, features, proposals, None)
                
                # Lấy các hộp dự đoán cuối cùng (Detected Boxes)
                pred_boxes = [x.pred_boxes for x in instances]
                
                # Trích xuất feature từ res4 (feature map backbone) dựa trên các hộp dự đoán
                box_features = self.model.roi_heads.pooler(
                    [features[f] for f in self.cfg.MODEL.ROI_HEADS.IN_FEATURES], 
                    pred_boxes
                )
                
                # Qua lớp Res5 head (lớp conv cuối của R101)
                box_features = self.model.roi_heads.res5(box_features)
                
                # Global Average Pooling -> (Total_Boxes, 2048)
                box_features = box_features.mean(dim=[2, 3]) 
                
                # Chọn top K features (Padding nếu thiếu)
                if box_features.size(0) < self.d_region:
                    padding = torch.zeros((self.d_region - box_features.size(0), self.d_model), device=self.device)
                    V_features = torch.cat([box_features, padding], dim=0)
                else:
                    V_features = box_features[:self.d_region]

                V_features = V_features.cpu().numpy()
                
                # Tính g_raw
                g_raw = np.mean(V_features, axis=0)

            return V_features, g_raw

        except Exception as e:
            return None, None

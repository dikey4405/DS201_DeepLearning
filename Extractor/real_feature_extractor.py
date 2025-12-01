import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn as nn

D_MODEL = 2048
D_REGION = 49

class FeatureExtractor:
    """
    Sử dụng ConvNeXt-Large (SOTA CNN) để trích xuất đặc trưng.
    Mô hình này mạnh hơn ResNet-101/ResNeXt rất nhiều.
    """

    def __init__(self, d_model=D_MODEL, d_region=D_REGION):
        self.d_model = d_model
        self.d_region = d_region
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        print(f"Khởi tạo SUPER Feature Extractor: ConvNeXt-Large trên {self.device}...")
        
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        self.model = models.convnext_large(weights=weights)
        
        self.feature_extractor = self.model.features
        
        self.adapter = nn.Conv2d(1536, 2048, kernel_size=1)
        
        self.feature_extractor.to(self.device)
        self.adapter.to(self.device)
        self.feature_extractor.eval()
        self.adapter.eval()

        self.transform = weights.transforms()

    def extract(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None, None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ConvNeXt hoạt động tốt nhất ở độ phân giải này hoặc lớn hơn
        image = cv2.resize(image, (224, 224)) 
        
        # Chuyển thành PIL Image để dùng transform chuẩn của ConvNeXt
        from PIL import Image
        image_pil = Image.fromarray(image)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 1. Lấy raw features (1, 1536, 7, 7)
            raw_features = self.feature_extractor(image_tensor)
            
            # 2. Chiếu lên 2048 chiều (1, 2048, 7, 7)
            feature_map = self.adapter(raw_features)
            
            # 3. Tạo V_raw (Đặc trưng vùng)
            # Flatten: (1, 2048, 49) -> (2048, 49) -> (49, 2048)
            feature_map_flat = feature_map.view(1, self.d_model, -1).squeeze(0).transpose(0, 1) 
            
            # ConvNeXt 7x7 = 49 vùng. 
            # Nếu N_REGIONS=36, ta lấy 36 vùng đầu. 
            # Tốt nhất là lấy hết 49 vùng nếu model GET cho phép (chỉ cần sửa N_REGIONS=49 ở config GET)
            V_features = feature_map_flat[:self.d_region, :].cpu().numpy() 

            # 4. Tạo g_raw (Đặc trưng toàn cục)
            g_raw = np.mean(V_features, axis=0).astype(np.float32)

        return V_features, g_raw

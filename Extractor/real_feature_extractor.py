# Trích xuất đặc trưng từ dữ liệu đầu vào
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

# configurations
D_MODEL = 2048
D_REGION = 36

class FeatureExtractor:
    """
    Sử dụng ResNet-50 (ImageNet) để trích xuất đặc trưng 2048D thô.
    Chúng ta giả lập 36 vùng bằng cách lấy các vị trí không gian từ feature map cuối.
    """

    def __init__(self, d_model=D_MODEL, d_region=D_REGION):
        self.d_model = d_model
        self.d_region = d_region
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Loại bỏ lớp fully connected và avgpool cuối cùng
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2]).to(self.device)
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không thể đọc ảnh tại {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature_map = self.feature_extractor(image) # (1, 2048, 7, 7)
            
            # 1. Tạo V_raw (Đặc trưng vùng giả lập)
            # Làm phẳng feature map (49, 2048) và lấy N_REGIONS=36 vị trí đầu tiên
            feature_map_flat = feature_map.view(1, self.d_model, -1).squeeze(0).transpose(0, 1) # (49, 2048)
            V_features = feature_map_flat[:self.d_region, :].cpu().numpy() # (36, 2048)

            # 2. Tạo g_raw (Đặc trưng toàn cục)
            g_raw = np.mean(V_features, axis=0).astype(np.float32)

        return V_features, g_raw
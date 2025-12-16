
# DataLoader/COCO_dataset.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import numpy as np 
from pyvi import ViTokenizer
import json
import os
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=1):
        # Cho phép tùy chỉnh ngưỡng tần suất khi khởi tạo
        self.freq_threshold = freq_threshold
        
        self.word_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

    def __len__(self):
        return len(self.word_to_idx)

    def build_vocab(self, all_captions):
        word_freq = {}
        print("Building vocabulary using ViTokenizer...")
        
        for caption in all_captions:
            if caption is None or not isinstance(caption, str):
                continue
            
            # Cải tiến 1: Chuyển về chữ thường (lower) để chuẩn hóa
            caption = caption.lower().strip()
            
            # Tokenize tiếng Việt
            tokens = ViTokenizer.tokenize(caption).split()
            
            for word in tokens:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        idx = 4
        for word, count in word_freq.items():
            # Sử dụng threshold được truyền vào từ __init__
            if count >= self.freq_threshold:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
                
        print(f"Total words in vocab (freq >= {self.freq_threshold}): {len(self.word_to_idx)}")

    # Cải tiến 2: Hàm quan trọng để chuyển câu văn thành list số
    def numericalize(self, text):
        """
        Input: "Một con mèo đang ngủ"
        Output: [1, 45, 23, 12, 99, 2] (Ví dụ: <SOS>, Một, con_mèo, đang, ngủ, <EOS>)
        """
        if not isinstance(text, str):
            return []
            
        # Tokenize và chuẩn hóa giống hệt lúc build_vocab
        text = text.lower().strip()
        tokens = ViTokenizer.tokenize(text).split()
        
        result = []
        
        # Thêm <SOS> ở đầu
        result.append(self.SOS_token)
        
        for token in tokens:
            # Nếu từ có trong từ điển thì lấy index, không có thì lấy <UNK>
            index = self.word_to_idx.get(token, self.UNK_token)
            result.append(index)
            
        # Thêm <EOS> ở cuối
        result.append(self.EOS_token)
        
        return result
        
class COCODataset(Dataset):
    def __init__(self, image_dir, features_dir, captions_file, vocabulary):
        super().__init__()
        self.image_dir = image_dir
        self.features_dir = features_dir
        self.vocabulary = vocabulary

        try:
            print(f"Loading annotations from: {captions_file}")
            with open(captions_file, 'r', encoding='utf-8') as f:
                raw_annotations = json.load(f)
            
            # Chỉ giữ lại các mục có caption tiếng Việt hợp lệ
            self.annotations = self._reformat_annotations(raw_annotations)
            print(f"Loaded {len(self.annotations)} images.")
            
        except Exception as e:
            print(f"Lỗi khi tải hoặc xử lý file JSON: {e}")
            self.annotations = {}

        self.image_ids = list(self.annotations.keys())
        # Tạo đường dẫn tới file đặc trưng .npz
        self.feature_paths = {img_id: os.path.join(features_dir, f"{os.path.splitext(img_id)[0]}.npz") 
                              for img_id in self.image_ids}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _reformat_annotations(self, raw_annotations):
        """Chuyển đổi list raw annotations thành dict {image_name: [viet_cap, ...]}"""
        reformatted = {}
        for item in raw_annotations:
            img_name = item.get('image_name')
            viet_cap = item.get('translate')
            
            # --- LỌC DỮ LIỆU NGHIÊM NGẶT (Nơi xảy ra việc giảm số lượng ảnh) ---
            if img_name and viet_cap and isinstance(viet_cap, str) and len(viet_cap.strip()) > 0:
                if img_name not in reformatted:
                    reformatted[img_name] = []
                
                # Tokenize ngay khi load để đảm bảo tính nhất quán (SCST và XE)
                tokenized_cap = ViTokenizer.tokenize(viet_cap.lower().strip())
                reformatted[img_name].append(tokenized_cap)
        return reformatted
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Tải Đặc trưng (.NPZ)
        feature_path = self.feature_paths.get(img_id)
        try:
            features = np.load(feature_path)
            V_raw = torch.tensor(features['V_features'], dtype=torch.float)
            g_raw = torch.tensor(features['g_raw'], dtype=torch.float)
        except Exception as e:
            # Nếu file .npz bị thiếu, trả về tensor ngẫu nhiên (chỉ nên xảy ra khi debug)
            V_raw = torch.randn(36, 2048)
            g_raw = torch.randn(2048)

        # Lấy TẤT CẢ caption tiếng Việt (dạng string đã tokenize) cho SCST/Evaluation
        viet_captions_list = self.annotations[img_id] 
        
        # Token hóa MỘT caption (caption đầu tiên) cho huấn luyện XE
        caption_xe = viet_captions_list[0]
        
        tokens_xe = [self.vocabulary.word_to_idx.get(word, self.vocabulary.UNK_token) 
                     for word in caption_xe.split()]
        
        tokens_xe = [self.vocabulary.SOS_token] + tokens_xe + [self.vocabulary.EOS_token]
        caption_tensor = torch.tensor(tokens_xe)

        # Trả về img_id, V_raw, g_raw, tensor (cho XE), len, list (cho SCST)
        return img_id, V_raw, g_raw, caption_tensor, len(tokens_xe), viet_captions_list
    
def collate_fn(data):
    # Sắp xếp data theo độ dài caption (tốt cho việc đệm)
    data.sort(key=lambda x: x[4], reverse=True)
    image_ids, V_batch, g_batch, captions, lengths, viet_captions_list = zip(*data)
    
    # Xếp chồng V_raw và g_raw
    V_batch = torch.stack(V_batch)
    g_batch = torch.stack(g_batch)

    # Đệm (Pad) captions (cho XE)
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len).long()
    for i, caption in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = caption[:end]
        
    return image_ids, V_batch, g_batch, padded_captions, torch.tensor(lengths), viet_captions_list


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
    def __init__(self):
        # Token đặc biệt
        self.word_to_idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        # Khởi tạo ngược
        self.idx_to_word = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3

    def __len__(self):
        return len(self.word_to_idx)

    def build_vocab(self, all_captions, threshold=1):
        """
        Xây dựng vocabulary từ danh sách caption tiếng Việt
        """
        print("Building vocabulary using ViTokenizer + Counter...")
        word_freq = Counter()

        # 1. Đếm tần suất
        for caption in all_captions:
            if not isinstance(caption, str):
                continue
            
            # Thêm lower() và strip() để chuẩn hóa trước khi tokenize
            tokens = ViTokenizer.tokenize(caption.lower().strip()).split()
            word_freq.update(tokens)

        # 2. Sắp xếp để đảm bảo index cố định (Deterministic)
        sorted_words = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))

        # 3. Gán index
        idx = 4
        for word, count in sorted_words:
            if count >= threshold:
                # word_to_idx: Giữ nguyên gạch dưới (để khớp input lúc train)
                self.word_to_idx[word] = idx
                
                # idx_to_word: XÓA gạch dưới (để output lúc eval sạch đẹp, tăng điểm)
                # Đây là dòng quan trọng nhất giúp bạn không phải sửa file evaluate.py
                self.idx_to_word[idx] = word.replace("_", " ")
                
                idx += 1

        print(f"Total words in vocab (freq >= {threshold}): {len(self.word_to_idx)}")

    def numericalize(self, tokens):
        """
        Chuyển danh sách token thành danh sách index
        Input tokens: list các từ (có thể có gạch dưới do ViTokenizer sinh ra)
        """
        return [
            self.word_to_idx.get(word, self.UNK_token)
            for word in tokens
        ]

    def decode(self, indices, stop_at_eos=True):
        """
        Chuyển index -> câu chữ
        """
        words = []
        for idx in indices:
            # Xử lý an toàn nếu idx là Tensor
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            if stop_at_eos and idx == self.EOS_token:
                break
            
            # Bỏ qua các token đặc biệt khi in ra
            if idx not in [self.PAD_token, self.SOS_token, self.UNK_token]:
                word = self.idx_to_word.get(idx, "<UNK>")
                words.append(word)
                
        return " ".join(words)

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

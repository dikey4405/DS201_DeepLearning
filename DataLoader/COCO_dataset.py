import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2 as cv
from torchvision import transforms
import numpy as np

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.PAD_idx = 0
        self.SOS_idx = 1
        self.EOS_idx = 2
        self.UNK_idx = 3

    def __len__(self):
        return len(self.word2idx)
    
    def build_vocab(self, all_captions_list):
        word_list = set()
        for caption in all_captions_list:
            word_list.update(caption.split())
        
        for word in sorted(list(word_list)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

class COCODataset(Dataset):
    def __init__(self, image_dir, features_dir, captions_file, vocabulary):
        super().__init__()
        self.image_dir = image_dir
        self.features_dir = features_dir
        self.vocabulary = vocabulary

        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                raw_annotations = json.load(f)
            
            # Cấu trúc lại annotations thành {image_name: [viet_cap1, viet_cap2, ...]}
            self.annotations = self._reformat_annotations(raw_annotations)
            
        except Exception as e:
            print(f"Lỗi khi tải hoặc xử lý file JSON: {e}")
            self.annotations = {}

        self.image_ids = list(self.annotations.keys())
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
            
            if img_name and viet_cap:
                if img_name not in reformatted:
                    reformatted[img_name] = []
                reformatted[img_name].append(viet_cap.lower().strip())
        return reformatted
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Tải Đặc trưng (.NPZ)
        feature_path = self.feature_paths.get(img_id)
        try:
            features = np.load(feature_path)
            V_raw = torch.tensor(features['V_features'], dtype=torch.float)
            g_raw = torch.tensor(features['g_raw'], dtype=torch.float)
        except:
            V_raw = torch.randn(36, 2048)
            g_raw = torch.randn(2048)

        # Lấy TẤT CẢ caption tiếng Việt (dạng string) cho SCST/Evaluation
        viet_captions_list = self.annotations[img_id] 
        
        # Token hóa MỘT caption (caption đầu tiên) cho huấn luyện XE
        caption_xe = viet_captions_list[0]
        tokens_xe = [self.vocab.word_to_idx.get(word, self.vocab.UNK_token) 
                     for word in caption_xe.split()]
        tokens_xe = [self.vocab.SOS_token] + tokens_xe + [self.vocab.EOS_token]
        caption_tensor = torch.tensor(tokens_xe)

        # Trả về img_id (để đánh giá), V_raw, g_raw, tensor (cho XE), list (cho SCST)
        return img_id, V_raw, g_raw, caption_tensor, len(tokens_xe), viet_captions_list
    
def collate_fn(data):
    # Sắp xếp data theo độ dài caption (tốt cho việc đệm)
    data.sort(key=lambda x: x[4], reverse=True)
    img_ids, V_batch, g_batch, captions, lengths, viet_captions_list = zip(*data)
    
    # Xếp chồng V_raw và g_raw
    V_batch = torch.stack(V_batch)
    g_batch = torch.stack(g_batch)

    # Đệm (Pad) captions (cho XE)
    max_len = max(lengths)
    padded_captions = torch.zeros(len(captions), max_len).long()
    for i, caption in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = caption[:end]
        
    # Trả về img_ids (string), features, tensor (XE), lengths, và list (SCST)
    return img_ids, V_batch, g_batch, padded_captions, torch.tensor(lengths), viet_captions_list


        
        
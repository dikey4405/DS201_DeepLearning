# train.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
import json
import os

# --- THÔNG SỐ CONFIG ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 

TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')

FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

# --- HÀM TRAIN_XE ---
def train_xe(model, data_loader, optimizer, criterion, device):
    model.train() 
    total_loss = 0
    for _, V_raw, g_raw, captions, _, _ in data_loader:
        V_raw, g_raw, captions = V_raw.to(device), g_raw.to(device), captions.to(device)
        targets = captions[:, 1:] 
        inputs = captions[:, :-1]  
        optimizer.zero_grad()
        outputs = model(V_raw, g_raw, inputs) 
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.flatten())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# --- HÀM EVALUATE_XE_LOSS ---
def evaluate_xe_loss(model, data_loader, criterion, device):
    model.eval() 
    total_loss = 0
    with torch.no_grad():
        for _, V_raw, g_raw, captions, _, _ in data_loader:
            V_raw, g_raw, captions = V_raw.to(device), g_raw.to(device), captions.to(device)
            targets = captions[:, 1:] 
            inputs = captions[:, :-1]  
            outputs = model(V_raw, g_raw, inputs) 
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.flatten())
            total_loss += loss.item()
    return total_loss / len(data_loader)

# --- HÀM TRAIN_SCST ---
def train_scst(model, data_loader, optimizer, cider_reward_metric, vocab, device, beam_size=5):
    model.train()
    total_loss = 0
    
    for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        optimizer.zero_grad()
        
        # 1. Lấy mẫu (Sampling)
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, beam_size=beam_size)
        
        # 2. Tính Phần thưởng (Reward)
        rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=beam_size)
        
        # 3. Tính Baseline (b)
        baseline = rewards.mean(dim=1, keepdim=True)
        
        # 4. Tính Loss (Công thức 23)
        reward_diff = (rewards - baseline) # (B, k)
        
        # Reshape log_probs (B * k, T) -> (B, k, T)
        log_probs = log_probs.view(V_raw.size(0), beam_size, -1)
        seq_log_probs = log_probs.sum(dim=2) # (B, k)
        
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

# --- HÀM MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    n_head = 8
    num_encoder_layers = 3 
    num_decoder_layers = 3 
    controller_type = 'MAC'
    BATCH_SIZE = 32
    XE_EPOCHS = 50
    SCST_EPOCHS = 10
    BEAM_SIZE_SCST = 5
    
    # --- 2. Chuẩn bị Dữ liệu và Từ vựng ---
    vocab = Vocabulary()
    try:
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw_train_annotations = json.load(f)
        all_train_captions = [item.get('translate') 
                              for item in raw_train_annotations if item.get('translate')]
        vocab.build_vocab(all_train_captions)
        vocab_size = len(vocab)
        print(f"Xây dựng Vocab thành công: {vocab_size} từ.")
    except Exception as e:
        print(f"Lỗi khi xây dựng vocab: {e}")
        return
    
    # --- SỬA ĐỔI Ở ĐÂY: Truyền đúng folder ảnh cho từng tập ---
    print(f"Loading Train data from: {TRAIN_IMAGE_DIR}")
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    print(f"Loading Val data from: {VAL_IMAGE_DIR}")
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # --- 3. Khởi tạo Mô hình và Optimizer ---
    model = GET(vocab_size, d_model, n_head, num_encoder_layers, num_decoder_layers, controller_type=controller_type).to(device)
    
    # --- 4. Pha 1: Pre-training (XE) với Validation ---
    print(f"Bắt đầu Pha 1: Pre-training (XE) trong {XE_EPOCHS} epochs...")
    optimizer_xe = Adam(model.parameters(), lr=1e-4) 
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token).to(device) 
    best_val_loss = float('inf')

    for epoch in range(XE_EPOCHS):
        train_loss = train_xe(model, train_loader, optimizer_xe, criterion, device)
        val_loss = evaluate_xe_loss(model, val_loader, criterion, device)
        print(f"XE Epoch {epoch+1}/{XE_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"  -> Đã lưu mô hình XE tốt nhất: {BEST_XE_MODEL_PATH}")
            
    # --- 5. Pha 2: Fine-tuning (SCST) ---
    print(f"\nBắt đầu Pha 2: Fine-tuning (SCST) trong {SCST_EPOCHS} epochs...")
    
    # Tải mô hình tốt nhất từ Pha 1
    model.load_state_dict(torch.load(BEST_XE_MODEL_PATH))
    
    optimizer_scst = Adam(model.parameters(), lr=5e-6) # Learning rate rất thấp
    cider_metric = CIDErReward(vocab, device)
    
    for epoch in range(SCST_EPOCHS):
        # Lưu ý: SCST cần train_loader có chứa gt_captions (list string)
        scst_loss = train_scst(model, train_loader, optimizer_scst, cider_metric, vocab, device, beam_size=BEAM_SIZE_SCST)
        print(f"SCST Epoch {epoch+1}/{SCST_EPOCHS} - SCST Loss: {scst_loss:.4f}")
        
        # Lưu checkpoint SCST
        torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)

    print("\nHoàn thành huấn luyện.")
    print(f"Mô hình SCST cuối cùng đã lưu tại: {BEST_SCST_MODEL_PATH}")

if __name__ == '__main__':
    main()

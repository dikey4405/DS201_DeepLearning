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

ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')

FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8 
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

print(f"Gradient Accumulation Config: Effective BS={EFFECTIVE_BATCH_SIZE}, Micro BS={MICRO_BATCH_SIZE}, Steps={GRAD_ACCUM_STEPS}")

# --- HÀM TRAIN_XE (Có Accumulation) ---
def train_xe(model, data_loader, optimizer, criterion, device):
    model.train() 
    total_loss = 0
    optimizer.zero_grad() 
    
    for i, (_, V_raw, g_raw, captions, _, _) in enumerate(data_loader):
        V_raw, g_raw, captions = V_raw.to(device), g_raw.to(device), captions.to(device)
        
        targets = captions[:, 1:] 
        inputs = captions[:, :-1]  
        
        outputs = model(V_raw, g_raw, inputs) 
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.flatten())
        
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    model.eval()
    all_rewards = []
    print("Evaluating CIDEr on Validation set...")
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1)
            rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=1)
            all_rewards.append(rewards.cpu())
    all_rewards = torch.cat(all_rewards)
    return all_rewards.mean().item()

# --- HÀM TRAIN_SCST (Có Accumulation) ---
def train_scst(model, data_loader, optimizer, cider_reward_metric, vocab, device, beam_size=5):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (_, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        
        # 1. Sampling & Reward
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, beam_size=beam_size)
        rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=beam_size)
        
        # 2. Baseline & Loss
        baseline = rewards.mean(dim=1, keepdim=True)
        reward_diff = (rewards - baseline)
        
        log_probs = log_probs.view(V_raw.size(0), beam_size, -1)
        seq_log_probs = log_probs.sum(dim=2)
        
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        # 3. Gradient Accumulation
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

# --- HÀM MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    n_head = 8
    num_encoder_layers = 3 
    num_decoder_layers = 3 
    controller_type = 'MAC'
    
    XE_EPOCHS = 50
    SCST_EPOCHS = 15
    BEAM_SIZE_SCST = 5 
    
    # --- 2. Chuẩn bị Dữ liệu ---
    vocab = Vocabulary()
    try:
        print(f"Đang đọc file caption từ: {CAPTION_TRAIN_JSON}")
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
    
    # --- DATALOADER VỚI MICRO BATCH SIZE ---
    # Lưu ý: Ở đây ta dùng MICRO_BATCH_SIZE (8) thay vì 32
    print(f"Loading Train data with Micro Batch Size: {MICRO_BATCH_SIZE}")
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    print(f"Loading Val data...")
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # --- 3. Khởi tạo Mô hình ---
    model = GET(vocab_size, d_model, n_head, num_encoder_layers, num_decoder_layers, controller_type=controller_type).to(device)
    
    # --- 4. Pha 1: Pre-training (XE) ---
    print(f"\n=== Bắt đầu Pha 1: XE Training ({XE_EPOCHS} epochs) ===")
    optimizer_xe = Adam(model.parameters(), lr=1e-4) 
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token).to(device) 
    cider_metric = CIDErReward(vocab, device)
    
    best_val_cider = 0.0

    for epoch in range(XE_EPOCHS):
        train_loss = train_xe(model, train_loader, optimizer_xe, criterion, device)
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"XE Epoch {epoch+1}/{XE_EPOCHS} | Train Loss: {train_loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"  --> New Best XE Model Saved (CIDEr: {val_cider:.4f})")
            
    # --- 5. Pha 2: Fine-tuning (SCST) ---
    print(f"\n=== Bắt đầu Pha 2: SCST Training ({SCST_EPOCHS} epochs) ===")
    
    if os.path.exists(BEST_XE_MODEL_PATH):
        print(f"Loading best XE model from {BEST_XE_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_XE_MODEL_PATH))
    else:
        print("Warning: Không tìm thấy XE model, bắt đầu SCST từ checkpoint hiện tại.")
    
    optimizer_scst = Adam(model.parameters(), lr=5e-6)
    best_scst_cider = best_val_cider
    
    for epoch in range(SCST_EPOCHS):
        scst_loss = train_scst(model, train_loader, optimizer_scst, cider_metric, vocab, device, beam_size=BEAM_SIZE_SCST)
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"SCST Epoch {epoch+1}/{SCST_EPOCHS} | SCST Loss: {scst_loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        if val_cider > best_scst_cider:
            best_scst_cider = val_cider
            torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)
            print(f"  --> New Best SCST Model Saved (CIDEr: {val_cider:.4f})")

    print("\nHoàn thành toàn bộ quá trình huấn luyện.")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

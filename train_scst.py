import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
import json
import os
import numpy as np

# --- CONFIG ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')
FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'

# Load đúng model 0.63
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

# Batch config
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# --- HÀM TRAIN SCST (SIÊU ỔN ĐỊNH) ---
def train_scst_epoch(model, data_loader, optimizer, cider_reward_metric, vocab, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (_, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        
        # --- 1. Greedy Baseline (Không tính gradient) ---
        model.eval()
        with torch.no_grad():
            # Greedy search (Beam=1)
            greedy_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1)
            reward_baseline = cider_reward_metric.compute(greedy_seqs, gt_captions_list, beam_size=1)
        
        # --- 2. Sampling (Có tính gradient) ---
        model.train()
        # Giữ beam_size=5 để explore tốt
        sample_beam_size = 5 
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, beam_size=sample_beam_size)
        reward_sample = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=sample_beam_size)
        
        # --- 3. Tính Advantage (Quan trọng) ---
        # Mở rộng baseline: (B, 1) -> (B, 5)
        reward_baseline = reward_baseline.expand(-1, sample_beam_size)
        
        # Advantage = Sample - Greedy
        reward_diff = reward_sample - reward_baseline
        
        # [THAY ĐỔI] Bỏ qua Normalization phức tạp, chỉ dùng diff thuần túy 
        # hoặc chia cho một hằng số cố định nếu muốn giảm biên độ
        # Lý do: Với Micro_Batch=8, tính mean/std của batch rất nhiễu.
        # reward_diff = reward_diff  <-- Dùng trực tiếp là an toàn nhất với batch nhỏ
        
        # --- 4. Masking & Loss ---
        # Reshape: (B, 5, T)
        log_probs = log_probs.view(V_raw.size(0), sample_beam_size, -1)
        
        # Masking: Tạo mask để không tính log_prob của PAD token (quan trọng)
        # sampled_seqs: (B*5, T) -> view (B, 5, T)
        reshaped_seqs = sampled_seqs.view(V_raw.size(0), sample_beam_size, -1)
        mask = (reshaped_seqs != vocab.PAD_token).float()
        
        # Chỉ tính log_prob của các từ thật (không tính PAD)
        # log_probs * mask -> sum
        seq_log_probs = (log_probs * mask).sum(dim=2) # (B, 5)
        
        # Loss = - Advantage * LogProb / (Batch size)
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        # --- 5. Gradient Accumulation & Clipping ---
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            # [THAY ĐỔI] Kẹp gradient cực chặt (0.1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

# --- EVALUATE ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    model.eval()
    all_rewards = []
    print("Evaluating CIDEr on Validation set...")
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            # Eval bằng Greedy (Beam=1) cho nhanh và chuẩn baseline
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1) 
            rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=1)
            all_rewards.append(rewards.cpu())
    return torch.cat(all_rewards).mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vocab
    vocab = Vocabulary()
    try:
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        clean_captions = [i.get('translate') for i in raw if i.get('translate') and isinstance(i.get('translate'), str) and len(i.get('translate').strip())>0]
        vocab.build_vocab(clean_captions)
    except Exception as e:
        print(f"Error: {e}"); return

    # 2. Dataloaders
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 3. Model Setup
    model = GET(
        vocab_size=len(vocab), 
        d_model=512, 
        n_head=8, 
        num_encoder_layers=3, 
        num_decoder_layers=3, 
        dropout=0.2, # Giữ dropout 0.2
        controller_type='MAC'
    ).to(device)
    
    # Load XE Model
    if os.path.exists(BEST_XE_MODEL_PATH):
        print(f"--> Loading BEST XE Model from: {BEST_XE_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_XE_MODEL_PATH, map_location=device))
    else:
        print("CRITICAL: XE Model not found!"); return

    # 4. SCST Config [QUAN TRỌNG NHẤT]
    # LR cực thấp: 5e-7 (Năm mũ trừ bảy)
    # Đây là chìa khóa để không phá vỡ model
    optimizer_scst = Adam(model.parameters(), lr=5e-6)
    cider_metric = CIDErReward(vocab, device)
    
    EPOCHS = 15
    
    print("Checking initial CIDEr of loaded model...")
    # Kiểm tra ngay lập tức xem load model có đúng không
    best_scst_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
    print(f"Initial CIDEr: {best_scst_cider:.4f}")
    
    # Nếu ban đầu load vào mà thấp (ví dụ 0.01) -> File .pth bị lỗi hoặc vocab không khớp
    if best_scst_cider < 0.1:
        print("WARNING: Model load vào có CIDEr quá thấp. Kiểm tra lại file path hoặc Vocab!")

    print(f"\n=== Starting Phase 2: SCST Training (Low LR & Masking) ===")
    
    for epoch in range(EPOCHS):
        scst_loss = train_scst_epoch(model, train_loader, optimizer_scst, cider_metric, vocab, device)
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"Epoch {epoch+1} | Loss: {scst_loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        # Chỉ lưu khi CIDEr thực sự cải thiện
        if val_cider > best_scst_cider:
            best_scst_cider = val_cider
            torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)
            print(f"--> Saved New Best SCST Model (CIDEr: {best_scst_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()


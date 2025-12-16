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

BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# --- HÀM TRAIN SCST (Đã sửa lỗi tham số) ---
def train_scst_epoch(model, data_loader, optimizer, cider_reward_metric, vocab, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (_, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        
        # 1. Greedy Baseline (Dùng sample_k=1)
        model.eval()
        with torch.no_grad():
            # SỬA LỖI: beam_size -> sample_k
            greedy_seqs, _ = model.sample(V_raw, g_raw, vocab, sample_k=1)
            reward_baseline = cider_reward_metric.compute(greedy_seqs, gt_captions_list, beam_size=1)
        
        # 2. Sample (Dùng sample_k=5)
        model.train()
        sample_k_val = 5
        # SỬA LỖI: beam_size -> sample_k
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, sample_k=sample_k_val)
        reward_sample = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=sample_k_val)
        
        # 3. Advantage
        # Mở rộng baseline để khớp kích thước: (B, 1) -> (B, 5) -> flatten
        reward_baseline = reward_baseline.expand(-1, sample_k_val).contiguous().view(-1)
        reward_sample = reward_sample.view(-1)
        
        reward_diff = reward_sample - reward_baseline
        
        # Normalization (Quan trọng cho SCST)
        if reward_diff.std() > 0:
             reward_diff = (reward_diff - reward_diff.mean()) / (reward_diff.std() + 1e-9)
        
        # 4. Loss
        # log_probs shape gốc: (B * sample_k, Max_Len)
        # Tính tổng log_prob của cả câu
        # mask những chỗ pad (nếu log_probs chưa xử lý mask, nhưng thường model trả về 0 ở pad rồi)
        seq_log_probs = log_probs.sum(dim=1) # (B * sample_k)
        
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        # 5. Accumulate
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

# --- EVALUATE (Đã sửa lỗi tham số) ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    model.eval()
    all_rewards = []
    print("Evaluating CIDEr on Validation set...")
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            # SỬA LỖI: beam_size -> sample_k
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, sample_k=1) 
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
        dropout=0.2, 
        controller_type='MAC'
    ).to(device)
    
    # Load XE Model
    if os.path.exists(BEST_XE_MODEL_PATH):
        print(f"--> Loading BEST XE Model from: {BEST_XE_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_XE_MODEL_PATH, map_location=device))
    else:
        print("CRITICAL: XE Model not found!"); return

    # --- TỐI ƯU HÓA QUAN TRỌNG: ĐÓNG BĂNG ENCODER ---
    print("Freezing Encoder for SCST stability...")
    # Đóng băng Encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    # Đóng băng các lớp Projection đầu vào
    for p in model.v_proj.parameters():
        p.requires_grad = False
    for p in model.g_proj.parameters():
        p.requires_grad = False
        
    # Chỉ đưa các tham số requires_grad=True (Decoder) vào Optimizer
    optimizer_scst = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-7) 
    
    cider_metric = CIDErReward(vocab, device)
    
    EPOCHS = 10
    
    print("Checking initial CIDEr...")
    best_scst_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
    print(f"Initial CIDEr: {best_scst_cider:.4f}")

    print(f"\n=== Starting Phase 2: SCST Training (Stable Greedy Baseline) ===")
    
    for epoch in range(EPOCHS):
        scst_loss = train_scst_epoch(model, train_loader, optimizer_scst, cider_metric, vocab, device)
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"Epoch {epoch+1} | Loss: {scst_loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        if val_cider > best_scst_cider:
            best_scst_cider = val_cider
            torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)
            print(f"--> Saved Best SCST Model (CIDEr: {best_scst_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

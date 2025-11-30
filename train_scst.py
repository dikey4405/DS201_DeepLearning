import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
import json
import os
import traceback
import numpy as np

ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')
FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'

# Load model XE
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
# Save model SCST mới
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

# Config SCST
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# --- HÀM TRAIN SCST ---
def train_scst_epoch(model, data_loader, optimizer, cider_reward_metric, vocab, device, beam_size):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (_, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        
        # 1. Lấy mẫu (Sampling) bằng Beam Search hoặc Multinomial
        # sampled_seqs: (Micro_BS * Beam, T)
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, beam_size=beam_size)
        
        # 2. Tính Reward (CIDEr)
        # rewards: (Micro_BS, Beam)
        rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=beam_size)
        
        # 3. Tính Baseline (Trung bình reward của các beam)
        baseline = rewards.mean(dim=1, keepdim=True)
        
        # 4. Tính Advantage (Lợi thế)
        reward_diff = rewards - baseline
        
        if reward_diff.std() > 0:
            reward_diff = (reward_diff - reward_diff.mean()) / (reward_diff.std() + 1e-9)
        
        # 5. Tính Loss SCST
        # Reshape log_probs: (Micro_BS, Beam, T)
        log_probs = log_probs.view(V_raw.size(0), beam_size, -1)
        
        # Tổng log_prob của cả câu: (Micro_BS, Beam)
        seq_log_probs = log_probs.sum(dim=2)
        
        # Loss = - Advantage * LogProb
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        # 6. Gradient Accumulation
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

# --- HÀM EVALUATE CIDEr ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    model.eval()
    all_rewards = []
    print("Evaluating CIDEr on Validation set...")
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            # Dùng Greedy (beam=1) hoặc Beam=3 để đánh giá
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1) 
            rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=1)
            all_rewards.append(rewards.cpu())
    all_rewards = torch.cat(all_rewards)
    return all_rewards.mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vocab
    vocab = Vocabulary()
    try:
        print(f"Loading vocab from: {CAPTION_TRAIN_JSON}")
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        clean_captions = []
        for i in raw:
            val = i.get('translate')
            if val is not None and isinstance(val, str) and len(val.strip()) > 0:
                clean_captions.append(val)
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Dataloaders
    print("Initializing Dataloaders...")
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 3. Model Setup
    # Cần khớp tham số với lúc train XE (512, 8, 3, 3)
    model = GET(len(vocab), 512, 8, 3, 3, controller_type='MAC').to(device)
    
    # --- LOAD MODEL XE ĐÃ TRAIN (0.59 CIDEr) ---
    if os.path.exists(BEST_XE_MODEL_PATH):
        print(f"--> Loading BEST XE Model from: {BEST_XE_MODEL_PATH}")
        state_dict = torch.load(BEST_XE_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("CRITICAL ERROR: Không tìm thấy file model XE! Vui lòng train XE trước.")
        return

    # 4. SCST Config
    # Learning rate rất nhỏ cho SCST (5e-6 hoặc 5e-7)
    optimizer_scst = Adam(model.parameters(), lr=5e-6)
    cider_metric = CIDErReward(vocab, device)
    
    EPOCHS = 15 
    BEAM_SIZE = 5 
    
    # Lấy baseline CIDEr hiện tại
    print("Checking initial CIDEr of XE model...")
    best_scst_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
    print(f"Initial CIDEr on Dev: {best_scst_cider:.4f}")

    print(f"\n=== Starting Phase 2: SCST Training ({EPOCHS} epochs) ===")
    
    for epoch in range(EPOCHS):
        scst_loss = train_scst_epoch(model, train_loader, optimizer_scst, cider_metric, vocab, device, BEAM_SIZE)
        
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"SCST Epoch {epoch+1}/{EPOCHS} | SCST Loss: {scst_loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        # Save if better
        if val_cider > best_scst_cider:
            best_scst_cider = val_cider
            torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)
            print(f"--> Saved Best SCST Model (CIDEr: {best_scst_cider:.4f})")

    print("\nHoàn thành Fine-tuning SCST.")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

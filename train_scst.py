# train_scst.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
import json
import os

# --- CONFIG ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')
FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'
# Load model XE tốt nhất
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'
# Save model SCST
BEST_SCST_MODEL_PATH = '/kaggle/working/get_model_best_scst.pth'

# Gradient Accumulation Config
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

def train_scst_epoch(model, data_loader, optimizer, cider_metric, vocab, device, beam_size):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (_, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        
        # 1. Sample & Reward
        # sampled_seqs: (Micro_BS * Beam, T)
        sampled_seqs, log_probs = model.sample(V_raw, g_raw, vocab, beam_size=beam_size)
        
        # Tính reward (đã fix lỗi assert)
        # rewards: (Micro_BS, Beam)
        rewards = cider_metric.compute(sampled_seqs, gt_captions_list, beam_size=beam_size)
        
        # [cite_start]2. SCST Loss [cite: 221]
        baseline = rewards.mean(dim=1, keepdim=True)
        reward_diff = rewards - baseline
        
        log_probs = log_probs.view(V_raw.size(0), beam_size, -1)
        seq_log_probs = log_probs.sum(dim=2)
        
        loss = -reward_diff * seq_log_probs
        loss = loss.mean()
        
        # 3. Accumulate
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        
    return total_loss / len(data_loader)

def evaluate_cider(model, data_loader, cider_metric, vocab, device):
    model.eval()
    all_rewards = []
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            # Greedy check
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1)
            rewards = cider_metric.compute(sampled_seqs, gt_captions_list, beam_size=1)
            all_rewards.append(rewards.cpu())
    return torch.cat(all_rewards).mean().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary()
    
    # Load Vocab
    try:
        print(f"Loading vocab from: {CAPTION_TRAIN_JSON}")
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        clean_captions = []
        for i in raw:
            val = i.get('translate')
            if val is not None and isinstance(val, str) and len(val.strip()) > 0:
                clean_captions.append(val)
        
        print(f"Found {len(clean_captions)} valid captions out of {len(raw)} items.")
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
        
    except Exception as e:
        print(f"Error loading vocab: {e}")
        # In chi tiết lỗi để debug nếu cần
        import traceback
        traceback.print_exc()
        return

    # Dataloaders
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model Init
    model = GET(len(vocab), 512, 8, 3, 3, controller_type='MAC').to(device)
    
    # Load XE Weights
    if os.path.exists(BEST_XE_MODEL_PATH):
        print(f"Loading XE weights from {BEST_XE_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_XE_MODEL_PATH))
    else:
        print("Warning: XE weights not found!")

    # [cite_start]SCST Config [cite: 238]
    optimizer_scst = Adam(model.parameters(), lr=5e-6)
    cider_metric = CIDErReward(vocab, device)
    
    EPOCHS = 15
    BEAM_SIZE = 5
    best_scst_cider = 0.0 
    print(f"=== Starting Phase 2: SCST Training ({EPOCHS} epochs) ===")
    
    for epoch in range(EPOCHS):
        loss = train_scst_epoch(model, train_loader, optimizer_scst, cider_metric, vocab, device, BEAM_SIZE)
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"Epoch {epoch+1} | SCST Loss: {loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        if val_cider > best_scst_cider:
            best_scst_cider = val_cider
            torch.save(model.state_dict(), BEST_SCST_MODEL_PATH)
            print(f"--> Saved Best SCST Model (CIDEr: {best_scst_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    main()

# train_xe.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
import json
import os
import traceback

# --- CONFIG ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')
FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'

EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# --- TRAINING FUNCTION ---
def train_xe_epoch(model, data_loader, optimizer, criterion, device):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
    return total_loss / len(data_loader)

# --- EVALUATION FUNCTION (Updated for Beam Search) ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device, beam_size=3):
    model.eval()
    all_rewards = []
    print(f"Evaluating CIDEr on Validation set (Beam Size={beam_size})...")
    
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=beam_size)
            
            rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=beam_size)
            
            # Lấy điểm trung bình của các beam tốt nhất (hoặc trung bình beam)
            all_rewards.append(rewards.cpu())
            
    all_rewards = torch.cat(all_rewards)
    return all_rewards.mean().item()

# --- MAIN ---
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
        
        print(f"Found {len(clean_captions)} valid captions.")
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
        
    except Exception as e:
        print(f"Error loading vocab: {e}")
        traceback.print_exc()
        return

    # 2. Dataloaders
    print(f"Loading Train data from: {TRAIN_IMAGE_DIR}")
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    print(f"Loading Val data from: {VAL_IMAGE_DIR}")
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = GET(
        vocab_size=len(vocab), 
        d_model=512, 
        n_head=8, 
        num_encoder_layers=3, 
        num_decoder_layers=4, 
        dropout=0.3,         
        controller_type='MAC'
    ).to(device)
    
    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token, label_smoothing=0.2).to(device)
    cider_metric = CIDErReward(vocab, device)
    
    base_lr = 5e-4 
    optimizer = Adam(model.parameters(), lr=base_lr)

    # 5. Config Training
    best_val_cider = 0.0
    EPOCHS = 50
    WARMUP_EPOCHS = 5
    
    # --- Scheduler: Warmup + Cosine Decay ---
    # Giai đoạn 1: Warmup từ LR rất nhỏ lên base_lr trong 5 epoch
    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    # Giai đoạn 2: Cosine Decay từ base_lr xuống min_lr trong các epoch còn lại
    scheduler2 = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    # Kết hợp
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[WARMUP_EPOCHS])

    print(f"=== Starting Phase 1: XE Training (Optimized) ===")
    print(f"Config: Layers=4, Dropout=0.3, Smooth=0.2, Warmup={WARMUP_EPOCHS}, Eval Beam=3")

    for epoch in range(EPOCHS):
        loss = train_xe_epoch(model, train_loader, optimizer, criterion, device)
        
        # Đánh giá CIDEr với Beam Search = 3
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device, beam_size=3)
        
        # Cập nhật Scheduler (Step theo epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val CIDEr (Beam3): {val_cider:.4f} | LR: {current_lr:.2e}")
        
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"--> Saved Best XE Model (CIDEr: {best_val_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

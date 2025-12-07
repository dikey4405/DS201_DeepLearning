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

# --- CONFIG (Đường dẫn Kaggle) ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TRAIN_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'train') 
VAL_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'dev')
FEATURE_DIR = '/kaggle/working/coco_features_2048d/'
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json'
CAPTION_VAL_JSON = '/kaggle/input/data-dl/Captions/dev.json'
BEST_XE_MODEL_PATH = '/kaggle/working/get_model_best_xe.pth'

# Cấu hình Batch Size để tránh OOM
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

# --- HÀM TRAIN ---
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
        
        # Accumulate Gradient
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
    return total_loss / len(data_loader)

# --- HÀM EVALUATE (Greedy Search) ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    model.eval()
    all_rewards = []
    print("Evaluating CIDEr (Greedy)...")
    
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            
            # Dùng Greedy Search (beam_size=1)
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
        
        print(f"Found {len(clean_captions)} valid captions.")
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
        
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return

    # 2. Dataloaders
    print("Initializing Dataloaders...")
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
    
    # 4. Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token, label_smoothing=0.1).to(device)
    
    cider_metric = CIDErReward(vocab, device)

    # 5. Training Loop
    best_val_cider = 0.0
    EPOCHS = 50
    
    print(f"=== Starting Phase 1: XE Training (With Label Smoothing 0.1) ===")
    for epoch in range(EPOCHS):
        loss = train_xe_epoch(model, train_loader, optimizer, criterion, device)
        
        # Đánh giá
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Val CIDEr: {val_cider:.4f}")
        
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"--> Saved Best XE Model (CIDEr: {best_val_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

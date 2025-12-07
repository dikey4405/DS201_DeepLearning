import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
# Import Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

# --- HÀM TRAIN (Tính Loss) ---
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
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM_STEPS
    return total_loss / len(data_loader)

# --- HÀM EVALUATE (Tính CIDEr) ---
def evaluate_cider(model, data_loader, cider_reward_metric, vocab, device):
    """
    Chạy sinh caption trên tập Validation và tính điểm CIDEr trung bình.
    Sử dụng Greedy Search (beam_size=1) để đánh giá nhanh trong quá trình train.
    """
    model.eval()
    all_rewards = []
    
    # Không cần tính gradient khi evaluate
    with torch.no_grad():
        for _, V_raw, g_raw, _, _, gt_captions_list in data_loader:
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            
            # 1. Sinh caption (Greedy search: beam_size=1)
            # Hàm sample trả về (B*1, T)
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1)
            
            # 2. Tính điểm CIDEr cho batch này
            # rewards shape: (B, 1)
            rewards = cider_reward_metric.compute(sampled_seqs, gt_captions_list, beam_size=1)
            
            all_rewards.append(rewards.cpu())
            
    # Tính trung bình CIDEr trên toàn bộ tập Val
    all_rewards = torch.cat(all_rewards)
    mean_cider = all_rewards.mean().item()
    
    return mean_cider

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
        dropout=0.1, 
        controller_type='MAC'
    ).to(device)
    
    # 4. Optimizer & Metric
    optimizer = Adam(model.parameters(), lr=3e-4) # LR khởi điểm 3e-4
    
    # Scheduler: Giảm LR nếu CIDEr không tăng (mode='max')
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token, label_smoothing=0.1).to(device)
    
    # Khởi tạo CIDEr metric để đánh giá
    cider_metric = CIDErReward(vocab, device)

    # 5. Training Loop
    best_val_cider = 0.0 # Theo dõi CIDEr cao nhất
    EPOCHS = 30
    
    print(f"=== Starting Phase 1: XE Training (Saving based on Val CIDEr) ===")
    
    for epoch in range(EPOCHS):
        # Train (vẫn dùng Loss để cập nhật trọng số)
        train_loss = train_xe_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate (Dùng CIDEr để chọn model)
        print(f"Evaluating Epoch {epoch+1}...")
        val_cider = evaluate_cider(model, val_loader, cider_metric, vocab, device)
        
        # Cập nhật Scheduler dựa trên CIDEr
        scheduler.step(val_cider)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val CIDEr: {val_cider:.4f} | LR: {current_lr:.2e}")
        
        # Lưu model nếu CIDEr tăng
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"--> Saved Best XE Model (New Best CIDEr: {best_val_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

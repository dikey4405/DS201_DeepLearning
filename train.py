import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from DataLoader.Cider_reward import CIDErReward 
# Import cần thiết cho evaluate_cider mới
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.cider.cider import Cider
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

# Batch Size
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE

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

# --- HÀM EVALUATE CIDEr (CORPUS LEVEL - CHUẨN) ---
def evaluate_cider(model, data_loader, vocab, device):
    """
    Đánh giá CIDEr trên TOÀN BỘ tập dữ liệu (Corpus-level).
    """
    model.eval()
    
    # Containers để chứa toàn bộ kết quả
    gts = {} # Ground Truths
    res = {} # Results (Predictions)
    
    print("Generating captions for Evaluation (Greedy)...")
    
    # Hàm decode từ index sang string
    def decode_seq(seq):
        words = []
        for idx in seq:
            idx = idx.item()
            if idx == vocab.EOS_token: break
            if idx not in [vocab.SOS_token, vocab.PAD_token]:
                words.append(vocab.idx_to_word.get(idx, "<UNK>"))
        return " ".join(words)

    with torch.no_grad():
        for batch_idx, (img_ids, V_raw, g_raw, _, _, gt_captions_list) in enumerate(data_loader):
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            
            # 1. Sinh caption (Greedy Search cho nhanh và ổn định ở Pha 1)
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, beam_size=1)
            
            # 2. Gom kết quả
            for i in range(len(img_ids)):
                # Tạo ID duy nhất cho ảnh (img_ids từ dataloader là tuple)
                image_id = str(img_ids[i])
                
                # Decode câu sinh ra
                pred_sent = decode_seq(sampled_seqs[i])
                
                # Lưu vào dict theo chuẩn COCO eval
                res[image_id] = [{'caption': pred_sent}]
                gts[image_id] = [{'caption': c} for c in gt_captions_list[i]]
                
                # --- DEBUG: In ra 2 mẫu đầu tiên để xem model nói gì ---
                if batch_idx == 0 and i < 2:
                    print(f"\n[DEBUG Sample {i}]")
                    print(f"  GT: {gt_captions_list[i][0]}")
                    print(f"  Pred: {pred_sent}")

    # 3. Tính điểm CIDEr một lần cho toàn bộ tập
    print(f"Computing CIDEr for {len(res)} images...")
    
    # Tokenize (Chuẩn hóa dấu câu, khoảng trắng...)
    # Lưu ý: PTBTokenizer có thể in ra một số log, không cần lo lắng
    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)
    
    # Compute Score
    scorer = Cider()
    score, _ = scorer.compute_score(gts_tokenized, res_tokenized)
    
    return score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vocab
    vocab = Vocabulary()
    try:
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        clean_captions = [i.get('translate') for i in raw if i.get('translate') and isinstance(i.get('translate'), str) and len(i.get('translate').strip())>0]
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
    except Exception as e:
        print(e); return

    # 2. Dataloaders
    train_dataset = COCODataset(TRAIN_IMAGE_DIR, FEATURE_DIR, CAPTION_TRAIN_JSON, vocab)
    train_loader = DataLoader(train_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataset = COCODataset(VAL_IMAGE_DIR, FEATURE_DIR, CAPTION_VAL_JSON, vocab)
    val_loader = DataLoader(val_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 3. Model Setup (Visual Genome Features rất tốt, nên dùng config chuẩn)
    model = GET(
        vocab_size=len(vocab), 
        d_model=512, 
        n_head=8, 
        num_encoder_layers=3, 
        num_decoder_layers=3, # 3 layers là đủ với features xịn
        dropout=0.2,          # Dropout vừa phải
        controller_type='MAC'
    ).to(device)
    
    # 4. Optimizer
    optimizer = Adam(model.parameters(), lr=3e-4) # LR 3e-4 cho XE là tốt
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_token, label_smoothing=0.1).to(device)
    
    # Lưu ý: Không cần khởi tạo CIDErReward ở đây vì hàm evaluate_cider tự lo

    # 5. Training Loop
    best_val_cider = 0.0
    EPOCHS = 30
    
    print(f"=== Starting Phase 1: XE Training (Corpus Eval) ===")
    for epoch in range(EPOCHS):
        loss = train_xe_epoch(model, train_loader, optimizer, criterion, device)
        
        # Gọi hàm evaluate mới (không cần truyền metric object)
        val_cider = evaluate_cider(model, val_loader, vocab, device)
        
        scheduler.step(val_cider)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val CIDEr: {val_cider:.4f} | LR: {current_lr:.2e}")
        
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(model.state_dict(), BEST_XE_MODEL_PATH)
            print(f"--> Saved Best XE Model (CIDEr: {best_val_cider:.4f})")

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()

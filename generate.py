import torch
import torch.nn as nn
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import numpy as np
from pyvi import ViTokenizer

# --- CONFIG (Cần khớp với môi trường Kaggle của bạn) ---
ROOT_IMAGE_DIR = '/kaggle/input/data-dl/Images/Images' 
TEST_IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, 'test') 

# Đường dẫn tới thư mục chứa features .npz
FEATURE_DIR = '/kaggle/working/coco_features_2048d' 

# File JSON Caption
CAPTION_TRAIN_JSON = '/kaggle/input/data-dl/Captions/train.json' 
CAPTION_TEST_JSON = '/kaggle/input/data-dl/Captions/test.json'   

# File Checkpoint
CHECKPOINT_PATH = '/kaggle/input/image-captioning/pytorch/default/5/get_model_best_scst.pth'
if not os.path.exists(CHECKPOINT_PATH):
    print("Không tìm thấy model SCST, chuyển sang dùng model XE...")
    CHECKPOINT_PATH = '/kaggle/input/image-captioning/pytorch/default/5/get_model_best_xe.pth'

OUTPUT_RESULT_FILE = '/kaggle/working/caption_results.json'

# Tham số mô hình
D_MODEL = 512
N_HEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.2
CONTROLLER_TYPE = 'MAC'
BEAM_SIZE_INFERENCE = 3 # Beam 3 hoặc 5 thường cho kết quả tốt nhất

def decode_caption(seq, vocab):
    """
    Chuyển đổi chuỗi index thành câu văn và hậu xử lý để tăng CIDEr.
    """
    words = []
    for idx in seq:
        # Xử lý nếu idx là tensor
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
            
        if idx == vocab.EOS_token:
            break
            
        # Bỏ qua các token đặc biệt
        if idx not in [vocab.SOS_token, vocab.PAD_token, vocab.UNK_token]:
            # Lấy từ từ dictionary
            word = vocab.idx_to_word.get(idx, "<UNK>")
            words.append(word)
    
    # 1. Ghép thành câu
    caption = " ".join(words)
    
    # 2. QUAN TRỌNG: Xóa dấu gạch dưới do ViTokenizer sinh ra
    # Nếu Vocab class của bạn chưa xử lý, dòng này sẽ cứu cánh cho điểm CIDEr
    caption = caption.replace("_", " ")
    
    # 3. HẬU XỬ LÝ (Post-processing): Xóa từ lặp liên tiếp
    # VD: "con mèo con mèo đang ngủ" -> "con mèo đang ngủ"
    cleaned_words = caption.split()
    if not cleaned_words:
        return ""
        
    final_words = [cleaned_words[0]]
    for i in range(1, len(cleaned_words)):
        if cleaned_words[i] != cleaned_words[i-1]:
            final_words.append(cleaned_words[i])
            
    return " ".join(final_words)

def main_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}")
    
    # 1. Xây dựng Vocab (Bắt buộc dùng tập TRAIN để khớp index)
    vocab = Vocabulary()
    try:
        print(f"Loading vocab from {CAPTION_TRAIN_JSON}...")
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        # Lọc và chuẩn hóa caption giống hệt Dataset
        clean_captions = []
        for i in raw:
            cap = i.get('translate')
            if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                # Đảm bảo logic này khớp với logic trong COCODataset
                clean_captions.append(cap)
                
        vocab.build_vocab(clean_captions)
        print(f"Vocab size: {len(vocab)}")
    except Exception as e:
        print(f"Lỗi khi xây dựng vocab: {e}"); return
        
    # 2. Tải Test Dataloader
    if not os.path.exists(FEATURE_DIR):
        print(f"CRITICAL ERROR: Không tìm thấy thư mục features tại {FEATURE_DIR}")
        return

    print("Initializing Test Dataloader...")
    try:
        test_dataset = COCODataset(TEST_IMAGE_DIR, FEATURE_DIR, CAPTION_TEST_JSON, vocab)
        # Batch size = 1 để xử lý từng ảnh một cách chính xác
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
    except Exception as e:
        print(f"Lỗi khởi tạo Dataloader: {e}")
        return
    
    # 3. Tải Mô hình
    print("Loading Model...")
    model = GET(
        vocab_size=len(vocab), 
        d_model=D_MODEL, 
        n_head=N_HEAD, 
        num_encoder_layers=NUM_ENCODER_LAYERS, 
        num_decoder_layers=NUM_DECODER_LAYERS, 
        dropout=DROPOUT,
        controller_type=CONTROLLER_TYPE
    )
    
    try:
        print(f"Loading weights from: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file checkpoint!")
        return
    except Exception as e:
        print(f"Lỗi khi load weights: {e}")
        return

    model.to(device)
    model.eval() # Quan trọng: Tắt Dropout

    results = [] # List chứa kết quả chuẩn format COCO
    results_dict = {} 

    print(f"Bắt đầu sinh caption với Beam Size = {BEAM_SIZE_INFERENCE}...")
    
    with torch.no_grad():
        for i, (img_ids, V_raw, g_raw, _, _, _) in enumerate(tqdm(test_loader)):
            V_raw, g_raw = V_raw.to(device), g_raw.to(device)
            img_id = str(img_ids[0]) 
            
            # --- BEAM SEARCH ---
            # Gọi hàm sample của model với sample_k > 1 để kích hoạt Beam Search
            # (Giả định model GET đã implement logic này bên trong)
            sampled_seqs, _ = model.sample(V_raw, g_raw, vocab, sample_k=BEAM_SIZE_INFERENCE)
            
            # Nếu model trả về batch, lấy phần tử đầu tiên
            seq = sampled_seqs[0]
            
            # Decode kết quả (Đã bao gồm xử lý xóa '_' và xóa lặp từ)
            caption = decode_caption(seq, vocab)
            
            # Lưu kết quả
            results_dict[img_id] = caption
            results.append({"image_id": int(img_id) if img_id.isdigit() else img_id, "caption": caption})
            
            if i < 3:
                print(f"\n[Img {img_id}]: {caption}")

    # 4. Lưu kết quả ra file
    print(f"Saving results to {OUTPUT_RESULT_FILE}...")
    with open(OUTPUT_RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Hoàn thành! Đã sinh caption cho {len(results)} ảnh.")

if __name__ == '__main__':
    # Fix lỗi phân mảnh bộ nhớ trên Kaggle
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main_generate()

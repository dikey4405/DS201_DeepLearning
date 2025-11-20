# generate.py

import torch
import torch.nn.functional as F
from model.model_vi import GET
from DataLoader.COCO_dataset import COCODataset, Vocabulary, collate_fn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

# --- CONFIG ---
IMAGE_DIR = './data/coco_images/'
FEATURE_DIR = './data/coco_features_2048d/'
CAPTION_TEST_JSON = './data/test_captions.json' # File JSON của tập TEST
CAPTION_TRAIN_JSON = './data/train_captions.json' # Cần để build vocab
CHECKPOINT_PATH = 'get_model_best_scst.pth' # Dùng checkpoint SCST cuối cùng
OUTPUT_RESULT_FILE = 'caption_results.json' # File JSON kết quả

# Tham số mô hình (PHẢI KHỚP VỚI train.py)
D_MODEL = 512
N_HEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
CONTROLLER_TYPE = 'MAC'
BEAM_SIZE_INFERENCE = 3 # Beam size 3 cho inference
MAX_LEN = 50

def generate_caption_beam_search(model, V_raw, g_raw, vocab):
    """Thực hiện Beam Search (B=1) cho inference."""
    model.eval()
    device = V_raw.device
    
    # 1. Chạy Encoder một lần
    V0 = model.v_proj(V_raw)
    g0 = model.g_proj(g_raw)
    V_L, g_F = model.encoder(V0, g0) # (1, N, D), (1, D)

    # 2. Khởi tạo Beam
    beams = [(0.0, [vocab.SOS_token])] 
    final_candidates = []
    
    for _ in range(MAX_LEN):
        new_beams = []
        current_beams = sorted(beams, key=lambda x: x[0], reverse=True)[:BEAM_SIZE_INFERENCE]
        
        if not current_beams: break
            
        for score, seq in current_beams:
            if seq[-1] == vocab.EOS_token:
                final_candidates.append((score / (len(seq) - 1), seq))
                continue

            input_seq = torch.tensor([seq], dtype=torch.long, device=device)
            
            with torch.no_grad():
                output = model(V_raw, g_raw, input_seq)
            log_probs = F.log_softmax(output[0, -1, :], dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, BEAM_SIZE_INFERENCE)

            for i in range(BEAM_SIZE_INFERENCE):
                next_word_idx = topk_indices[i].item()
                next_log_prob = topk_log_probs[i].item()
                new_score = score + next_log_prob
                new_seq = seq + [next_word_idx]
                new_beams.append((new_score, new_seq))
        beams = new_beams

    if not final_candidates:
        final_candidates = [ (score / (len(seq) - 1), seq) for score, seq in beams]

    _, final_best_seq = max(final_candidates, key=lambda x: x[0]) 
    
    caption_words = [vocab.idx_to_word[idx] for idx in final_best_seq 
                     if idx not in [vocab.SOS_token, vocab.EOS_token, vocab.PAD_token]]
    return " ".join(caption_words)


def main_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Xây dựng Vocab (từ tập train)
    vocab = Vocabulary()
    try:
        with open(CAPTION_TRAIN_JSON, 'r', encoding='utf-8') as f:
            raw_train_annotations = json.load(f)
        all_train_captions = [item.get('translate') 
                              for item in raw_train_annotations if item.get('translate')]
        vocab.build_vocab(all_train_captions)
        vocab_size = len(vocab)
    except Exception as e:
        print(f"Lỗi khi xây dựng vocab: {e}"); return
        
    # 2. Tải Test Dataloader
    test_dataset = COCODataset(IMAGE_DIR, FEATURE_DIR, CAPTION_TEST_JSON, vocab)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 3. Tải Mô hình
    model = GET(vocab_size, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, controller_type=CONTROLLER_TYPE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy checkpoint tại {CHECKPOINT_PATH}. Vui lòng huấn luyện mô hình trước.")
        return
    model.to(device).eval()

    results = {}
    print("Bắt đầu sinh caption cho tập Test...")
    
    for img_ids, V_raw, g_raw, _, _, _ in tqdm(test_loader):
        V_raw, g_raw = V_raw.to(device), g_raw.to(device)
        img_id = img_ids[0] # Vì batch_size=1
        
        caption = generate_caption_beam_search(model, V_raw, g_raw, vocab)
        results[img_id] = caption
            
    # Lưu kết quả {image_name: caption}
    with open(OUTPUT_RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\nHoàn thành sinh caption. Đã lưu kết quả tại: {OUTPUT_RESULT_FILE}")
    return results

if __name__ == '__main__':
    main_generate()
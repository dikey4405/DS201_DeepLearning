import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class CIDErReward:
    """Lớp Wrapper để tính toán phần thưởng CIDEr-D cho SCST."""
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        # Khởi tạo metric CIDEr-D
        self.cider_scorer = Cider()
        self.tokenizer = PTBTokenizer() 
        print("Khởi tạo CIDErReward Scorer thành công.")

    def _decode_batch(self, sequences):
        """Chuyển đổi tensor chỉ số (B, T) sang list các chuỗi string."""
        decoded = []
        for i in range(sequences.size(0)):
            seq = sequences[i]
            words = []
            for idx in seq:
                idx = idx.item()
                if idx == self.vocabulary.EOS_idx:
                    break
                if idx not in [self.vocabulary.SOS_idx, self.vocabulary.PAD_idx]:
                    words.append(self.vocabulary.idx2word.get(idx, "<UNK>"))
            decoded.append(" ".join(words))
        return decoded

    def compute(self, sampled_seqs, gt_captions_list, beam_size=5):
        """
        Tính toán phần thưởng CIDEr.
        sampled_seqs: (B * k, T) - Tensor chỉ số từ .sample()
        gt_captions_list: (B,) - List các list string (ground truth)
        """
        batch_size = len(gt_captions_list)
        
        # 1. Decode các chuỗi đã lấy mẫu (sampled)
        decoded_sampled = self._decode_batch(sampled_seqs) # (B * k) strings
        
        # 2. Định dạng Ground Truth (GT)
        gts = {}
        for i in range(batch_size):
            gts[str(i)] = gt_captions_list[i] # { '0': [cap1, cap2], '1': [cap3, cap4] }
            
        # 3. Định dạng kết quả đã sinh (Res)
        res = {}
        for i in range(len(decoded_sampled)):
            img_id_key = str(i // beam_size) # ID của ảnh (0, 1, 2...)
            caption = decoded_sampled[i]
            if img_id_key not in res:
                res[img_id_key] = []
            res[img_id_key].append(caption) # { '0': [samp1, samp2...], '1': [...] }

        # Tokenize (Cần thiết cho CIDEr)
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)
        
        # 4. Tính điểm CIDEr
        scores, _ = self.cider_scorer.compute_score(gts_tokenized, res_tokenized)
        
        # 5. Định dạng lại điểm số
        rewards = torch.tensor(scores).view(batch_size, beam_size).to(self.device)
        return rewards

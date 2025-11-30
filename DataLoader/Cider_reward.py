# DataLoader/Cider_reward.py

import torch
import numpy as np
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class CIDErReward:
    def __init__(self, vocab, device):
        self.vocab = vocab
        self.device = device
        self.cider_scorer = Cider()
        self.tokenizer = PTBTokenizer()

    def _decode_batch(self, sequences):
        decoded = []
        for i in range(sequences.size(0)):
            seq = sequences[i]
            words = []
            for idx in seq:
                idx = idx.item()
                if idx == self.vocab.EOS_token:
                    break
                if idx not in [self.vocab.SOS_token, self.vocab.PAD_token]:
                    words.append(self.vocab.idx_to_word.get(idx, "<UNK>"))
            decoded.append(" ".join(words))
        return decoded

    def compute(self, sampled_seqs, gt_captions_list, beam_size):
        """
        Tính reward.
        sampled_seqs: (B * K, T)
        gt_captions_list: List độ dài B, mỗi phần tử là list các caption gốc.
        """
        batch_size = len(gt_captions_list)
        
        # 1. Decode thành string
        decoded_sampled = self._decode_batch(sampled_seqs) # List (B*K) strings
        
        # 2. "Làm phẳng" Ground Truth và Results
        gts = {}
        res = {}
        
        for i in range(len(decoded_sampled)):

            img_idx = i // beam_size 
            
            unique_id = str(i) 
            
            gts[unique_id] = [{'caption': c} for c in gt_captions_list[img_idx]]
            
            res[unique_id] = [{'caption': decoded_sampled[i]}]

        # 3. Tokenize
        gts_tokenized = self.tokenizer.tokenize(gts)
        res_tokenized = self.tokenizer.tokenize(res)
        
        # 4. Tính điểm
        _, scores = self.cider_scorer.compute_score(gts_tokenized, res_tokenized)
        
        # 5. Reshape lại về (Batch, Beam)
        rewards = torch.tensor(scores, dtype=torch.float32).view(batch_size, beam_size).to(self.device)
        
        return rewards

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 5.1. Kiến trúc phụ trợ (PE, FFN) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

# --- 5.2. Encoder Components ---
class GlobalEnhancedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
    def forward(self, V_in, g_in):
        O_input = torch.cat([V_in, g_in.unsqueeze(1)], dim=1)
        attn_output, _ = self.multihead_attn(O_input, O_input, O_input) 
        V_out = attn_output[:, :-1, :] 
        g_out = attn_output[:, -1, :] 
        return V_out, g_out 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.gea = GlobalEnhancedAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm1_g = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, V_in, g_in):
        V_out_raw, g_out_raw = self.gea(V_in, g_in)
        V_mid = self.norm1(V_in + self.dropout1(V_out_raw))
        g_mid = self.norm1_g(g_in + self.dropout1(g_out_raw)) 
        V_out = self.norm2(V_mid + self.dropout2(self.ffn(V_mid)))
        g_out = self.norm1_g(g_mid + self.dropout1(self.ffn(g_mid))) 
        return V_out, g_out 

class GlobalEnhancedEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)])
        self.lstm = nn.LSTMCell(d_model, d_model) 
    def forward(self, V0, g0):
        V_l = V0
        g_l_intra = g0
        B, D = g0.shape
        hx = torch.zeros(B, D, device=g0.device)
        cx = torch.zeros(B, D, device=g0.device)
        for layer in self.layers:
            V_l, g_l_intra = layer(V_l, g_l_intra)
            hx, cx = self.lstm(g_l_intra, (hx, cx))
        g_F = hx 
        return V_l, g_F 

# --- 5.3. Decoder Components ---
class GlobalAdaptiveController(nn.Module):
    def __init__(self, d_model, n_head, dropout, controller_type='MAC'):
        super().__init__()
        self.controller_type = controller_type
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        if controller_type == 'GAC':
            self.sigmoid = nn.Sigmoid()
    def forward(self, a_tl, V_L, g_F):
        if self.controller_type == 'GAC':
            hat_e_tl, _ = self.multihead_attn(a_tl, V_L, V_L)
            a_t_last = a_tl[:, -1, :] 
            alpha = self.sigmoid(torch.sum(a_t_last * g_F, dim=-1, keepdim=True)).unsqueeze(1) 
            g_F_expanded = g_F.unsqueeze(1).expand_as(hat_e_tl)
            e_tl = hat_e_tl + alpha * g_F_expanded 
        elif self.controller_type == 'MAC':
            V_g = torch.cat([V_L, g_F.unsqueeze(1)], dim=1) 
            e_tl, _ = self.multihead_attn(a_tl, V_g, V_g) 
        return e_tl

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, controller_type):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.gac = GlobalAdaptiveController(d_model, n_head, dropout, controller_type)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, V_L, g_F, tgt_mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        a_tl = self.norm1(x + self.dropout1(attn_output)) 
        gac_output = self.gac(a_tl, V_L, g_F) 
        e_tl = self.norm2(a_tl + self.dropout2(gac_output)) 
        ffn_output = self.ffn(e_tl)
        h_tl = self.norm3(e_tl + self.dropout3(ffn_output)) 
        return h_tl

class GlobalAdaptiveDecoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout, controller_type):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout, controller_type) for _ in range(num_layers)])
    def forward(self, x, V_L, g_F, tgt_mask):
        for layer in self.layers:
            x = layer(x, V_L, g_F, tgt_mask)
        return x

# --- 5.4. Global Enhanced Transformer (GET) ---
class GET(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, num_encoder_layers=3, num_decoder_layers=3, d_ff=2048, dropout=0.1, controller_type='MAC'):
        super().__init__()
        self.v_proj = nn.Linear(2048, d_model) 
        self.g_proj = nn.Linear(2048, d_model)
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = GlobalEnhancedEncoder(num_encoder_layers, d_model, n_head, d_ff, dropout)
        self.decoder = GlobalAdaptiveDecoder(num_decoder_layers, d_model, n_head, d_ff, dropout, controller_type)
        self.classifier = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, V_raw, g_raw, captions):
        # 1. Projection đầu vào ảnh
        V0 = self.v_proj(V_raw)
        g0 = self.g_proj(g_raw)

        # 2. Encoder
        V_L, g_F = self.encoder(V0, g0)
        
        # 3. Decoder
        caption_embed = self.word_embedding(captions) * math.sqrt(V_L.size(-1))
        caption_embed = self.pos_encoder(caption_embed)
        tgt_len = captions.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(V_L.device)
        
        h_L = self.decoder(caption_embed, V_L, g_F, tgt_mask) 
        
        # 4. Classifier
        output = self.classifier(h_L)
        return output

    def sample(self, V_raw, g_raw, vocab, max_len=50, beam_size=5):
        """
        Sinh caption (sampling) cho SCST.
        Sử dụng Greedy Search (beam_size=1) cho Batch (để đơn giản hóa).
        """
        self.eval()
        device = V_raw.device
        batch_size = V_raw.size(0)

        # 1. Chạy Encoder một lần
        V0 = self.v_proj(V_raw)
        g0 = self.g_proj(g_raw)
        V_L, g_F = self.encoder(V0, g0) # (B, N, D), (B, D)

        # 2. Khởi tạo
        sampled_seqs = torch.zeros(batch_size * beam_size, max_len, dtype=torch.long).to(device)
        log_probs_sum = torch.zeros(batch_size * beam_size, max_len).to(device)
        
        # Lặp qua từng beam (trong trường hợp này k=beam_size)
        for k in range(beam_size):
            # Khởi tạo token <SOS>
            input_seq = torch.full((batch_size, 1), vocab.SOS_token, dtype=torch.long).to(device)
            
            for t in range(max_len):
                # Tạo mask
                tgt_len = input_seq.size(1)
                tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
                
                # Chạy Decoder
                caption_embed = self.word_embedding(input_seq) * math.sqrt(V_L.size(-1))
                caption_embed = self.pos_encoder(caption_embed)
                
                h_L = self.decoder(caption_embed, V_L, g_F, tgt_mask)
                
                # Classifier cho từ cuối cùng
                output = self.classifier(h_L[:, -1, :]) # (B, V)
                log_probs = F.log_softmax(output, dim=-1) # (B, V)
                
                if t == 0 and k > 0:
                    # Dùng Multinomial sampling cho các beam k>0
                    sampled_word = torch.multinomial(log_probs.exp(), 1) # (B, 1)
                else:
                    # Dùng Greedy (Argmax) cho beam k=0
                    _, sampled_word = torch.max(log_probs, dim=1)
                    sampled_word = sampled_word.unsqueeze(1) # (B, 1)
                
                # Lấy log_prob của từ đã chọn
                current_log_probs = log_probs.gather(1, sampled_word).squeeze(1) # (B,)
                
                # Lưu trữ
                idx = k * batch_size
                sampled_seqs[idx:idx+batch_size, t] = sampled_word.squeeze(1)
                log_probs_sum[idx:idx+batch_size, t] = current_log_probs

                # Thêm từ đã chọn vào input cho bước tiếp theo
                input_seq = torch.cat([input_seq, sampled_word], dim=1)
                
                # Dừng nếu tất cả trong batch đã sinh <EOS>
                if (sampled_word.squeeze(1) == vocab.EOS_token).all():
                    break
                    
        return sampled_seqs, log_probs_sum
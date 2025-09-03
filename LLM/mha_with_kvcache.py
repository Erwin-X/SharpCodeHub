import math
import torch
import torch.nn.functional as F

class MultiHeadAttentionKV:
    def __init__(self, hidden_size, head_size, attn_dropout):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.head_dim = self.hidden_size // self.head_size
        self.q_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.k_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.v_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.output_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)

    def forward(self, input, input_mask, kv_cache=None, use_cache=False):
        """
            input: [B, T, D]
            input_mask: [B, T]
        """
        b,t,_ = input.shape
        h,h_d = self.head_size, self.head_dim

        Q = self.q_linear(input).reshape(b,t,h,h_d).transpose(1,2)    # [B, H, T, h_d]
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            K = torch.cat(k_cache, self.k_linear(input).reshape(b,t,h,h_d).transpose(1,2), dim=2)
            V = torch.cat(v_cache, self.v_linear(input).reshape(b,t,h,h_d).transpose(1,2), dim=2)
        else:
            K = self.k_linear(input).reshape(b,t,h,h_d).transpose(1,2)
            V = self.v_linear(input).reshape(b,t,h,h_d).transpose(1,2)
        new_kv_cache = (K,V) if use_cache else None

        attn_logits = torch.einsum("bhtd, bhtd -> bhtt", Q, K)/math.sqrt(self.h_d)  # smooth
        if input_mask:
            attn_masks = torch.einsum("btd,btd -> btt", input_mask[:,:,None], input_mask[:,:,None])
            attn_logits -= attn_masks[:,None,:] * 1e9
        attn_weights = F.softmax(attn_logits, dim=-1)    # [B, H, T, T]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum("bhtt, bhtd -> bt(hd)", attn_weights, V)
        output = self.output_linear(output)
        return output, new_kv_cache

import math
import torch
import torch.nn.functional as F

class MultiHeadAttentionKV:
    def __init__(self, hidden_size, head_size, attn_dropout):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.head_dim = self.hidden_size // self.head_size
        self.q_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.k_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.v_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.output_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)

    def forward(self, input, input_mask, prev_K, prev_V, kv_cache=False, return_kv=False):
        """
            Params:
                input: [B, L, D]
            Return:
                output: [B, L, D]
        """
        b,l,_ = input.shape
        h,h_d = self.head_size, self.head_dim

        Q = self.q_linear(input).reshape(b,l,h,h_d).transpose(1,2)    # [B, H, L, h_d]
        if kv_cache:
            K = torch.cat(prev_K, self.k_linear(input).reshape(b,l,h,h_d).transpose(1,2), dim=2)
            V = torch.cat(prev_V, self.k_linear(input).reshape(b,l,h,h_d).transpose(1,2), dim=2)
        else:
            K = self.k_linear(input).reshape(b,l,h,h_d).transpose(1,2)
            V = self.v_linear(input).reshape(b,l,h,h_d).transpose(1,2)

        attn_logits = torch.einsum("bhld, bhld -> bhll", Q, K)/math.sqrt(self.hidden_size)  # smooth
        if input_mask:
            attn_masks = torch.einsum("bld,bld -> bll", input_mask, input_mask)
            attn_logits -= attn_masks[:,None,:] * 1e9
        attn_weights = F.softmax(attn_logits, dim=-1)    # [B, H, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum("bhll, bhld -> bl(hd)", attn_weights, V)
        output = self.output_linear(output)
        if kv_cache or return_kv:
            return output, K, V
        return output

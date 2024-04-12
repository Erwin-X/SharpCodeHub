import math
import torch
import torch.nn.functional as F

class GroupQueryAttention:
    def __init__(self, hidden_size, head_size, group_size, attn_dropout=0.1):
        assert head_size % group_size == 0, "head_size mod group_size must be zero"
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.head_dim = self.hidden_size // self.head_size
        self.kv_out_dim = group_size * self.head_dim
        self.q_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.k_linear = torch.nn.Dense(hidden_size, self.kv_out_dim)
        self.v_linear = torch.nn.Dense(hidden_size, self.kv_out_dim)
        self.output_linear = torch.nn.Dense(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)

    def forward(self, input, input_mask, prev_K, prev_V, kv_cache=False, return_kv=False):
        """
            Params:
                input: [B, L, D]
                input_mask: [B, L]
            Return:
                output: [B, L, D]
        """
        b,l,_ = input.shape
        h,h_d = self.head_size, self.head_dim
        g,g_d = self.group_size,h//self.group_size

        Q = self.q_linear(input).reshape(b,l,g,g_d,h_d).transpose(1,2)    # [B, G, J, L, h_d], H=G*J
        if kv_cache:
            K = torch.cat(prev_K, self.k_linear(input).reshape(b,l,g,h_d).transpose(1,2), dim=2)
            V = torch.cat(prev_V, self.v_linear(input).reshape(b,l,g,h_d).transpose(1,2), dim=2)
        else:
            K = self.k_linear(input).reshape(b,l,g,h_d).transpose(1,2)    # [B, G, L, h_d]
            V = self.v_linear(input).reshape(b,l,g,h_d).transpose(1,2)    # [B, G, L, h_d]
 
        attn_logits = torch.einsum("bgjld, bgld -> bgjll", Q, K)/math.sqrt(self.hidden_size)  # smooth
        if input_mask:
            attn_masks = torch.einsum("bld,bld -> bll", input_mask, input_mask)
            attn_logits -= attn_masks[:,None,None,:] * 1e9
        attn_weights = F.softmax(attn_logits, dim=-1)    # [B, G, J, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum("bgjll, bgld -> bl(gjd)", attn_weights, V)
        output = self.output_linear(output)
        if kv_cache or return_kv:
            return output, K, V
        return output

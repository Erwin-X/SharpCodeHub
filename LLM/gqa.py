import math
import torch
import torch.nn.functional as F

class GroupQueryAttention:
    def __init__(self, hidden_size, head_num, group_num, attn_dropout=0.1):
        assert head_num % group_num == 0, "head_num mod group_num must be zero"
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = self.hidden_size // self.head_num
        self.group_num = group_num
        self.group_size = head_num // group_num
        self.kv_out_dim = group_num * self.head_dim
        self.q_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.k_linear = torch.nn.Linear(hidden_size, self.kv_out_dim)
        self.v_linear = torch.nn.Linear(hidden_size, self.kv_out_dim)
        self.output_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)

    def forward(self, input, input_mask):
        """
            Params:
                input: [B, S, D]
                input_mask: [B, S]
            Return:
                output: [B, S, D]
        """
        b,s,_ = input.shape
        h,h_dim = self.head_num, self.head_dim
        g,g_dim = self.group_num, self.group_size

        Q = self.q_linear(input).reshape(b,s,g,g_dim,h_dim).transpose(1,2)    # [B, G, Gsize, S, h_dim], H=G*Gsize
        K = self.k_linear(input).reshape(b,s,g,h_dim).transpose(1,2)    # [B, G, S, h_dim]
        V = self.v_linear(input).reshape(b,s,g,h_dim).transpose(1,2)    # [B, G, S, h_dim]
 
        attn_logits = torch.einsum("bgjld, bgld -> bgjll", Q, K)/math.sqrt(self.h_d)  # smooth
        if input_mask:
            attn_masks = torch.einsum("bld,bld -> bll", input_mask, input_mask)
            attn_logits -= attn_masks[:,None,None,:] * 1e9
        attn_weights = F.softmax(attn_logits, dim=-1)    # [B, G, J, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.einsum("bgjll, bgld -> bl(gjd)", attn_weights, V)
        output = self.output_linear(output)
        return output

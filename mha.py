import math
import torch
import torch.nn.functional as F

class MultiHeadAttention:
    def __init__(self, hidden_size, head_size, attn_dropout=0.1):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.head_dim = self.hidden_size // self.head_size
        self.q_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.k_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.v_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.output_linear = torch.nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = torch.nn.Dropout(p=attn_dropout)

    def forward(self, input, input_mask):
        """
            Params:
                input: [B, L, D]
                input_mask: [B, L]
            Return:
                output: [B, L, D]
        """
        b,l,_ = input.shape
        h,h_d = self.head_size, self.head_dim

        Q = self.q_linear(input).reshape(b,l,h,h_d).transpose(1,2)    # [B, H, L, h_d]
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
        return output

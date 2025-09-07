import math
import torch
import torch.nn as nn

def rotate_half(x):
    """将输入张量的后一半维度进行旋转"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, sin, cos):
    """应用旋转位置编码"""
    return (x * cos) + (rotate_half(x) * sin)

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码模块"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 预计算正弦和余弦函数
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 缓存余弦和正弦值
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(self, x, seq_len=None):
        """前向传播"""
        if seq_len is None:
            seq_len = x.size(2)
        
        # 返回对应序列长度的位置编码
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )

# 使用示例
def rope_example():
    # 参数设置
    batch_size = 4
    seq_len = 128
    n_heads = 8
    head_dim = 64
    dim = n_heads * head_dim
    
    # 创建RoPE模块
    rope = RotaryPositionEmbedding(head_dim, max_seq_len=2048)
    
    # 创建随机查询和键向量
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)
    
    # 获取位置编码
    cos, sin = rope(q)
    
    # 应用位置编码到查询和键
    q_rotated = apply_rotary_pos_emb(q, sin, cos)
    k_rotated = apply_rotary_pos_emb(k, sin, cos)
    
    print("Original query shape:", q.shape)
    print("Rotated query shape:", q_rotated.shape)
    print("Position encoding applied successfully!")

if __name__ == "__main__":
    rope_example()
import re
from collections import defaultdict

class BBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        
    def train(self, texts, vocab_size, verbose=False):
        """训练BBPE tokenizer，接受文本列表作为输入"""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
            
        # 初始化词汇表为所有字节
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        
        # 将所有文本编码为字节序列并合并
        all_tokens = []
        for text in texts:
            tokens = list(text.encode('utf-8'))
            all_tokens.extend(tokens)
        
        # 转换为初始token IDs
        token_ids = [t for t in all_tokens]
        
        # 迭代合并直到达到目标词汇表大小
        num_merges = vocab_size - 256
        merges_done = 0
        
        while merges_done < num_merges:
            # 统计相邻token对的出现频率
            pairs = defaultdict(int)
            for i in range(len(token_ids) - 1):
                pairs[(token_ids[i], token_ids[i+1])] += 1
            
            if not pairs:
                break
                
            # 找到最常见的对
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            
            if best_count < 2:
                break
                
            # 创建新token
            new_id = 256 + merges_done
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.inverse_vocab[self.vocab[new_id]] = new_id
            self.merges[best_pair] = new_id
            merges_done += 1
            
            if verbose:
                print(f"合并 {merges_done}/{num_merges}: {best_pair} -> {new_id} (出现次数: {best_count})")
            
            # 在token序列中合并最佳对
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == best_pair:
                    new_token_ids.append(new_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = new_token_ids
            
        # 确保词汇表大小正确
        self.vocab = {k: self.vocab[k] for k in sorted(self.vocab.keys())}
        
        return merges_done
    
    def encode(self, text):
        """将文本编码为token IDs"""
        # 首先将文本转换为字节
        byte_data = text.encode('utf-8')
        token_ids = [b for b in byte_data]
        
        # 应用合并规则
        while True:
            # 找到可以合并的相邻对
            min_idx = None
            min_rank = float('inf')
            
            for i in range(len(token_ids) - 1):
                pair = (token_ids[i], token_ids[i+1])
                if pair in self.merges:
                    # 获取合并的优先级（合并顺序）
                    merged_id = self.merges[pair]
                    rank = merged_id - 256  # 先合并的优先级更高
                    if rank < min_rank:
                        min_rank = rank
                        min_idx = i
            
            if min_idx is None:
                break
                
            # 执行合并
            pair = (token_ids[min_idx], token_ids[min_idx+1])
            merged_id = self.merges[pair]
            
            token_ids = token_ids[:min_idx] + [merged_id] + token_ids[min_idx+2:]
        
        return token_ids
    
    def decode(self, token_ids):
        """将token IDs解码为文本"""
        # 将token IDs转换为字节
        byte_data = b''
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_data += self.vocab[token_id]
            else:
                # 处理未知token（回退到字节）
                byte_data += bytes([token_id])
        
        # 将字节解码为文本
        try:
            return byte_data.decode('utf-8', errors='replace')
        except:
            return byte_data.decode('utf-8', errors='ignore')
    
    def get_vocab(self):
        """获取词汇表"""
        return self.vocab
    
    def save(self, filepath):
        """保存tokenizer到文件"""
        import json
        
        data = {
            'vocab': {k: list(v) for k, v in self.vocab.items()},
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath):
        """从文件加载tokenizer"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}


# 示例用法
if __name__ == "__main__":
    # 示例文本列表
    texts = [
        "自然语言处理是人工智能的一个重要领域。",
        "BBPE是一种有效的tokenization方法。",
        "深度学习中，tokenization是非常重要的一步。"
    ]
    
    # 创建和训练tokenizer
    tokenizer = BBPETokenizer()
    num_merges = tokenizer.train(texts, vocab_size=300, verbose=True)
    print(f"完成了 {num_merges} 次合并")
    
    # 编码和解码
    test_text = "自然语言处理和BBPE都是重要的NLP技术。"
    encoded = tokenizer.encode(test_text)
    print("编码结果:", encoded)
    
    decoded = tokenizer.decode(encoded)
    print("解码结果:", decoded)
    
    # 验证重建
    print("原始文本等于解码文本:", test_text == decoded)
    
    # 保存和加载
    tokenizer.save("bbpe_tokenizer.json")
    
    new_tokenizer = BBPETokenizer()
    new_tokenizer.load("bbpe_tokenizer.json")
    
    # 测试加载的tokenizer
    new_encoded = new_tokenizer.encode(test_text)
    print("加载后的编码结果:", new_encoded)
    print("编码结果一致:", encoded == new_encoded)
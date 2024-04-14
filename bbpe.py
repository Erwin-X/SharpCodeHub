from collections import defaultdict

class BBPEConstructor:
    def __init__(self, vocab_size):
        assert vocab_size >= 256
        self.initial_vocab = [bytes([byte]) for byte in range(256)]    # 构建初始词汇表，包含一个字节的256个表示
        self.vocab = set(self.initial_vocab.copy())
        self.merges = {}
        self.vocab_size = vocab_size

    def construct(self, sentences):
        stats = self._build_stats(sentences)
        splits = self._init_splits(stats)
        pair_freqs = self._compute_pair_freqs(stats, splits)

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs(stats, splits)
            best_pair = ()
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = self._merge_pair(best_pair, stats, splits)
            merged_byte = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_byte
            self.vocab.add(merged_byte)
        return self.vocab

    def tokenize(self, sentence):
        splits = [[bytes([byte]) for byte in word.encode('utf-8')] for word in sentence.split()]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i:i+2] == list(pair):
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

    def _build_stats(self, sentences):
        stats = defaultdict(int)
        for sentence in sentences:
            symbols = sentence.split()
            for symbol in symbols:
                stats[symbol.encode("utf-8")] += 1
        return stats

    def _init_splits(self, stats):
        splits = {word: [bytes([byte]) for byte in word] for word in stats.keys()}
        return splits

    def _compute_pair_freqs(self, stats, splits):
        pair_freqs = defaultdict(int)
        for word, freq in stats.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair, stats, splits):
        merged_byte = pair[0] + pair[1]
        for word in stats:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i:i+2] == list(pair):
                    split = split[:i] + [merged_byte] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

if __name__ == "__main__":
    sentences = [
    "花卉 " * 100,
    "英国 " * 20,
    "小猫 小猪 " * 30,
    "我是谁 我在英国养小狗。。。。。。。",
    "I like to eat apples, I like to eat apples, I like to eat apples",
    "She has a cute cat",
    "you are very cute",
    "give you a hug",
    ]

    bbpe_constructor = BBPEVocaConstructor(vocab_size=300)
    vocab = bbpe_constructor.construct(sentences)
    print("vocab:")
    for i,t in enumerate(vocab):
        try:
            print(f"{i},{t.decode()},End")
        except:
            print(f"{i},{t},End")
    print("\n")

    print("merges:")
    for k,v in bbpe_constructor.merges.items():
        try:
            print(f"{k}: {v.decode()}")
        except:
            pass
    print("\n")
    
    sentence = "我在英国养了花卉和小猫，但没养cat和小猪。"
    tokens = bbpe_constructor.tokenize(sentence)
    print("tokens: ", tokens)
    print("\n")
    for t in tokens:
        try:
            print(f"{t.decode()}")
        except:
            print(t)

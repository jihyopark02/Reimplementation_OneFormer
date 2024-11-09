import torch
import torch.nn as nn

class TaskTokenizer:
    def __init__(self, vocab_size=30000, embed_dim=256, max_seq_len=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, task_texts):
        token_ids = [self._tokenize(text) for text in task_texts]
        token_ids = torch.tensor(token_ids) 
        return self.tokenizer(token_ids)  

    def _tokenize(self, text):
        token_id = min(sum(ord(c) for c in text) % self.vocab_size, self.vocab_size - 1)
        return [token_id] * self.max_seq_len

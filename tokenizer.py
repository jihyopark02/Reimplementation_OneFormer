import torch
import torch.nn as nn

class TaskTokenizer:
    def __init__(self, vocab_size=30000, embed_dim=256, max_seq_len=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = nn.Embedding(self.vocab_size, self.embed_dim)

    def forward(self, task_texts):
        # Assuming task_texts is a list of strings ["panoptic", "instance", "semantic"]
        token_ids = [self._tokenize(text) for text in task_texts]
        token_ids = torch.tensor(token_ids)  # Shape: [batch_size, max_seq_len]
        return self.tokenizer(token_ids)  # Shape: [batch_size, max_seq_len, embed_dim]

    def _tokenize(self, text):
        # Simple tokenizer to map text to indices (this is a placeholder)
        # In practice, you'd use a tokenizer from Hugging Face or similar
        token_id = min(sum(ord(c) for c in text) % self.vocab_size, self.vocab_size - 1)
        return [token_id] * self.max_seq_len

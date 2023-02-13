from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 4096
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 512
    dropout: float = 0.3
    bias: bool = False

class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)
        
        y = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=None, dropout_p=self.dropout, is_causal=True
        )
        
        return y

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config.head_size) for _ in range(config.n_head)])
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = torch.cat([model(x) for model in self.heads], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadedAttention(config.n_head , config.n_embd//config.n_head)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.self_attention(x)
        x = x + self.feedforward(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.lm = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, tokens, targets=None):
        num_blocks, num_tokens = tokens.shape
        pos = torch.arange(0,num_tokens,dtype=torch.long,device=tokens.device)

        token_embd = self.token_embedding(tokens)
        position_embd = self.position_embedding(pos)
        embedding = token_embd + position_embd
        logits = self.blocks(embedding)
        logits = self.lm(logits)

        if targets == None:
            return logits, None

        num_blocks, num_tokens, num_chars = logits.shape
        logits = logits.view(num_blocks * num_tokens, num_chars)
        targets = targets.view(num_blocks * num_tokens)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, tokens, length):
        for _ in range(length):
            logits, _ = self(tokens[:,-self.config.block_size:])
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, 1)
            tokens = torch.cat((tokens, sample), dim=1)
        return tokens
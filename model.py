import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 128
    vocab_size: int = 50280
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False


config = ModelConfig()


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear = nn.Linear(
            config.n_embd, config.n_embd*3, bias=config.bias)
        self.attention = nn.MultiheadAttention(
            config.n_embd, config.n_head, config.dropout, config.bias, batch_first=True)
        mask = torch.triu(torch.ones(config.block_size,
                          config.block_size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        self.register_buffer('mask', mask)

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        b, t, c = x.shape
        query, key, value = self.linear(x).split(config.n_embd, dim=2)
        x = x + self.attention(query, key, value,
                               need_weights=False, attn_mask=self.mask[:t, :t])[0]
        x = self.layer_norm(x)
        x = x + self.mlp(x)
        x = self.layer_norm(x)
        return x


class GenerativeTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size

        self.text_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_enc = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)])
        self.linear = nn.Linear(
            config.n_embd, config.vocab_size, bias=config.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.shape
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = self.text_embed(x) + self.pos_enc(pos)
        x = self.dropout(x)
        x = self.blocks(x)

        if targets is None:
            x = self.linear(x[:, -1, :])
            x = F.softmax(x, dim=-1)
            loss = None
        else:
            x = self.linear(x)
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1)

        return x, loss

    @torch.no_grad()
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            new_token, _ = self(idx_cond)
            new_token = torch.multinomial(new_token, num_samples=1)
            idx = torch.cat((idx, new_token), dim=1)
        return idx

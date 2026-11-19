"""Decoder-only transformer. Defaults are exactly GPT-2 (learned positions, LayerNorm, GELU, biases, tied head);
flags switch in the Llama-style upgrades: RoPE, RMSNorm, SwiGLU, no biases."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True  # biases in Linear and LayerNorm (GPT-2: True, Llama: False)
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp: str = "gelu"  # "gelu" | "swiglu"
    pos: str = "learned"  # "learned" | "rope"
    flash: bool = True  # F.scaled_dot_product_attention instead of the hand-written attention


class RMSNorm(nn.Module):
    """LayerNorm without mean-centering or bias: x / rms(x) * weight."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()  # compute the statistic in fp32 even under bf16 autocast
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def make_norm(cfg):
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias) if cfg.norm == "layernorm" else RMSNorm(cfg.n_embd)


def rope_tables(head_dim, max_len, base=10000.0):
    """cos/sin of angle pos * base^(-2i/d) for each position and each of the head_dim/2 rotation pairs."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate pairs (x[i], x[i + d/2]) by a position-dependent angle (GPT-NeoX/Llama "half" layout).
    q·k then depends only on the relative distance between positions."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class KVCache:
    """Keys/values of already-processed positions, so each new token attends to them without recomputing."""

    def __init__(self, n_layer, max_len):
        self.k, self.v = [None] * n_layer, [None] * n_layer
        self.max_len = max_len
        self.pos = 0  # number of positions already stored

    def update(self, layer, k, v):
        """Store k, v (B, nh, T, hs) at positions [pos, pos+T); return all keys/values so far."""
        if self.k[layer] is None:
            B, nh, _, hs = k.shape
            self.k[layer] = k.new_zeros(B, nh, self.max_len, hs)
            self.v[layer] = v.new_zeros(B, nh, self.max_len, hs)
        T = k.size(2)
        self.k[layer][:, :, self.pos : self.pos + T] = k
        self.v[layer][:, :, self.pos : self.pos + T] = v
        return self.k[layer][:, :, : self.pos + T], self.v[layer][:, :, : self.pos + T]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.layer, self.flash, self.dropout = cfg.n_head, layer, cfg.flash, cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.keep_attn, self.last_attn = False, None  # analysis: set keep_attn (and flash=False) to record weights
        self.rope = cfg.pos == "rope"
        if self.rope:
            cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        start = cache.pos if cache is not None else 0
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, nh, T, hs)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            cos, sin = self.cos[start : start + T], self.sin[start : start + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)
        # Rows: the T new positions; columns: every position so far. Causal = row i sees columns <= start + i.
        mask = self.mask[start : start + T, : start + T]

        if self.flash:
            dropout_p = self.dropout if self.training else 0.0
            if start == 0:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
            else:  # is_causal assumes a square mask, which is wrong once cached positions are prepended
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, nh, T, start+T)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.keep_attn:
                self.last_attn = att.detach()  # (B, nh, T, start+T)
            y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # back to (B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.swiglu = cfg.mlp == "swiglu"
        if self.swiglu:
            # 2/3 of 4C keeps the parameter count of a 4C GELU MLP despite the extra gate matrix.
            hidden = 64 * math.ceil(8 * cfg.n_embd / 3 / 64)
            self.fc = nn.Linear(cfg.n_embd, 2 * hidden, bias=cfg.bias)  # gate and up projections, fused
        else:
            hidden = 4 * cfg.n_embd
            self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        if self.swiglu:
            gate, up = self.fc(x).chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.gelu(self.fc(x), approximate="tanh")  # GPT-2's GELU
        return self.drop(self.proj(h))


class Block(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg, layer)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2: scale residual output projections so the residual stream doesn't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Returns (model in eval mode, raw checkpoint dict)."""
        ckpt = torch.load(path, map_location=device)
        model = cls(GPTConfig(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model"])
        return model.eval(), ckpt

    def grow_vocab(self, n_new):
        """Append n_new token rows (e.g. new special tokens), initialised to the mean embedding so they start as
        a 'typical' token. Existing tokens' logits are unchanged; the head stays tied to the embedding."""
        old = self.tok_emb.weight.data
        new = torch.cat([old, old.mean(0, keepdim=True).expand(n_new, -1)])
        self.tok_emb = nn.Embedding.from_pretrained(new.clone(), freeze=False)
        self.lm_head = nn.Linear(self.cfg.n_embd, new.size(0), bias=False, device=new.device)
        self.lm_head.weight = self.tok_emb.weight
        self.cfg.vocab_size += n_new

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.block_size, f"sequence length {start + T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, cache)
        if cache is not None:
            cache.pos += T
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True, eos_id=None):
        """temperature 0 = greedy. With use_cache, each step feeds only the newest token; once the context is
        full, the cache is rebuilt from the last block_size tokens (the same window the uncached path sees).
        eos_id: stop early once every sequence has produced it."""
        bs, cache = self.cfg.block_size, None
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -bs:])
            elif cache is None or cache.pos >= bs:
                cache = KVCache(self.cfg.n_layer, bs)
                logits, _ = self(idx[:, -bs:], cache=cache)  # prefill
            else:
                logits, _ = self(idx[:, -1:], cache=cache)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                if top_p is not None:  # keep the smallest set of tokens whose probability sums to >= top_p
                    sorted_p, order = torch.sort(probs, descending=True)
                    sorted_p[sorted_p.cumsum(-1) - sorted_p > top_p] = 0
                    probs = torch.zeros_like(probs).scatter(-1, order, sorted_p)
                next_id = torch.multinomial(probs, num_samples=1)  # renormalizes internally
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None:
                done |= next_id[:, 0] == eos_id
                if done.all():
                    break
        return idx

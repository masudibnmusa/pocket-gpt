# pocket-gpt
A GPT-style language model built from scratch: BPE tokenizer, transformer, training loop, and sampling, trained on a small corpus.

## Current state: minimal char-level GPT
| File | What |
|------|------|
| `data.py` | Download Tiny Shakespeare, char tokenizer, `train.bin`/`val.bin`, random-window batches |
| `model.py` | Decoder-only transformer: hand-written causal attention, pre-norm blocks, tied head, `generate()` with temperature/top-k |
| `train.py` | AdamW loop with periodic train/val eval, saves `runs/<name>/ckpt.pt` |
| `sample.py` | Generate from a checkpoint |
| `tests/test_model.py` | Initial loss ≈ ln(vocab), no future-token leak, overfit one batch, generate shape |

## Quickstart
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python data.py                 # download + tokenize
.venv/bin/python -m pytest -q tests      # sanity checks
.venv/bin/python train.py                # ~0.8M params, 5000 steps, ~1 min on an RTX 5060
.venv/bin/python sample.py --prompt "ROMEO:" --temperature 0.8
```
Every model field is a flag, e.g. `train.py --n_layer 6 --n_embd 256 --dropout 0.1 --out runs/bigger`.

Baseline (default config): val loss ≈ 1.56 at step ~4250, then slowly overfits.

## Next
LR warmup + cosine, grad clipping, bf16 → BPE tokenizer + TinyStories → RoPE/RMSNorm/SwiGLU → KV cache → GPT-2 parity.

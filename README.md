![pocket-gpt: a tiny GPT language model, built from scratch](assets/banners/pocket-gpt/gradient-1280x320.png)

![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.15-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-bf16-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Model](https://img.shields.io/badge/model-12M_params-6366F1?style=flat-square)
![Data](https://img.shields.io/badge/data-TinyStories_·_Tiny_Shakespeare-A21CAF?style=flat-square)
![GPT-2](https://img.shields.io/badge/GPT--2_parity-verified-0EA5E9?style=flat-square)
![Tests](https://img.shields.io/badge/tests-32_passing-3FB950?style=flat-square&logo=pytest&logoColor=white)

A GPT-style language model built from scratch: BPE tokenizer, transformer (GPT-2 and Llama-style variants), training loop, KV-cached sampling, and a GPT-2 weight-loading parity check. A 12M-parameter model trained for ~10 minutes on one RTX 5060 writes coherent children's stories.

## Project structure
```text
pocket-gpt/
├── bpe.py               # Byte-level BPE from scratch: GPT-4 regex pre-split, incremental merge training, special tokens
├── data.py              # Download corpus (Tiny Shakespeare / TinyStories), tokenize -> data/<name>/, random-window batches
├── model.py             # Decoder-only transformer: GPT-2 defaults, flags for RoPE / RMSNorm / SwiGLU / no-bias,
│                        #   hand-written or SDPA (Flash) attention, KV cache, temperature / top-k / top-p sampling
├── train.py             # AdamW param groups + warmup/cosine, grad clipping, bf16, torch.compile, eval (loss + bits/byte), checkpoint/resume
├── sample.py            # Generate text from a checkpoint with its own tokenizer (KV cache on by default)
├── gpt2.py              # Load OpenAI GPT-2 (124M) weights from Hugging Face into model.GPT
├── scaling.py           # Scaling sweep: 5 model sizes, same token budget, power-law fit L(N) = c + a·N^-alpha
├── analysis.py          # Attention maps, induction / previous-token head scores, embedding neighbours + PCA
├── finetune.py          # Instruction fine-tuning on TinyStories-Instruct (loss on the story only) + word-following eval
├── tests/
│   ├── test_model.py    # per config (GPT-2/Llama x manual/SDPA): initial loss, no future leak, overfit one batch,
│   │                    #   manual == SDPA attention, KV cache == full forward, cached == uncached generation, top-p
│   ├── test_bpe.py      # first merge, unicode round-trip, compression, reload, incremental == naive training, specials
│   ├── test_finetune.py # instruct parsing, loss mask covers only response tokens
│   └── test_gpt2_parity.py  # our logits == Hugging Face GPT-2 logits (skipped without transformers)
├── assets/banners/      # README banners (PNG) + their HTML sources
├── assets/analysis/     # plots written by scaling.py and analysis.py
├── requirements.txt     # torch, numpy, regex, pytest (+ transformers, matplotlib, optional)
│
├── data/                # generated, git-ignored
│   ├── input.txt, tinystories_*.txt   # raw corpora
│   └── <name>/          #   train.bin, val.bin (uint16 token ids), meta.json (vocab/merges + byte counts)
└── runs/                # generated, git-ignored
    └── <run>/ckpt.pt    #   model, optimizer, step, RNG, config, args (saved every eval)
```

**Data flow:** raw text → `data.py` (tokenizer) → `train.bin`/`val.bin` → `train.py` (batches → `model.py` → cross-entropy → AdamW) → `ckpt.pt` → `sample.py` → text

## Quickstart
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests      # sanity checks (GPT-2 parity downloads 548 MB on first run)

# TinyStories, 12M params (~400 MB of text -> 98M tokens; ~1 min to prepare)
.venv/bin/python data.py --dataset tinystories --tokenizer bpe --vocab_size 4096
.venv/bin/python train.py --data data/tinystories_bpe4096 --out runs/ts_llama \
    --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --steps 6000 --warmup 200 --eval_interval 500 --compile \
    --bias false --norm rmsnorm --mlp swiglu --pos rope          # drop this line for GPT-2 style
.venv/bin/python sample.py --ckpt runs/ts_llama/ckpt.pt --prompt "Once upon a time" --temperature 0.8 --top_p 0.9

# Tiny Shakespeare, character-level (~1 min)
.venv/bin/python data.py
.venv/bin/python train.py
.venv/bin/python sample.py --prompt "ROMEO:"

# Real GPT-2 weights running in this codebase
.venv/bin/python gpt2.py "Hello, I'm a language model,"

# Experiments (need the TinyStories data above)
.venv/bin/python scaling.py                                      # 5 sizes, ~15 min, resumable -> assets/analysis/scaling.png
.venv/bin/python analysis.py --ckpt runs/ts_llama/ckpt.pt        # attention / induction / embedding plots
.venv/bin/python finetune.py --base runs/ts_llama/ckpt.pt        # instruction SFT + base-vs-SFT eval, ~5 min
```
Every model field is a flag (`train.py --help`). A checkpoint is saved at every eval; continue a stopped run with
`train.py --out <same dir> --resume` (keep the same `--steps` so the LR schedule matches). Training refuses to overwrite an existing run without `--resume`.

## Results (RTX 5060, bf16)
Compare across tokenizers by **val bits per byte (bpb)**: loss is per token, and tokens differ in size.

**TinyStories, 12M params, 6000 steps x 16k tokens (98M tokens, 1 epoch), identical settings:**

| Architecture | Val loss | Val bpb | Throughput | Wall time |
|---|---|---|---|---|
| GPT-2 style (learned pos, LayerNorm, GELU) | 1.512 | 0.543 | 256k tok/s | ~6.5 min |
| Llama style (RoPE, RMSNorm, SwiGLU, no bias) | **1.480** | **0.532** | 171k tok/s | ~9.7 min |

Llama style is better at every step (it reached GPT-2's final 0.543 at step 5000). Those runs were eager mode, where
Llama style is ~33% slower because RoPE and RMSNorm run as separate unfused ops. Train/val stay within 0.03: no overfitting.

**`torch.compile`** (`--compile`, same 12M models, 600-step benchmark; loss unchanged):

| | Eager | Compiled | Speedup |
|---|---|---|---|
| GPT-2 style | 252k tok/s | 295k tok/s | 1.17x |
| Llama style | 169k tok/s | 290k tok/s | **1.71x** |

Compilation fuses the elementwise RoPE/RMSNorm/SwiGLU ops, which closes almost the whole gap. It costs ~20 s of warmup, so skip it for short debug runs.

> Once upon a time, there was a little girl named Lily. She had a big, soft teddy bear that she loved very much. One day, she went to the park to play with her teddy bear. While playing, Lily saw a little boy who was sad. He had lost his teddy bear. Lily wanted to help [...] The moral of the story is that when we help others, we make new friends and make them happy.
>
> — Llama-style 12M, temperature 0.8, top-p 0.9

**GPT-2 parity:** with OpenAI's 124M weights loaded, logits match Hugging Face's within 1e-3, and greedy decoding gives GPT-2's known output.

**KV cache:** GPT-2 124M generating 600 tokens: 445 vs 82 tok/s (5.4x). For the 12M model with a 256-token context there is
almost nothing to save (~375 vs ~405 tok/s on GPU): per-token time is kernel-launch overhead, not compute.

**Tiny Shakespeare, 0.8M params:**

| Tokenizer | Bytes/token | Best val bpb |
|---|---|---|
| char (65) | 1.00 | 2.21 (~step 4000) |
| BPE 1024 | 2.62 | 2.22 (~step 1000), then overfits hard (383k train tokens) |

- Warmup + cosine + clipping vs constant LR: val loss 1.54 at ~3000 steps vs 1.56 at ~4250.
- bf16 vs fp32: ~1.20M vs ~0.73M tok/s, same loss.

## Scaling sweep
`python scaling.py` trains five Llama-style models on the same 49M TinyStories tokens (3000 steps, identical
hyperparameters, `--compile`, ~15 min total) and fits loss against non-embedding parameters N:

| Model | Shape (layers, heads, dim) | Non-emb params | Val loss | Val bpb |
|---|---|---|---|---|
| xs | 2, 2, 64 | 0.11M | 2.658 | 0.955 |
| s | 4, 4, 128 | 0.85M | 2.045 | 0.735 |
| m | 4, 4, 256 | 3.21M | 1.728 | 0.621 |
| l | 6, 6, 384 | 10.6M | 1.600 | 0.575 |
| xl | 8, 8, 512 | 25.7M | 1.516 | 0.545 |

**Fit: L(N) = 1.19 + 36.1 · N^-0.28.** Every 10x in parameters buys less than the last, and loss flattens toward an
irreducible ~1.19 for this data budget. The larger models are still improving at 49M tokens, so they are data-limited here.

![Scaling sweep](assets/analysis/scaling.png)

## Inside the model
`python analysis.py --ckpt runs/ts_llama/ckpt.pt` (12M Llama-style model):

**Embeddings** learn meaning with no supervision. Nearest neighbours by cosine similarity:

| Token | Nearest neighbours |
|---|---|
| happy | glad, grateful, cheerful, proud, excited, sad |
| dog | cat, dogs, puppy, horse, bird, squirrel |
| red | yellow, blue, green, pink, black, purple |
| mom | mommy, dad, Mom, mother, mum, grandma |
| big | small, large, huge, little, pretty, long |

![Embedding PCA](assets/analysis/embeddings.png)

**Attention:** every head of every layer on a story prompt (row = query token, column = key token). Layer 3 head 4 is a
clean *previous-token head* (one step off the diagonal).

![Attention maps](assets/analysis/attention_all.png)

**Induction heads** (attend to the token after the previous occurrence of the current token, so a repeated sequence can
be copied) are tested on a random 50-token sequence repeated twice. Across the sweep, previous-token heads strengthen
with size (best score 0.32 → 0.68), but induction scores stay small (0.01 → 0.06) and the repeat gets only modestly
easier (loss −0.65 at 25.7M). The first half of the induction circuit has formed, the copying half barely has: TinyStories
rarely repeats arbitrary sequences, and 49M tokens is early in training.

## Instruction fine-tuning
`python finetune.py --base runs/ts_llama/ckpt.pt` fine-tunes the 12M model on TinyStories-Instruct: a prompt of
constraints (required words, features like Dialogue/BadEnding, a summary, a sentence to include) followed by the story.
The loss covers only the story tokens. 3000 steps x 32 examples (61k examples that fit in 256 tokens), ~2 min.

| Model | Story val loss | Required words used (100 held-out prompts) |
|---|---|---|
| base (pretrained only) | 1.80 | 15% |
| fine-tuned | **1.42** | **51%** |

The base model has never seen the prompt format and mostly continues it as text. The fine-tuned model writes a story
and uses about half the required words. It follows features and summaries only loosely: at 12M parameters, that is
about as far as instruction-following goes. Same held-out prompt, both models:

```text
Words: decide, hero, charming
Features: BadEnding
Summary: Timmy meets a charming hero during a walk, but the hero turns out to be a bad person [...]
Story:
```
> **base:** Store: Deal: Deal. Deal. Deal?" The hero nodded and said: "Deal.. I'm sorry. I didn't mean to take your toy. [...]
>
> **fine-tuned:** Timmy's friends wanted to play with him, but they didn't want to play with him. Timmy felt sad and started to cry. [...] Suddenly, the hero came out and said, "I'm sorry for your friend. I didn't mean to scare you. Let's go away and be friends." Timmy and his friends were happy again and they all went home to play together. The end.

The fine-tuned model writes a proper story with the hero in it, but misses "decide" and "charming" and ignores the bad ending.

## Next
Longer pretraining (to see induction heads form) → compute-optimal sweep (vary tokens as well as parameters) → chat-style SFT with special tokens.

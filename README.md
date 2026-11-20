![pocket-gpt: a tiny GPT language model, built from scratch](assets/banners/pocket-gpt/gradient-1280x320.png)

![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-2.15-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-bf16-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Model](https://img.shields.io/badge/model-28M_params-6366F1?style=flat-square)
![Data](https://img.shields.io/badge/data-TinyStories_·_Tiny_Shakespeare-A21CAF?style=flat-square)
![GPT-2](https://img.shields.io/badge/GPT--2_parity-verified-0EA5E9?style=flat-square)
![Tests](https://img.shields.io/badge/tests-44_passing-3FB950?style=flat-square&logo=pytest&logoColor=white)

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
├── isoflop.py           # Compute-optimal (Chinchilla IsoFLOP) sweep: 3 budgets x 5 sizes, optimal N and D vs compute
├── analysis.py          # Attention maps, induction / previous-token heads (also over training), embeddings + PCA
├── finetune.py          # SFT on TinyStories-Instruct: plain-text or chat format (<|user|>/<|assistant|> tokens)
├── multiturn.py         # Multi-turn chat SFT: follow-ups answerable only from earlier turns + name-recall eval
├── judge.py             # Learned judge (reward model): does a story fit its request? Features, summary, words
├── dpo.py               # DPO preference tuning, multi-round on-policy; rule-based or learned-judge reward
├── chat.py              # Multi-turn terminal chat (history kept until the context is full, /reset)
├── tests/
│   ├── test_model.py    # per config (GPT-2/Llama x manual/SDPA): initial loss, no future leak, overfit one batch,
│   │                    #   manual == SDPA attention, KV cache == full forward, cached == uncached generation, top-p
│   ├── test_bpe.py      # first merge, unicode round-trip, compression, reload, incremental == naive training, specials
│   ├── test_finetune.py # instruct parsing, response-only loss mask, chat requests name every word, grow_vocab
│   ├── test_multiturn.py # name facts, per-turn loss masks, conversations fit the context
│   ├── test_dpo.py      # sequence log-prob counts only the reply, DPO loss = ln 2 when policy == reference
│   ├── test_judge.py    # judge scores ignore padding/batch mates, feature-only requests
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

# Full TinyStories (2.2 GB -> 546M tokens, ~35 s on 16 cores, streamed so memory stays flat)
.venv/bin/python data.py --dataset tinystories --tokenizer bpe --vocab_size 4096 --train_mb 2300
.venv/bin/python isoflop.py                                      # compute-optimal sweep, ~30 min -> isoflop.png
.venv/bin/python train.py --data data/tinystories_bpe4096_2300mb --out runs/ts_llama_long --steps 18000 \
    --n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --warmup 200 --eval_interval 500 --compile --induction \
    --bias false --norm rmsnorm --mlp swiglu --pos rope          # 295M tokens, ~17 min, logs induction scores
.venv/bin/python finetune.py --base runs/ts_llama_long/ckpt.pt --out runs/ts_llama_long_chat --chat
.venv/bin/python chat.py --ckpt runs/ts_llama_long_chat/ckpt.pt

# Compute-optimal 28M model with a 1024-token context, then multi-turn chat, then DPO (~75 min total)
.venv/bin/python train.py --data data/tinystories_bpe4096_2300mb --out runs/ts_xl_1024 --steps 22000 \
    --n_layer 8 --n_head 8 --n_embd 512 --block_size 1024 --batch_size 16 --warmup 300 --eval_interval 1000 \
    --compile --induction --bias false --norm rmsnorm --mlp swiglu --pos rope   # 360M tokens, ~45 min
.venv/bin/python multiturn.py --base runs/ts_xl_1024/ckpt.pt --out runs/ts_xl_chat              # ~6 min
.venv/bin/python dpo.py --sft runs/ts_xl_chat/ckpt.pt --reward words --rounds 1 --out runs/ts_xl_dpo  # ~15 min

# Learned judges + 3 rounds of judge-rewarded DPO (~40 min)
.venv/bin/python multiturn.py --base runs/ts_xl_1024/ckpt.pt --out runs/ts_xl_chat2
.venv/bin/python judge.py --sft runs/ts_xl_chat2/ckpt.pt --half 0 --out runs/judge_a
.venv/bin/python judge.py --sft runs/ts_xl_chat2/ckpt.pt --half 1 --out runs/judge_b
.venv/bin/python dpo.py --sft runs/ts_xl_chat2/ckpt.pt --reward judge --judge_rm runs/judge_a/ckpt.pt \
    --eval_judge runs/judge_b/ckpt.pt --rounds 3 --out runs/ts_xl_dpo_judge

# Chat model that also handles greetings, off-topic questions and casual story requests (~7 min)
.venv/bin/python multiturn.py --base runs/ts_xl_1024/ckpt.pt --out runs/ts_xl_chat3
.venv/bin/python chat.py --ckpt runs/ts_xl_chat3/ckpt.pt
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
easier (loss −0.65 at 25.7M).

**Do they form with more training?** `train.py --induction` logs the scores at every eval:

| Run | Data | Steps | Best induction score | Repeat gain (nats) |
|---|---|---|---|---|
| 12M Llama-style, long run | TinyStories, 295M tokens | 18000 | 0.07 | 0.91 |
| 2-layer control | TinyStories | 3000 | 0.01 | 0.08 |
| 2-layer + repeats (`--repeat_frac 0.25`) | TinyStories + 25% random sequences repeated with a random period | 3000 | **0.88** | **8.24** |

![Induction over training](assets/analysis/induction_training.png)

On plain TinyStories, 3x more training only nudges them (repeat gain 0.65 → 0.91; induction score stays ~0.07): the data
rarely rewards copying arbitrary sequences. When 25% of each batch is random tokens repeated with a random period, the
same 2-layer model shows the classic **phase change**: nothing for ~1700 steps, then within ~300 steps a previous-token
head (0.03 → 0.85) and an induction head (0.03 → 0.88) appear together and repeated sequences become almost free to
predict (loss −8.2 nats). The circuit needs both halves, so neither helps alone and they form at once.

A first version of this control used a fixed repeat period (129 tokens) and showed nothing: with RoPE the model solved it
with a positional "look exactly 128 back" head that does not transfer to other periods. A random period per row forces
the content-based mechanism.

## Compute-optimal sweep (Chinchilla)
`python isoflop.py`: for each compute budget C, every size N trains for exactly D = C/(6N) tokens with its cosine
schedule matched to that length (14 runs on the full 546M-token TinyStories, ~30 min). A parabola in log N per budget
gives the loss-minimising size.

| Budget (FLOPs) | Optimal params N | Optimal tokens D | Tokens / param |
|---|---|---|---|
| 3e14 | 1.6M | 32M | 21 |
| 1e15 | 3.1M | 54M | 18 |
| 3e15 | 5.8M | 86M | 15 |

**N_opt ∝ C^0.57, D_opt ∝ C^0.43**, and ~15–21 tokens per parameter: the Chinchilla result (≈0.5 / 0.5, ≈20 tokens per
parameter) reproduced at 1/10^8 of its scale. By this measure the 12M models above, trained on 98M tokens (8 tokens/param),
were undertrained for their size, which the long run confirms (val loss 1.480 → 1.337 with 295M tokens).
Three budgets give a rough exponent; more budgets would tighten it.

![IsoFLOP curves](assets/analysis/isoflop.png)

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

**Better pretraining, better fine-tune.** The same plain SFT on the long-run model (295M tokens instead of 98M) reaches
story val loss 1.28 (vs 1.42) and uses **59%** of required words (vs 51%).

## Chat fine-tuning
`finetune.py --chat` adds two special tokens, `<|user|>` and `<|assistant|>`: the embedding table (and the tied output
head) grows from 4096 to 4098 rows, the new rows starting at the mean embedding so existing logits are unchanged. Each
example's constraints become a natural request with varied phrasing; the loss covers only the assistant's story:

```text
<|user|>Tell me a story. It should be about this: [summary] Use the words "tremble", "friend" and "cold". Use some foreshadowing.<|assistant|>[story]<|endoftext|>
```

`chat.py` encodes the user's text with `encode_ordinary`, so typing `<|assistant|>` can never forge a role token.

```text
you> Tell me a story about a little dragon who is afraid of the dark.
model> Once upon a time there was a little dragon who was very frightened. He was scared of the dark [...] One day, the
dragon decided to be brave. [...] Finally, they found a small cave. The dragon crawled inside and was no longer frightened.

you> Write a short story for kids. Use the words "moon", "boat" and "brave". Include some dialogue.
model> Once upon a time there was a sailor who loved to go sailing. Every morning the sailor would get out his boat [...]
he remembered his boat and how he was brave enough to stay in the boat.
```

| Format (on the long-run model) | Story val loss | Required words used |
|---|---|---|
| plain `Words: ... Story:` | 1.28 | 59% |
| chat request | 1.38 | 37% |

The chat model handles free-form requests (topic, lesson, sad ending) that the plain format cannot express, but follows
word constraints less reliably: the words are buried in a sentence instead of a dedicated field, the summary is dropped
from 30% of requests, and the two role tokens start untrained. In the dragon story it kept the topic; in the second it
used "boat" and "brave" but missed "moon" and the dialogue.

## Compute-optimal pretraining with a longer context
At ~6e16 FLOPs (45 min on the RTX 5060), the IsoFLOP fit above puts the optimal size near 30M parameters, so the
largest sweep shape (8 layers, 8 heads, 512 dim; 27.8M total) was trained on 360M tokens (13 tokens/param) with a
**1024-token context** (16 x 1024 tokens per step, 133k tok/s with `--compile`, 1.1 GB of GPU memory):

| Model | Params | Context | Tokens | Val loss | Val bpb |
|---|---|---|---|---|---|
| ts_llama | 12.2M | 256 | 98M | 1.480 | 0.532 |
| ts_llama_long | 12.2M | 256 | 295M | 1.337 | 0.489 |
| **ts_xl_1024** | 27.8M | 1024 | 360M | **1.170** | **0.420** |

Its induction score peaked at 0.18 (repeat gain 1.65 nats), the most of any TinyStories run here, but still nowhere near
the 0.88 of the repeat-trained control.

## Multi-turn chat
`multiturn.py` builds conversations from TinyStories-Instruct whose follow-up turns can only be answered from earlier
turns: a character's name from the story the assistant told ("a dog named Max" → "What was the dog's name?"), a one-line
summary, the words the user asked for, or a second story with the first still in context. Every assistant turn ends
with `<|endoftext|>` and only assistant tokens are learned. 90k training conversations (340 tokens on average),
6000 steps x 8, ~6 min; chat val loss 2.04 → 1.16.

**Does it read the history?** On 200 held-out conversations, turn 1 is a dataset story and turn 2 asks for a character's
name (greedy decoding):

| History in context | Correct name |
|---|---|
| the real story | **98%** |
| a different conversation's story swapped in | 21% |

So the answer comes from the conversation, not from a memorised favourite name.

```text
you> Tell me a story about a little bunny named Pip who wants to fly.
model> Once upon a time, there was a little bunny called Pip. Pip was a very enthusiastic bunny and loved to explore. [...]
you> What was the bunny's name?
model> The bunny's name was Pip.
you> Can you sum it up in one sentence?
model> Pip, a enthusiastic bunny, discovers a beautiful butterfly and follows it to a beautiful park [...]
you> Thanks! Now tell me a story about a dragon. Use the words "moon" and "brave".
model> Once upon a time there was a little girl named Louise. [...] a friendly dragon sitting on a branch nearby. [...]
you> Which words did I ask you to use?
model> You asked me to use the words "match", "Pip" and "kip".
```

The last answer is wrong: in training, "which words" always referred to the *first* request, and here the words were
in the fourth. The model learned the pattern it was shown, not the general skill.

## DPO preference tuning
`dpo.py` tunes the multi-turn chat model with Direct Preference Optimization, using a rule-based reward instead of
human labels:

1. **Sample** 6 replies for each of 1500 story requests (temperature 1.0, top-p 0.95), batching requests of equal token length.
2. **Rank** each finished reply by the fraction of required words it uses. Best = chosen, worst = rejected; ties skipped.
   → 1216 pairs; chosen replies use 60% of the words, rejected ones 15%.
3. **Train** 300 steps x 8 pairs, β = 0.1, lr 5e-6, against a frozen copy of the starting model.
   DPO loss 0.69 → 0.26, reward accuracy 0.75 → ~1.0.
4. **Evaluate** 200 held-out requests, plus fluency: the per-token loss of each reply under the pretrained base model,
   which would rise sharply if DPO learned to just list the words.

| Model | Required words used | Reply length (tokens) | Fluency (judge loss, lower = more natural) |
|---|---|---|---|
| multi-turn chat (SFT) | 38% | 195 | 0.997 |
| + DPO | **43%** | 184 | 1.065 |

A real but modest gain (+5 points) for a small fluency cost and no length inflation. The pairs only teach "use more of
the words", so that is all that moves; the summary/feature constraints are not rewarded and do not improve.

### Word questions about any turn
In the first multi-turn data, "Which words did I ask you to use?" always referred to the first request, and the model
learned exactly that. Conversations are now built turn by turn: unqualified questions ("which words", "sum it up")
refer to the latest story, "...in the first story?" / "...in the last story?" name it, and a name question is only
asked when its noun picks out a single story. On 200 held-out two-story conversations (both with required words):

| Question | Old chat model | Retrained (`ts_xl_chat2`) |
|---|---|---|
| "Which words did I ask you to use?" (= latest) | 0% | **100%** |
| "...in the first story?" | 92% | 95% |
| "...in the last story?" | 0% | **99%** |

Name recall is unchanged (98% with the real history, 16% with a swapped one).

## A learned judge
The word reward ignores most of a request (features like a sad ending or dialogue, the summary). `judge.py` trains a
reward model for the whole request: the chat model's transformer plus a linear head on the hidden state at the closing
`<|endoftext|>`, with a pairwise (Bradley-Terry) loss, `-log σ(r(request, its story) - r(request, other story))`.
The "other story" depends on the request type: a random story for full and summary-only requests, a story *missing a
requested feature* for feature-only requests ("Give it a sad ending. Include some dialogue."), so features must be
detected rather than inferred from the topic.

Two judges are trained on disjoint halves of 87k entries (2500 steps x 16 pairs, ~8 min each). **Judge A** is the DPO
reward; **judge B** is used only for evaluation, so the policy cannot score well by exploiting one judge's quirks.
Held-out accuracy (picks the matching story):

| Judge | Full request | Features only | Summary only |
|---|---|---|---|
| A | 96% | 92% | 100% |
| B | 97% | 92% | 100% |

## Iterative DPO with the judge
`dpo.py --reward judge --rounds 3`: each round samples 6 replies to 1500 *fresh* requests from the current policy
(on-policy), rewards them with judge A + 2 x word fraction, trains 300 DPO steps against the policy as it was at the start
of the round, then evaluates 300 held-out requests (temperature 0.7):

| Model | Words used | Dialogue (when asked) | Judge B score | Judge B win rate vs SFT | Fluency (lower = better) | Length |
|---|---|---|---|---|---|---|
| multi-turn SFT | 41% | 90% | 10.1 | – | **0.98** | 183 |
| round 1 | 49% | **95%** | 12.3 | 82% | 1.11 | 162 |
| round 2 | 59% | 83% | 13.0 | 87% | 1.22 | 178 |
| round 3 | **65%** | 86% | **13.2** | **89%** | 1.37 | 221 |

Every round follows the request better by the held-out judge and by word use, and every round is **less fluent**. The
example shows how: the final model packs the request's own phrases into the story.

```text
Request: It should be about this: Tom and Anna are excited to go on a holiday with their parents, and they fly on a big
plane to a place with sun and sand. Include some dialogue. Include the sentence: "They are very excited and want to fly too."

round 3: Tom and Anna are very excited. They are very excited and want to go on a holiday with their mom and dad. [...]
They are very excited and want to fly on a big plane. They look for a place to go and they see a place with sun and
sand and sand. [...]
```

Copying summary wording convinces both judges (they learned "the summary's words appear" = "this story matches"),
so this is **reward hacking** of a pattern both judges share, which a held-out judge cannot catch. The fluency check
does catch it. Round 1 is the best trade-off here (+8 points of words, +5 of dialogue, small fluency cost); more rounds
would need a KL or fluency penalty in the reward, or judges trained on hard negatives that copy the summary.

## Small talk and casual requests
Chatting with `ts_xl_chat2` showed a failure the evals missed: every message after a story got the story's summary.

```
you> bella's male ?     model> Bella explores a beautiful rug in her garden and meets a rude voice.
you> hello              model> Bella explores a beautiful rug in her garden and meets a friendly voice.
```

Its training only had story requests and questions about stories, so "summary" was the safe answer to anything.
`ts_xl_chat3` adds greetings and off-topic questions or random letters, each with a fixed reply. It also makes a
quarter of the story requests short and casual ("tell me a story"), with no "which words?" questions after them.
A first try without the casual requests turned "write short children story" down as off-topic: every story request
it had seen listed words to use.

On phrasings that never appear in training, these are the rates of correct replies:

| Held-out message (after a story) | Without casual requests | `ts_xl_chat3` |
|---|---|---|
| Greeting ("hiya", "howdy") | 48% | 45% |
| Off-topic question | 90% | 100% |
| Random letters | 100% | 100% |
| Story request ("story please") | 55% | 88% |
| Greeting as the first message | 42% | 55% |

Name recall (98%) and words recall (93-97%) are unchanged. Your replayed chat now gets a greeting for "hi" and
"hello", a story for "write short children story", and the "I only tell stories" reply for the rest. Greetings it
never saw still fail about half the time; more varied greetings in training would be the next fix.

## Next
Add a fluency penalty or KL budget to the reward → judges trained against summary-copying negatives → a bigger
pretraining run (the IsoFLOP fit says ~45M parameters for 2x the compute).

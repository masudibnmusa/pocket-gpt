"""Load OpenAI's pretrained GPT-2 weights (via Hugging Face) into model.GPT to verify the architecture.

python gpt2.py "Hello, I'm a language model,"    # needs: pip install transformers
"""
import re
import sys

import torch

from model import GPT, GPTConfig

# HF name inside transformer.h.<i>. -> our name inside blocks.<i>.
_BLOCK_RENAME = {"ln_1": "ln1", "ln_2": "ln2", "attn.c_attn": "attn.qkv", "attn.c_proj": "attn.proj",
                 "mlp.c_fc": "mlp.fc", "mlp.c_proj": "mlp.proj"}
# HF GPT-2 stores these as Conv1D, i.e. (in, out); nn.Linear wants (out, in).
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2(name="gpt2"):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(name)
    c = hf.config
    cfg = GPTConfig(vocab_size=c.vocab_size, block_size=c.n_positions, n_layer=c.n_layer, n_head=c.n_head, n_embd=c.n_embd)
    model = GPT(cfg)  # GPTConfig's defaults are the GPT-2 architecture

    ours = {}
    for k, v in hf.state_dict().items():
        if k == "lm_head.weight" or k.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # head is tied to the embedding; the others are HF's causal-mask buffers
        if k.endswith(_TRANSPOSED):
            v = v.t()
        m = re.fullmatch(r"transformer\.h\.(\d+)\.(\w+\.\w+|\w+)\.(weight|bias)", k)
        if m:
            layer, sub, kind = m.groups()
            k = f"blocks.{layer}.{_BLOCK_RENAME[sub]}.{kind}"
        else:
            k = {"transformer.wte.weight": "tok_emb.weight", "transformer.wpe.weight": "pos_emb.weight",
                 "transformer.ln_f.weight": "ln_f.weight", "transformer.ln_f.bias": "ln_f.bias"}[k]
        ours[k] = v
    missing, unexpected = model.load_state_dict(ours, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected, (missing, unexpected)  # lm_head comes via tying
    return model.eval(), hf.eval()


if __name__ == "__main__":
    from transformers import GPT2TokenizerFast

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I'm a language model,"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model, _ = load_gpt2()
    idx = torch.tensor([tok.encode(prompt)])
    print(tok.decode(model.generate(idx, 40, temperature=0)[0].tolist()))

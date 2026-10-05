#!/usr/bin/env python3
"""Generate ``reference/pins.json`` from a Hugging Face transformers reference forward.

Run once on the cluster (needs the weights and GPUs or enough host memory):

    python3 accuracy_checker/make_pins.py --model-path "$MODEL_PATH"

For each fixed prompt it applies the chat template with thinking disabled,
greedy-decodes exactly ``--n-tokens`` tokens (EOS suppressed, matching the
benchmark's ``ignore_eos``), one prompt at a time (no padding effects), and
records the token ids. The output schema is documented in
``accuracy_checker/README.md``.

The pinned checkpoint is normally the same one candidates serve. If
transformers cannot load it (for example a quantized MXFP4 export), pass
``--model-path`` pointing at a loadable copy of the same model (bf16 or FP8)
and ``--tokenizer-path`` pointing at the served checkpoint; the checker's
budget tolerates the resulting numeric noise but not a different model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import EXACT_PREFIX_TOKENS, HOLDOUT_WORD_BANK, PINS_SCHEMA_VERSION  # noqa: E402

sys.path.insert(0, str(BUNDLE_DIR))  # mxfp4.py and weights.py, for --mxfp4

PINS_SEED = 20260301  # distinct from the benchmark and holdout seeds.

STATIC_PROMPTS: tuple[list[dict], ...] = (
    [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
    [
        {
            "role": "user",
            "content": "Write a Python function that returns the n-th Fibonacci number.",
        }
    ],
    [{"role": "user", "content": "List three differences between TCP and UDP."}],
    [{"role": "user", "content": "Translate to French: The meeting has been moved to Thursday."}],
    [{"role": "user", "content": "What is the capital of Australia, and what is it known for?"}],
    [{"role": "user", "content": "Summarize the plot of Romeo and Juliet in one paragraph."}],
    [
        {"role": "user", "content": "My favorite color is green. Please remember it."},
        {"role": "assistant", "content": "Got it, your favorite color is green."},
        {"role": "user", "content": "Suggest a hobby that matches my favorite color."},
    ],
    [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Give me a one-line tip for writing clear commit messages."},
    ],
)
N_GENERATED_PROMPTS = 8


def generated_prompts() -> list[list[dict]]:
    """Longer word-bank prompts, deterministic, so pins also cover multi-hundred-token prefills."""
    prompts = []
    for index in range(N_GENERATED_PROMPTS):
        rng = random.Random(f"{PINS_SEED}-{index}")
        words = " ".join(rng.choice(HOLDOUT_WORD_BANK) for _ in range(rng.randint(60, 240)))
        prompts.append(
            [{"role": "user", "content": f"Summarize this text in one sentence: {words}."}]
        )
    return prompts


def load_mxfp4_reference(model_path: str, dtype):  # noqa: ANN001, ANN201
    """Transformers `Qwen3_5MoeForCausalLM` with MXFP4 experts dequantized on use.

    transformers cannot load the Quark MXFP4 export, and the dense bf16 experts (773 GB) do not
    fit in 4 x 128 GB. So the HF modules (attention, DeltaNet, router, shared expert, norms,
    generate) run unchanged, and only `Qwen3_5MoeExperts` is replaced by a module that keeps the
    packed uint8 payloads and dequantizes each hit expert per call. The dequantization is
    `mxfp4.dequant_mxfp4`, shared with the seed; it was checked against the FP8 source
    checkpoint (reference/README.md). Layers are split contiguously over all GPUs and
    activations are moved between them by forward pre-hooks.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from mxfp4 import dequant_mxfp4
    from torch import nn
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
    from transformers.activations import ACT2FN
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    from weights import Checkpoint

    raw = json.loads((Path(model_path) / "config.json").read_text())
    cfg = Qwen3_5MoeTextConfig(**{k: v for k, v in raw["text_config"].items() if k != "model_type"})
    cfg._attn_implementation = "eager"  # noqa: SLF001
    ck = Checkpoint(model_path)
    devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [
        torch.device("cpu")
    ]
    n_layers = cfg.num_hidden_layers

    class PackedExperts(nn.Module):
        def __init__(self, layer: int, dev: torch.device) -> None:
            super().__init__()
            p, n = f"model.language_model.layers.{layer}.mlp.experts", cfg.num_experts
            self.act_fn = ACT2FN[cfg.hidden_act]

            def stack(read):  # noqa: ANN001, ANN202
                first = read(0)
                out = torch.empty((n, *first.shape), dtype=first.dtype, device=dev)
                out[0] = first

                def fill(e: int) -> None:
                    out[e] = read(e)

                with ThreadPoolExecutor(32) as pool:
                    list(pool.map(fill, range(1, n)))
                return out

            def rd(e: int, proj: str, suffix: str = "weight"):  # noqa: ANN202
                return ck.load(f"{p}.{e}.{proj}.{suffix}", dev)

            cat = lambda e, s: torch.cat([rd(e, "gate_proj", s), rd(e, "up_proj", s)], 0)  # noqa: E731
            self.gate_up = stack(lambda e: cat(e, "weight"))
            self.gate_up_scale = stack(lambda e: cat(e, "weight_scale"))
            self.down = stack(lambda e: rd(e, "down_proj"))
            self.down_scale = stack(lambda e: rd(e, "down_proj", "weight_scale"))

        def forward(self, hidden_states, top_k_index, top_k_weights):  # noqa: ANN001, ANN202
            out = torch.zeros_like(hidden_states)
            for e in top_k_index.unique().tolist():
                pos, tok = torch.where(top_k_index.T == e)  # [top_k, T] layout as the HF module
                gate_up = dequant_mxfp4(self.gate_up[e], self.gate_up_scale[e], dtype)
                down = dequant_mxfp4(self.down[e], self.down_scale[e], dtype)
                gate, up = nn.functional.linear(hidden_states[tok], gate_up).chunk(2, dim=-1)
                y = nn.functional.linear(self.act_fn(gate) * up, down)
                out.index_add_(0, tok, (y * top_k_weights[tok, pos, None]).to(out.dtype))
            return out

    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(cfg).to(dtype)

    def put(module: nn.Module, prefix: str, dev: torch.device) -> None:
        """Materialize `module`'s own params on dev from checkpoint names under `prefix`."""
        module.to_empty(device=dev)
        for name, param in module.named_parameters():
            src = f"{prefix}.{name}" if prefix else name
            if ck.has(src):
                param.data.copy_(ck.load(src, dev, param.dtype))
            else:
                raise KeyError(f"checkpoint has no tensor {src}")

    lm = "model.language_model."
    layer_dev = [devs[i * len(devs) // n_layers] for i in range(n_layers)]
    put(model.model.embed_tokens, lm + "embed_tokens", devs[0])
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(cfg, device=devs[0])
    for i, layer in enumerate(model.model.layers):
        dev = layer_dev[i]
        layer.mlp.experts = PackedExperts(i, dev)
        # experts are real already; materialize everything else in the layer
        for name, child in layer.named_children():
            if name != "mlp":
                put(child, f"{lm}layers.{i}.{name}", dev)
        for name, child in layer.mlp.named_children():
            if name != "experts":
                put(child, f"{lm}layers.{i}.mlp.{name}", dev)

        def move(_m, args, kwargs, dev=dev):  # noqa: ANN001, ANN202
            def to(x):  # noqa: ANN001, ANN202
                if isinstance(x, torch.Tensor):
                    return x.to(dev)
                if isinstance(x, (tuple, list)):
                    return type(x)(to(v) for v in x)
                return x

            return to(args), {k: to(v) for k, v in kwargs.items()}

        layer.register_forward_pre_hook(move, with_kwargs=True)
        print(f"loaded layer {i} on {dev}", file=sys.stderr, flush=True)
    last = devs[-1]
    put(model.model.norm, lm + "norm", last)
    model.lm_head.to_empty(device=last)
    model.lm_head.weight.data = ck.load("lm_head.weight", last, dtype)
    model.model.norm.register_forward_pre_hook(
        lambda _m, a, k: (tuple(x.to(last) for x in a), k), with_kwargs=True
    )
    model.lm_head.register_forward_hook(lambda _m, _a, out: out.to(devs[0]))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Checkpoint transformers loads.")
    parser.add_argument("--tokenizer-path", default=None, help="Defaults to --model-path.")
    parser.add_argument("--n-tokens", type=int, default=EXACT_PREFIX_TOKENS)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--mxfp4",
        action="store_true",
        help="Load the MXFP4 checkpoint with experts dequantized on use (see load_mxfp4_reference).",
    )
    parser.add_argument("--output", default=str(BUNDLE_DIR / "reference" / "pins.json"))
    args = parser.parse_args()
    if args.n_tokens < EXACT_PREFIX_TOKENS:
        parser.error(f"--n-tokens must be >= {EXACT_PREFIX_TOKENS} (the checker's prefix length)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_path, trust_remote_code=True
    )
    if args.mxfp4:
        model = load_mxfp4_reference(args.model_path, getattr(torch, args.dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=getattr(torch, args.dtype),
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    pins = []
    for index, messages in enumerate((*STATIC_PROMPTS, *generated_prompts())):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.n_tokens,
                min_new_tokens=args.n_tokens,  # suppress EOS: mirrors ignore_eos
            )
        ids = out[0, inputs["input_ids"].shape[1] :].tolist()
        if set(ids) & set(tokenizer.all_special_ids):
            # The chat API returns text without special tokens, so a pin that contains one
            # (the model ended its turn before n_tokens) cannot be compared by retokenizing.
            print(f"pin{index:02d}: skipped, reference emitted a special token", file=sys.stderr)
            continue
        pins.append({"id": f"pin{index:02d}", "messages": messages, "expected_token_ids": ids})
        print(f"pin{index:02d}: {tokenizer.decode(ids)[:80]!r}", file=sys.stderr)

    payload = {
        "version": PINS_SCHEMA_VERSION,
        "model": args.model_path,
        "n_tokens": args.n_tokens,
        "enable_thinking": False,
        "pins": pins,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(pins)} pins to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a tiny random checkpoint of an architecture and load the result.

The checkpoint is written from a config alone (random weights, a byte-level
vocabulary), so the test needs no download. It checks that the converter
resolves the architecture, maps every tensor, and that llama.cpp loads the
GGUF and decodes a few tokens with it.

usage: test-convert-tiny.py <path to llama-completion> [architecture ...]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import gguf  # noqa: E402
from conversion import get_model_class  # noqa: E402

N_VOCAB = 288


def bailing_v2_like(hf_arch: str) -> tuple[dict, dict[str, torch.Tensor]]:
    n_embd, n_head, n_head_kv, head_dim, n_layer = 64, 4, 2, 16, 2
    n_expert, n_ff_exp = 4, 32
    config = {
        "architectures": [hf_arch],
        "vocab_size": N_VOCAB,
        "hidden_size": n_embd,
        "intermediate_size": 96,
        "moe_intermediate_size": n_ff_exp,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head_kv,
        "head_dim": head_dim,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "first_k_dense_replace": 1,
        "num_experts": n_expert,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "score_function": "sigmoid",
        "use_qk_norm": True,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
    }
    g = torch.Generator().manual_seed(0)

    def w(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g) * 0.02

    t: dict[str, torch.Tensor] = {
        "model.word_embeddings.weight": w(N_VOCAB, n_embd),
        "model.norm.weight": torch.ones(n_embd),
        "lm_head.weight": w(N_VOCAB, n_embd),
    }
    qkv = (n_head + 2 * n_head_kv) * head_dim
    for i in range(n_layer):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = torch.ones(n_embd)
        t[p + "post_attention_layernorm.weight"] = torch.ones(n_embd)
        t[p + "attention.query_key_value.weight"] = w(qkv, n_embd)
        t[p + "attention.dense.weight"] = w(n_embd, n_head * head_dim)
        t[p + "attention.query_layernorm.weight"] = torch.ones(head_dim)
        t[p + "attention.key_layernorm.weight"] = torch.ones(head_dim)
        if i < config["first_k_dense_replace"]:
            for name, shape in (("gate_proj", (96, n_embd)), ("up_proj", (96, n_embd)), ("down_proj", (n_embd, 96))):
                t[p + f"mlp.{name}.weight"] = w(*shape)
            continue
        t[p + "mlp.gate.weight"] = w(n_expert, n_embd)
        t[p + "mlp.gate.expert_bias"] = torch.zeros(n_expert)
        for x in range(n_expert):
            for name, shape in (("gate_proj", (n_ff_exp, n_embd)), ("up_proj", (n_ff_exp, n_embd)), ("down_proj", (n_embd, n_ff_exp))):
                t[p + f"mlp.experts.{x}.{name}.weight"] = w(*shape)
        for name, shape in (("gate_proj", (n_ff_exp, n_embd)), ("up_proj", (n_ff_exp, n_embd)), ("down_proj", (n_embd, n_ff_exp))):
            t[p + f"mlp.shared_experts.{name}.weight"] = w(*shape)
    return config, t


CASES = {
    # architecture in the GGUF: (HF architecture, checkpoint builder)
    "llada2": ("LLaDA2MoeModelLM", bailing_v2_like),
    "bailingmoe2": ("BailingMoeV2ForCausalLM", bailing_v2_like),
}


def write_byte_vocab(writer: gguf.GGUFWriter) -> None:
    # 256 byte tokens plus a few specials; the converter's own vocabulary
    # handling is covered by the tokenizer tests
    tokens = [f"<0x{b:02X}>" for b in range(256)] + [f"<special_{i}>" for i in range(N_VOCAB - 256)]
    types = [gguf.TokenType.BYTE] * 256 + [gguf.TokenType.CONTROL] * (N_VOCAB - 256)
    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * N_VOCAB)
    writer.add_token_types(types)
    writer.add_bos_token_id(256)
    writer.add_eos_token_id(257)


def run_case(gguf_arch: str, completion: Path, work: Path) -> None:
    hf_arch, build = CASES[gguf_arch]
    model_dir = work / gguf_arch
    model_dir.mkdir()
    config, tensors = build(hf_arch)
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(model_dir / "model.safetensors"))

    cls = get_model_class(hf_arch)
    if gguf.MODEL_ARCH_NAMES[cls.model_arch] != gguf_arch:
        raise SystemExit(f"{hf_arch}: converter writes {gguf.MODEL_ARCH_NAMES[cls.model_arch]}, expected {gguf_arch}")

    class Tiny(cls):  # type: ignore[misc, valid-type]
        model_arch = cls.model_arch

        def set_vocab(self):
            write_byte_vocab(self.gguf_writer)

    out = work / f"{gguf_arch}.gguf"
    model = Tiny(model_dir, gguf.LlamaFileType.ALL_F32, out)
    model.write()

    reader = gguf.GGUFReader(out)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    if arch != gguf_arch:
        raise SystemExit(f"{out.name}: general.architecture is {arch}")

    res = subprocess.run(
        [str(completion), "-m", str(out), "-p", "ab", "-n", "4", "--temp", "0", "-no-cnv", "--no-warmup", "-ngl", "0"],
        capture_output=True, text=True, errors="replace", timeout=300,
    )
    if res.returncode != 0:
        raise SystemExit(f"{gguf_arch}: llama-completion failed ({res.returncode})\n{res.stderr[-2000:]}")
    print(f"{gguf_arch}: converted {len(reader.tensors)} tensors from {hf_arch}, loaded and decoded")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    completion = Path(sys.argv[1])
    archs = sys.argv[2:] or list(CASES)
    with tempfile.TemporaryDirectory() as tmp:
        for a in archs:
            run_case(a, completion, Path(tmp))


if __name__ == "__main__":
    main()

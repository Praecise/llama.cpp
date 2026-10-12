#!/usr/bin/env python3
"""The omni talker's waveform decoder tensors are mapped and folded as the graph reads them.

Every SnakeBeta in the decoder (each upsampling block's, each residual unit's two, and
the final one) keeps alpha and beta in log scale in the checkpoint; the converter folds
them to exp(alpha) and 1 / (exp(beta) + 1e-9). A checkpoint whose parameters are all zero
cannot tell the two apart, so this uses distinct non-zero values.

usage: test-convert-omni-talker-decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import torch  # noqa: E402

import gguf  # noqa: E402
from conversion.qwen3omni_talker import Qwen3OmniTalkerGenModel  # noqa: E402


def main() -> None:
    n_blocks = 2
    fake = SimpleNamespace(
        _wav_config={"upsample_rates": [2] * n_blocks},
        model_arch=gguf.MODEL_ARCH.MMPROJ,
    )
    fake.format_tensor_name = lambda key, bid=None, suffix=".weight": Qwen3OmniTalkerGenModel.format_tensor_name(fake, key, bid, suffix)
    fake._snake = Qwen3OmniTalkerGenModel._snake

    alpha = torch.tensor([0.5, -0.25])
    beta = torch.tensor([-0.75, 1.0])
    want_a = torch.exp(alpha)
    want_b = 1.0 / (torch.exp(beta) + 1e-9)

    cases = [
        ("1.block.0.alpha", "a.gen.wav.dac.blk.0.snake.alpha", want_a),
        ("1.block.0.beta", "a.gen.wav.dac.blk.0.snake.beta", want_b),
        ("2.block.3.act1.alpha", "a.gen.wav.dac.blk.1.res.1.act1.alpha", want_a),
        ("2.block.4.act2.beta", "a.gen.wav.dac.blk.1.res.2.act2.beta", want_b),
        (f"{n_blocks + 1}.alpha", "a.gen.wav.dac.post_snake.alpha", want_a),
        (f"{n_blocks + 1}.beta", "a.gen.wav.dac.post_snake.beta", want_b),
    ]
    for rest, name, want in cases:
        data = alpha if rest.endswith("alpha") else beta
        out = list(Qwen3OmniTalkerGenModel._decoder_tensor(fake, rest, data))
        assert len(out) == 1, (rest, out)
        got_name, got = out[0]
        assert got_name == name, (rest, got_name)
        assert torch.allclose(got, want), (rest, got, want)

    # convolutions and the transposed convolutions pass through unchanged
    w = torch.randn(3, 3, 7)
    (name, got), = Qwen3OmniTalkerGenModel._decoder_tensor(fake, "0.conv.weight", w)
    assert name == "a.gen.wav.dac.entry.weight" and torch.equal(got, w), name
    (name, got), = Qwen3OmniTalkerGenModel._decoder_tensor(fake, "1.block.1.conv.weight", w)
    assert name == "a.gen.wav.dac.blk.0.conv.weight" and torch.equal(got, w), name
    print("ok")


if __name__ == "__main__":
    main()

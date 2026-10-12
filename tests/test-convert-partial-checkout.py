#!/usr/bin/env python3
"""A conversion that keeps part of a sharded checkpoint reads only what it keeps.

A model class that converts one part of a checkpoint (the speech part of an omni
model, say) declares the tensors it needs; a local copy holding only the shards
with those tensors, or shards trimmed to those tensors, must index cleanly, and a
needed tensor that is missing must still be refused.

usage: test-convert-partial-checkout.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gguf-py"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from conversion.base import ModelBase  # noqa: E402


def checkout(root: Path, shards: dict[str, dict[str, torch.Tensor]], index: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, tensors in shards.items():
        save_file(tensors, str(root / name))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


def indexer(root: Path, needed: tuple[str, ...]):
    # the parts of ModelBase that index_tensors reads, with a class keeping `needed`
    keep = lambda name: name.startswith(needed)  # noqa: E731
    return SimpleNamespace(
        dir_model=root,
        is_mistral_format=False,
        lazy=True,
        mtp_only=False,
        filter_tensors=lambda item: item if keep(item[0]) else None,
        needs_tensor=keep,
    )


def index(fake) -> dict:
    return ModelBase.index_tensors(fake)


def main() -> None:
    t = lambda: torch.zeros(2, 2)  # noqa: E731
    full_index = {
        "reader.a": "model-00001-of-00003.safetensors",
        "reader.b": "model-00002-of-00003.safetensors",
        "speaker.a": "model-00002-of-00003.safetensors",
        "speaker.b": "model-00003-of-00003.safetensors",
    }
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        # only the shards holding the speaker's tensors, the second trimmed to them
        part = work / "part"
        checkout(part, {
            "model-00002-of-00003.safetensors": {"speaker.a": t()},
            "model-00003-of-00003.safetensors": {"speaker.b": t()},
        }, full_index)
        got = index(indexer(part, ("speaker.",)))
        assert sorted(got) == ["speaker.a", "speaker.b"], sorted(got)

        # a needed tensor whose shard is absent is still refused
        broken = work / "broken"
        checkout(broken, {"model-00002-of-00003.safetensors": {"speaker.a": t()}}, full_index)
        try:
            index(indexer(broken, ("speaker.",)))
        except ValueError as e:
            assert "model-00003-of-00003.safetensors" in str(e), e
        else:
            raise AssertionError("a missing needed shard was accepted")

        # a class that needs everything still refuses a partial copy
        try:
            index(indexer(part, ("",)))
        except ValueError:
            pass
        else:
            raise AssertionError("a partial copy was accepted for a full conversion")

    print("ok")


if __name__ == "__main__":
    main()

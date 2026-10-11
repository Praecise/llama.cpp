from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, MmprojModel, ModelBase, TextModel, gguf, logger

from .qwen import QwenModel


@ModelBase.register("HunYuanMoEV1ForCausalLM")
@ModelBase.example("tencent/Hunyuan-A13B-Instruct")
class HunYuanMoEModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HUNYUAN_MOE

    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        # 1. Get the pre-tokenizer identifier hash
        tokpre = self.get_vocab_base_pre(tokenizer)

        # 2. Reverse-engineer the merges list from mergeable_ranks
        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks  # ty: ignore[unresolved-attribute]
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            if len(merged) == 2: # todo this is an assert in Qwen, why?
                merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

        # 3. Generate the tokens and toktypes lists
        vocab_size = self.hparams["vocab_size"]
        assert tokenizer.vocab_size == vocab_size  # ty: ignore[unresolved-attribute]
        special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
        reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token = reverse_vocab[i]
                tokens.append(token)
                if i in special_tokens.values():
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)

        # 4. Write all vocab-related fields to the GGUF writer
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)

        # 5. Add special tokens and chat templates
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)
        # FIX for BOS token: Overwrite incorrect id read from config.json
        self.gguf_writer.add_bos_token_id(127959) # <|bos|>

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["intermediate_size"])

        moe_intermediate_size = hparams["moe_intermediate_size"]
        assert all(n == moe_intermediate_size[0] for n in moe_intermediate_size)
        self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size[0])

        moe_topk = hparams["moe_topk"]
        assert all(topk == moe_topk[0] for topk in moe_topk)
        self.gguf_writer.add_expert_used_count(moe_topk[0])

        moe_shared_expert = hparams["num_shared_expert"]
        assert all(n == moe_shared_expert[0] for n in moe_shared_expert)
        self.gguf_writer.add_expert_shared_count(moe_shared_expert[0])

        # Rope
        if self.rope_parameters.get("rope_type") == "dynamic":
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = self.rope_parameters.get("alpha", 1000)
            base = self.rope_parameters.get("rope_theta", 10000.0)
            dim = (hparams["hidden_size"] // hparams["num_attention_heads"]) # 128
            scaled_base = base * (alpha ** (dim / (dim - 2))) # 10000 * (1000 ** (128 / 126)) = 11158839.9251
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
            self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
            self.gguf_writer.add_context_length(256 * 1024) # 256k context length

            # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
            assert alpha == 1000 and base == 10000.0 and dim == 128 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return

        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("HunYuanDenseV1ForCausalLM")
@ModelBase.example("tencent/Hunyuan-4B-Instruct")
class HunYuanModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HUNYUAN_DENSE

    def set_vocab(self):
        # Also called by draft models (e.g. DFlash), with dir_model pointing at
        # the target model.
        config = ModelBase.load_hparams(self.dir_model, self.is_mistral_format)
        config = {**config, **config.get("text_config", {})}
        self.hparams["pad_token_id"] = config.get("pad_token_id")
        self.hparams["eod_token_id"] = config.get("eod_token_id")

        if (self.dir_model / "tokenizer.json").is_file():
            tokens, toktypes, tokpre = self.get_vocab_base()
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)

            # Some HunYuanVL variants (e.g. OCR-style configs) have pad_token_id=-1;
            # guard SpecialVocab so it doesn't try to emit an invalid pad id.
            token_types = None
            if (self.hparams.get("pad_token_id") or 0) < 0:
                token_types = ('bos', 'eos', 'unk', 'sep', 'cls', 'mask')
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True, special_token_types=token_types)
            special_vocab.add_to_gguf(self.gguf_writer)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

            # 1. Get the pre-tokenizer identifier hash
            tokpre = self.get_vocab_base_pre(tokenizer)

            # 2. Reverse-engineer the merges list from mergeable_ranks
            merges = []
            vocab = {}
            mergeable_ranks = tokenizer.mergeable_ranks  # ty: ignore[unresolved-attribute]
            for token, rank in mergeable_ranks.items():
                vocab[QwenModel.token_bytes_to_string(token)] = rank
                if len(token) == 1:
                    continue
                merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
                if len(merged) == 2:
                    merges.append(' '.join(map(QwenModel.token_bytes_to_string, merged)))

            # 3. Generate the tokens and toktypes lists
            vocab_size = self.hparams["vocab_size"]
            assert tokenizer.vocab_size == vocab_size  # ty: ignore[unresolved-attribute]
            special_tokens = tokenizer.special_tokens  # ty: ignore[unresolved-attribute]
            reverse_vocab = {id_ : encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
            tokens: list[str] = []
            toktypes: list[int] = []
            for i in range(vocab_size):
                if i not in reverse_vocab:
                    tokens.append(f"[PAD{i}]")
                    toktypes.append(gguf.TokenType.UNUSED)
                else:
                    token = reverse_vocab[i]
                    tokens.append(token)
                    if i in special_tokens.values():
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        toktypes.append(gguf.TokenType.NORMAL)

            # 4. Write all vocab-related fields to the GGUF writer
            self.gguf_writer.add_tokenizer_model("gpt2")
            self.gguf_writer.add_tokenizer_pre(tokpre)
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_types(toktypes)
            self.gguf_writer.add_token_merges(merges)

            # 5. Add special tokens and chat templates
            special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
            special_vocab.add_to_gguf(self.gguf_writer)
            # FIX for BOS token: Overwrite incorrect id read from config.json
            if self.hparams['hidden_size'] == 4096:
                self.gguf_writer.add_bos_token_id(127958) # only for 7b dense, fix <|bos|> token

        # Fix EOS/EOT tokens that are incorrect in upstream configs.
        eod_id = self.hparams.get("eod_token_id")
        if eod_id is not None:
            self.gguf_writer.add_eos_token_id(eod_id)

        gen_cfg = self.dir_model / "generation_config.json"
        if gen_cfg.is_file():
            with open(gen_cfg, encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id")
            if isinstance(eos, list) and len(eos) >= 2:
                self.gguf_writer.add_eot_token_id(eos[0])

    def set_gguf_parameters(self):
        # Some HunYuanVL variants set num_experts=1 (not real MoE);
        # prevent the parent class from emitting expert_count metadata in that case.
        saved_num_experts = self.hparams.pop("num_experts", None)
        super().set_gguf_parameters()
        if saved_num_experts is not None and saved_num_experts > 1:
            self.hparams["num_experts"] = saved_num_experts
        hparams = self.hparams

        # Rope
        if self.rope_parameters.get("rope_type") in ("dynamic", "xdrope"):
            # HunYuan uses NTK Aware Alpha based scaling. Original implementation: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
            # 1000 corresponds to a usable context length of 256k (https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)
            alpha = self.rope_parameters.get("alpha", 50)
            base = self.rope_parameters.get("rope_theta", 10000.0)
            dim = hparams["head_dim"]
            scaled_base = base * (alpha ** (dim / (dim - 2)))
            self.gguf_writer.add_rope_freq_base(scaled_base)
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
            self.gguf_writer.add_rope_scaling_factor(1)
            if self.rope_parameters.get("rope_type") == "dynamic":
                # There is no consistent way to calculate ctx from alpha, and the config is incorrectly set to 32k
                self.gguf_writer.add_rope_scaling_orig_ctx_len(256 * 1024) # 256k context length
                self.gguf_writer.add_context_length(256 * 1024) # 256k context length

                # if any of our assumptions about the values are wrong, something has changed and this may need to be updated
                assert base == 10000.0 and self.hparams["max_position_embeddings"] in [32 * 1024, 256 * 1024] , \
                    "HunYuan dynamic RoPE scaling assumptions changed, please update the logic or context length manually"

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight":
            if self.hparams.get("tie_word_embeddings", False):
                logger.info("Skipping tied output layer 'lm_head.weight'")
                return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("HunYuanVLForConditionalGeneration")
@ModelBase.example("tencent/HunyuanOCR")
class HunyuanVLVisionModel(MmprojModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        # HunyuanVL uses max_image_size instead of image_size
        if "image_size" not in self.hparams_vision:
            self.hparams_vision["image_size"] = self.hparams_vision.get("max_image_size", 2048)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        assert self.hparams_vision is not None
        vcfg = self.hparams_vision
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.HUNYUANVL)
        self.gguf_writer.add_vision_use_gelu(True)
        self.gguf_writer.add_vision_attention_layernorm_eps(vcfg.get("rms_norm_eps", 1e-5))
        self.gguf_writer.add_vision_spatial_merge_size(vcfg.get("spatial_merge_size", 2))
        self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
        self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("vit."):
            return None

        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # strip CLS token (row 0) from position embeddings so resize_position_embeddings works
        if "position_embedding" in name:
            data_torch = data_torch[1:]  # [n_patches+1, n_embd] -> [n_patches, n_embd]
        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # force conv weights to F32 or F16 to avoid BF16 IM2COL issues on Metal
        # HunyuanVL emit the ViT -> LLM projection as mm.0/mm.2.
        if ("mm.0." in new_name or "mm.2." in new_name) and new_name.endswith(".weight"):
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("HunYuanVLForConditionalGeneration")
@ModelBase.example("tencent/HunyuanOCR")
class HunyuanVLTextModel(HunYuanModel):
    model_arch = gguf.MODEL_ARCH.HUNYUAN_VL

    def __init__(self, dir_model: Path, *args, **kwargs):
        super().__init__(dir_model, *args, **kwargs)
        # transformers 5.13.0 encodes HunyuanVL XD-RoPE as dynamic + mrope_section.
        # Normalize it to avoid the HunYuan dynamic-RoPE context assertion.
        if self.rope_parameters.get("rope_type") == "dynamic" and "mrope_section" in self.rope_parameters:
            self.rope_parameters["rope_type"] = "xdrope"
            self.rope_parameters["type"] = "xdrope"
            self.rope_parameters["xdrope_section"] = list(self.rope_parameters["mrope_section"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # XD-RoPE metadata for the HunyuanVL;
        if self.rope_parameters.get("rope_type") != "xdrope":
            return

        self.gguf_writer.add_rope_freq_base(float(self.rope_parameters["rope_theta"]))
        self.gguf_writer.add_rope_scaling_alpha(float(self.rope_parameters["alpha"]))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_rope_scaling_factor(float(self.rope_parameters.get("factor", 1)))

        ctx_len = int(self.hparams["max_position_embeddings"])
        self.gguf_writer.add_rope_scaling_orig_ctx_len(ctx_len)
        self.gguf_writer.add_context_length(ctx_len)

        self.gguf_writer.add_rope_dimension_sections(list(self.rope_parameters["xdrope_section"]))


@ModelBase.register("HYV3ForCausalLM")
@ModelBase.example("tencent/Hy3")
class HYV3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.HY_V3
    supports_mtp_export = True

    # Trunk layer count, stashed before indexing so the classmethod
    # filter_tensors can identify the appended MTP block(s) (mirrors
    # Step35Model).
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NextN/MTP layers are appended past num_hidden_layers; extend the
        # tensor map so the MTP block's tensors resolve to blk.<n>.* names.
        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0))
        if n_nextn > 0 and not self.no_mtp:
            self.block_count += n_nextn
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(
            self.hparams["moe_intermediate_size"] * self.hparams.get("num_shared_experts", 1)
        )
        self.gguf_writer.add_expert_weights_norm(self.hparams.get("route_norm", True))
        self.gguf_writer.add_expert_weights_scale(float(self.hparams.get("router_scaling_factor", 1.0)))
        # sigmoid router with expert selection bias
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        n_nextn = int(self.hparams.get("num_nextn_predict_layers", 0))
        if n_nextn > 0 and not self.no_mtp:
            self.gguf_writer.add_nextn_predict_layers(n_nextn)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        # HY V3 appends the MTP block(s) past num_hidden_layers.
        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers

        # --no-mtp: drop the appended MTP block(s) entirely.
        if is_mtp and cls.no_mtp:
            return None
        # --mtp: keep ONLY MTP-block tensors plus the shared embeddings/norm/
        # lm_head (so the resulting GGUF carries just the draft head).
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        # The MTP block's trailing final_layernorm (applied after the decoder
        # block, before the shared LM head) maps to nextn.shared_head_norm.
        if is_mtp:
            name = name.replace(".final_layernorm.", ".shared_head.norm.")

        return name, gen

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # merge the per-expert tensors into stacked 3d tensors
        if name.startswith("model.layers.") and ".mlp.experts." in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ("down_proj", "gate_proj", "up_proj"):
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    merged = torch.stack(datas, dim=0)
                    yield from super().modify_tensors(merged, f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("HunyuanImage3ForCausalMM")
@ModelBase.example("tencent/HunyuanImage-3.0")
class HunyuanImage3Model(TextModel):
    """HunyuanImage 3.0: the mixture-of-experts backbone goes to the GGUF file; the image head
    (latent patch embedding, timestep embedders, final layer) and the latent decoder are written
    unchanged to `<output>.image.safetensors` next to it, for the engine's image side.

    Two-axis rotary positions: each head's query/key dimensions are reordered to
    [row pairs, column pairs] and a frequency-factor table restores the original interleaved
    frequencies, so the backbone rotates as a two-section M-RoPE."""
    model_arch = gguf.MODEL_ARCH.HUNYUAN_IMAGE3

    image_side_prefixes = ("patch_embed.", "time_embed.", "time_embed_2.", "timestep_emb.", "final_layer.", "vae.")
    skipped_prefixes = ("vision_model.", "vision_aligner.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rope = dict(self.rope_parameters)
        if rope.get("type", rope.get("rope_type")) == "custom":
            if float(rope.get("alpha", 1.0)) != 1.0 or float(rope.get("factor", 1.0)) != 1.0:
                raise ValueError(f"HunyuanImage 3.0 RoPE scaling {rope} is not supported")
            for k in ("type", "rope_type"):
                self.rope_parameters.pop(k, None)
        if self.hparams.get("rope_type", "2d") != "2d":
            raise ValueError(f"HunyuanImage 3.0 rope_type {self.hparams.get('rope_type')} is not supported")
        # Group-limited routing is off in this family (stored as false).
        for k in ("n_group", "topk_group"):
            if not self.hparams.get(k):
                self.hparams.pop(k, None)
        if isinstance(self.hparams.get("num_experts"), list):
            raise ValueError("per-layer expert counts are not supported")
        self._image_side: dict[str, Tensor] = {}
        self._experts: list[dict[str, Tensor]] | None = None

    def set_vocab(self):
        if (self.dir_model / "tokenizer.json").is_file():
            self._set_vocab_gpt2()
        else:
            self._set_vocab_none()
            self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])

    def _head_dim(self) -> int:
        return int(self.hparams.get("attention_head_dim", self.hparams.get("head_dim")))

    def _pair_order(self) -> list[int]:
        half = self._head_dim() // 2
        return list(range(0, half, 2)) + list(range(1, half, 2))

    def _dim_order(self) -> list[int]:
        half = self._head_dim() // 2
        pairs = self._pair_order()
        return pairs + [p + half for p in pairs]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        hd = self._head_dim()
        if hp.get("head_dim") is None:
            self.gguf_writer.add_key_length(hd)
            self.gguf_writer.add_value_length(hd)

        def same(key):
            v = hp[key]
            if isinstance(v, list):
                assert all(x == v[0] for x in v), f"per-layer {key} is not supported"
                return v[0]
            return v

        moe_ff = same("moe_intermediate_size")
        self.gguf_writer.add_expert_used_count(same("moe_topk"))
        self.gguf_writer.add_expert_feed_forward_length(moe_ff)
        n_shared = same("num_shared_expert")
        self.gguf_writer.add_expert_shared_count(n_shared)
        self.gguf_writer.add_expert_shared_feed_forward_length(moe_ff * n_shared)
        quarter = hd // 4
        self.gguf_writer.add_rope_dimension_sections([quarter, quarter, 0, 0])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        hd = self._head_dim()
        base = float(self.hparams.get("rope_theta", 10000.0))
        pairs = self._pair_order()
        factors = [base ** (2.0 * (orig - new) / hd) for new, orig in enumerate(pairs)]
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS), torch.tensor(factors, dtype=torch.float32))

    def _permute_rows(self, w: Tensor, n_heads: int) -> Tensor:
        hd = self._head_dim()
        order = torch.tensor(self._dim_order())
        w = w.reshape(n_heads, hd, *w.shape[1:])
        return w[:, order].reshape(n_heads * hd, *w.shape[2:])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        if name.startswith(self.image_side_prefixes):
            self._image_side[name] = data_torch
            return
        if name.startswith(self.skipped_prefixes):
            return

        if name == "model.wte.weight":
            yield (self.format_tensor_name(T.TOKEN_EMBD), data_torch)
            return
        if name == "lm_head.weight":
            yield (self.format_tensor_name(T.OUTPUT), data_torch)
            return
        if name == "model.ln_f.weight":
            yield (self.format_tensor_name(T.OUTPUT_NORM), data_torch)
            return

        assert bid is not None, f"unexpected tensor {name}"
        n_head = self.hparams["num_attention_heads"]
        n_kv = self.hparams["num_key_value_heads"]
        hd = self._head_dim()
        groups = n_head // n_kv
        suffix = name.split(f"model.layers.{bid}.", 1)[1]

        if suffix == "self_attn.qkv_proj.weight":
            w = data_torch.reshape(n_kv, groups + 2, hd, -1)
            q = w[:, :groups].reshape(n_head * hd, -1)
            k = w[:, groups].reshape(n_kv * hd, -1)
            v = w[:, groups + 1].reshape(n_kv * hd, -1)
            yield (self.format_tensor_name(T.ATTN_Q, bid), self._permute_rows(q, n_head))
            yield (self.format_tensor_name(T.ATTN_K, bid), self._permute_rows(k, n_kv))
            yield (self.format_tensor_name(T.ATTN_V, bid), v)
            return
        if suffix in ("self_attn.query_layernorm.weight", "self_attn.key_layernorm.weight"):
            key = T.ATTN_Q_NORM if suffix.startswith("self_attn.query") else T.ATTN_K_NORM
            yield (self.format_tensor_name(key, bid), data_torch[torch.tensor(self._dim_order())])
            return
        simple = {
            "self_attn.o_proj.weight": T.ATTN_OUT,
            "input_layernorm.weight": T.ATTN_NORM,
            "post_attention_layernorm.weight": T.FFN_NORM,
            "mlp.gate.wg.weight": T.FFN_GATE_INP,
            "mlp.shared_mlp.down_proj.weight": T.FFN_DOWN_SHEXP,
        }
        if suffix in simple:
            yield (self.format_tensor_name(simple[suffix], bid), data_torch)
            return
        if suffix == "mlp.shared_mlp.gate_and_up_proj.weight":
            # The activation applies to the second half: down(first * act(second)).
            up, gate = data_torch.chunk(2, dim=0)
            yield (self.format_tensor_name(T.FFN_UP_SHEXP, bid), up)
            yield (self.format_tensor_name(T.FFN_GATE_SHEXP, bid), gate)
            return
        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            n_experts = n_experts if isinstance(n_experts, int) else n_experts[bid]
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) < n_experts * 2:
                return
            ups, gates, downs = [], [], []
            for xid in range(n_experts):
                gu = self._experts[bid].pop(f"model.layers.{bid}.mlp.experts.{xid}.gate_and_up_proj.weight")
                up, gate = gu.chunk(2, dim=0)
                ups.append(up)
                gates.append(gate)
                downs.append(self._experts[bid].pop(f"model.layers.{bid}.mlp.experts.{xid}.down_proj.weight"))
            yield (self.format_tensor_name(T.FFN_UP_EXP, bid), torch.stack(ups, dim=0))
            yield (self.format_tensor_name(T.FFN_GATE_EXP, bid), torch.stack(gates, dim=0))
            yield (self.format_tensor_name(T.FFN_DOWN_EXP, bid), torch.stack(downs, dim=0))
            return
        raise ValueError(f"unexpected tensor {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None and any(len(d) for d in self._experts):
            raise ValueError("unprocessed expert tensors")
        if not self._image_side:
            raise ValueError("the checkpoint has no image head tensors")

    def write(self):
        super().write()
        from safetensors.torch import save_file
        hp = self.hparams
        config = {
            "hidden_size": hp["hidden_size"],
            "patch_embed_hidden_dim": hp.get("patch_embed_hidden_dim", 1024),
            "patch_size": hp.get("patch_size", 1),
            "vae": hp["vae"],
        }
        if hp.get("img_proj_type", "unet") != "unet":
            raise ValueError(f"image projection {hp.get('img_proj_type')} is not supported")
        out = Path(str(self.fname_out).removesuffix(".gguf") + ".image.safetensors")
        tensors = {k: LazyTorchTensor.to_eager(v).contiguous() for k, v in sorted(self._image_side.items())}
        save_file(tensors, str(out), metadata={"format": "pt", "config": json.dumps(config)})
        logger.info(f"image head and latent decoder: {len(tensors)} tensors -> {out}")

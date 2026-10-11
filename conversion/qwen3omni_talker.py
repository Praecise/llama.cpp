from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, gguf
from .qwen import Qwen2MoeModel

# The speech side of Qwen3-Omni, converted with --talker.
#
# The talker reads the thinker's turn and speaks it:
#   backbone   --> libllama text model (MoE with a shared expert), arch qwen3omnitalker
#   code_predictor, code2wav, hidden_projection --> gen-audio mmproj (qwen3omni_gen)
#
# Vocabulary of the backbone: [thinker vocab | codec vocab]. The thinker's token
# embedding passed through text_projection is folded into the first block of rows,
# so a thinker token id selects the talker's conditioning row directly; the codec
# embedding fills the second block. The codec head is smaller than the vocabulary,
# the graph pads its logits with -inf over the thinker rows.
#
# hidden_projection maps the thinker's inner hidden state (layer accept_hidden_layer)
# of a picture, audio or video position to the talker's width; it runs as the
# mmproj's conditioning process.

_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}

# metadata the speech pipeline reads from the backbone
_THINKER_TOKEN_KEYS = (
    "tts_bos_token_id",
    "tts_eos_token_id",
    "tts_pad_token_id",
    "im_start_token_id",
    "im_end_token_id",
    "assistant_token_id",
    "user_token_id",
    "system_token_id",
)


def _codec_token_names(talker_config: dict[str, Any], n_codec: int) -> list[str]:
    names = [f"<|codec_{i}|>" for i in range(n_codec)]
    for key, val in talker_config.items():
        if key.startswith("codec_") and key.endswith("_id") and isinstance(val, int):
            names[val] = f"<|{key[:-len('_id')]}|>"
    for speaker, val in (talker_config.get("speaker_id") or {}).items():
        names[val] = f"<|speaker_{speaker.lower()}|>"
    return names


def _rope_theta(config: dict[str, Any]) -> float:
    for source in (config, config.get("rope_parameters") or {}, config.get("rope_scaling") or {}):
        if "rope_theta" in source:
            return float(source["rope_theta"])
    raise KeyError("rope_theta")


class Qwen3OmniTalkerModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3OMNITALKER

    _TEXT_PROJ_KEYS = (
        "thinker_embed_tokens",
        "text_projection.linear_fc1.weight",
        "text_projection.linear_fc1.bias",
        "text_projection.linear_fc2.weight",
        "text_projection.linear_fc2.bias",
    )

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        self._root_config = {k: v for k, v in hparams.items() if not isinstance(v, dict)}
        self._talker_config = dict(hparams["talker_config"])
        talker_text = dict(self._talker_config["text_config"])
        self.n_codec_vocab = talker_text["vocab_size"]
        self.n_thinker_vocab = hparams["thinker_config"]["text_config"]["vocab_size"]
        talker_text["vocab_size"] = self.n_thinker_vocab
        hparams["text_config"] = talker_text
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._text_proj_buffer: dict[str, Tensor] = {}
        self._folded_text_embed: Tensor | None = None
        self._codec_embed: Tensor | None = None

    def get_vocab_base_pre(self, tokenizer) -> str:
        # the thinker's byte-level BPE; the talker never tokenizes text itself, it takes token ids
        return "qwen2"

    def _newline_token_id(self) -> int:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(self.dir_model).encode("\n", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("the newline must be a single token")
        return int(ids[0])

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        assert len(tokens) == self.n_thinker_vocab
        codec_tokens = _codec_token_names(self._talker_config, self.n_codec_vocab)
        tokens += codec_tokens
        toktypes += [gguf.TokenType.CONTROL] * len(codec_tokens)
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        if not special_vocab.merges:
            # The loader needs a merge list for a byte-level BPE vocabulary. The talker
            # selects rows by token id and never tokenizes text, so one merge of the
            # first two single-byte tokens stands in for a tokenizer published without any.
            singles = [t for t in tokens[:256] if len(t) == 1][:2]
            if len(singles) < 2:
                raise ValueError("the tokenizer has no merges and no single-byte tokens")
            self.gguf_writer.add_token_merges([f"{singles[0]} {singles[1]}"])

        # the talker is not a chat model
        self.gguf_writer.add_chat_template(None)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_weights_norm(bool(self.hparams.get("norm_topk_prob", False)))

        n_vocab = self.n_thinker_vocab + self.n_codec_vocab
        codec_eos = self.n_thinker_vocab + self._talker_config["codec_eos_token_id"]
        # the last 1024 codec rows are control codes; only end-of-speech may be sampled among them
        self.gguf_writer.add_suppress_tokens([i for i in range(n_vocab - 1024, n_vocab) if i != codec_eos])
        self.gguf_writer.add_eos_token_id(codec_eos)
        self.gguf_writer.add_add_eos_token(False)

        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        for key in _THINKER_TOKEN_KEYS:
            if key in self._root_config:
                self.gguf_writer.add_uint32(f"{arch}.{key}", int(self._root_config[key]))
        self.gguf_writer.add_uint32(f"{arch}.codec_offset", self.n_thinker_vocab)
        self.gguf_writer.add_uint32(f"{arch}.newline_token_id", self._newline_token_id())
        self.gguf_writer.add_uint32(f"{arch}.code_groups", int(self._talker_config["num_code_groups"]))
        self.gguf_writer.add_uint32(f"{arch}.accept_hidden_layer", int(self._talker_config["accept_hidden_layer"]))
        self.gguf_writer.add_uint32(f"{arch}.thinker_embedding_length", int(self._talker_config["thinker_hidden_size"]))
        for key in ("audio_token_id", "image_token_id", "video_token_id"):
            self.gguf_writer.add_uint32(f"{arch}.{key}", int(self._talker_config[key]))

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name == "thinker.model.embed_tokens.weight":
            return "thinker_embed_tokens", gen
        if not name.startswith("talker.") or name.startswith(("talker.code_predictor.", "talker.hidden_projection.")):
            return None
        return super().filter_tensors((name[len("talker."):], gen))

    def _emit_token_embd(self) -> Iterable[tuple[str, Tensor]]:
        if self._folded_text_embed is None or self._codec_embed is None:
            return
        combined = torch.cat([self._folded_text_embed, self._codec_embed], dim=0)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), combined)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.codec_embedding.weight":
            self._codec_embed = data_torch
            yield from self._emit_token_embd()
            return

        if name == "codec_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return

        if name in self._TEXT_PROJ_KEYS:
            self._text_proj_buffer[name] = data_torch
            if len(self._text_proj_buffer) < len(self._TEXT_PROJ_KEYS):
                return
            b = self._text_proj_buffer
            act = _ACT2FN[self.hparams["hidden_act"]]
            hidden = act(F.linear(b["thinker_embed_tokens"].float(),
                                  b["text_projection.linear_fc1.weight"].float(),
                                  b["text_projection.linear_fc1.bias"].float()))
            self._folded_text_embed = F.linear(hidden,
                                               b["text_projection.linear_fc2.weight"].float(),
                                               b["text_projection.linear_fc2.bias"].float())
            self._text_proj_buffer = {}
            yield from self._emit_token_embd()
            return

        yield from super().modify_tensors(data_torch, name, bid)


class Qwen3OmniTalkerGenModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = False

    _CODE_LAYER_TENSOR_MAP = {
        "input_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "self_attn.q_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q,
        "self_attn.q_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q_NORM,
        "self_attn.k_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K,
        "self_attn.k_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K_NORM,
        "self_attn.v_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_V,
        "self_attn.o_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "post_attention_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "mlp.gate_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "mlp.up_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "mlp.down_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }

    _WAV_LAYER_TENSOR_MAP = {
        "input_layernorm.weight":          gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_NORM,
        "self_attn.q_proj.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_Q,
        "self_attn.k_proj.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_K,
        "self_attn.v_proj.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_V,
        "self_attn.o_proj.weight":         gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_OUT,
        "self_attn_layer_scale.scale":     gguf.MODEL_TENSOR.A_GEN_WAV_TFM_ATTN_SCALE,
        "post_attention_layernorm.weight": gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_NORM,
        "mlp.gate_proj.weight":            gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_GATE,
        "mlp.up_proj.weight":              gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_UP,
        "mlp.down_proj.weight":            gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_DOWN,
        "mlp_layer_scale.scale":           gguf.MODEL_TENSOR.A_GEN_WAV_TFM_FFN_SCALE,
    }

    _UP_TENSOR_MAP = {
        "0.conv.weight":        (gguf.MODEL_TENSOR.A_GEN_WAV_UP_CONV, ".weight"),
        "0.conv.bias":          (gguf.MODEL_TENSOR.A_GEN_WAV_UP_CONV, ".bias"),
        "1.dwconv.conv.weight": (gguf.MODEL_TENSOR.A_GEN_WAV_UP_DWCONV, ".weight"),
        "1.dwconv.conv.bias":   (gguf.MODEL_TENSOR.A_GEN_WAV_UP_DWCONV, ".bias"),
        "1.norm.weight":        (gguf.MODEL_TENSOR.A_GEN_WAV_UP_NORM, ".weight"),
        "1.norm.bias":          (gguf.MODEL_TENSOR.A_GEN_WAV_UP_NORM, ".bias"),
        "1.pwconv1.weight":     (gguf.MODEL_TENSOR.A_GEN_WAV_UP_PW1, ".weight"),
        "1.pwconv1.bias":       (gguf.MODEL_TENSOR.A_GEN_WAV_UP_PW1, ".bias"),
        "1.pwconv2.weight":     (gguf.MODEL_TENSOR.A_GEN_WAV_UP_PW2, ".weight"),
        "1.pwconv2.bias":       (gguf.MODEL_TENSOR.A_GEN_WAV_UP_PW2, ".bias"),
        "1.gamma":              (gguf.MODEL_TENSOR.A_GEN_WAV_UP_GAMMA, ""),
    }

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        talker = hparams["talker_config"]
        hparams["text_config"] = {"hidden_size": talker["text_config"]["hidden_size"]}
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._talker_config = talker
        self._wav_config = self.global_config["code2wav_config"]
        self.n_code_groups = int(talker["num_code_groups"])
        self._code_embed_buffer: dict[int, Tensor] = {}
        self._code_head_buffer: dict[int, Tensor] = {}

    def get_audio_config(self) -> dict[str, Any] | None:
        # no encoder; the code predictor's config stands in for the block count
        return self.global_config["talker_config"]["code_predictor_config"]

    def set_gguf_parameters(self):
        cp = self._talker_config["code_predictor_config"]
        wav = self._wav_config
        w = self.gguf_writer
        w.add_file_type(self.ftype)
        w.add_clip_has_gen_audio_encoder(True)
        w.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.QWEN3OMNI_GEN)
        w.add_gen_audio_projection_dim(self.n_embd_text)
        w.add_gen_audio_embedding_length(cp["hidden_size"])
        w.add_gen_audio_feed_forward_length(cp["intermediate_size"])
        w.add_gen_audio_block_count(cp["num_hidden_layers"])
        w.add_gen_audio_head_count(cp["num_attention_heads"])
        w.add_gen_audio_head_count_kv(cp["num_key_value_heads"])
        w.add_gen_audio_attention_layernorm_eps(cp["rms_norm_eps"])
        w.add_gen_audio_rope_theta(_rope_theta(cp))
        w.add_gen_audio_wav_block_count(wav["num_hidden_layers"])
        w.add_gen_audio_wav_head_count(wav["num_attention_heads"])
        w.add_gen_audio_wav_head_count_kv(wav["num_key_value_heads"])
        w.add_gen_audio_wav_layernorm_eps(wav["rms_norm_eps"])
        w.add_gen_audio_wav_rope_theta(_rope_theta(wav))
        w.add_gen_audio_wav_sliding_window(wav["sliding_window"])
        w.add_gen_audio_wav_upsample_block_count(len(wav["upsampling_ratios"]))
        w.add_gen_audio_wav_decoder_block_count(len(wav["upsample_rates"]))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # conv1d/conv1d_dw kernels must be F16, ggml_conv_1d(_dw) has no BF16 path
        if new_name.endswith(".weight") and (
            new_name in ("a.gen.wav.dac.entry.weight", "a.gen.wav.dac.post_conv.weight")
            or (".up.blk." in new_name and new_name.endswith(".dwconv.weight"))
            or (".dac.blk." in new_name and (new_name.endswith(".conv1.weight") or new_name.endswith(".conv2.weight")))
        ):
            return gguf.GGMLQuantizationType.F16
        # ConvTranspose1d kernels: only F16/F32 are implemented, no BF16
        if new_name.endswith(".conv.weight") and (".up.blk." in new_name or ".dac.blk." in new_name):
            return gguf.GGMLQuantizationType.F32
        # tables read with get_rows and the small conditioning MLP stay exact
        if new_name.startswith(("a.gen.wav.code_embd", "a.gen.cond.")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not (
            name.startswith(("talker.code_predictor.", "talker.hidden_projection.", "code2wav."))
            or name == "talker.model.codec_embedding.weight"
        ):
            return None
        return super().filter_tensors((name, gen))

    @staticmethod
    def _snake(name: str, data: Tensor) -> Tensor:
        # SnakeBeta keeps alpha and beta in log scale; fold exp() and the reciprocal here
        if name.endswith(".alpha"):
            return torch.exp(data.float())
        return 1.0 / (torch.exp(data.float()) + 1e-9)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        n_rest = self.n_code_groups - 1

        if name == "talker.model.codec_embedding.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUT_EMBD), data_torch)
            return

        if name.startswith("talker.hidden_projection.linear_fc"):
            which, suffix = name[len("talker.hidden_projection.linear_fc"):].split(".", 1)
            tensor = T.A_GEN_COND_FC1 if which == "1" else T.A_GEN_COND_FC2
            yield (self.format_tensor_name(tensor, suffix="." + suffix), data_torch)
            return

        if name == "talker.code_predictor.model.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        if name.startswith("talker.code_predictor.model.codec_embedding."):
            idx = int(name.split("codec_embedding.")[1].split(".")[0])
            self._code_embed_buffer[idx] = data_torch
            if len(self._code_embed_buffer) < n_rest:
                return
            stacked = torch.stack([self._code_embed_buffer.pop(i) for i in range(n_rest)], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), stacked)
            return

        if name.startswith("talker.code_predictor.lm_head."):
            idx = int(name.split("lm_head.")[1].split(".")[0])
            self._code_head_buffer[idx] = data_torch
            if len(self._code_head_buffer) < n_rest:
                return
            stacked = torch.stack([self._code_head_buffer.pop(i) for i in range(n_rest)], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), stacked)
            return

        if name.startswith("talker.code_predictor.model.layers."):
            layer, key_with_suffix = name[len("talker.code_predictor.model.layers."):].split(".", 1)
            tensor = self._CODE_LAYER_TENSOR_MAP.get(key_with_suffix.rsplit(".", 1)[0])
            if tensor is None:
                raise ValueError(f"unexpected code predictor tensor {name}")
            yield (self.format_tensor_name(tensor, int(layer)), data_torch)
            return

        if name == "code2wav.code_embedding.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_CODE_EMBD), data_torch)
            return

        if name == "code2wav.pre_transformer.norm.weight":
            yield (self.format_tensor_name(T.A_GEN_WAV_TFM_OUTPUT_NORM), data_torch)
            return

        if name.startswith("code2wav.pre_transformer.layers."):
            layer, key = name[len("code2wav.pre_transformer.layers."):].split(".", 1)
            tensor = self._WAV_LAYER_TENSOR_MAP.get(key)
            if tensor is None:
                raise ValueError(f"unexpected code2wav tensor {name}")
            yield (self.format_tensor_name(tensor, int(layer)), data_torch)
            return

        if name.startswith("code2wav.upsample."):
            blk, key = name[len("code2wav.upsample."):].split(".", 1)
            tensor, suffix = self._UP_TENSOR_MAP[key]
            yield (self.format_tensor_name(tensor, int(blk), suffix=suffix), data_torch)
            return

        if name.startswith("code2wav.decoder."):
            yield from self._decoder_tensor(name[len("code2wav.decoder."):], data_torch)
            return

        raise ValueError(f"unexpected tensor {name}")

    def _decoder_tensor(self, rest: str, data: Tensor) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        n_blocks = len(self._wav_config["upsample_rates"])
        idx_s, key = rest.split(".", 1)
        idx = int(idx_s)

        # decoder.0: entry conv; decoder.1..n: upsampling blocks; then the final SnakeBeta and conv
        if idx == 0:
            suffix = "." + key.rsplit(".", 1)[1]
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_ENTRY, suffix=suffix), data)
            return
        if idx == n_blocks + 1:
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_SNAKE, suffix="." + key), self._snake(key, data))
            return
        if idx == n_blocks + 2:
            suffix = "." + key.rsplit(".", 1)[1]
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_POST_CONV, suffix=suffix), data)
            return

        bid = idx - 1
        m = re.fullmatch(r"block\.(\d+)\.(.+)", key)
        if m is None:
            raise ValueError(f"unexpected decoder tensor {rest}")
        sub, tail = int(m.group(1)), m.group(2)
        if sub == 0:
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_SNAKE, bid, suffix="." + tail), self._snake(tail, data))
            return
        if sub == 1:
            suffix = "." + tail.rsplit(".", 1)[1]
            yield (self.format_tensor_name(T.A_GEN_WAV_DAC_UP_CONV, bid, suffix=suffix), data)
            return

        xid = sub - 2  # block.2/3/4 are the residual units
        part, leaf = tail.split(".", 1)
        names = {
            "act1": T.A_GEN_WAV_DAC_RES_ACT1,
            "act2": T.A_GEN_WAV_DAC_RES_ACT2,
            "conv1": T.A_GEN_WAV_DAC_RES_CONV1,
            "conv2": T.A_GEN_WAV_DAC_RES_CONV2,
        }
        base = gguf.TENSOR_NAMES[names[part]].format(bid=bid, xid=xid)
        if part.startswith("act"):
            yield (base + "." + leaf, self._snake(leaf, data))
        else:
            yield (base + "." + leaf.rsplit(".", 1)[1], data)


def talker_model_class(arch: str, mmproj: bool) -> type[ModelBase]:
    if arch != "Qwen3OmniMoeForConditionalGeneration":
        raise NotImplementedError(f"{arch} has no speech output part")
    return Qwen3OmniTalkerGenModel if mmproj else Qwen3OmniTalkerModel

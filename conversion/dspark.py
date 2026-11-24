from __future__ import annotations

import json

from typing import Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, ModelType, TextModel, get_model_architecture, gguf, logger


class DFlashModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH

    def set_vocab(self):
        if self.target_model_dir is None:
            raise ValueError(
                "DFlash draft model requires --target-model-dir to be specified. "
                "Please provide the path to the target model directory containing the tokenizer."
            )
        logger.info(f"DFlash: Using tokenizer from target model: {self.target_model_dir}")
        original_dir = self.dir_model
        self.dir_model = self.target_model_dir

        # Reuse the target model's own vocab handler
        from . import get_model_class
        with open(self.target_model_dir / "config.json", "r", encoding="utf-8") as f:
            target_hparams = json.load(f)
        target_arch = get_model_architecture(target_hparams, ModelType.TEXT)
        target_cls = get_model_class(target_arch)

        if target_cls is not type(self):
            target_cls.set_vocab(self)  # ty: ignore[missing-attribute]
        else:
            super().set_vocab()

        self.dir_model = original_dir

        mask_token_id = self.hparams.get("dflash_config", {}).get("mask_token_id")
        if mask_token_id is not None:
            self.gguf_writer.add_mask_token_id(mask_token_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        dflash_config = self.hparams.get("dflash_config", {})
        if (partial_rotary_factor := self.rope_parameters.get("partial_rotary_factor")) is not None:
            head_dim = self.hparams.get("head_dim") or self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            self.gguf_writer.add_rope_dimension_count(int(head_dim * partial_rotary_factor))
        if (value_scale := dflash_config.get("attention_value_scale")) is not None:
            self.gguf_writer.add_attn_value_scale(float(value_scale))

        block_size = dflash_config.get("block_size", self.hparams.get("block_size", 16))
        self.gguf_writer.add_block_size(block_size)

        output_multiplier = dflash_config.get(
            "output_multiplier", self.hparams.get("output_multiplier")
        )
        if output_multiplier is not None:
            self.gguf_writer.add_logit_scale(float(output_multiplier))
        softcap = dflash_config.get(
            "final_logit_softcapping", self.hparams.get("final_logit_softcapping")
        )
        if softcap is not None and float(softcap) > 0:
            self.gguf_writer.add_final_logit_softcapping(float(softcap))
        embedding_scale = dflash_config.get(
            "input_embedding_scale", self.hparams.get("input_embedding_scale")
        )
        if embedding_scale is not None:
            self.gguf_writer.add_embedding_scale(float(embedding_scale))

        target_layer_ids = dflash_config.get("target_layer_ids", self.hparams.get("target_layer_ids", []))
        if target_layer_ids:
            extract_layer_ids = [i + 1 for i in target_layer_ids]
            self.gguf_writer.add_target_layers(extract_layer_ids)

        use_sliding_window = self.hparams.get("use_sliding_window", False) or dflash_config.get("use_swa", False)
        sliding_window = dflash_config.get("swa_window_size") or self.hparams.get("sliding_window")
        layer_types = self.hparams.get("layer_types")
        if use_sliding_window and sliding_window:
            is_swa = ([True] * self.block_count if dflash_config.get("use_swa", False)
                      else [lt == "sliding_attention" for lt in layer_types or []])
            self.gguf_writer.add_sliding_window(sliding_window)
            self.gguf_writer.add_sliding_window_pattern(is_swa)

        causal = self.hparams.get("is_causal")
        if causal is None:
            causal = dflash_config.get("causal")
        if causal is not None:
            self.gguf_writer.add_causal_attention(bool(causal))

        # M-RoPE target: the draft ropes on the temporal dim only, so write
        # degenerate sections [n_rot/2, 0, 0, 0]
        if self._target_uses_mrope():
            head_dim = self.hparams.get("head_dim") or self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
            self.gguf_writer.add_rope_dimension_sections([head_dim // 2, 0, 0, 0])

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        yield from super().generate_extra_tensors()

        mask_path = self.dir_model / "mask_embedding.pt"
        if not mask_path.is_file():
            return

        mask = torch.load(mask_path, map_location="cpu", weights_only=True)
        mask_id = self.hparams.get("dflash_config", {}).get("mask_token_id")
        if mask_id is None or mask["mask_token_id"] != mask_id:
            raise ValueError("mask_embedding.pt mask_token_id does not match dflash_config")
        if tuple(mask["embedding"].shape) != (self.hparams["hidden_size"],):
            raise ValueError("mask_embedding.pt has an unexpected embedding shape")
        if not 0 <= mask_id < self.hparams["vocab_size"]:
            raise ValueError("mask_embedding.pt mask_token_id is outside the vocabulary")

        def target_tensor(name: str) -> Tensor:
            if self.target_model_dir is None:
                raise ValueError("mask_embedding.pt requires --target-model-dir with the target embeddings and output head")
            index_path = self.target_model_dir / "model.safetensors.index.json"
            if index_path.is_file():
                with open(index_path, encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                part_names = [weight_map[name]]
            else:
                part_names = self.get_model_part_names(self.target_model_dir, "model", ".safetensors")

            for part_name in part_names:
                with gguf.utility.SafetensorsLocal(self.target_model_dir / part_name) as part:
                    if name in part:
                        return LazyTorchTensor.from_local_tensor(part[name])
            raise ValueError(f"Target tensor {name!r} was not found in safetensors")

        embedding_name = "model.embed_tokens.weight"
        if embedding_name in self.model_tensors:
            embeddings = self.model_tensors.pop(embedding_name)()
        else:
            embeddings = target_tensor(embedding_name)

        if "model.lm_head.weight" not in self.model_tensors:
            if self.target_model_dir is None:
                raise ValueError("mask_embedding.pt requires --target-model-dir to obtain the output head")
            target_config = ModelBase.load_hparams(self.target_model_dir)
            target_config = {**target_config, **target_config.get("text_config", {})}
            head_name = embedding_name if target_config.get("tie_word_embeddings", False) else "lm_head.weight"
            # Keep the output head separate from the patched input embedding table.
            yield "model.lm_head.weight", target_tensor(head_name)

        embeddings = LazyTorchTensor.to_eager(embeddings).clone()
        if tuple(embeddings.shape) != (self.hparams["vocab_size"], self.hparams["hidden_size"]):
            raise ValueError("Target token embedding shape does not match the DFlash draft")
        # the target's mask row can be untrained, the draft provides its own vector
        embeddings[mask_id] = mask["embedding"].to(embeddings.dtype)
        self.hparams["has_embed_tokens"] = True
        yield embedding_name, embeddings

    def _target_uses_mrope(self) -> bool:
        if self.target_model_dir is None:
            return False
        with open(self.target_model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg)
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        return "mrope_section" in rope

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("model."):
            name = "model." + name
        if "sink" in name and not name.endswith(".weight"):
            name += ".weight"
        return super().filter_tensors((name, gen))

    _ROPE_PERMUTE_SUFFIXES = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.embed_tokens.weight" and not self.hparams.get("has_embed_tokens", True):
            return

        # interleaved-rope checkpoints (rope_is_neox_style = false) -> NeoX layout: per head, even dims first then odd
        if not self.hparams.get("rope_is_neox_style", True) and name.endswith(self._ROPE_PERMUTE_SUFFIXES):
            head_dim = self.hparams["head_dim"]
            shape = data_torch.shape
            data_torch = data_torch.reshape(-1, head_dim // 2, 2, *shape[1:]).transpose(1, 2).reshape(shape)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register(
    "Qwen3DSparkModel",
    "DSparkDraftModel",
    "DSparkSpeculator",
)
@ModelBase.example("satgeze/Qwen3.6-27B-DSpark")
class DSparkModel(DFlashModel):
    # DSpark = DFlash + a semi-autoregressive Markov head.
    model_arch = gguf.MODEL_ARCH.DFLASH

    def __init__(self, dir_model, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model)

        # EAGLE3-style exports use the 1+N bonus-anchor block, DFlash-lineage exports sample from the anchor
        self._sample_from_anchor = hparams.get(
            "sample_from_anchor",
            "transformer_layer_config" not in hparams and "aux_hidden_state_layer_ids" not in hparams)
        if "transformer_layer_config" in hparams:
            hparams = {**hparams, **hparams["transformer_layer_config"]}

        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

        # normalize both schemas to DFlash's nested dflash_config
        if "aux_hidden_state_layer_ids" in self.hparams:
            self.hparams.setdefault("dflash_config", {
                "mask_token_id": self.hparams.get("mask_token_id"),
                "target_layer_ids": [i - 1 for i in self.hparams["aux_hidden_state_layer_ids"]],
            })
        else:
            self.hparams.setdefault("dflash_config", {
                k: self.hparams[k] for k in ("target_layer_ids", "mask_token_id") if k in self.hparams
            })

        if (markov_head_type := self.hparams.get("markov_head_type", "vanilla")) != "vanilla":
            raise ValueError(f"unsupported markov_head_type {markov_head_type!r} (only 'vanilla' is supported)")

        n_vocab = self.hparams["vocab_size"]
        self._n_vocab_draft = self.hparams.get("draft_vocab_size") or n_vocab
        if self._n_vocab_draft > n_vocab:
            raise ValueError(f"draft_vocab_size {self._n_vocab_draft} exceeds vocab_size {n_vocab}")
        self._d2t: Tensor | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_sample_from_anchor(self._sample_from_anchor)

        # confidence head is optional: vanilla-markov exports ship without it
        has_conf = any("confidence_head.proj" in name for name in self.model_tensors)
        self.gguf_writer.add_has_confidence_head(has_conf)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0] == "t2d":  # not used at runtime
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.d2t":
            self._d2t = data_torch
            return

        if self._n_vocab_draft == self.hparams["vocab_size"] and name.endswith("lm_head.weight"):
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        n_vocab = self.hparams["vocab_size"]
        if self._n_vocab_draft < n_vocab and self._d2t is None:
            raise ValueError(f"draft_vocab_size {self._n_vocab_draft} < vocab_size {n_vocab} but no d2t table found")

        # write d2t as absolute target token ids
        if self._d2t is not None:
            data = LazyTorchTensor.to_eager(self._d2t).to(torch.int64).cpu().numpy().reshape(-1)
            if data.size != self._n_vocab_draft:
                raise ValueError(f"d2t size {data.size} does not match draft_vocab_size {self._n_vocab_draft}")
            data = data + np.arange(data.size, dtype=np.int64)
            if np.any((data < 0) | (data >= n_vocab)):
                raise ValueError(f"d2t target ids out of range for target vocab size {n_vocab}")
            if np.unique(data).size != data.size:
                raise ValueError("d2t contains duplicate target ids")
            logger.info(f"{'d2t,':<30} --> I64, shape = {{{data.size}}}")
            self.gguf_writer.add_tensor("d2t", data, raw_dtype=gguf.GGMLQuantizationType.I64)

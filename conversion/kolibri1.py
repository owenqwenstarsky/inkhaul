from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1")
class Kolibri1Model(TextModel):
    """Kolibri-1: sliding-window and NoPE full-attention layers, sigmoid-routed MoE with a shared expert."""

    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # without a dense prefix, the FFN length is the length of an expert
        self.hparams["intermediate_size"] = self.hparams["moe_intermediate_size"]
        self.n_dense_lead = 0

        # A dense prefix replaces the MoE of the leading layers by dense FFNs
        if dense_layers := self.hparams.get("mlp_only_layers"):
            if dense_layers != list(range(len(dense_layers))):
                raise ValueError(f"Dense FFN layers must be the leading layers, not {dense_layers}")
            self.n_dense_lead = len(dense_layers)
            self.hparams["intermediate_size"] = self.hparams["mlp_only_intermediate_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        if (rope_dim := hp.get("head_dim")) is None:
            rope_dim = hp["hidden_size"] // hp["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["shared_expert_intermediate_size"])
        if self.n_dense_lead > 0:
            self.gguf_writer.add_leading_dense_block_count(self.n_dense_lead)
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        self.gguf_writer.add_sliding_window(hp["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in hp["layer_types"]])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # the selection bias of the router
        if name.endswith(".expert_bias"):
            name += ".bias"

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # stack the experts of a layer into one 3D tensor per projection
        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ("gate_proj", "up_proj", "down_proj"):
                    datas = [self._experts[bid].pop(f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight") for xid in range(n_experts)]
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(torch.stack(datas, dim=0), merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("Kolibri1_1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1.1")
class Kolibri11Model(Kolibri1Model):
    """Kolibri-1.1: Kolibri-1 with sparse attention in all full-attention layers."""

    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if bid is not None and self.hparams["kv_source_layers"][bid] != bid:
            if name.endswith((".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.k_norm.weight")):
                # Checkpoints retain K/V projections that shared-KV layers do not use.
                logger.info("Skipping unused shared-KV tensor %s", name)
                return
        yield from super().modify_tensors(data_torch, name, bid)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        # The sparse policy llama.cpp implements: NVFP4 K/V and routing keys in the sparse layers,
        # 8 sink keys, 128 recent keys and routing over the query heads of the first KV head.
        supported = {
            "kv_cache_qat": True,
            "index_num_sink_tokens": 8,
            "index_num_local_tokens": 128,
            "index_num_heads": hp["num_attention_heads"] // hp["num_key_value_heads"],
        }
        for key, value in supported.items():
            if hp.get(key) != value:
                raise ValueError(f"Unsupported sparse attention: {key} is {hp.get(key)!r}, expected {value!r}")

        n_layers = hp["num_hidden_layers"]
        swa_layers = {i for i, t in enumerate(hp["layer_types"]) if t == "sliding_attention"}
        kv_sources = hp["kv_source_layers"]
        indexers = set(hp["index_layers"])
        if len(kv_sources) != n_layers or any(kv_sources[i] != i for i in swa_layers) or indexers & swa_layers:
            raise ValueError("Sliding-window layers must neither share K/V nor be indexers")
        if hp["index_pool_size"] % hp["index_compression"] != 0:
            raise ValueError("The pool must hold whole blocks")

        self.gguf_writer.add_indexer_types([i in indexers for i in range(n_layers)])
        self.gguf_writer.add_kv_source_map(kv_sources)
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_candidate_block_size(hp["index_compression"])
        self.gguf_writer.add_indexer_candidate_top_k_blocks(hp["index_pool_size"] // hp["index_compression"])

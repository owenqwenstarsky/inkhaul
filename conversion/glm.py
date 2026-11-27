from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Glm5NextForConditionalGeneration")
@ModelBase.example("zai-org/GLM-5.3-Flash")
class Glm5NextModel(TextModel):
    """GLM-5.3-Flash text model: KDA linear attention, nope MLA with a k-pool DSA indexer, mHC and DeepSeek MoE. The vision tower is skipped."""

    model_arch = gguf.MODEL_ARCH.GLM5_NEXT
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nextn_layers = self.hparams.get("num_nextn_predict_layers", 0) or 0
        if self.no_mtp:
            nextn_layers = 0
        self.nextn_layers = nextn_layers
        self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        # head_dim is 0 and the KV heads are per layer: both are written below, not by TextModel
        self.hparams.pop("head_dim", None)
        self.hparams.pop("num_key_value_heads", None)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        # ModelBase.__init__ calls this before TextModel hoists text_config
        type(self)._main_layers = self.hparams.get("text_config", self.hparams)["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def set_vocab(self):
        # the released tokenizer_config names TokenizersBackend, so this needs transformers >= 5
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        assert tokenizer is not None
        added_vocab = tokenizer.get_added_vocab()

        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # same special tokens as GLM-4.5: <|endoftext|> as eot never stops generation
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab._set_special_token("bos", added_vocab["[gMASK]"])
        special_vocab._set_special_token("eot", added_vocab["<|user|>"])
        special_vocab._set_special_token("unk", added_vocab["<|endoftext|>"])
        special_vocab._set_special_token("eom", added_vocab["<|observation|>"])
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        lin = hp["linear_attn_config"]
        if not hp.get("mla_use_nope", False) or hp["qk_rope_head_dim"] != 0:
            raise ValueError("GLM-5.3 requires nope-only MLA (mla_use_nope, qk_rope_head_dim 0)")
        if not hp.get("mhc", True) or hp["hc_mult"] != 4:
            raise ValueError("GLM-5.3 requires mHC with hc_mult 4")
        if not hp.get("index_kpool_compress", True):
            raise ValueError("GLM-5.3 requires the compressed k-pool indexer")
        if lin["num_heads"] != hp["num_attention_heads"]:
            raise ValueError("GLM-5.3 requires as many KDA heads as attention heads")

        super().set_gguf_parameters()

        layer_types = hp["layer_types"]
        if len(layer_types) != hp["num_hidden_layers"]:
            raise ValueError(f"layer_types has {len(layer_types)} entries, expected {hp['num_hidden_layers']}")
        # 0 KV heads marks a KDA layer, the MTP layer is MLA
        n_head_kv = [0 if t == "linear_attention" else 1 for t in layer_types]
        self.gguf_writer.add_head_count_kv(n_head_kv + [1] * self.nextn_layers)
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        # the indexer key LayerNorm
        self.gguf_writer.add_layer_norm_eps(1e-6)

        # KDA
        self.gguf_writer.add_ssm_conv_kernel(lin["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(lin["head_dim"])
        if (lower_bound := lin.get("gate_lower_bound")) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(lower_bound)

        # MLA, the cache holds the compressed latent
        kv_lora_rank = hp["kv_lora_rank"]
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(0)
        self.gguf_writer.add_key_length(kv_lora_rank)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"])
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])

        # DSA indexer with k-pool compression
        self.gguf_writer.add_indexer_head_count(hp["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["index_topk"])
        self.gguf_writer.add_indexer_kpool(hp["index_kpool"])
        self.gguf_writer.add_indexer_kpool_select_tail(hp.get("index_kpool_always_select_tail", True))
        if (indexer_types := hp.get("indexer_types")) is not None:
            self.gguf_writer.add_indexer_types([t == "full" for t in indexer_types])

        # mHC
        self.gguf_writer.add_hyper_connection_count(hp["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hp["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hp["hc_eps"])

        # MoE
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["moe_intermediate_size"] * hp["n_shared_experts"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["norm_topk_prob"])
        if (limit := hp.get("swiglu_limit")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp([limit] * self.block_count)
            self.gguf_writer.add_swiglu_clamp_shexp([limit] * self.block_count)

        if self.nextn_layers > 0:
            self.gguf_writer.add_nextn_predict_layers(self.nextn_layers)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        if cls._main_layers is None:
            return titem

        m = re.match(r"model\.layers\.(\d+)\.", name)
        is_mtp = m is not None and int(m.group(1)) >= cls._main_layers

        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
        ):
            return None

        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if ".mlp.experts." in name:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for weight_name in ("down_proj", "gate_proj", "up_proj"):
                    tensors = []
                    for expert_id in range(n_experts):
                        expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                        tensors.append(self._experts[bid].pop(expert_name))
                    merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                    yield from super().modify_tensors(torch.stack(tensors, dim=0), merged_name, bid)
            return

        if name.endswith(".self_attn.kv_b_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            name_k = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid)
            name_v = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid)
            yield from super().modify_tensors(k_b.transpose(1, 2), name_k, bid)
            yield from super().modify_tensors(v_b, name_v, bid)
            return

        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim not in (2, 3):
                raise ValueError(f"unexpected conv1d rank {data_torch.ndim} for {name}")
            data_torch = data_torch.reshape(1, data_torch.shape[0], 1, data_torch.shape[-1])

        # the graph uses -exp(A_log) as the per-head decay
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float().flatten()[:self.hparams["num_attention_heads"]])

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # mHC and k-pool parameters have no .weight suffix in the checkpoint
        if re.search(r"\.(hc_(?:attn|ffn)_(?:fn|base|scale)|index_kpool_compress_(?:ape|gate))$", name):
            yield self.map_tensor_name(name) + ".weight", data_torch
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # keep the small mHC, k-pool and gating parameters exact
        exact_keys = ("hc_attn_", "hc_ffn_", "indexer_compressor_", "ssm_a", "ssm_dt", "exp_probs_b")
        if new_name.startswith("blk.") and any(key in new_name for key in exact_keys):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")

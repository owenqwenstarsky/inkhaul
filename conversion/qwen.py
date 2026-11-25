from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


# only the Qwen 3.5 / 3.6 subclasses below are registered; the base classes keep the upstream names
class Qwen2MoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE  # overridden by subclasses

    def _map_fp8_weight_names(self, name: str) -> tuple[str, ...]:
        if name.removesuffix(".weight").endswith(".mlp.experts.gate_up_proj"):
            bid = next(int(part) for part in self.map_tensor_name(name).split(".") if part.isdecimal())
            return (
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid),
                self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid),
            )
        return super()._map_fp8_weight_names(name)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (shared_expert_intermediate_size := self.hparams.get('shared_expert_intermediate_size')) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_expert_intermediate_size)
            logger.info(f"gguf: expert shared feed forward length = {shared_expert_intermediate_size}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # handle aggregated expert tensors
        # GGUF stores dimensions reversed from PyTorch, so:
        # PyTorch (A,B,C) -> GGUF writes [C,B,A] -> GGML reads ne={C,B,A}
        # Input shapes from HF: (n_expert, n_ff_exp, n_embd) or (n_expert, n_embd, n_ff_exp)
        # Expected GGML ne: {n_embd, n_ff_exp, n_expert} for gate/up, {n_ff_exp, n_embd, n_expert} for down
        if name.endswith("mlp.experts.down_proj") or name.endswith("mlp.experts.down_proj.weight"):
            mapped = f"{name}.weight" if not name.endswith(".weight") else name
            # HF: [n_expert, n_embd, n_ff] -> GGML: {n_ff, n_embd, n_expert}
            yield from super().modify_tensors(data_torch, mapped, bid)
            return

        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.gate_up_proj.weight"):
            if data_torch.ndim < 3 or data_torch.shape[-2] % 2 != 0:
                raise ValueError(f"Unexpected gate_up_proj shape for {name}: {tuple(data_torch.shape)}")
            # HF: [n_expert, 2*n_ff, n_embd] -> split on dim=-2
            n_ff = data_torch.shape[-2] // 2
            gate = data_torch[..., :n_ff, :].contiguous()
            up = data_torch[..., n_ff:, :].contiguous()
            # gate/up: [n_expert, n_ff, n_embd] -> GGML: {n_embd, n_ff, n_expert}
            base_name = name.removesuffix(".weight").removesuffix(".gate_up_proj")
            mapped_gate = f"{base_name}.gate_proj.weight"
            mapped_up = f"{base_name}.up_proj.weight"
            yield from super().modify_tensors(gate, mapped_gate, bid)
            yield from super().modify_tensors(up, mapped_up, bid)
            return

        if name.find("experts") != -1:
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
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


class Qwen3NextModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE  # overridden by subclasses

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        # the MTP head is not exported
        if titem[0].startswith(("mtp.", "model.mtp.")):
            return None
        return titem

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(self.hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(self.hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(self.hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(self.hparams["linear_value_head_dim"] * self.hparams["linear_num_value_heads"])
        if (layer_types := self.hparams.get("layer_types")) is not None:
            n_layer = self.hparams["num_hidden_layers"]
            if len(layer_types) != n_layer:
                raise ValueError(f"layer_types has {len(layer_types)} entries, expected num_hidden_layers ({n_layer})")
            self.gguf_writer.add_recurrent_layers([t == "linear_attention" for t in layer_types])
        self.gguf_writer.add_full_attention_interval(self.hparams.get("full_attention_interval", 4))
        if (rope_dim := self.hparams.get("head_dim")) is None:
            rope_dim = self.hparams["hidden_size"] // self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.rope_parameters.get("partial_rotary_factor", 0.25)))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif "conv1d" in name:
            data_torch = data_torch.squeeze()
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)


class _LinearAttentionVReorderBase(Qwen3NextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE  # overridden by subclasses
    """reorders V heads from grouped to tiled order for ggml broadcast

    see https://github.com/ggml-org/llama.cpp/pull/19468#discussion_r2786394306

    Linear attention may has num_k_heads < num_v_heads. The HF weights store
    V heads grouped by K head: [G0_v0..v{r-1}, G1_v0..v{r-1}, ...].
    ggml binary ops use tiled broadcast: [K0, K1, ..., K0, K1, ...].
    We reorder V heads to tiled order so ggml_repeat can replace the expensive
    interleaved repeat: [G0_v0, G1_v0, ..., G0_v1, G1_v1, ...].
    """

    _SHARED_INPUT_MERGES: tuple[tuple[gguf.MODEL_TENSOR, tuple[gguf.MODEL_TENSOR, gguf.MODEL_TENSOR], bool], ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._merge_parts: dict[tuple[int, gguf.MODEL_TENSOR], dict[int, Tensor]] = {}

    # caches one half of a merged weight until the other half arrives, then yields [a | b]
    def _merge_shared_input(self, new_name: str, data: Tensor, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # a pre-quantized checkpoint carries per-tensor scales, which one merged weight cannot hold
        if not self._SHARED_INPUT_MERGES or bid is None or self.hparams.get("quantization_config") or self._fp8_e4m3_preserved or hasattr(data, "get_lora_A_B"):
            yield (new_name, data)
            return

        layer_types = self.hparams.get("layer_types")
        if layer_types is None:
            is_linear = bid < self.hparams["num_hidden_layers"] and (bid + 1) % self.hparams.get("full_attention_interval", 4) != 0
        else:
            is_linear = bid < len(layer_types) and layer_types[bid] == "linear_attention"

        match = next(((merged, parts.index(part)) for merged, parts, linear_only in self._SHARED_INPUT_MERGES
                      if merged in gguf.MODEL_TENSORS[self.model_arch]
                      for part in parts
                      if (is_linear or not linear_only) and self.match_model_tensor_name(new_name, part, bid)), None)
        if match is None:
            yield (new_name, data)
            return

        merged, idx = match
        halves = self._merge_parts.setdefault((bid, merged), {})
        halves[idx] = data
        if len(halves) == 2:
            del self._merge_parts[(bid, merged)]
            yield (self.format_tensor_name(merged, bid), torch.cat([halves[0], halves[1]], dim=0))

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._merge_parts:
            raise ValueError(f"merged weights missing a half: {sorted((bid, t.name) for bid, t in self._merge_parts)}")

    @staticmethod
    def _reorder_v_heads(tensor: Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> Tensor:
        """Reorder V heads from grouped (by K head) to tiled order along the given dimension."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)

        # LoRA tensors (W = B @ A) cannot reshape their row dimension.
        # Instead, build a permutation index and apply it to A (column reorder) or B (row reorder) directly.
        if hasattr(tensor, 'get_lora_A_B'):
            n = shape[dim]
            idx = torch.arange(n).reshape(num_k_heads, num_v_per_k, head_dim)
            idx = idx.permute(1, 0, 2).contiguous().reshape(n)
            lora_A, lora_B = tensor.get_lora_A_B()  # ty: ignore[call-non-callable]
            if dim == len(shape) - 1:
                return type(tensor)(lora_A[:, idx], lora_B)
            elif dim == 0:
                return type(tensor)(lora_A, lora_B[idx])
            else:
                raise NotImplementedError(f"_reorder_v_heads on dim={dim} not supported for LoRA tensors")

        new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
        tensor = tensor.reshape(*new_shape)
        perm = list(range(len(new_shape)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return tensor.permute(*perm).contiguous().reshape(*shape)

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        # packed NVFP4 linear attention weights would need the V head reorder on the packed blocks
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)
        if num_k_heads != num_v_heads and name.endswith((
            ".linear_attn.in_proj_qkv.weight",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.out_proj.weight",
        )):
            raise NotImplementedError(f"NVFP4 linear attention weights are not supported: {name!r}")
        super()._repack_nvfp4(name, weight, scale, scale2, input_scale)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)

        if num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads and "linear_attn." in name:
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_per_k = num_v_heads // num_k_heads

            if ".in_proj_qkv." in name:
                # QKV weight: reorder only the V rows
                q_dim = head_k_dim * num_k_heads
                k_dim = head_k_dim * num_k_heads
                q = data_torch[:q_dim]
                k = data_torch[q_dim:q_dim + k_dim]
                v = data_torch[q_dim + k_dim:]
                v = self._reorder_v_heads(v, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([q, k, v], dim=0)

            elif ".in_proj_z." in name:
                # Z gate weight: reorder rows (num_v_heads * head_v_dim)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, head_v_dim)

            elif ".in_proj_b." in name or ".in_proj_a." in name:
                # Beta/Alpha weight: reorder rows (num_v_heads, head_dim=1)
                data_torch = self._reorder_v_heads(data_torch, 0, num_k_heads, num_v_per_k, 1)

            elif ".A_log" in name or ".dt_bias" in name or ".dt_proj" in name:
                # A_log / dt_bias: 1D parameters with num_v_heads elements
                if data_torch.ndim == 1:
                    data_torch = self._reorder_v_heads(
                        data_torch.unsqueeze(-1), 0, num_k_heads, num_v_per_k, 1
                    ).squeeze(-1)
                else:
                    data_torch = self._reorder_v_heads(data_torch, -1, num_k_heads, num_v_per_k, 1)

            elif ".conv1d" in name:
                # Conv1d kernel: reorder only the V channel portion
                data = data_torch.squeeze()
                qk_channels = head_k_dim * num_k_heads * 2
                qk_part = data[:qk_channels]
                v_part = data[qk_channels:]
                v_part = self._reorder_v_heads(v_part, 0, num_k_heads, num_v_per_k, head_v_dim)
                data_torch = torch.cat([qk_part, v_part], dim=0)

            elif ".out_proj." in name:
                # Out projection weight: reorder columns (input dimension)
                data_torch = self._reorder_v_heads(data_torch, 1, num_k_heads, num_v_per_k, head_v_dim)

        for new_name, data in super().modify_tensors(data_torch, name, bid):
            yield from self._merge_shared_input(new_name, data, bid)


class _Qwen35MRopeMixin:
    # Qwen3.5 always applies interleaved MRoPE (see Qwen3_5RotaryEmbedding in transformers);
    # the upstream default mrope_section is [11, 11, 10] and llama.cpp's QWEN35 / QWEN35MOE
    # loaders treat qwen35.rope.dimension_sections as required, so make sure it is always
    # written even when a particular checkpoint omits the field in `rope_parameters`.
    _QWEN35_DEFAULT_MROPE_SECTION = [11, 11, 10, 0]

    gguf_writer: gguf.GGUFWriter
    rope_parameters: dict

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # ty: ignore[missing-attribute]
        if "mrope_section" not in self.rope_parameters:
            self.gguf_writer.add_rope_dimension_sections(self._QWEN35_DEFAULT_MROPE_SECTION)


@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5TextModel")
@ModelBase.example("Qwen/Qwen3.5-9B")
class Qwen3_5TextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35

    _SHARED_INPUT_MERGES = (
        (gguf.MODEL_TENSOR.ATTN_QKV_GATE,      (gguf.MODEL_TENSOR.ATTN_QKV, gguf.MODEL_TENSOR.ATTN_GATE), True),
        (gguf.MODEL_TENSOR.SSM_BETA_ALPHA_CAT, (gguf.MODEL_TENSOR.SSM_BETA, gguf.MODEL_TENSOR.SSM_ALPHA), True),
    )

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()


@ModelBase.register("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM")
@ModelBase.example("Qwen/Qwen3.5-35B-A3B")
class Qwen3_5MoeTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE

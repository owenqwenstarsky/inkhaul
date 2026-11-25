#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import contextlib
import json
import math
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Iterator, Sequence, TypeVar, cast
from itertools import chain
from math import prod
from transformers import AutoConfig

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
import gguf


logger = logging.getLogger("hf-to-gguf")


AnyModel = TypeVar("AnyModel", bound="type[ModelBase]")


# for checkpoints that ship no config.json, we will try to provide a synthetic one
HparamsMatcher = Callable[[Path], bool]
HparamsLoader = Callable[[Path], dict[str, Any]]


class ModelType(IntEnum):
    TEXT = 1
    MMPROJ = 2


class ModelBase:
    _model_classes: dict[ModelType, dict[str, type[ModelBase]]] = {
        ModelType.TEXT: {},
        ModelType.MMPROJ: {},
    }
    _hparams_loaders: list[tuple[HparamsMatcher, HparamsLoader]] = []

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    dry_run: bool
    hparams: dict[str, Any]
    model_tensors: dict[str, Callable[[], Tensor]]
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    metadata: gguf.Metadata
    dir_model_card: Path
    remote_hf_model_id: str | None
    target_model_dir: Path | None

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    # subclasses should initialize this!
    block_count: int
    tensor_map: gguf.TensorNameMap

    # MTP (multi-token prediction) export modes; set by main() before instantiation.
    # Architectures that implement the filtering/export behavior opt in by
    # setting supports_mtp_export = True on their model class or a mixin.
    supports_mtp_export: bool = False
    mtp_only: bool = False
    no_mtp: bool = False

    def __init__(self, dir_model: Path, ftype: gguf.LlamaFileType, fname_out: Path, *, is_big_endian: bool = False,
                 use_temp_file: bool = False, eager: bool = False,
                 metadata_override: Path | None = None, model_name: str | None = None,
                 split_max_tensors: int = 0, split_max_size: int = 0, dry_run: bool = False,
                 small_first_shard: bool = False, hparams: dict[str, Any] | None = None, remote_hf_model_id: str | None = None,
                 target_model_dir: Path | None = None,
                 fuse_gate_up_exps: bool = False,
                 fp8_as_q8: bool = False):
        if type(self) is ModelBase or \
                type(self) is TextModel or \
                type(self) is MmprojModel:
            raise TypeError(f"{type(self).__name__!r} should not be directly instantiated")

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        self.use_temp_file = use_temp_file
        self.lazy = not eager or (remote_hf_model_id is not None)
        self.dry_run = dry_run
        self.remote_hf_model_id = remote_hf_model_id
        self.target_model_dir = target_model_dir
        self.fuse_gate_up_exps = fuse_gate_up_exps
        self._gate_exp_buffer: dict[int, Tensor] = {}
        self._up_exp_buffer: dict[int, Tensor] = {}
        self._unfusable_gate_up_layers: set[int] = set()
        self.hparams = ModelBase.load_hparams(self.dir_model) if hparams is None else hparams
        # source tensor name to filtered tensor name mapping.
        self._source_tensor_names: dict[str, str | None] = {}
        self.model_tensors = self.index_tensors(remote_hf_model_id=remote_hf_model_id)
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py
        self._is_nvfp4 = False
        self._is_mxfp4 = False
        self._nvfp4_global_algo: str | None = None # checkpoint-wide NVFP4 quant_algo
        self._nvfp4_layer_algo: dict[str, str | None] = {} # per-layer quant_algo, keyed by HF module path
        self._prec_a4: dict[str, bool] = {} # gguf tensor name -> can use 4-bit (A4) activations
        self._fp8_as_q8 = fp8_as_q8
        self._preserve_fp8 = ftype == gguf.LlamaFileType.GUESSED and not fp8_as_q8
        self._fp8_dequantized: set[str] = set()
        self._fp8_e4m3_preserved: set[str] = set()

        # Apply heuristics to figure out typical tensor encoding based on first tensor's dtype
        # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
        if self.ftype == gguf.LlamaFileType.GUESSED:
            for _, tensor in self.get_tensors():
                if tensor.dim() < 2:
                    continue

                if tensor.dtype == torch.bfloat16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_BF16
                    logger.info("heuristics detected bfloat16 tensor dtype, setting --outtype bf16")
                    break
                elif tensor.dtype == torch.float16:
                    self.ftype = gguf.LlamaFileType.MOSTLY_F16
                    logger.info("heuristics detected float16 tensor dtype, setting --outtype f16")
                    break
            else:
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
                logger.info("heuristics unable to detect tensor dtype, defaulting to --outtype f16")

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(path=None, arch=gguf.MODEL_ARCH_NAMES[self.model_arch], endianess=self.endianess, use_temp_file=self.use_temp_file,
                                           split_max_tensors=split_max_tensors, split_max_size=split_max_size, dry_run=dry_run, small_first_shard=small_first_shard)

    @classmethod
    def add_prefix_to_filename(cls, path: Path, prefix: str) -> Path:
        stem, suffix = path.stem, path.suffix
        new_name = f"{prefix}{stem}{suffix}"
        return path.with_name(new_name)

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}

        if remote_hf_model_id is not None:
            logger.info(f"Using remote model with HuggingFace id: {remote_hf_model_id}")
            remote_tensors = gguf.utility.SafetensorRemote.get_list_tensors_hf_model(remote_hf_model_id)
            for name, remote_tensor in remote_tensors.items():
                data_gen = lambda r=remote_tensor: LazyTorchTensor.from_remote_tensor(r)  # noqa: E731
                titem = self.filter_tensors((name, data_gen))
                self._source_tensor_names[name] = titem[0] if titem else None
                if titem:
                    tname, tgen = titem
                    tensors[tname] = tgen

            return tensors

        return self.index_local_tensors("model")

    # index the local "<prefix>*.safetensors" files (or pytorch_model*.bin), through filter_tensors
    def index_local_tensors(self, prefix: str) -> dict[str, Callable[[], Tensor]]:
        tensors: dict[str, Callable[[], Tensor]] = {}
        part_names: list[str] = ModelBase.get_model_part_names(self.dir_model, prefix, ".safetensors")
        is_safetensors: bool = len(part_names) > 0 or (self.dir_model / f"{prefix}.safetensors.index.json").is_file()
        if not is_safetensors:
            part_names = ModelBase.get_model_part_names(self.dir_model, "pytorch_model", ".bin")

        tensor_names_from_index: set[str] = set()
        tensor_names_from_parts: set[str] = set()

        index_name = f"{prefix}.safetensors" if is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name

        if index_file.is_file():
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                tensor_names_from_index.update(weight_map.keys())
                part_dict: dict[str, None] = dict.fromkeys(weight_map.values(), None)
                part_names = sorted(part_dict.keys())
        else:
            weight_map = {}

        for part_name in part_names:
            logger.info(f"gguf: indexing model part '{part_name}'")
            ctx: ContextManager[Any]
            if is_safetensors:
                ctx = cast(ContextManager[Any], gguf.utility.SafetensorsLocal(self.dir_model / part_name))
            else:
                ctx = contextlib.nullcontext(torch.load(str(self.dir_model / part_name), map_location="cpu", mmap=True, weights_only=True))

            with ctx as model_part:
                assert model_part is not None

                for name in model_part.keys():
                    tensor_names_from_parts.add(name)
                    if is_safetensors:
                        data: gguf.utility.LocalTensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                        else:
                            dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                            data_gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
                    else:
                        data_torch: Tensor = model_part[name]
                        if self.lazy:
                            data_gen = lambda data=data_torch: LazyTorchTensor.from_eager(data)  # noqa: E731
                        else:
                            data_gen = lambda data=data_torch: data  # noqa: E731
                    titem = self.filter_tensors((name, data_gen))
                    self._source_tensor_names[name] = titem[0] if titem else None
                    if titem:
                        tname, tgen = titem
                        tensors[tname] = tgen

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_index) > 0:
            if len(tensor_names_from_parts.symmetric_difference(tensor_names_from_index)) > 0:
                missing = sorted(tensor_names_from_index.difference(tensor_names_from_parts))
                extra = sorted(tensor_names_from_parts.difference(tensor_names_from_index))
                missing_files = sorted(set(weight_map[n] for n in missing if n in weight_map))
                if len(extra) == 0 and len(missing_files) > 0:
                    raise ValueError(f"Missing or incomplete model files: {missing_files}\n"
                                     f"Missing tensors: {missing}")
                else:
                    raise ValueError("Mismatch between weight map and model parts for tensor names:\n"
                                     f"Missing tensors: {missing}\n"
                                     f"Extra tensors: {extra}")

        return tensors

    @staticmethod
    def _scale_is_trivial(scale: Tensor) -> bool:
        return scale.numel() <= 1 and abs(float(scale.float().sum()) - 1.0) < 1e-6

    def _write_scale_tensor(self, scale_name: str, scale: Tensor):
        if not self._scale_is_trivial(scale):
            scale_f32 = scale.float().numpy().flatten()
            logger.info(f"  + {scale_name} (per-tensor scale, shape [{scale_f32.size}])")
            self.gguf_writer.add_tensor(scale_name, scale_f32)

    def _write_scales_tensor(self, scale_name: str, scales: list[float]):
        if not np.allclose(scales, 1.0, atol=1e-6):
            scale_vals = np.array(scales, dtype=np.float32)
            logger.info(f"  + {scale_name} (per-expert scale, shape [{len(scales)}])")
            self.gguf_writer.add_tensor(scale_name, scale_vals)

    def _prepare_kv_cache_scales(self):
        for name in list(self.model_tensors.keys()):
            if not name.endswith((".k_scale", ".v_scale")):
                continue

            mapped = self.tensor_map.get_type_and_name(key=name, try_suffixes=(".k_scale", ".v_scale"))
            if mapped is None:
                continue

            tensor_type, new_name = mapped
            if tensor_type not in (gguf.MODEL_TENSOR.ATTN_K, gguf.MODEL_TENSOR.ATTN_V):
                continue

            bid = int(new_name.split(".")[1])
            scale_type = gguf.MODEL_TENSOR.ATTN_K if name.endswith(".k_scale") else gguf.MODEL_TENSOR.ATTN_V
            suffix = ".k_scale" if name.endswith(".k_scale") else ".v_scale"
            new_name = gguf.TENSOR_NAMES[scale_type].format(bid=bid) + suffix

            scale = LazyTorchTensor.to_eager(self.model_tensors.pop(name)())
            if scale.dtype not in (torch.bfloat16, torch.float32) or scale.numel() != 1:
                raise ValueError(f"KV cache scale {name!r} must be a scalar BF16 or FP32 tensor")

            scale = scale.float()
            value = float(scale.item())
            if not math.isfinite(value) or value == 0.0:
                raise ValueError(f"KV cache scale {name!r} must be finite and nonzero")

            logger.info(f"  + {new_name} (KV cache scale, shape [1])")
            self.gguf_writer.add_tensor(new_name, scale.flatten().numpy())

    def dequant_model(self):
        # If all quantized tensors were already handled (e.g. pure NVFP4), skip
        if self._is_nvfp4 and not any(k.endswith((".weight_scale", ".weight_scale_inv", ".input_scale", ".activation_scale", "_activation_scale", ".k_scale", ".v_scale")) for k in self.model_tensors):
            return

        tensors_to_remove: list[str] = []
        new_tensors: dict[str, Callable[[], Tensor]] = {}

        if (quant_config := self.hparams.get("quantization_config")) and isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")

            def dequant_bitnet(weight: Tensor, scale: Tensor) -> Tensor:
                weight = weight.view(torch.uint8)
                orig_shape = weight.shape

                shift = torch.tensor([0, 2, 4, 6], dtype=torch.uint8).reshape((4, *(1 for _ in range(len(orig_shape)))))
                data = weight.unsqueeze(0).expand((4, *orig_shape)) >> shift
                data = data & 3
                data = (data.float() - 1).reshape((orig_shape[0] * 4, *orig_shape[1:]))

                # The scale is inverted
                return data / scale.float()

            def dequant_simple(weight: Tensor, scale: Tensor, block_size: Sequence[int] | None = None) -> Tensor:
                if scale.dtype == torch.uint8:
                    # E8M0 exponent stored as raw bytes (MXFP8)
                    scale = (scale.float() - 127.0).exp2()
                scale = scale.float()

                # ModelOpt block scales are pre-shaped for broadcasting, e.g. [out/128, 1, in/128, 1]
                if block_size is None and scale.ndim == 2 * weight.ndim and all(n == 1 for n in scale.shape[1::2]):
                    assert all(w % n == 0 for w, n in zip(weight.shape, scale.shape[0::2]))
                    block_size = [w // n for w, n in zip(weight.shape, scale.shape[0::2])]
                    scale = scale.reshape(tuple(scale.shape[0::2]))

                if block_size is not None:
                    dim_offset = scale.ndim - len(block_size)
                    for i, size in enumerate(block_size):
                        scale = scale.repeat_interleave(size, dim_offset + i)
                    # unpad the scale (e.g. when the tensor size isn't a multiple of the block size)
                    scale = scale[tuple(slice(0, size) for size in weight.shape)]

                # align scale dims to weight for correct broadcasting (e.g. [128] -> [128, 1, 1])
                while scale.ndim < weight.ndim:
                    scale = scale.unsqueeze(-1)

                return weight.float() * scale

            # ref: https://github.com/ModelCloud/GPTQModel/blob/037c5c0f6c9e33c500d975b038d02e7ca437546d/gptqmodel/nn_modules/qlinear/__init__.py#L437-L476
            def dequant_gptq(g_idx: Tensor, qweight: Tensor, qzeros: Tensor, scales: Tensor) -> Tensor:
                bits = quant_config["bits"]
                assert bits in (2, 3, 4, 8)
                assert qweight.dtype == qzeros.dtype
                maxq = (2 ** bits) - 1
                weight = None
                zeros = None
                pack_dtype_bits = qweight.dtype.itemsize * 8

                if bits in [2, 4, 8]:
                    pack_factor = pack_dtype_bits // bits
                    wf = torch.tensor(list(range(0, pack_dtype_bits, bits)), dtype=torch.int32).unsqueeze(0)
                    if self.lazy:
                        wf = LazyTorchTensor.from_eager(wf)

                    zeros = torch.bitwise_right_shift(
                        qzeros.unsqueeze(2).expand(-1, -1, pack_factor),
                        wf.unsqueeze(0)
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    zeros = torch.bitwise_and(zeros, maxq).reshape(scales.shape)

                    weight = torch.bitwise_and(
                        torch.bitwise_right_shift(
                            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
                            wf.unsqueeze(-1)
                        ).to(torch.int16 if bits == 8 else torch.int8),
                        maxq
                    )
                elif bits == 3:
                    raise NotImplementedError("3-bit gptq dequantization is not yet implemented")

                assert weight is not None
                assert zeros is not None

                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

                # gptq_v2 doesn't need to offset zeros
                if quant_config.get("checkpoint_format", "gptq") == "gptq":
                    zeros += 1

                return (scales[g_idx].float() * (weight - zeros[g_idx]).float()).T

            def dequant_packed(w: Tensor, scale: Tensor, shape_tensor: Tensor, zero_point: Tensor | None, num_bits: int, group_size: int):
                assert w.dtype == torch.int32
                shape = tuple(shape_tensor.tolist())
                assert len(shape) == 2
                mask = (1 << num_bits) - 1

                shifts = torch.arange(0, 32 - (num_bits - 1), num_bits, dtype=torch.int32)
                if self.lazy:
                    shifts = LazyTorchTensor.from_eager(shifts)

                if zero_point is None:
                    offset = 1 << (num_bits - 1)
                else:
                    assert len(zero_point.shape) == 2
                    offset = (zero_point.unsqueeze(1) >> shifts.reshape(1, -1, 1)) & mask
                    offset = offset.reshape(-1, zero_point.shape[1])
                    # trim padding, and prepare for broadcast
                    # NOTE: the zero-point is packed along dim 0
                    offset = offset[:shape[0], :].unsqueeze(-1)

                # extract values
                # NOTE: the weights are packed along dim 1
                unpacked = (w.unsqueeze(-1) >> shifts.reshape(1, 1, -1)) & mask
                unpacked = unpacked.reshape(shape[0], -1)

                # trim padding
                unpacked = unpacked[:, :shape[1]]

                # prepare for broadcast of the scale
                unpacked = unpacked.reshape(shape[0], (unpacked.shape[-1] + group_size - 1) // group_size, group_size)
                unpacked = unpacked - offset

                return (unpacked * scale.unsqueeze(-1).float()).reshape(shape)

            def dequant_fp8(fp8_configs: list[dict[str, Any]] | None = None) -> None:
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        if weight_name not in self.model_tensors:
                            tensors_to_remove.append(name)
                            continue
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        block_size = None
                        is_fp8_weight = False
                        if self._fp8_as_q8 or fp8_configs is not None:
                            weight = w()
                            is_fp8_weight = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                        if fp8_configs is not None:
                            if not is_fp8_weight:
                                raise ValueError(f"Expected remaining weight {weight_name!r} to be FP8, got {weight.dtype}")
                            scale = s()
                            is_channel_scale = scale.ndim >= 1 and scale.ndim <= weight.ndim and scale.shape[0] == weight.shape[0] and all(size == 1 for size in scale.shape[1:])
                            if prod(scale.shape) != 1 and not is_channel_scale:
                                block_configs = [c for c in fp8_configs if c.get("strategy") == "block"]
                                if not block_configs or any(not c.get("block_structure") or c.get("group_size") is not None for c in block_configs):
                                    raise ValueError(f"FP8 weight {weight_name!r} with scale shape {list(scale.shape)} requires a block layout")
                                block_sizes = {tuple(c["block_structure"]) for c in block_configs}
                                if len(block_sizes) != 1:
                                    raise ValueError(f"FP8 weight {weight_name!r} has ambiguous block layouts")
                                block_size = next(iter(block_sizes))
                        self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                        tensors_to_remove.append(name)
                        if self._fp8_as_q8 and is_fp8_weight:
                            self._fp8_dequantized.add(weight_name)
                    if name.endswith((".input_scale", ".activation_scale", "_activation_scale", ".k_scale", ".v_scale")):
                        tensors_to_remove.append(name)

            if quant_method == "bitnet":
                for name in self.model_tensors.keys():
                    if name.endswith(".weight_scale"):
                        weight_name = name.removesuffix("_scale")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s: dequant_bitnet(w(), s())
                        tensors_to_remove.append(name)
            elif quant_method in ("fp8", "bsgemm-moe", "mxfp8"):  # bsgemm-moe: drama-3, same block layout
                block_size = quant_config.get("weight_block_size")
                for name in self.model_tensors.keys():
                    if name.endswith("_scale_inv"):
                        weight_name = name.removesuffix("_scale_inv")
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                        tensors_to_remove.append(name)
                        if self._fp8_as_q8:
                            self._fp8_dequantized.add(weight_name)
                    if name.endswith(".activation_scale"):  # unused
                        tensors_to_remove.append(name)
                    if name.endswith("_activation_scale"):  # Mistral-Small-4-119B-2602, unused
                        tensors_to_remove.append(name)
                    # mistral format
                    if name.endswith(".qscale_weight"):
                        weight_name = name.removesuffix("qscale_weight") + "weight"
                        w = self.model_tensors[weight_name]
                        s = self.model_tensors[name]
                        self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                        tensors_to_remove.append(name)
                        if self._fp8_as_q8:
                            self._fp8_dequantized.add(weight_name)
                    if name.endswith(".qscale_act"):
                        tensors_to_remove.append(name)
            elif quant_method == "gptq":
                for name in self.model_tensors.keys():
                    if name.endswith(".qweight"):
                        base_name = name.removesuffix(".qweight")
                        g_idx = self.model_tensors[base_name + ".g_idx"]
                        qweight = self.model_tensors[base_name + ".qweight"]
                        qzeros = self.model_tensors[base_name + ".qzeros"]
                        scales = self.model_tensors[base_name + ".scales"]
                        new_tensors[base_name + ".weight"] = (
                            lambda g=g_idx, z=qzeros, w=qweight, s=scales: dequant_gptq(
                                g(), w(), z(), s()
                            )
                        )
                        tensors_to_remove += [
                            base_name + n
                            for n in (
                                ".g_idx",
                                ".qzeros",
                                ".qweight",
                                ".scales",
                            )
                        ]
            elif quant_method == "compressed-tensors":
                quant_format = quant_config["format"]
                groups = quant_config["config_groups"]
                nvfp4_compressed_tensors = self._is_nvfp4_compressed_tensors(quant_method, quant_format, groups)

                if nvfp4_compressed_tensors:
                    fp8_configs = [
                        g["weights"] for g in groups.values()
                        if isinstance(g, dict) and g.get("format") == "float-quantized"
                        and g.get("weights", {}).get("type") == "float"
                        and g.get("weights", {}).get("num_bits") == 8
                    ]
                    dequant_fp8(fp8_configs)
                elif len(groups) > 1:
                    raise NotImplementedError("Can't handle multiple config groups for compressed-tensors yet")
                elif quant_format == "float-quantized" or quant_format == "int-quantized" or quant_format == "naive-quantized":
                    weight_config = tuple(groups.values())[0]["weights"]
                    block_size = weight_config.get("block_structure", None)
                    strategy = weight_config.get("strategy")
                    is_fp8 = (
                        quant_format == "float-quantized"
                        and weight_config.get("type") == "float"
                        and weight_config.get("num_bits") == 8
                    )
                    assert strategy in ("channel", "block") or (is_fp8 and strategy == "tensor")
                    assert weight_config.get("group_size") is None  # didn't find a model using this yet
                    for name in self.model_tensors.keys():
                        if name.endswith(".weight_scale"):
                            weight_name = name.removesuffix("_scale")
                            w = self.model_tensors[weight_name]
                            s = self.model_tensors[name]
                            self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                            tensors_to_remove.append(name)
                            if self._fp8_as_q8 and is_fp8:
                                self._fp8_dequantized.add(weight_name)
                        if is_fp8 and name.endswith((".input_scale", ".activation_scale", "_activation_scale", ".k_scale", ".v_scale")):
                            tensors_to_remove.append(name)
                elif quant_format == "pack-quantized":
                    weight_config = tuple(groups.values())[0]["weights"]
                    assert weight_config.get("strategy") == "group"
                    assert weight_config.get("type", "int") == "int"
                    num_bits = weight_config.get("num_bits")
                    group_size = weight_config.get("group_size")
                    assert isinstance(num_bits, int)
                    assert isinstance(group_size, int)
                    for name in self.model_tensors.keys():
                        if name.endswith(".weight_packed"):
                            base_name = name.removesuffix("_packed")
                            w = self.model_tensors[name]
                            scale = self.model_tensors[base_name + "_scale"]
                            shape = self.model_tensors[base_name + "_shape"]
                            zero_point = self.model_tensors.get(base_name + "_zero_point", lambda: None)
                            new_tensors[base_name] = (
                                lambda w=w, scale=scale, shape=shape, zero_point=zero_point: dequant_packed(
                                    w(), scale(), shape(), zero_point(), num_bits, group_size,
                                )
                            )
                            tensors_to_remove += [base_name + n for n in ("_packed", "_shape", "_scale")]
                            if (base_name + "_zero_point") in self.model_tensors:
                                tensors_to_remove.append(base_name + "_zero_point")
                else:
                    raise NotImplementedError(f"Quant format {quant_format!r} for method {quant_method!r} is not yet supported")
            elif quant_method in ("modelopt", "modelopt_mixed"):
                dequant_fp8()
                # FP8_PB_WO: one scale per block, the block size follows from the two shapes
                for name in self.model_tensors.keys():
                    if not name.endswith(".weight_scale_inv"):
                        continue
                    weight_name = name.removesuffix("_scale_inv")
                    if weight_name not in self.model_tensors:
                        tensors_to_remove.append(name)
                        continue
                    w = self.model_tensors[weight_name]
                    s = self.model_tensors[name]
                    if w().dtype != torch.float8_e4m3fn:
                        continue
                    w_shape, s_shape = w().shape, s().shape
                    block_size = None
                    if prod(s_shape) != 1:
                        if len(s_shape) != len(w_shape) or any(ws % ss for ws, ss in zip(w_shape, s_shape)):
                            raise ValueError(f"FP8 weight {weight_name!r} {list(w_shape)} is not tiled by its scale {list(s_shape)}")
                        block_size = [ws // ss for ws, ss in zip(w_shape, s_shape)]
                    self.model_tensors[weight_name] = lambda w=w, s=s, bs=block_size: dequant_simple(w(), s(), bs)
                    tensors_to_remove.append(name)
                    if self._fp8_as_q8:
                        self._fp8_dequantized.add(weight_name)
            elif quant_method is not None:
                raise NotImplementedError(f"Quant method is not yet supported: {quant_method!r}")

        for name in tensors_to_remove:
            if name in self.model_tensors:
                del self.model_tensors[name]

        for name, value in new_tensors.items():
            self.model_tensors[name] = value

    def _map_fp8_weight_names(self, name: str) -> tuple[str, ...]:
        new_name = self.tensor_map.get_name(name, try_suffixes=(".weight",))
        return (new_name,) if new_name is not None else ()

    def _prepare_fp8_e4m3_tensors(self):
        if not self._preserve_fp8:
            return

        scale_tensors: dict[str, list[tuple[int, float]] | np.ndarray] = {}
        input_scale_tensors: dict[str, list[tuple[int, float]] | np.ndarray] = {}
        consumed: list[str] = []
        candidates: dict[str, tuple[str, Tensor, tuple[str, ...], torch.Size]] = {}
        weight_dtypes: dict[str, torch.dtype] = {}
        scaled_weights: set[str] = set()
        expert_pattern = re.compile(r"(\.[^.]*experts)\.(\d+)\.")
        scale_types = {
            gguf.MODEL_TENSOR.ATTN_Q, gguf.MODEL_TENSOR.ATTN_K, gguf.MODEL_TENSOR.ATTN_V,
            gguf.MODEL_TENSOR.ATTN_OUT, gguf.MODEL_TENSOR.ATTN_QKV, gguf.MODEL_TENSOR.ATTN_GATE,
            gguf.MODEL_TENSOR.FFN_GATE, gguf.MODEL_TENSOR.FFN_UP, gguf.MODEL_TENSOR.FFN_DOWN,
            gguf.MODEL_TENSOR.FFN_GATE_SHEXP, gguf.MODEL_TENSOR.FFN_UP_SHEXP, gguf.MODEL_TENSOR.FFN_DOWN_SHEXP,
            gguf.MODEL_TENSOR.FFN_GATE_EXP, gguf.MODEL_TENSOR.FFN_UP_EXP, gguf.MODEL_TENSOR.FFN_DOWN_EXP,
            gguf.MODEL_TENSOR.SSM_OUT, gguf.MODEL_TENSOR.SSM_ALPHA, gguf.MODEL_TENSOR.SSM_BETA,
            gguf.MODEL_TENSOR.NEXTN_EH_PROJ, gguf.MODEL_TENSOR.OUTPUT,
        }

        for scale_name in list(self.model_tensors.keys()):
            weight_name = None
            if scale_name.endswith(".weight_scale"):
                weight_name = scale_name.removesuffix("_scale")
            elif scale_name.endswith("_scale_inv"):
                weight_name = scale_name.removesuffix("_scale_inv")
            elif scale_name.endswith(".qscale_weight"):
                weight_name = scale_name.removesuffix("qscale_weight") + "weight"

            if weight_name is None or weight_name not in self.model_tensors:
                continue

            scaled_weights.add(weight_name)
            weight = self.model_tensors[weight_name]()
            weight_dtypes[weight_name] = weight.dtype
            if weight.dtype != torch.float8_e4m3fn or weight.ndim < 2:
                continue

            scale = self.model_tensors[scale_name]()
            if prod(scale.shape) != 1:
                continue

            merged_name = expert_pattern.sub(r"\1.", weight_name)
            new_names = self._map_fp8_weight_names(merged_name)
            if not new_names:
                continue
            if any((mapped := self.tensor_map.get_type_and_name(new_name, try_suffixes=(".weight",))) is None or mapped[0] not in scale_types for new_name in new_names):
                continue
            bid = next((int(part) for part in new_names[0].split(".") if part.isdecimal()), None)
            self._fp8_e4m3_preserved.add(weight_name)
            if any(new_name.endswith("_norm.weight") or self.tensor_force_quant(weight_name, new_name, bid, weight.ndim) != gguf.GGMLQuantizationType.F8_E4M3 for new_name in new_names):
                self._fp8_e4m3_preserved.discard(weight_name)
                continue
            candidates[weight_name] = (scale_name, scale, new_names, weight.shape)

        expert_groups: dict[str, list[str]] = {}
        for name in self.model_tensors:
            if (match := expert_pattern.search(name)) and name.endswith(".weight"):
                expert_groups.setdefault(name[:match.end(1)], []).append(name)
        for names in expert_groups.values():
            for name in names:
                if name not in weight_dtypes:
                    weight_dtypes[name] = self.model_tensors[name]().dtype
            dtypes = {weight_dtypes[name] for name in names}
            if torch.float8_e4m3fn not in dtypes:
                continue
            if dtypes == {torch.float8_e4m3fn} and not any(name in scaled_weights for name in names):
                continue
            # The converter merges expert projections together.
            if not all(name in candidates for name in names):
                for name in names:
                    candidates.pop(name, None)
                    self._fp8_e4m3_preserved.discard(name)
                    if name not in scaled_weights:
                        gen = self.model_tensors[name]
                        self.model_tensors[name] = lambda gen=gen: gen().float()

        for weight_name, (scale_name, scale, new_names, shape) in candidates.items():
            scale = LazyTorchTensor.to_eager(scale).float().flatten()

            weight_prefix = weight_name.removesuffix(".weight")
            # Transformers fine-grained FP8 uses activation_scale while ModelOpt uses input_scale.
            # Ref: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/mistral3.py#L357-L361
            input_scale_name = next((name for name in (
                weight_prefix + ".input_scale",
                weight_prefix + ".activation_scale",
                weight_prefix + "_activation_scale",
                weight_prefix + ".qscale_act",
            ) if name in self.model_tensors), None)
            input_scale = None
            if input_scale_name is not None:
                input_scale = self.model_tensors[input_scale_name]()
                if prod(input_scale.shape) != 1:
                    raise ValueError(f"FP8 input scale {input_scale_name!r} must be a scalar")
                input_scale = LazyTorchTensor.to_eager(input_scale).float().flatten()
                consumed.append(input_scale_name)

            consumed.append(scale_name)

            expert_match = expert_pattern.search(weight_name)
            if expert_match:
                expert_id = int(expert_match.group(2))
                for new_name in new_names:
                    target_name = new_name.replace(".weight", ".scale")
                    entries = scale_tensors.setdefault(target_name, [])
                    assert isinstance(entries, list)
                    cast(list[tuple[int, float]], entries).append((expert_id, float(scale[0])))
                    if input_scale is not None:
                        target_name = new_name.replace(".weight", ".input_scale")
                        entries = input_scale_tensors.setdefault(target_name, [])
                        assert isinstance(entries, list)
                        cast(list[tuple[int, float]], entries).append((expert_id, float(input_scale[0])))
            else:
                for new_name in new_names:
                    mapped = self.tensor_map.get_type_and_name(new_name, try_suffixes=(".weight",)) if len(shape) == 3 else None
                    is_packed_expert = mapped is not None and mapped[0] in (
                        gguf.MODEL_TENSOR.FFN_GATE_EXP, gguf.MODEL_TENSOR.FFN_UP_EXP, gguf.MODEL_TENSOR.FFN_DOWN_EXP,
                    )
                    # Packed expert weights need one scale value per expert.
                    n_scales = shape[0] if is_packed_expert else 1
                    scale_tensors[new_name.replace(".weight", ".scale")] = np.repeat(scale.numpy(), n_scales)
                    if input_scale is not None:
                        input_scale_tensors[new_name.replace(".weight", ".input_scale")] = np.repeat(input_scale.numpy(), n_scales)

        for name in consumed:
            self.model_tensors.pop(name, None)

        for name, values in chain(scale_tensors.items(), input_scale_tensors.items()):
            if isinstance(values, list):
                values.sort(key=lambda item: item[0])
                scale = np.array([item[1] for item in values], dtype=np.float32)
            else:
                scale = values.astype(np.float32)
            if not np.allclose(scale, 1.0, atol=1e-6):
                logger.info(f"  + {name} (FP8 scale, shape [{scale.size}])")
                self.gguf_writer.add_tensor(name, scale)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        if "language_model." in name:
            name = name.replace("language_model.", "")

        return name, gen

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, gen in self.model_tensors.items():
            yield name, gen()

    def format_tensor_name(self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight") -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}")
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(self, name: str, key: gguf.MODEL_TENSOR, bid: int | None, suffix: str = ".weight") -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def prepare_gate_up_fusion(self) -> None:
        self._unfusable_gate_up_layers.clear()
        if not self.fuse_gate_up_exps or self._fp8_as_q8:
            return

        for name, gen in self.model_tensors.items():
            merged_name = re.sub(r"(\.experts)\.\d+\.", r"\1.", name)
            mapped = self.tensor_map.get_type_and_name(merged_name, try_suffixes=(".weight",))
            if mapped is None:
                continue
            tensor_type, new_name = mapped
            if tensor_type not in (gguf.MODEL_TENSOR.FFN_GATE_EXP, gguf.MODEL_TENSOR.FFN_UP_EXP, gguf.MODEL_TENSOR.FFN_GATE_UP_EXP) or not new_name.endswith(".weight"):
                continue
            if gen().dtype == torch.float8_e4m3fn:
                bid = next(int(part) for part in new_name.split(".") if part.isdecimal())
                self._unfusable_gate_up_layers.add(bid)

        for bid in sorted(self._unfusable_gate_up_layers):
            logger.warning(f"Skipping --fuse-gate-up-exps for layer {bid}: fusion of preserved FP8 expert projections is not supported")

    def _tag_prec_a4(self, hf_name: str, gguf_name: str) -> None:
        # W4A16_NVFP4 should not use 4-bit activations
        name = hf_name.removesuffix(".weight").removesuffix(".bias")
        algo = self._nvfp4_global_algo
        while name:
            if name in self._nvfp4_layer_algo:
                algo = self._nvfp4_layer_algo[name]
                break
            name = name.rpartition(".")[0]
        if algo == "W4A16_NVFP4":
            self._prec_a4[gguf_name] = False

    def check_no_hadamard(self) -> None:
        # the runtime has no Hadamard-aware matmul path
        if (self.dir_model / "hadamard_packing.json").is_file():
            raise ValueError("Hadamard-folded checkpoints (hadamard_packing.json) are not supported")

    def set_gguf_parameters(self):
        raise NotImplementedError("set_gguf_parameters() must be implemented in subclasses")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)

        # Handle gate/up expert tensor fusion if enabled
        # Preserved FP8 gate/up scales must stay with their separate projections.
        if self.fuse_gate_up_exps and bid is not None and bid not in self._unfusable_gate_up_layers:
            if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_GATE_EXP, bid):
                self._gate_exp_buffer[bid] = data_torch
            elif self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_UP_EXP, bid):
                self._up_exp_buffer[bid] = data_torch

            # Check if both gate and up are buffered for this layer
            if bid in self._gate_exp_buffer and bid in self._up_exp_buffer:
                gate_data = self._gate_exp_buffer.pop(bid)
                up_data = self._up_exp_buffer.pop(bid)
                # gate/up shape: (n_expert, n_ff, n_embd), concatenate to (n_expert, n_ff*2, n_embd)
                fused_data = torch.cat([gate_data, up_data], dim=1)
                fused_name = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_UP_EXP, bid)
                logger.info(f"Fused gate_exps and up_exps for layer {bid}")
                return [(fused_name, fused_data)]

            # If we buffered a gate/up tensor, wait for the other
            if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_GATE_EXP, bid) or \
               self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.FFN_UP_EXP, bid):
                return []

        return [(new_name, data_torch)]

    def _tensor_requires_f32(self, new_name: str, bid: int | None, n_dims: int) -> bool:
        # Conditions should closely match those in llama_model_quantize_internal in llama.cpp.
        return n_dims <= 1 or new_name.endswith("_norm.weight") or (
            any(
                self.match_model_tensor_name(new_name, key, bid)
                for key in (
                    gguf.MODEL_TENSOR.FFN_GATE_INP,
                    gguf.MODEL_TENSOR.FFN_GATE_INP_SHEXP,
                    gguf.MODEL_TENSOR.SSM_CONV1D,
                    gguf.MODEL_TENSOR.V_ENC_EMBD_POS,
                    # Kimi KDA conv weights should be F32
                    gguf.MODEL_TENSOR.SSM_CONV1D_Q,
                    gguf.MODEL_TENSOR.SSM_CONV1D_K,
                    gguf.MODEL_TENSOR.SSM_CONV1D_V,
                )
            )
            or new_name[-7:] not in (".weight", ".lora_a", ".lora_b")
        )

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if name in self._fp8_e4m3_preserved and n_dims >= 2:
            if self._tensor_requires_f32(new_name, bid, n_dims):
                return gguf.GGMLQuantizationType.F32
            return gguf.GGMLQuantizationType.F8_E4M3
        # Force FP8-original tensors to Q8_0 when requested; Q8_0 is faster than F16/BF16.
        if self._fp8_as_q8 and name in self._fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.Q8_0
        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    @staticmethod
    def repack_mxfp4_blocks(packed: Tensor, scale: Tensor) -> np.ndarray:
        """
        Repack 4-bit MX weights into ggml `block_mxfp4`. Lossless - only moves bits.

        Source (compressed-tensors "mxfp4-pack-quantized", also used by DeepSeek-V4):
          packed  uint8 [rows, cols/2]  element 2i in the low nibble, 2i+1 in the high one
          scale   uint8 [rows, cols/32] one E8M0 biased exponent per 32-element group

        Destination, per group: one scale byte then 16 code bytes, where byte j holds
        element j in the low nibble and element j+16 in the high one.

        The 4-bit codes need no remapping: both sides index into ggml's kvalues_mxfp4
        order. ggml doubles the kvalues and halves the scale, so the value is the same.
        """
        p = packed.contiguous().view(torch.uint8)
        s = scale.contiguous().view(torch.uint8)

        rows, packed_cols = p.shape
        cols = packed_cols * 2
        if cols % 32 != 0:
            raise ValueError(f"MXFP4 source row has {cols} values, expected a multiple of 32")

        n_blocks = cols // 32
        if tuple(s.shape) != (rows, n_blocks):
            raise ValueError(f"MXFP4 scale shape {tuple(s.shape)} does not match {(rows, n_blocks)}")

        src = p.reshape(rows, n_blocks, 16)
        lo = src & 0x0F           # elements 0, 2, 4, ...
        hi = (src >> 4) & 0x0F    # elements 1, 3, 5, ...

        vals = torch.stack((lo, hi), dim=-1).reshape(rows, n_blocks, 32)
        qs = vals[:, :, :16] | (vals[:, :, 16:] << 4)

        raw = torch.cat((s.unsqueeze(-1), qs.to(torch.uint8)), dim=-1)
        return raw.reshape(rows, n_blocks * 17).cpu().numpy()

    def _mxfp4_expert_tensor(self, loaders: list[tuple[Callable[[], Tensor], Callable[[], Tensor]]]):
        """
        One stacked [n_expert, rows, cols] MXFP4 tensor, built lazily.

        gguf_writer holds every added tensor until the final write, so building
        this eagerly (like the DeepSeek-V4 path does) keeps every expert in
        memory at once. lazy means only the tensor being written is resident.
        """
        # meta shapes, so this does not read any weights
        rows, packed_cols = loaders[0][0]().shape
        n_blocks = (packed_cols * 2) // 32
        byte_shape = (len(loaders), rows, n_blocks * 17)

        def load(fns: list[tuple[Callable[[], Tensor], Callable[[], Tensor]]]) -> np.ndarray:
            out = np.empty(byte_shape, dtype=np.uint8)
            for eid, (packed_fn, scale_fn) in enumerate(fns):
                out[eid] = self.repack_mxfp4_blocks(
                    LazyTorchTensor.to_eager(packed_fn()),
                    LazyTorchTensor.to_eager(scale_fn()),
                )
            return out

        # loaders goes through args, not the closure, so that `func` matches
        # LazyBase's single-argument shape
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, byte_shape),
            args=(loaders,),
            func=load,
        )

    @staticmethod
    def _is_nvfp4_compressed_tensors(quant_method, quant_format, groups) -> bool:
        # Some models use per-tensor quant_algo (e.g. "MIXED_PRECISION" with
        # per-layer NVFP4/FP8) instead of a single global "NVFP4" value.
        if quant_method != "compressed-tensors":
            return False
        if quant_format == "nvfp4-pack-quantized":
            return True
        if quant_format != "mixed-precision" or not groups:
            return False
        return any(g.get("format") == "nvfp4-pack-quantized" for g in groups.values() if isinstance(g, dict))

    @staticmethod
    def _nvfp4_pack(weight: Tensor, scale: Tensor) -> tuple[np.ndarray, list[int]]:
        """Repack NVFP4 ModelOpt tensors into ggml super-block layout.
        Preserves original E4M3 scale bits as UE4M3 (strip sign bit).
        The per-tensor scale2 factor is stored as a separate tensor and applied at inference time via ggml_mul().
        Returns (raw_data, logical_shape)."""

        out_features = weight.shape[0]
        n_blocks = scale.shape[1]

        # Unpack ModelOpt nibble-packed weights
        w = weight.reshape(out_features, n_blocks, 8)
        vals = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(out_features, n_blocks, 16)

        # Preserve original E4M3 scale bits as UE4M3 (strip sign bit)
        d_ue = scale.view(torch.uint8).numpy().reshape(out_features, n_blocks) & 0x7F
        qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).to(torch.uint8).numpy()

        # Pack into super-blocks: [4 UE4M3 scales, 32 qs bytes] = 36 bytes per 64 elements
        n_super = n_blocks // 4
        d_grouped = d_ue.reshape(out_features, n_super, 4)
        qs_grouped = qs.reshape(out_features, n_super, 4, 8).reshape(out_features, n_super, 32)
        raw = np.concatenate([d_grouped, qs_grouped], axis=-1).reshape(out_features, n_super * 36)
        return raw, [out_features, n_super * 64]

    def _repack_nvfp4(self, name: str, weight: Tensor, scale: Tensor, scale2: Tensor, input_scale: Tensor):
        new_name = self.map_tensor_name(name)

        raw, shape = self._nvfp4_pack(weight, scale)
        logger.info(f"Repacked {new_name} with shape {shape} and quantization NVFP4")
        self.gguf_writer.add_tensor(new_name, raw, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        self._tag_prec_a4(name, new_name)

        self._write_scale_tensor(new_name.replace(".weight", ".scale"), scale2)
        self._write_scale_tensor(new_name.replace(".weight", ".input_scale"), input_scale)

    def _generate_nvfp4_tensors(self):
        # Per-layer expert merging to avoid holding all experts in memory
        expert_blocks: dict[tuple[int, str], list[tuple[int, np.ndarray]]] = {}
        expert_scales: dict[tuple[int, str], list[tuple[int, float]]] = {}
        expert_input_scales: dict[tuple[int, str], list[tuple[int, float]]] = {}
        expert_shapes: dict[tuple[int, str], list[int]] = {}
        expert_names: dict[tuple[int, str], str] = {}
        n_experts = self.find_hparam(["num_local_experts", "num_experts", "n_routed_experts"], optional=True) or 0
        consumed: list[str] = []

        for name in self.model_tensors.keys():
            if not name.endswith(".weight"):
                continue
            scale_name = name.replace(".weight", ".weight_scale")
            scale2_name = name.replace(".weight", ".weight_scale_2")
            input_scale_name = name.replace(".weight", ".input_scale")
            if scale_name not in self.model_tensors:
                continue
            weight = self.model_tensors[name]()
            scale = self.model_tensors[scale_name]()

            # Leave FP8 weights and their scales for _prepare_fp8_e4m3_tensors.
            if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or scale.ndim < 2:
                continue
            if weight.ndim != 2 or scale.shape[0] != weight.shape[0] or scale.shape[1] * 8 != weight.shape[1]:
                raise ValueError(f"NVFP4 weight {name!r} has incompatible shapes: weight {list(weight.shape)}, scale {list(scale.shape)}")

            weight = LazyTorchTensor.to_eager(weight)
            scale = LazyTorchTensor.to_eager(scale)
            scale2 = LazyTorchTensor.to_eager(self.model_tensors.get(scale2_name, lambda: torch.tensor(1.0))())
            input_scale = LazyTorchTensor.to_eager(self.model_tensors.get(input_scale_name, lambda: torch.tensor(1.0))())

            # Mark tensors for removal from model_tensors (already written to gguf)
            consumed.extend([name, scale_name])
            if scale2_name in self.model_tensors:
                consumed.append(scale2_name)
            if input_scale_name in self.model_tensors:
                consumed.append(input_scale_name)

            # Check if this is a per-expert tensor
            m = re.search(r'\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$', name)
            if m:
                expert_id = int(m.group(1))
                proj_type = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}.get(m.group(2), m.group(2))
                bid_m = re.search(r'\.layers\.(\d+)\.', name)
                bid = int(bid_m.group(1)) if bid_m else 0
                key = (bid, proj_type)

                raw, shape = self._nvfp4_pack(weight, scale)

                if key not in expert_blocks:
                    expert_blocks[key] = []
                    expert_scales[key] = []
                    expert_input_scales[key] = []
                    expert_shapes[key] = shape
                    expert_names[key] = name
                expert_blocks[key].append((expert_id, raw.copy()))
                # Collect per-expert scale2 (scalar per expert)
                expert_scales[key].append((expert_id, float(scale2.float().sum())))
                # Collect per-expert input_scale (scalar per expert)
                expert_input_scales[key].append((expert_id, float(input_scale.float().sum())))

                # Flush when all experts for this (layer, proj) are collected
                if n_experts > 0 and len(expert_blocks[key]) >= n_experts:
                    self._flush_nvfp4_experts(key, expert_blocks, expert_scales, expert_input_scales, expert_shapes, expert_names, bid, proj_type)
            else:
                self._repack_nvfp4(name, weight, scale, scale2, input_scale)

        # Flush any remaining experts (fallback if n_experts was unknown)
        for bid, proj_type in list(expert_blocks.keys()):
            self._flush_nvfp4_experts((bid, proj_type), expert_blocks, expert_scales, expert_input_scales, expert_shapes, expert_names, bid, proj_type)

        # Remove consumed tensors so get_tensors/modify_tensors won't see them
        for name in consumed:
            self.model_tensors.pop(name, None)

        # Remove any remaining unused auxiliary tensors
        for name in list(self.model_tensors.keys()):
            if name.endswith((".k_scale", ".v_scale")):
                del self.model_tensors[name]

    def _flush_nvfp4_experts(self, key, expert_blocks, expert_scales, expert_input_scales, expert_shapes, expert_names, bid, proj_type):
        experts = expert_blocks.pop(key)
        scales = expert_scales.pop(key)
        input_scales = expert_input_scales.pop(key)
        shape = expert_shapes.pop(key)

        experts.sort(key=lambda x: x[0])
        merged = np.stack([e[1] for e in experts], axis=0)
        merged_name = f"model.layers.{bid}.mlp.experts.{proj_type}.weight"
        new_name = self.map_tensor_name(merged_name)
        logger.info(f"Repacked {new_name} with shape [{len(experts)}, {shape[0]}, {shape[1]}] and quantization NVFP4")
        self.gguf_writer.add_tensor(new_name, merged, raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        self._tag_prec_a4(expert_names.pop(key), new_name)

        scales.sort(key=lambda x: x[0])
        self._write_scales_tensor(new_name.replace(".weight", ".scale"), [s[1] for s in scales])

        input_scales.sort(key=lambda x: x[0])
        self._write_scales_tensor(new_name.replace(".weight", ".input_scale"), [s[1] for s in input_scales])

        del experts, merged

    def prepare_tensors(self):
        # detect NVFP4 quantization (ModelOpt and Compressed-tensors formats)
        quantization_config = self.hparams.get("quantization_config") or {}
        quant_algo = quantization_config.get("quant_algo")
        quant_method = quantization_config.get("quant_method")
        quant_format = quantization_config.get("format")
        quant_groups = quantization_config.get("config_groups") or {}
        quant_layers = quantization_config.get("quantized_layers") or {}
        quant_config_file = self.dir_model / "hf_quant_config.json"

        if (not quant_algo or not quant_layers) and quant_config_file.is_file():
            with open(quant_config_file, "r", encoding="utf-8") as f:
                hf_quant_config = json.load(f)
                quant_config = hf_quant_config.get("quantization") or {}
                producer = hf_quant_config.get("producer") or {}
                producer_name = (producer.get("name") or "").lower()
                if quant_method is None:
                    self.hparams.setdefault("quantization_config", {})["quant_method"] = producer_name
                    quant_method = producer_name
                quant_algo = quant_config.get("quant_algo", quant_algo)
                quant_method = quant_config.get("quant_method", quant_method)
                quant_format = quant_config.get("format", quant_format)
                quant_groups = quant_config.get("config_groups", quant_groups) or {}
                quant_layers = quant_config.get("quantized_layers", quant_layers) or {}

        nvfp4_compressed_tensors = self._is_nvfp4_compressed_tensors(quant_method, quant_format, quant_groups)
        # e.g. MXFP8 dense weights with compressed-tensors NVFP4 experts
        nvfp4_compressed_tensors |= quantization_config.get("moe_quant_format") == "nvfp4-pack-quantized"

        self._nvfp4_global_algo = quant_algo

        if quant_algo != "NVFP4":
            if nvfp4_compressed_tensors:
                quant_algo = "NVFP4"
            elif any(str(v.get("quant_algo")).endswith("NVFP4") for v in quant_layers.values() if isinstance(v, dict)):
                quant_algo = "NVFP4"

        self._is_nvfp4 = quant_algo in ("NVFP4", "W4A16_NVFP4")
        self._is_mxfp4 = quant_method == "mxfp4"

        # Per-tensor NVFP4 precision.
        self._nvfp4_layer_algo = {}
        if quant_layers:
            # quantized_layers keys are module paths in the source checkpoint, not tensor names
            # resolve each source tensor to the closest quantized_layers parent and record the algo under its filtered name for _tag_prec_a4
            layer_algo = {k: v.get("quant_algo") for k, v in quant_layers.items() if isinstance(v, dict)}
            matched: set[str] = set()
            for src_name, filtered_name in self._source_tensor_names.items():
                module = src_name
                while module := module.rpartition(".")[0]:
                    if module in layer_algo:
                        break
                if not module:
                    continue

                matched.add(module)
                # filter_tensors may drop a tensors so we need to check for that here.
                if filtered_name is not None:
                    self._nvfp4_layer_algo[filtered_name.removesuffix(".weight").removesuffix(".bias")] = layer_algo[module]

            unmatched = layer_algo.keys() - matched
            assert not unmatched, f"quantized_layers entries not in the model tensors: {sorted(unmatched)}"

        self._prepare_kv_cache_scales()

        # NVFP4 weights are repacked and written directly to gguf_writer.
        # This must run before dequant_model so NVFP4 tensors are removed
        # from model_tensors, leaving only non-NVFP4 (e.g. FP8) for dequant.
        if self._is_nvfp4:
            if nvfp4_compressed_tensors:
                # Convert compressed-tensors 'global' scales into the reciprocal
                def inverse_scale(gen):
                    def load():
                        scale = LazyTorchTensor.to_eager(gen()).float()
                        return 1.0 / scale
                    return load

                # Change the compressed-tensors names to the ModelOpt names for handling consistently later
                for name in list(self.model_tensors.keys()):
                    if name.endswith(".weight_packed"):
                        weight_name = name.removesuffix("_packed")
                        if weight_name not in self.model_tensors:
                            self.model_tensors[weight_name] = self.model_tensors.pop(name)
                    elif name.endswith(".weight_global_scale"):
                        scale2_name = name.replace(".weight_global_scale", ".weight_scale_2")
                        if scale2_name not in self.model_tensors:
                            self.model_tensors[scale2_name] = inverse_scale(self.model_tensors.pop(name))
                    elif name.endswith(".input_global_scale"):
                        input_scale_name = name.replace(".input_global_scale", ".input_scale")
                        if input_scale_name not in self.model_tensors:
                            self.model_tensors[input_scale_name] = inverse_scale(self.model_tensors.pop(name))
            self._generate_nvfp4_tensors()

        self._prepare_fp8_e4m3_tensors()
        self.dequant_model()

        self.prepare_gate_up_fusion()

        # Handle empty tensor_map for models with block_count=0 (like MobileNetV5)
        if self.tensor_map.mapping:
            max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(".weight,")
        else:
            max_name_len = len("vision_encoder.weight,")  # Default reasonable length

        for name, data_torch in chain(self.generate_extra_tensors(), self.get_tensors()):
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
                continue

            # FP8 transforms must only reorder values; norms and vectors use numeric F32 transforms.
            preserve_fp8_e4m3 = self._preserve_fp8 and data_torch.dtype == torch.float8_e4m3fn and data_torch.ndim >= 2 and not name.endswith(("norm.weight", "norm_f.weight"))
            if preserve_fp8_e4m3:
                for new_name in self._map_fp8_weight_names(name):
                    mapped_bid = next((int(part) for part in new_name.split(".") if part.isdecimal()), None)
                    if self._tensor_requires_f32(new_name, mapped_bid, data_torch.ndim):
                        preserve_fp8_e4m3 = False
                        break
                if preserve_fp8_e4m3:
                    self._fp8_e4m3_preserved.add(name)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32) and name not in self._fp8_e4m3_preserved:
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in (self.modify_tensors(data_torch, name, bid)):
                n_dims = len(data_torch.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(name, new_name, bid, n_dims)

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and self._tensor_requires_f32(new_name, bid, n_dims):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_TQ1_0,
                        gguf.LlamaFileType.MOSTLY_TQ2_0,
                    ):
                        # TODO: use Q4_K and Q6_K
                        data_qtype = gguf.GGMLQuantizationType.F16

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ1_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ1_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ2_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ2_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                if data_torch.dtype == torch.float8_e4m3fn:
                    data = data_torch.view(torch.uint8).numpy() if data_qtype == gguf.GGMLQuantizationType.F8_E4M3 else data_torch.float().numpy()
                else:
                    data = data_torch.numpy()

                # a chunked tensor quantizes as one chunk at a time, while it is written
                quantize = data.quantize if isinstance(data, gguf.LazyChunkedTensor) else (
                    lambda qtype, d=data: gguf.quants.quantize(d, qtype))

                if not (data_qtype == gguf.GGMLQuantizationType.F8_E4M3 and data.dtype == np.uint8):
                    try:
                        data = quantize(data_qtype)
                    except gguf.QuantError as e:
                        logger.warning("%s, %s", e, "falling back to F16")
                        data_qtype = gguf.GGMLQuantizationType.F16
                        data = quantize(data_qtype)

                shape = gguf.quant_shape_from_byte_shape(data.shape, data_qtype) if data.dtype == np.uint8 else data.shape

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}")

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):

        total_params, shared_params, expert_params, expert_count = self.gguf_writer.get_total_parameter_count()

        self.metadata = gguf.Metadata.load(self.metadata_override, self.dir_model_card, self.model_name, total_params)

        # If we are using HF model id, set the metadata name to the model id
        if self.remote_hf_model_id:
            self.metadata.name = self.remote_hf_model_id

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        if self.ftype in (gguf.LlamaFileType.ALL_F32, gguf.LlamaFileType.MOSTLY_F16, gguf.LlamaFileType.MOSTLY_BF16):
            if self._is_nvfp4:
                self.ftype = gguf.LlamaFileType.MOSTLY_NVFP4
            elif self._is_mxfp4:
                self.ftype = gguf.LlamaFileType.MOSTLY_MXFP4_MOE
            elif self._fp8_e4m3_preserved:
                self.ftype = gguf.LlamaFileType.MOSTLY_F8_E4M3

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(total_params, shared_params, expert_params, expert_count)

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

        self.check_no_hadamard()

        if self._prec_a4:
            names = sorted(self._prec_a4.keys())
            values = [self._prec_a4[n] for n in names]
            logger.info(f"Set prec_a4 metadata for {len(names)} tensor(s)")
            self.gguf_writer.add_tensor_extra_prec_a4(names, values)

    def write_vocab(self):
        raise NotImplementedError("write_vocab() must be implemented in subclasses")

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams_guess(dir_model: Path) -> dict[str, Any] | None:
        # some models ship no config.json, will try to guess them
        from conversion import load_all_models
        load_all_models()

        for matcher, loader in ModelBase._hparams_loaders:
            if matcher(dir_model):
                return loader(dir_model)
        return None

    @classmethod
    def register_hparams_loader(cls, matcher: HparamsMatcher) -> Callable[[HparamsLoader], HparamsLoader]:
        def inner(loader: HparamsLoader) -> HparamsLoader:
            cls._hparams_loaders.append((matcher, loader))
            return loader
        return inner

    @staticmethod
    def load_hparams(dir_model: Path, guess: bool = True):
        # checkpoints with a non-HF layout are matched by their own loader
        # models with a HF layout can also register a hparams loader to switch to a custom class
        config = ModelBase.load_hparams_guess(dir_model) if guess and dir_model.is_dir() else None
        if config is not None:
            return config

        try:
            # for security reason, we don't allow loading remote code by default
            # if a model need remote code, we will fallback to config.json
            config = AutoConfig.from_pretrained(dir_model, trust_remote_code=False).to_dict()
        except Exception as e:
            logger.warning(f"Failed to load model config from {dir_model}: {e}")
            logger.warning("Trying to load config.json instead")
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        return config

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            model_type = ModelType.MMPROJ if modelcls.model_arch == gguf.MODEL_ARCH.MMPROJ else ModelType.TEXT
            for name in names:
                cls._model_classes[model_type][name] = modelcls
            return modelcls
        return func

    @classmethod
    def example(cls, *hf_repos: str) -> Callable[[AnyModel], AnyModel]:
        del hf_repos  # unused

        def func(modelcls: AnyModel) -> AnyModel:
            return modelcls
        return func

    @classmethod
    def print_registered_models(cls):
        for model_type, model_classes in cls._model_classes.items():
            logger.error(f"{model_type.name} models:")
            for name in sorted(model_classes.keys()):
                logger.error(f"  - {name}")

    @classmethod
    def from_model_architecture(cls, arch: str, model_type = ModelType.TEXT) -> type[ModelBase]:
        try:
            return cls._model_classes[model_type][arch]
        except KeyError:
            raise NotImplementedError(f'Architecture {arch!r} not supported!') from None


class TextModel(ModelBase):
    model_type = ModelType.TEXT
    hf_arch: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_arch = get_model_architecture(self.hparams, self.model_type)

        if "text_config" in self.hparams:
            # move the text_config to the root level
            self.hparams = {**self.hparams, **self.hparams["text_config"]}

        self.block_count = self.find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self.rope_parameters = self.hparams.get("rope_parameters", self.hparams.get("rope_scaling")) or {}

        rope_theta = self.find_hparam(["global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base"], optional=True)
        local_rope_theta = self.find_hparam(["local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq"], optional=True)
        partial_rotary_factor = self.find_hparam(["partial_rotary_factor", "rope_pct", "rope_percent"], optional=True)
        original_max_position_embeddings = self.find_hparam(["original_max_position_embeddings"], optional=True)

        # Ensure global params are mirrored in rope_parameters
        if "full_attention" not in self.rope_parameters and "sliding_attention" not in self.rope_parameters:
            if local_rope_theta is not None:
                self.rope_parameters["sliding_attention"] = {"rope_theta": local_rope_theta}
            if "rope_theta" not in self.rope_parameters and rope_theta is not None:
                self.rope_parameters["rope_theta"] = rope_theta
            if "rope_type" not in self.rope_parameters and (rope_type := self.rope_parameters.get("type")) is not None:
                self.rope_parameters["rope_type"] = rope_type
            if "partial_rotary_factor" not in self.rope_parameters and partial_rotary_factor is not None:
                self.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
            if "original_max_position_embeddings" not in self.rope_parameters and original_max_position_embeddings is not None:
                self.rope_parameters["original_max_position_embeddings"] = original_max_position_embeddings

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip multimodal tensors
        # strip the "model." wrapper so the prefixes below match (name is not returned)
        if name.startswith("model."):
            name = name[len("model."):]
        if name.startswith(("mlp", "vit.", "vpm.", "siglip2.", "conformer.", "connector.", "merger.", "resampler.", "sound_encoder.", "sound_projection.", "speech_embeddings.")) \
                or "visual." in name or "vision." in name or "audio." in name or "talker." in name \
                or "vision_" in name or "audio_" in name \
                or "token2wav." in name or "code2wav." in name \
                or "projector." in name or "pre_mm_projector_norm" in name \
                or "image_newline" in name or "view_seperator" in name \
                or "patch_embed" in name or "patch_embedding" in name \
                or "patch_merger." in name or "patch_merge_mlp." in name:
            return None

        return super().filter_tensors(item)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        total_params = self.gguf_writer.get_total_parameter_count()[0]
        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, self.metadata.size_label, output_type, model_type="LoRA" if total_params < 0 else None)
            else:
                fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=None, model_type="vocab")

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

        logger.info("Set model tokenizer")
        self.set_vocab()

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (n_ctx := self.find_hparam(["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"], optional=True)) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        if (n_embd := self.find_hparam(["hidden_size", "n_embd", "dim"], optional=True)) is not None:
            self.gguf_writer.add_embedding_length(n_embd)
            logger.info(f"gguf: embedding length = {n_embd}")

        if (n_ff := self.find_hparam(["prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim"], optional=True)) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        if (n_head := self.find_hparam(["num_attention_heads", "n_head", "n_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count(n_head)
            logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.find_hparam(["num_key_value_heads", "n_kv_heads"], optional=True)) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if self.hparams.get("is_causal") is False or self.hparams.get("use_bidirectional_attention") is True:
            self.gguf_writer.add_causal_attention(False)
            logger.info("gguf: causal attention = False")

        # TODO: Handle "sliding_attention" similarly when models start implementing it
        rope_params = self.rope_parameters.get("full_attention", self.rope_parameters)
        if (rope_type := rope_params.get("rope_type")) is not None:
            rope_factor = rope_params.get("factor")
            rope_gguf_type = gguf.RopeScalingType.NONE
            if rope_type == "linear" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.LINEAR
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
            elif rope_type == "yarn" and rope_factor is not None:
                rope_gguf_type = gguf.RopeScalingType.YARN
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
                self.gguf_writer.add_rope_scaling_factor(rope_factor)
                self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_params["original_max_position_embeddings"])
                if (truncate := rope_params.get("truncate")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_truncate(truncate)
                if (yarn_ext_factor := rope_params.get("extrapolation_factor")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_ext_factor(yarn_ext_factor)
                if (yarn_attn_factor := rope_params.get("attention_factor", rope_params.get("attn_factor"))) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_attn_factor(yarn_attn_factor)
                if (yarn_beta_fast := rope_params.get("beta_fast")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_fast(yarn_beta_fast)
                if (yarn_beta_slow := rope_params.get("beta_slow")) is not None:
                    self.gguf_writer.add_rope_scaling_yarn_beta_slow(yarn_beta_slow)
                # self.gguf_writer.add_rope_scaling_yarn_log_mul(rope_params["mscale_all_dim"])
            elif rope_type == "su" or rope_type == "longrope":
                rope_gguf_type = gguf.RopeScalingType.LONGROPE
                self.gguf_writer.add_rope_scaling_type(rope_gguf_type)
            elif rope_type == "dynamic":
                # HunYuan, handled in model class
                pass
            elif rope_type.lower() == "llama3":
                # Handled in generate_extra_tensors
                pass
            else:
                logger.warning(f"Unknown RoPE type: {rope_type}")
            logger.info(f"gguf: rope scaling type = {rope_gguf_type.name}")

        if "mrope_section" in self.rope_parameters:
            mrope_section = self.rope_parameters["mrope_section"]
            # Pad to 4 dimensions [time, height, width, extra]
            while len(mrope_section) < 4:
                mrope_section.append(0)
            self.gguf_writer.add_rope_dimension_sections(mrope_section[:4])
            logger.info(f"gguf: mrope sections: {mrope_section[:4]}")

        if (rope_theta := rope_params.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (local_rope_theta := self.rope_parameters.get("sliding_attention", {}).get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base_swa(local_rope_theta)
            logger.info(f"gguf: rope theta swa = {local_rope_theta}")
        # Add rope_pattern from no_rope_layers
        if no_rope_layers := self.hparams.get("no_rope_layers"):
            if len(no_rope_layers) == self.block_count:  # "no_rope_layers": [0, 0, 0, 1]
                rope_pattern = [bool(x) for x in no_rope_layers]
            else:  # "no_rope_layers": [3]
                rope_pattern = [i not in no_rope_layers for i in range(self.block_count)]
            self.gguf_writer.add_rope_pattern(rope_pattern)
            logger.info(f"gguf: rope_pattern = {sum(rope_pattern)} RoPE layers / {len(rope_pattern)} total")
        if (f_rms_eps := self.find_hparam(["rms_norm_eps", "norm_eps"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (f_norm_eps := self.find_hparam(["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True)) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.find_hparam(["num_local_experts", "num_experts", "n_routed_experts"], optional=True)) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.find_hparam(["num_experts_per_tok", "num_experts_per_token", "top_k_experts"], optional=True)) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")
        if (n_expert_groups := self.hparams.get("n_group")) is not None:
            self.gguf_writer.add_expert_group_count(n_expert_groups)
            logger.info(f"gguf: expert groups count = {n_expert_groups}")
        if (n_group_used := self.hparams.get("topk_group")) is not None:
            self.gguf_writer.add_expert_group_used_count(n_group_used)
            logger.info(f"gguf: expert groups used count = {n_group_used}")

        if (score_func := self.find_hparam(["score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func", "expert_selection_fn", "router_score_func", "router_score_function"], optional=True)) is not None:
            if score_func == "sigmoid":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
            elif score_func == "softmax":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
            elif score_func == "sqrtsoftplus":
                self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)
            else:
                raise ValueError(f"Unsupported expert score gating function value: {score_func}")
            logger.info(f"gguf: expert score gating function = {score_func}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError('Splitting the vocabulary is not supported')

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>", "<2mass>", "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
        seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        assert tokenizer is not None
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
        added_vocab = tokenizer.get_added_vocab()

        added_tokens_decoder = tokenizer.added_tokens_decoder

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    # The tokenizer in llama.cpp assumes the CONTROL and USER_DEFINED tokens are pre-normalized.
                    # To avoid unexpected issues - we make sure to normalize non-normalized tokens
                    if not added_tokens_decoder[i].normalized:
                        previous_token = token
                        token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))  # ty: ignore[invalid-assignment]
                        if previous_token != token:
                            logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using AutoTokenizer")

                    if added_tokens_decoder[i].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        # NOTE: this was added for Gemma.
                        # Encoding and decoding the tokens above isn't sufficient for this case.
                        token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggml-org/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f":
            # ref: https://huggingface.co/openbmb/MiniCPM-V-4_6
            res = "qwen35"
        if chkhsh == "5635e1c8d621dff7b4825f41f2f97b6ceb39bcf2d5c541db28577080d6efd5e5":
            # ref: https://huggingface.co/thinkingmachines/Inkling-2-Small
            res = "inkling2"
        if chkhsh == "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206":
            # ref: https://huggingface.co/inclusionAI/Ling-mini-base-2.0
            res = "bailingmoe2"
        if chkhsh == "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4":
            # ref: https://huggingface.co/Qwen/Qwen3.5-9B-Instruct
            res = "qwen35"

        if res is None:
            logger.warning("\n")
            logger.warning("**************************************************************************************")
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning("**          - the model has not been added to convert_hf_to_gguf_update.py yet")
            logger.warning("**          - the pre-tokenization config has changed upstream")
            logger.warning("**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly.")
            logger.warning("** ref:     https://github.com/ggml-org/llama.cpp/pull/6920")
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning("**************************************************************************************")
            logger.warning("\n")
            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_none(self) -> None:
        self.gguf_writer.add_tokenizer_model("none")

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _try_set_pooling_type(self) -> None:
        # get pooling path
        pooling_path = None
        module_path = self.dir_model / "modules.json"
        if module_path.is_file():
            with open(module_path, encoding="utf-8") as f:
                modules = json.load(f)
            for mod in modules:
                if mod["type"].endswith("Pooling"):
                    pooling_path = mod["path"]
                    break

        mode_mapping = {
            "mean": gguf.PoolingType.MEAN,
            "cls": gguf.PoolingType.CLS,
            "lasttoken": gguf.PoolingType.LAST,
        }

        # get pooling type
        if pooling_path is not None:
            with open(self.dir_model / pooling_path / "config.json", encoding="utf-8") as f:
                pooling = json.load(f)
            if pooling.get("pooling_mode_mean_tokens"):
                pooling_type = gguf.PoolingType.MEAN
            elif pooling.get("pooling_mode_cls_token"):
                pooling_type = gguf.PoolingType.CLS
            elif pooling.get("pooling_mode_lasttoken"):
                pooling_type = gguf.PoolingType.LAST
            elif (pooling_mode := pooling.get("pooling_mode")) in mode_mapping:
                pooling_type = mode_mapping[pooling_mode]
            else:
                raise NotImplementedError("Only MEAN, CLS, and LAST pooling types supported")
            self.gguf_writer.add_pooling_type(pooling_type)
        else:
            embedding_config_path = self.dir_model / "embedding_config.json"
            if embedding_config_path.is_file():
                with open(embedding_config_path, encoding="utf-8") as f:
                    embedding_config = json.load(f)
                pooling = embedding_config.get("pooling")
                if pooling == "last_token":
                    self.gguf_writer.add_pooling_type(gguf.PoolingType.LAST)
                elif pooling is not None:
                    raise NotImplementedError(f"unsupported embedding_config.json pooling {pooling!r}")


class MmprojModel(ModelBase):
    model_type = ModelType.MMPROJ
    model_arch = gguf.MODEL_ARCH.MMPROJ
    preprocessor_config: dict[str, Any]
    global_config: dict[str, Any]

    n_block_keys = ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth", "layers", "encoder_layers", "vt_num_hidden_layers"]

    has_vision_encoder: bool = True # by default
    has_audio_encoder: bool = False

    # for models having multiple encoders, we need to separate their hparams
    hparams_vision: dict[str, Any] | None = None
    hparams_audio: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model_arch != gguf.MODEL_ARCH.MMPROJ:
            raise TypeError("MmprojModel must be subclassed with model_arch = gguf.MODEL_ARCH.MMPROJ")

        # get n_embd of the text model
        if "text_config" not in self.hparams:
            self.hparams["text_config"] = {}
        if "audio_config" not in self.hparams:
            self.hparams["audio_config"] = {}
        text_config = {**self.hparams, **self.hparams["text_config"]}
        self.n_embd_text = text_config.get("hidden_size", text_config.get("n_embd", 0))

        assert self.n_embd_text > 0, "n_embd not found in hparams"

        # move vision config to the top level, while preserving the original hparams in global_config
        import copy
        self.global_config = copy.deepcopy(self.hparams)
        self.hparams_vision = self.get_vision_config()
        self.hparams_audio = self.get_audio_config()

        if self.hparams_vision is None and self.hparams_audio is None:
            raise ValueError("vision_config / audio_config not found in hparams")

        # for compat with vision-only models
        self.hparams = self.hparams_vision or self.hparams_audio or self.hparams

        # TODO @ngxson : this is a hack to support both vision and audio encoders
        have_multiple_encoders = self.has_audio_encoder and self.has_vision_encoder
        self.block_count = 128 if have_multiple_encoders else self.find_hparam(self.n_block_keys, True)
        self.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, self.block_count)

        # load preprocessor config
        self.preprocessor_config = {}

        # prefer preprocessor_config.json if possible
        preprocessor_config_path = self.dir_model / "preprocessor_config.json"
        if preprocessor_config_path.is_file():
            with open(preprocessor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move media_proc_cfg to root level for compat
                if "media_proc_cfg" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["media_proc_cfg"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

        # prefer processor_config.json if possible
        processor_config_path = self.dir_model / "processor_config.json"
        if processor_config_path.is_file():
            with open(processor_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                # move image_processor to root level for compat
                if "image_processor" in cfg:
                    cfg = {
                        **cfg,
                        **cfg["image_processor"],
                    }
                # merge configs
                self.preprocessor_config = {**self.preprocessor_config, **cfg}

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # Skip non-multimodal tensors
        if "language_model." in name:
            return None

        return super().filter_tensors(item)

    def get_vision_config(self) -> dict[str, Any] | None:
        return self.global_config.get("vision_config")

    def get_audio_config(self) -> dict[str, Any] | None:
        mm_config_key = "whisper_config" if "whisper_config" in self.hparams else "audio_config"
        return self.global_config.get(mm_config_key)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MMPROJ)

    def prepare_metadata(self, vocab_only: bool):
        super().prepare_metadata(vocab_only=vocab_only)

        output_type: str = self.ftype.name.partition("_")[2]

        if self.fname_out.is_dir():
            fname_default: str = gguf.naming_convention(self.metadata.name, self.metadata.basename, self.metadata.finetune, self.metadata.version, size_label=None, output_type=output_type, model_type=None)
            self.fname_out = self.fname_out / f"mmproj-{fname_default}.gguf"
        else:
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(self.fname_out.name, output_type)

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)

        if self.has_vision_encoder:
            self.gguf_writer.add_clip_has_vision_encoder(True)
            self.gguf_writer.add_vision_projection_dim(self.n_embd_text)

            # vision config
            self.image_size = self.find_vparam(["image_size"])
            self.gguf_writer.add_vision_image_size(self.image_size)
            self.gguf_writer.add_vision_patch_size(self.find_vparam(["patch_size"]))
            self.gguf_writer.add_vision_embedding_length(self.find_vparam(["hidden_size", "width", "vt_hidden_size"]))
            self.gguf_writer.add_vision_feed_forward_length(self.find_vparam(["intermediate_size", "vt_intermediate_size"]))
            self.gguf_writer.add_vision_block_count(self.find_vparam(self.n_block_keys))
            self.gguf_writer.add_vision_head_count(self.find_vparam(["num_attention_heads", "num_heads", "heads", "vt_num_attention_heads"]))

            # preprocessor config
            image_mean = self.preprocessor_config["image_mean"]
            image_std = self.preprocessor_config["image_std"]

            self.gguf_writer.add_vision_image_mean(image_mean)
            self.gguf_writer.add_vision_image_std(image_std)

        if self.has_audio_encoder:
            self.gguf_writer.add_clip_has_audio_encoder(True)
            self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

            # audio config
            self.gguf_writer.add_audio_embedding_length(self.find_aparam(["hidden_size"]))
            self.gguf_writer.add_audio_feed_forward_length(self.find_aparam(["intermediate_size"]))
            self.gguf_writer.add_audio_block_count(self.find_aparam(self.n_block_keys))
            self.gguf_writer.add_audio_head_count(self.find_aparam(["num_attention_heads"]))

        if not self.has_vision_encoder and not self.has_audio_encoder:
            raise ValueError("MmprojModel must have either vision or audio encoder")

    def write_vocab(self):
        raise ValueError("MmprojModel does not support vocab writing")

    def find_vparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_vision is not None
        return self._find_param(self.hparams_vision, keys, optional)

    def find_aparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        assert self.hparams_audio is not None
        return self._find_param(self.hparams_audio, keys, optional)

    def _find_param(self, obj: dict[str, Any], keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in obj), None)
        if key is not None:
            return obj[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".patch_embd.weight" in new_name or ".patch_merger.weight" in new_name:
            return gguf.GGMLQuantizationType.F16 if self.ftype == gguf.LlamaFileType.MOSTLY_F16 else gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.uint8: np.uint8,
        torch.int64: np.int64,
    }

    # only used when byteswapping data. Only correct size is needed
    # TODO: uncomment uint64, uint32, and uint16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_byteswap_map: dict[torch.dtype, type] = {
        torch.float64: np.float64,
        torch.float32: np.float32,
        torch.bfloat16: np.float16,
        torch.float16: np.float16,
        torch.int64: np.int64,
        # torch.uint64: np.uint64,
        torch.int32: np.int32,
        # torch.uint32: np.uint32,
        torch.int16: np.int16,
        # torch.uint16: np.uint16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.uint8,
        torch.float8_e4m3fn: np.uint8,
        torch.float8_e5m2: np.uint8,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[...] if len(s.get_shape()) == 0 else s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_local_tensor(cls, t: gguf.utility.LocalTensor) -> Tensor:
        def load_tensor(tensor: gguf.utility.LocalTensor) -> Tensor:
            def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
                if sys.byteorder == 'big':
                    # switch data back to big endian
                    tensor = tensor.view(dtype).byteswap(inplace=False)
                return tensor
            dtype = cls._dtype_str_map[tensor.dtype]
            numpy_dtype = cls._dtype_byteswap_map[dtype]
            return torch.from_numpy(byteswap_tensor(tensor.mmap_bytes(), numpy_dtype)).view(dtype).reshape(tensor.shape)
        dtype = cls._dtype_str_map[t.dtype]
        shape = t.shape
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(t,), func=lambda r: load_tensor(r))
        return cast(torch.Tensor, lazy)

    @classmethod
    def from_remote_tensor(cls, remote_tensor: gguf.utility.RemoteTensor):
        def byteswap_tensor(tensor: np.ndarray, dtype: type) -> np.ndarray:
            if sys.byteorder == 'big':
                # switch data back to big endian
                tensor = tensor.view(dtype).byteswap(inplace=False)
            return tensor
        dtype = cls._dtype_str_map[remote_tensor.dtype]
        numpy_dtype = cls._dtype_byteswap_map[dtype]
        shape = remote_tensor.shape
        meta = cls.meta_with_dtype_and_shape(dtype, shape)
        lazy = cls(meta=meta, args=(remote_tensor,), func=lambda r: torch.from_numpy(byteswap_tensor(np.frombuffer(r.data(), dtype=numpy_dtype), numpy_dtype)).view(dtype).reshape(shape))
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            assert len(args)
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


if hasattr(torch, "float8_e8m0fnu"):
    _torch_float8_e8m0 = torch.float8_e8m0fnu
    LazyTorchTensor._dtype_map[_torch_float8_e8m0] = np.uint8
    LazyTorchTensor._dtype_byteswap_map[_torch_float8_e8m0] = np.uint8
    LazyTorchTensor._dtype_str_map["F8_E8M0"] = _torch_float8_e8m0
else:
    # Older torch builds do not expose F8_E8M0. Keep the raw bytes so callers
    # that know the format can decode them explicitly.
    LazyTorchTensor._dtype_str_map["F8_E8M0"] = torch.uint8


def get_model_architecture(hparams: dict[str, Any], model_type: ModelType) -> str:
    text_config = hparams.get("text_config", {})
    vision_config = hparams.get("vision_config", {})
    arch = None
    if (arches := hparams.get("architectures")) is not None and len(arches) > 0:
        arch = arches[0]

    # if "architectures" is found in the sub-config, use that instead
    if model_type == ModelType.TEXT and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    elif model_type == ModelType.MMPROJ and vision_config.get("architectures") is not None:
        arch = vision_config["architectures"][0]
    # some configs only declare the architecture inside text_config
    if arch is None and text_config.get("architectures") is not None:
        arch = text_config["architectures"][0]
    # some configs only have a model_type
    if arch is None:
        arch = hparams.get("model_type")
    if arch is None:
        raise ValueError("Failed to detect model architecture")
    return arch

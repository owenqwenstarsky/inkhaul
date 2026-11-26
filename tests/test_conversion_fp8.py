"""Test FP8 conversion with small synthetic checkpoints; no GPU or model downloads required.

Install dependencies and run from the repository root:
    python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python -m unittest discover -s tests -p 'test_conversion_fp8.py' -v

Use unittest discovery so the repository root is available for imports.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from conversion import get_model_class
from conversion.base import LazyTorchTensor, ModelType, get_model_architecture, gguf
from conversion.bailingmoe3 import BailingMoeV3Model, BailingMoeV3VLModel
from conversion.inkling2 import Inkling2Model, _E4M3, _E2M1
from conversion.kolibri1 import Kolibri1Model, Kolibri11Model


class ConversionTestCase(unittest.TestCase):
    def convert(self, model_class, tensors, eager=False, check_lazy=False, scan_nvfp4=False, ftype=gguf.LlamaFileType.GUESSED, hparams_override=None, with_kv=False, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            return self._convert(Path(directory), model_class, tensors, eager, check_lazy, scan_nvfp4, ftype, hparams_override, with_kv, **kwargs)

    def _convert(self, path, model_class, tensors, eager, check_lazy, scan_nvfp4, ftype, hparams_override, with_kv, **kwargs):
        hparams = {
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_experts": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 2,
            "quantization_config": {"quant_method": "modelopt"},
        }
        hparams.update(hparams_override or {})
        (path / "config.json").write_text(json.dumps(hparams))
        save_file(tensors, str(path / "model.safetensors"))
        model = model_class(path, ftype, path / "out.gguf", hparams=hparams, eager=eager, **kwargs)
        reads: list[str] = []
        if check_lazy:
            self.assertFalse(eager)

            def track(name, gen):
                tensor = gen()

                def load(source):
                    reads.append(name)
                    return LazyTorchTensor.to_eager(source())

                return LazyTorchTensor(meta=torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta"), args=(gen,), func=load)

            model.model_tensors = {
                name: lambda name=name, gen=gen: track(name, gen)
                for name, gen in model.model_tensors.items()
            }
        if scan_nvfp4:
            model._generate_nvfp4_tensors()
            self.assertEqual(reads, [])
        model.prepare_tensors()
        if check_lazy:
            deferred = {name for name, tensor in tensors.items() if tensor.numel() != 1 or name.endswith((".weight", ".bias"))}
            self.assertFalse(deferred.intersection(reads), f"Materialized during preparation: {reads}")
            for name, info in model.gguf_writer.tensors[0].items():
                if name.endswith((".weight", ".bias")):
                    self.assertIsInstance(info.tensor, gguf.LazyNumpyTensor)
        result = {
            name: (info.dtype, gguf.LazyNumpyTensor.to_eager(info.tensor).copy())
            for name, info in model.gguf_writer.tensors[0].items()
        }
        if check_lazy:
            self.assertTrue(deferred.intersection(reads))
        if with_kv:
            model.set_gguf_parameters()
            return result, {key: value.value for key, value in model.gguf_writer.kv_data[0].items()}
        return result


class TestInkling2PackedConversion(unittest.TestCase):
    """Conversion of the packed Inkling2 expert weights (see _dequant_expert_chunk in conversion/inkling2.py).

    The checkpoint stores `mlp.experts.w13_weight` (gate and up rows) and `mlp.experts.w2_weight` (down rows) packed:
    - NVFP4: two 4-bit E2M1 codes per byte, one E4M3 scale per 16 weights, times amax[expert] / outer_scale_denominator
    - MXFP8: one E4M3 byte per weight, one power-of-two scale per 32 weights, stored as the exponent byte (2^(byte - 127))

    What the tests pin:
    - decode: packed_checkpoint() decodes the same bytes independently, with torch float8 views and a literal E2M1 table
    - bit-exactness: the converter output must equal that decode bit for bit, for every outtype
    - row interleave: w13 rows alternate gate, up, gate, up; the GGUF tensor holds all gate rows, then all up rows
    """

    N_EXPERT, N_EMBD, N_FF = 2, 64, 160
    E2M1_VALUES = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]

    @staticmethod
    def gate_then_up(w13):
        """Reorders interleaved w13 rows (gate, up, gate, up) to the GGUF layout (all gate rows, then all up rows)."""
        n_expert, n_row, n_col = w13.shape
        return w13.reshape(n_expert, n_row // 2, 2, n_col).transpose(1, 2).reshape(w13.shape)

    @staticmethod
    def interleaved(gate_up):
        """Inverse of gate_then_up()."""
        n_expert, n_row, n_col = gate_up.shape
        return gate_up.reshape(n_expert, 2, n_row // 2, n_col).transpose(1, 2).reshape(gate_up.shape)

    @staticmethod
    def written_bytes(tensor, path):
        """Streams a lazy tensor to a file like the GGUF writer does and returns the bytes."""
        with path.open("wb") as f:
            tensor.tofile(f)
        return path.read_bytes()

    def nvfp4(self, shape, denominator):
        """Packs one NVFP4 projection. Returns the checkpoint tensors by name suffix and the decoded weights."""
        n_expert, n_row, n_col = shape
        index = torch.arange(n_expert * n_row * n_col).reshape(shape)
        codes = ((index * 7 + index // n_col) % 16).to(torch.uint8)
        packed = codes[:, :, ::2] | (codes[:, :, 1::2] << 4)  # low nibble first
        scales = (torch.arange(n_expert * n_row * (n_col // 16)).reshape(n_expert, n_row, n_col // 16) % 126 + 1).to(torch.uint8).view(torch.float8_e4m3fn)
        amax = torch.linspace(0.7, 1.9, n_expert)
        denominator = torch.tensor([denominator])
        decoded = torch.tensor(self.E2M1_VALUES)[codes.long()] * scales.float().repeat_interleave(16, -1)
        decoded *= (amax / denominator)[:, None, None]
        return {"_amax": amax, "_outer_scale_denominator": denominator, "": packed, "_scale": scales}, decoded

    def mxfp8(self, shape):
        """Packs one MXFP8 projection. Returns the checkpoint tensors by name suffix and the decoded weights."""
        n_expert, n_row, n_col = shape
        index = torch.arange(n_expert * n_row * n_col).reshape(shape)
        packed = ((index * 13 + index // n_col) % 256).to(torch.uint8)
        packed[(packed & 127) == 127] = 126  # 0x7f and 0xff are NaN in E4M3
        packed = packed.view(torch.float8_e4m3fn)
        scales = (torch.arange(n_expert * n_row * (n_col // 32)).reshape(n_expert, n_row, n_col // 32) % 7 + 123).to(torch.uint8)
        scales[0, 0, 0] = 0
        decoded = packed.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        return {"": packed, "_scale": scales}, decoded

    def packed_checkpoint(self):
        """Layer 0 is NVFP4, layer 1 is MXFP8. Returns the checkpoint tensors, the hf_quant_config groups and the expected GGUF tensors."""
        tensors, quant_groups, expected = {}, [], {}
        w13_shape = (self.N_EXPERT, 2 * self.N_FF, self.N_EMBD)
        w2_shape = (self.N_EXPERT, self.N_EMBD, self.N_FF)
        for layer, scheme in enumerate(("nvfp4_nearest", "mxfp8_nearest")):
            names = []
            for projection, shape, target in (("w13", w13_shape, "gate_up"), ("w2", w2_shape, "down")):
                name = f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"
                names.append(name)
                if scheme == "nvfp4_nearest":
                    stored, decoded = self.nvfp4(shape, denominator=3.0 if projection == "w13" else 7.0)
                else:
                    stored, decoded = self.mxfp8(shape)
                for suffix, tensor in stored.items():
                    tensors[name + suffix] = tensor
                if projection == "w13":
                    decoded = self.gate_then_up(decoded)
                expected[f"blk.{layer}.ffn_{target}_exps.weight"] = decoded.numpy()
            # the MXFP8 group says [32, 32] but the scales are stored per [1, 32] block: the converter follows the stored shape and warns
            block_shape = [1, 16] if scheme == "nvfp4_nearest" else [32, 32]
            quant_groups.append({"tensors": names, "weight_transform": scheme, "weight_scale_format": "amax_denominator", "weight_block_shape": block_shape})
        return tensors, quant_groups, expected

    def converter(self, path, tensors, quant_groups, ftype=gguf.LlamaFileType.ALL_F32, eager=False):
        """Writes the checkpoint to path and returns an Inkling2Model on it."""
        hp = {"model_type": "inkling2_text", "architectures": ["Inkling2ForCausalLM"], "num_hidden_layers": 2, "hidden_size": self.N_EMBD, "moe_intermediate_size": self.N_FF, "n_routed_experts": self.N_EXPERT, "layer_types": ["hybrid_sliding", "hybrid"]}
        (path / "config.json").write_text(json.dumps(hp))
        (path / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": quant_groups}}))
        save_file(tensors, path / "model.safetensors")
        return Inkling2Model(path, ftype, path / "unused.gguf", hparams=hp, eager=eager)

    def test_lookup_semantics(self):
        """The converter decode tables agree with torch.float8_e4m3fn; bytes 127 and 255 are NaN; E2M1 code 8 is -0."""
        reference = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
        np.testing.assert_array_equal(_E4M3.view(np.uint32)[np.isfinite(reference)], reference.view(np.uint32)[np.isfinite(reference)])
        np.testing.assert_array_equal(np.flatnonzero(np.isnan(_E4M3)), [127, 255])
        self.assertTrue(np.signbit(_E2M1[8]))

    def test_mixed_schemes_and_streaming_outtypes(self):
        """Both schemes convert bit-exact to every outtype, lazily, in chunks of at most 128 rows."""
        tensors, quant_groups, expected = self.packed_checkpoint()
        outtypes = (
            (gguf.LlamaFileType.GUESSED, gguf.GGMLQuantizationType.F16),
            (gguf.LlamaFileType.ALL_F32, gguf.GGMLQuantizationType.F32),
            (gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16),
            (gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16),
        )
        for eager in (False, True):
            for ftype, qtype in outtypes:
                with self.subTest(eager=eager, outtype=qtype), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    model = self.converter(path, tensors, quant_groups, ftype, eager)
                    with self.assertLogs("hf-to-gguf", level="WARNING") as logs:
                        model.prepare_tensors()
                    self.assertTrue(any("[32, 32]" in m and "[1, 32]" in m for m in logs.output))
                    infos = model.gguf_writer.tensors[0]
                    self.assertEqual(infos.keys(), expected.keys())
                    for name, info in infos.items():
                        self.assertEqual(info.dtype, qtype)
                        self.assertEqual(tuple(info.shape), expected[name].shape)
                        # the tensor must stay lazy: it cannot be materialized as one array, only streamed
                        self.assertIsInstance(info.tensor, gguf.LazyChunkedTensor)
                        with self.assertRaises(TypeError):
                            np.asarray(info.tensor)
                        for load_chunk in info.tensor._chunks:
                            self.assertLessEqual(load_chunk().shape[0], 128)
                        reference = gguf.quantize(expected[name], qtype)
                        self.assertEqual(self.written_bytes(info.tensor, path / "tensor.bin"), reference.tobytes(), name)

    def test_projection_and_companions_in_separate_shards(self):
        """A packed weight and its scale, amax and denominator can be in different safetensors shards."""
        tensors, quant_groups, expected = self.packed_checkpoint()
        name = quant_groups[0]["tensors"][0]
        tensors = {k: v for k, v in tensors.items() if k.startswith(name)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.converter(path, tensors, quant_groups)
            (path / "model.safetensors").rename(path / "model-00001-of-00002.safetensors")
            save_file({name: tensors.pop(name)}, path / "model-00001-of-00002.safetensors")
            save_file(tensors, path / "model-00002-of-00002.safetensors")
            model = Inkling2Model(path, gguf.LlamaFileType.ALL_F32, path / "unused.gguf")
            model.prepare_tensors()
            key = "blk.0.ffn_gate_up_exps.weight"
            infos = model.gguf_writer.tensors[0]
            self.assertEqual(list(infos), [key])
            tensor = infos[key].tensor
            assert tensor is not None
            self.assertEqual(self.written_bytes(tensor, path / "tensor.bin"), expected[key].tobytes())

    def test_float_experts_with_stale_quant_config(self):
        """Float experts convert unchanged even when hf_quant_config.json still lists them as packed."""
        _, quant_groups, expected = self.packed_checkpoint()
        tensors = {}
        for layer in range(2):
            for projection, target in (("w13", "gate_up"), ("w2", "down")):
                values = torch.from_numpy(expected[f"blk.{layer}.ffn_{target}_exps.weight"]).double()
                if projection == "w13":
                    values = self.interleaved(values)
                tensors[f"model.llm.layers.{layer}.mlp.experts.{projection}_weight"] = values
        with tempfile.TemporaryDirectory() as directory:
            model = self.converter(Path(directory), tensors, quant_groups)
            model.prepare_tensors()
            for name, info in model.gguf_writer.tensors[0].items():
                actual = gguf.LazyNumpyTensor.to_eager(info.tensor)
                np.testing.assert_array_equal(actual.view(np.uint32), expected[name].view(np.uint32))

    def test_invalid_metadata_and_companions(self):
        """Missing, mistyped or misshapen companions and unknown schemes are rejected before any tensor is written."""
        cases = ("missing", "dtype", "amax_shape", "den_shape", "grid", "block", "scheme", "orphan", "mxfp8_amax")
        for case in cases:
            tensors, quant_groups, _ = self.packed_checkpoint()
            name = quant_groups[0]["tensors"][0]
            if case == "missing":
                del tensors[name + "_amax"]
            elif case == "dtype":
                tensors[name + "_scale"] = tensors[name + "_scale"].view(torch.uint8)
            elif case == "amax_shape":
                tensors[name + "_amax"] = torch.ones(1)
            elif case == "den_shape":
                tensors[name + "_outer_scale_denominator"] = torch.ones(2)
            elif case == "grid":
                tensors[name + "_scale"] = tensors[name + "_scale"][:, :, :3].contiguous()
            elif case == "block":
                tensors[name + "_scale"] = tensors[name + "_scale"][:, ::2].contiguous()
            elif case == "scheme":
                quant_groups[0]["weight_transform"] = "unknown"
            elif case == "orphan":
                del tensors[name]
            else:
                tensors[quant_groups[1]["tensors"][0] + "_amax"] = torch.ones(2)
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                model = self.converter(Path(directory), tensors, quant_groups)
                with self.assertRaisesRegex(ValueError, "Inkling2:"):
                    model.prepare_tensors()

    def test_invalid_values_name_expert(self):
        """NaN codes and bad scales fail when the tensor is written, and the error names the tensor and the expert."""
        bad_values = (
            (0, "_scale", 127), (0, "_scale", 255),  # NVFP4 scale is E4M3: NaN bytes
            (1, "", 127), (1, "", 255),              # MXFP8 weight is E4M3: NaN bytes
            (1, "_scale", 255),                      # MXFP8 scale exponent 255 is NaN
            (0, "_amax", float("nan")), (0, "_outer_scale_denominator", 0.0),
        )
        for layer, suffix, value in bad_values:
            tensors, quant_groups, _ = self.packed_checkpoint()
            name = quant_groups[layer]["tensors"][0]
            target = tensors[name + suffix]
            if suffix in ("_amax", "_outer_scale_denominator"):
                target[-1] = value
            else:
                target.view(torch.uint8)[1, 0, 0] = value
            with self.subTest(layer=layer, suffix=suffix, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                model = self.converter(path, tensors, quant_groups)
                model.prepare_tensors()
                tensor = model.gguf_writer.tensors[0][f"blk.{layer}.ffn_gate_up_exps.weight"].tensor
                with self.assertRaisesRegex(ValueError, r"w13_weight.*expert [01]"):
                    self.written_bytes(tensor, path / "tensor.bin")


class TestInkling2TextNames(unittest.TestCase):
    """The checkpoint names (model.llm.*) and the transformers names (model.language_model.*) give the same GGUF tensors."""

    # checkpoint name, transformers name, shape; a 4th entry is the transformers shape when it differs
    NAMES = (
        ("embed.weight", "embed_tokens.weight", (16, 8)),
        ("embed_norm.weight", "embed_tokens.embed_norm.weight", (8,)),
        ("norm.weight", "norm.weight", (8,)),
        ("unembed.weight", "lm_head.weight", (16, 8)),
        ("attn_res.attn_queries.0", "layers.1.attention_input.query", (3, 8)),
        ("attn_res.mlp_queries.0", "layers.0.mlp_input.query", (8,)),
        ("attn_res.mlp_queries.1", "layers.1.mlp_input.query", (8,)),
        ("attn_res.output_query", "final_input.query", (8,)),
        ("layers.0.attn.k_sconv.weight", "layers.0.self_attn.k_sconv.conv1d.weight", (4, 4), (4, 1, 4)),
        ("layers.1.attn.k_sconv.static_w", "layers.1.self_attn.k_sconv.static_weight", (3, 4)),
    ) + tuple((f"layers.{i}.{old}", f"layers.{i}.{new}", *shapes) for i in range(2) for old, new, *shapes in (
        ("attn_norm.weight", "input_layernorm.weight", (8,)),
        ("mlp_norm.weight", "post_attention_layernorm.weight", (8,)),
        ("q_input_norm.weight", "q_input_layernorm.weight", (8,)),
        ("k_input_norm.weight", "k_input_layernorm.weight", (8,)),
        ("v_input_norm.weight", "v_input_layernorm.weight", (8,)),
        ("attn.wq_du.weight", "self_attn.q_proj.weight", (8, 8)),
        ("attn.wk_dv.weight", "self_attn.k_proj.weight", (4, 8)),
        ("attn.wv_dv.weight", "self_attn.v_proj.weight", (4, 8)),
        ("attn.wr_du.weight", "self_attn.r_proj.weight", (4, 8)),
        ("attn.wo_ud.weight", "self_attn.o_proj.weight", (8, 8)),
        ("attn.gate_attn.weight", "self_attn.gate_proj.weight", (8, 8)),
        ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight", (2, 8)),
        ("attn.q_l2_clip.scale", "self_attn.q_l2_clip.scale", (2, 4)),
        ("attn.rel_l2_clip.scale", "self_attn.rel_l2_clip.scale", (2, 2)),
        ("attn.rel_logits_proj.proj", "self_attn.rel_logits_proj.proj", (2, 5)),
        ("attn.k_sconv.weight_proj", "self_attn.k_sconv.weight_proj.weight", (6, 8)),
        ("attn.v_sconv.weight", "self_attn.v_sconv.conv1d.weight", (4, 3), (4, 1, 3)),
        ("attn.v_sconv.weight_proj", "self_attn.v_sconv.weight_proj.weight", (6, 8)),
        ("attn_sconv.weight", "post_attention_convolution.conv1d.weight", (8, 3), (8, 1, 3)),
        ("mlp_sconv.weight", "post_mlp_convolution.conv1d.weight", (8, 3), (8, 1, 3)),
        ("mlp.gate.weight", "mlp.gate.weight", (2, 8)),
        ("mlp.gate.bias", "mlp.gate.e_score_correction_bias", (2,)),
    ))

    def convert(self, tensors):
        with tempfile.TemporaryDirectory() as directory:
            return self._convert(Path(directory), tensors)

    def _convert(self, path, tensors):
        hp = {"model_type": "inkling2_text", "architectures": ["Inkling2ForCausalLM"], "num_hidden_layers": 2, "hidden_size": 8, "layer_types": ["hybrid_sliding", "hybrid"]}
        (path / "config.json").write_text(json.dumps(hp))
        save_file(tensors, path / "model.safetensors")
        model = Inkling2Model(path, gguf.LlamaFileType.ALL_F32, path / "unused.gguf", hparams=hp)
        model.prepare_tensors()
        # copy: the tensors are views of the mmapped checkpoint, which must be closed before the directory is removed
        return {name: gguf.LazyNumpyTensor.to_eager(info.tensor).copy() for name, info in model.gguf_writer.tensors[0].items()}

    def test_checkpoint_and_transformers_names(self):
        checkpoint, transformers = {}, {}
        for index, (old, new, shape, *new_shape) in enumerate(self.NAMES):
            tensor = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape) / 100 + index
            checkpoint["model.llm." + old] = tensor
            transformers[new if new.startswith("lm_head.") else "model.language_model." + new] = tensor.reshape(*new_shape) if new_shape else tensor
        actual = self.convert(checkpoint)
        self.assertEqual(len(actual), len(self.NAMES))
        source = {name: tensor.numpy() for name, tensor in checkpoint.items()}
        for target, expected in (
            ("token_embd.weight", source["model.llm.embed.weight"]),
            ("output.weight", source["model.llm.unembed.weight"]),
            ("output_res_query.weight", source["model.llm.attn_res.output_query"]),
            ("blk.1.attn_res_query.weight", source["model.llm.attn_res.attn_queries.0"]),  # attn_queries.{i} feeds block i + 1
            ("blk.1.ffn_res_query.weight", source["model.llm.attn_res.mlp_queries.1"]),
            ("blk.0.attn_q.weight", source["model.llm.layers.0.attn.wq_du.weight"]),
            ("blk.0.attn_rel.weight", source["model.llm.layers.0.attn.wr_du.weight"]),
            ("blk.0.attn_rel_bias.weight", source["model.llm.layers.0.attn.rel_logits_proj.proj"].T),
            ("blk.0.attn_k_conv.weight", source["model.llm.layers.0.attn.k_sconv.weight"]),
            ("blk.1.attn_k_conv.weight", source["model.llm.layers.1.attn.k_sconv.static_w"][::-1].T),  # static taps are [kernel, channels], newest first
            ("blk.0.attn_k_conv_proj.weight", source["model.llm.layers.0.attn.k_sconv.weight_proj"]),
            ("blk.0.attn_post_conv.weight", source["model.llm.layers.0.attn_sconv.weight"]),
            ("blk.0.ffn_post_conv.weight", source["model.llm.layers.0.mlp_sconv.weight"]),
            ("blk.0.attn_q_inp_norm.weight", source["model.llm.layers.0.q_input_norm.weight"]),
            ("blk.0.ffn_gate_inp.weight", source["model.llm.layers.0.mlp.gate.weight"]),
            ("blk.0.exp_probs_b.bias", source["model.llm.layers.0.mlp.gate.bias"]),
        ):
            np.testing.assert_array_equal(actual[target], expected, target)
        renamed = self.convert(transformers)
        self.assertEqual(renamed.keys(), actual.keys())
        for name in actual:
            np.testing.assert_array_equal(renamed[name], actual[name], name)


class TestBailingMoeV3Conversion(ConversionTestCase):
    """Synthetic Ling-3.2-air (bailingmoe3) checkpoint: layer 0 is KDA with a dense MLP, layer 1 is MLA with MoE, layer 2 is the MTP layer."""

    N_EMBD, N_VOCAB, HEAD_DIM, N_FF, N_FF_EXP, N_FF_SHEXP, N_EXPERT = 128, 32, 128, 256, 160, 256, 4
    QK_NOPE, QK_ROPE, V_HEAD, KV_RANK, KERNEL = 48, 16, 32, 64, 4
    FP8_CONFIG = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
    KEYS = {
        "block_count", "context_length", "embedding_length", "feed_forward_length", "attention.head_count",
        "attention.head_count_kv", "rope.freq_base", "attention.layer_norm_rms_epsilon", "expert_count", "expert_used_count",
        "expert_group_count", "expert_group_used_count", "expert_gating_func", "attention.key_length", "attention.value_length",
        "vocab_size", "ssm.conv_kernel", "kda.head_dim", "kda.safe_gate", "kda.gate_lower_bound", "attention.kv_lora_rank",
        "rope.dimension_count", "attention.key_length_mla", "attention.value_length_mla", "expert_feed_forward_length",
        "expert_shared_feed_forward_length", "expert_shared_count", "leading_dense_block_count", "expert_weights_scale",
        "expert_weights_norm", "swiglu_clamp_exp", "swiglu_clamp_shexp", "nextn_predict_layers", "rope.dimension_sections",
    }

    def text_config(self):
        return {
            "model_type": "bailing_hybrid",
            "vocab_size": self.N_VOCAB,
            "hidden_size": self.N_EMBD,
            "intermediate_size": self.N_FF,
            "moe_intermediate_size": self.N_FF_EXP,
            "moe_shared_expert_intermediate_size": self.N_FF_SHEXP,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "layer_group_size": 2,
            "first_k_dense_replace": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": self.HEAD_DIM,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "num_experts": self.N_EXPERT,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "n_group": 2,
            "topk_group": 1,
            "score_function": "sigmoid",
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "q_lora_rank": None,
            "kv_lora_rank": self.KV_RANK,
            "qk_nope_head_dim": self.QK_NOPE,
            "qk_rope_head_dim": self.QK_ROPE,
            "v_head_dim": self.V_HEAD,
            "no_kda_lora": True,
            "kda_safe_gate": True,
            "kda_lower_bound": -5.0,
            "short_conv_kernel_size": self.KERNEL,
            "gated_attention_proj_granularity_type": "head_wise",
            "expert_swiglu_limit_list": [None, 7.0],
            "share_expert_swiglu_limit_list": [None, 5.0],
        }

    def vl_config(self):
        text_config = {**self.text_config(), "mrope_section": [4, 2, 2]}
        return {"model_type": "bailing_hybrid_vl", "architectures": ["BailingMoeV3VLForConditionalGeneration"], "text_config": text_config, "vision_config": {"depth": 1}, "quantization_config": self.FP8_CONFIG}

    def causal_config(self):
        return {**self.text_config(), "architectures": ["BailingMoeV3ForCausalLM"], "quantization_config": self.FP8_CONFIG}

    @staticmethod
    def fp8(shape, scale):
        """FP8 weight with one scale per 128x128 block. Returns the weight, the scale and the reference dequant."""
        weight = (torch.linspace(-2, 2, shape[0] * shape[1]).reshape(shape) * torch.cos(torch.arange(shape[1]).float())).to(torch.float8_e4m3fn)
        scale = torch.tensor(scale, dtype=torch.float32)
        reference = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:shape[0], :shape[1]]
        return weight, scale, reference.numpy()

    def checkpoint(self, vision):
        """Returns the checkpoint tensors and the FP8 reference values by source weight name."""
        generator = torch.Generator().manual_seed(0)
        tensors: dict[str, torch.Tensor] = {}
        fp8_reference: dict[str, np.ndarray] = {}

        def add(name, *shape, dtype=torch.bfloat16):
            tensors[name] = torch.randn(shape, generator=generator).to(dtype)

        def add_fp8(name, shape, scale):
            weight, scale_inv, reference = self.fp8(shape, scale)
            tensors[name + ".weight"] = weight
            tensors[name + ".weight_scale_inv"] = scale_inv
            fp8_reference[name + ".weight"] = reference

        H = self.N_EMBD
        add("model.word_embeddings.weight", self.N_VOCAB, H)
        add("model.norm.weight", H)
        add("lm_head.weight", self.N_VOCAB, H)

        # layer 0: KDA attention, dense MLP
        p = "model.layers.0."
        add(p + "input_layernorm.weight", H)
        add(p + "post_attention_layernorm.weight", H)
        for proj in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj"):
            add(p + f"attention.{proj}.weight", self.HEAD_DIM, H)
        for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
            add(p + f"attention.{conv}.weight", self.HEAD_DIM, 1, self.KERNEL)
        add(p + "attention.b_proj.weight", 1, H)
        tensors[p + "attention.A_log"] = torch.tensor([0.3])
        add(p + "attention.dt_bias", self.HEAD_DIM, dtype=torch.float32)
        add(p + "attention.o_norm.weight", self.HEAD_DIM)
        add(p + "attention.o_proj.weight", H, self.HEAD_DIM)
        add_fp8(p + "mlp.gate_proj", (self.N_FF, H), [[0.5], [2.0]])
        add_fp8(p + "mlp.up_proj", (self.N_FF, H), [[0.25], [4.0]])
        add_fp8(p + "mlp.down_proj", (H, self.N_FF), [[1.5, 0.125]])

        # layer 1: MLA attention, MoE; layer 2: MTP
        for bid in (1, 2):
            p = f"model.layers.{bid}."
            add(p + "input_layernorm.weight", H)
            add(p + "post_attention_layernorm.weight", H)
            add(p + "attention.q_proj.weight", self.QK_NOPE + self.QK_ROPE, H)
            add_fp8(p + "attention.kv_a_proj_with_mqa", (self.KV_RANK + self.QK_ROPE, H), [[0.75]])
            add(p + "attention.kv_a_layernorm.weight", self.KV_RANK)
            add(p + "attention.kv_b_proj.weight", self.QK_NOPE + self.V_HEAD, self.KV_RANK)
            add(p + "attention.dense.weight", H, self.V_HEAD)
            add(p + "attention.g_proj.weight", 1, H)
            add(p + "mlp.gate.weight", self.N_EXPERT, H, dtype=torch.float32)
            add(p + "mlp.gate.expert_bias", self.N_EXPERT, dtype=torch.float32)
            for xid in range(self.N_EXPERT):
                add_fp8(p + f"mlp.experts.{xid}.gate_proj", (self.N_FF_EXP, H), [[0.5 + xid], [2.0]])
                add_fp8(p + f"mlp.experts.{xid}.up_proj", (self.N_FF_EXP, H), [[1.0], [0.25 * (xid + 1)]])
                add_fp8(p + f"mlp.experts.{xid}.down_proj", (H, self.N_FF_EXP), [[2.0 + xid, 0.5]])
            add(p + "mlp.shared_experts.gate_proj.weight", self.N_FF_SHEXP, H)
            add_fp8(p + "mlp.shared_experts.up_proj", (self.N_FF_SHEXP, H), [[3.0], [0.5]])
            add(p + "mlp.shared_experts.down_proj.weight", H, self.N_FF_SHEXP)

        p = "model.layers.2."
        add_fp8(p + "eh_proj", (H, 2 * H), [[0.5, 2.0]])
        add(p + "enorm.weight", H)
        add(p + "hnorm.weight", H)
        add(p + "final_layernorm.weight", H)

        if vision:
            add("model.visual.patch_embed.proj.weight", 16, 3, 2, 2)
            add("model.visual.blocks.0.attn.qkv.weight", 48, 16)
            add("linear_proj.0.weight", 32, 64)
            add("linear_proj.2.weight", H, 32)
        return tensors, fp8_reference

    def expected_shapes(self, layers):
        H = self.N_EMBD
        shapes = {
            "token_embd.weight": (self.N_VOCAB, H),
            "output_norm.weight": (H,),
            "output.weight": (self.N_VOCAB, H),
        }
        for bid in layers:
            shapes.update({
                f"blk.{bid}.attn_norm.weight": (H,),
                f"blk.{bid}.ffn_norm.weight": (H,),
            })
            if bid == 0:
                shapes.update({
                    "blk.0.attn_q.weight": (self.HEAD_DIM, H),
                    "blk.0.attn_k.weight": (self.HEAD_DIM, H),
                    "blk.0.attn_v.weight": (self.HEAD_DIM, H),
                    "blk.0.ssm_f_a.weight": (self.HEAD_DIM, H),
                    "blk.0.ssm_g_a.weight": (self.HEAD_DIM, H),
                    "blk.0.ssm_conv1d_q.weight": (1, self.HEAD_DIM, 1, self.KERNEL),
                    "blk.0.ssm_conv1d_k.weight": (1, self.HEAD_DIM, 1, self.KERNEL),
                    "blk.0.ssm_conv1d_v.weight": (1, self.HEAD_DIM, 1, self.KERNEL),
                    "blk.0.ssm_beta.weight": (1, H),
                    "blk.0.ssm_a": (1, 1),
                    "blk.0.ssm_dt.bias": (self.HEAD_DIM,),
                    "blk.0.ssm_norm.weight": (self.HEAD_DIM,),
                    "blk.0.attn_output.weight": (H, self.HEAD_DIM),
                    "blk.0.ffn_gate.weight": (self.N_FF, H),
                    "blk.0.ffn_up.weight": (self.N_FF, H),
                    "blk.0.ffn_down.weight": (H, self.N_FF),
                })
                continue
            shapes.update({
                f"blk.{bid}.attn_q.weight": (self.QK_NOPE + self.QK_ROPE, H),
                f"blk.{bid}.attn_kv_a_mqa.weight": (self.KV_RANK + self.QK_ROPE, H),
                f"blk.{bid}.attn_kv_a_norm.weight": (self.KV_RANK,),
                f"blk.{bid}.attn_k_b.weight": (1, self.KV_RANK, self.QK_NOPE),
                f"blk.{bid}.attn_v_b.weight": (1, self.V_HEAD, self.KV_RANK),
                f"blk.{bid}.attn_output.weight": (H, self.V_HEAD),
                f"blk.{bid}.attn_gate.weight": (1, H),
                f"blk.{bid}.ffn_gate_inp.weight": (self.N_EXPERT, H),
                f"blk.{bid}.exp_probs_b.bias": (self.N_EXPERT,),
                f"blk.{bid}.ffn_gate_exps.weight": (self.N_EXPERT, self.N_FF_EXP, H),
                f"blk.{bid}.ffn_up_exps.weight": (self.N_EXPERT, self.N_FF_EXP, H),
                f"blk.{bid}.ffn_down_exps.weight": (self.N_EXPERT, H, self.N_FF_EXP),
                f"blk.{bid}.ffn_gate_shexp.weight": (self.N_FF_SHEXP, H),
                f"blk.{bid}.ffn_up_shexp.weight": (self.N_FF_SHEXP, H),
                f"blk.{bid}.ffn_down_shexp.weight": (H, self.N_FF_SHEXP),
            })
            if bid == 2:
                shapes.update({
                    "blk.2.nextn.eh_proj.weight": (H, 2 * H),
                    "blk.2.nextn.enorm.weight": (H,),
                    "blk.2.nextn.hnorm.weight": (H,),
                    "blk.2.layer_output_norm.weight": (H,),
                })
        return shapes

    def run_converter(self, model_class, config, tensors, ftype=gguf.LlamaFileType.ALL_F32, **kwargs):
        result, kv = self.convert(model_class, tensors, ftype=ftype, hparams_override=config, with_kv=True, **kwargs)
        prefix = "bailingmoe3."
        return result, {key[len(prefix):]: value for key, value in kv.items() if key.startswith(prefix)}

    def test_registered_classes(self):
        for config, expected in ((self.vl_config(), BailingMoeV3VLModel), (self.causal_config(), BailingMoeV3Model)):
            arch = get_model_architecture(config, ModelType.TEXT)
            self.assertIs(get_model_class(arch), expected)
        with self.assertRaises(NotImplementedError):
            get_model_class("BailingMoeV3VLForConditionalGeneration", mmproj=True)

    def test_vl_text_model(self):
        tensors, fp8_reference = self.checkpoint(vision=True)
        result, kv = self.run_converter(BailingMoeV3VLModel, self.vl_config(), tensors, check_lazy=True)

        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, self.expected_shapes(range(3)))
        for name, (dtype, _) in result.items():
            self.assertEqual(dtype, gguf.GGMLQuantizationType.F32, name)

        self.assertEqual(kv.keys(), self.KEYS)
        self.assertEqual(kv["block_count"], 3)
        self.assertEqual(list(kv["attention.head_count_kv"]), [0, 1, 1])
        self.assertEqual(kv["attention.key_length"], self.KV_RANK + self.QK_ROPE)
        self.assertEqual(kv["attention.value_length"], self.HEAD_DIM)
        self.assertEqual(kv["attention.key_length_mla"], self.QK_NOPE + self.QK_ROPE)
        self.assertEqual(kv["attention.value_length_mla"], self.V_HEAD)
        self.assertEqual(kv["rope.dimension_count"], self.QK_ROPE)
        self.assertEqual(list(kv["rope.dimension_sections"]), [4, 2, 2, 0])
        self.assertEqual(list(kv["swiglu_clamp_exp"]), [0.0, 7.0, 0.0])
        self.assertEqual(list(kv["swiglu_clamp_shexp"]), [0.0, 5.0, 0.0])
        self.assertEqual((kv["expert_count"], kv["expert_used_count"]), (self.N_EXPERT, 2))
        self.assertEqual((kv["expert_group_count"], kv["expert_group_used_count"]), (2, 1))
        self.assertEqual(kv["expert_gating_func"], gguf.ExpertGatingFuncType.SIGMOID)
        self.assertEqual((kv["leading_dense_block_count"], kv["nextn_predict_layers"]), (1, 1))
        self.assertEqual((kv["expert_shared_count"], kv["expert_shared_feed_forward_length"]), (1, self.N_FF_SHEXP))
        self.assertEqual((kv["kda.head_dim"], kv["kda.safe_gate"], kv["kda.gate_lower_bound"]), (self.HEAD_DIM, True, -5.0))
        self.assertEqual((kv["expert_weights_scale"], kv["expert_weights_norm"]), (2.5, True))
        self.assertEqual((kv["ssm.conv_kernel"], kv["vocab_size"]), (self.KERNEL, self.N_VOCAB))

        def source(name):
            return tensors[name].float().numpy()

        # no rope row permutation
        np.testing.assert_array_equal(result["blk.1.attn_q.weight"][1], source("model.layers.1.attention.q_proj.weight"))
        np.testing.assert_array_equal(result["blk.0.ssm_f_a.weight"][1], source("model.layers.0.attention.f_proj.weight"))
        np.testing.assert_array_equal(result["blk.0.ssm_g_a.weight"][1], source("model.layers.0.attention.g_proj.weight"))
        np.testing.assert_array_equal(result["blk.1.attn_gate.weight"][1], source("model.layers.1.attention.g_proj.weight"))
        np.testing.assert_array_equal(result["blk.0.ssm_conv1d_k.weight"][1], source("model.layers.0.attention.k_conv1d.weight").reshape(1, self.HEAD_DIM, 1, self.KERNEL))
        np.testing.assert_array_equal(result["blk.0.ssm_dt.bias"][1], source("model.layers.0.attention.dt_bias"))
        np.testing.assert_allclose(result["blk.0.ssm_a"][1], np.exp([[0.3]]), rtol=1e-6)

        kv_b = source("model.layers.1.attention.kv_b_proj.weight").reshape(1, self.QK_NOPE + self.V_HEAD, self.KV_RANK)
        np.testing.assert_array_equal(result["blk.1.attn_k_b.weight"][1], kv_b[:, :self.QK_NOPE].transpose(0, 2, 1))
        np.testing.assert_array_equal(result["blk.1.attn_v_b.weight"][1], kv_b[:, self.QK_NOPE:])

        for bid, proj, target in ((1, "gate_proj", "ffn_gate_exps"), (2, "down_proj", "ffn_down_exps")):
            experts = np.stack([fp8_reference[f"model.layers.{bid}.mlp.experts.{xid}.{proj}.weight"] for xid in range(self.N_EXPERT)])
            np.testing.assert_array_equal(result[f"blk.{bid}.{target}.weight"][1], experts)
        np.testing.assert_array_equal(result["blk.1.exp_probs_b.bias"][1], source("model.layers.1.mlp.gate.expert_bias"))

        for source_name, target in (
            ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
            ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
            ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
            ("model.layers.1.attention.kv_a_proj_with_mqa.weight", "blk.1.attn_kv_a_mqa.weight"),
            ("model.layers.1.mlp.shared_experts.up_proj.weight", "blk.1.ffn_up_shexp.weight"),
            ("model.layers.2.mlp.shared_experts.up_proj.weight", "blk.2.ffn_up_shexp.weight"),
            ("model.layers.2.eh_proj.weight", "blk.2.nextn.eh_proj.weight"),
        ):
            np.testing.assert_array_equal(result[target][1], fp8_reference[source_name], target)

    def test_vl_auto_outtype(self):
        """With the auto outtype, 128x128 block FP8 weights are dequantized and KDA conv weights stay F32."""
        tensors, fp8_reference = self.checkpoint(vision=True)
        result, _ = self.run_converter(BailingMoeV3VLModel, self.vl_config(), tensors, ftype=gguf.LlamaFileType.GUESSED)
        values = {name: gguf.dequantize(data, dtype) for name, (dtype, data) in result.items()}
        self.assertEqual({name: data.shape for name, data in values.items()}, self.expected_shapes(range(3)))
        self.assertEqual(result["blk.0.ssm_conv1d_q.weight"][0], gguf.GGMLQuantizationType.F32)
        for source_name, target in (
            ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
            ("model.layers.1.attention.kv_a_proj_with_mqa.weight", "blk.1.attn_kv_a_mqa.weight"),
            ("model.layers.1.mlp.shared_experts.up_proj.weight", "blk.1.ffn_up_shexp.weight"),
            ("model.layers.2.eh_proj.weight", "blk.2.nextn.eh_proj.weight"),
        ):
            self.assertEqual(result[target][0], gguf.GGMLQuantizationType.BF16, target)
            np.testing.assert_array_equal(values[target], torch.from_numpy(fp8_reference[source_name]).bfloat16().float().numpy(), target)
        experts = np.stack([fp8_reference[f"model.layers.1.mlp.experts.{xid}.up_proj.weight"] for xid in range(self.N_EXPERT)])
        self.assertEqual(result["blk.1.ffn_up_exps.weight"][0], gguf.GGMLQuantizationType.BF16)
        np.testing.assert_array_equal(values["blk.1.ffn_up_exps.weight"], torch.from_numpy(experts).bfloat16().float().numpy())
        self.assertFalse([name for name in result if name.endswith(".scale")])

    def test_vl_no_mtp(self):
        tensors, _ = self.checkpoint(vision=True)
        with mock.patch.object(BailingMoeV3VLModel, "no_mtp", True):
            result, kv = self.run_converter(BailingMoeV3VLModel, self.vl_config(), tensors)
        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, self.expected_shapes(range(2)))
        self.assertEqual(kv.keys(), self.KEYS - {"nextn_predict_layers"})
        self.assertEqual(kv["block_count"], 2)
        self.assertEqual(list(kv["attention.head_count_kv"]), [0, 1])
        self.assertEqual(list(kv["swiglu_clamp_exp"]), [0.0, 7.0])

    def test_vl_mtp_only(self):
        tensors, _ = self.checkpoint(vision=True)
        with mock.patch.object(BailingMoeV3VLModel, "mtp_only", True):
            result, kv = self.run_converter(BailingMoeV3VLModel, self.vl_config(), tensors)
        expected = {name: shape for name, shape in self.expected_shapes(range(3)).items() if not name.startswith(("blk.0.", "blk.1."))}
        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, expected)
        self.assertEqual((kv["block_count"], kv["nextn_predict_layers"]), (3, 1))

    def test_causal_lm(self):
        tensors, fp8_reference = self.checkpoint(vision=False)
        result, kv = self.run_converter(BailingMoeV3Model, self.causal_config(), tensors)
        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, self.expected_shapes(range(3)))
        self.assertEqual(kv.keys(), self.KEYS - {"rope.dimension_sections"})
        np.testing.assert_array_equal(result["blk.2.nextn.eh_proj.weight"][1], fp8_reference["model.layers.2.eh_proj.weight"])

    def test_mtp_flags(self):
        import convert_hf_to_gguf
        for flag in ("--no-mtp", "--no-nextn"):
            with mock.patch.object(sys, "argv", ["convert_hf_to_gguf.py", flag, "model"]):
                args = convert_hf_to_gguf.parse_args()
            self.assertTrue(args.no_mtp)
            self.assertFalse(args.mtp)
        with mock.patch.object(sys, "argv", ["convert_hf_to_gguf.py", "--mtp", "model"]):
            self.assertTrue(convert_hf_to_gguf.parse_args().mtp)


class TestKolibri1Conversion(ConversionTestCase):
    """Synthetic Kolibri-1.1 checkpoint: layer 0 is a sliding layer with a dense FFN, layer 1 an indexer, layer 2 a sliding layer
    and layer 3 a reindexer that reads the K/V of layer 1 but keeps unused K/V projections, as the released checkpoints do."""

    N_EMBD, N_VOCAB, HEAD_DIM, N_HEAD, N_HEAD_KV, N_FF, N_FF_EXP, N_FF_SHEXP, N_EXPERT = 128, 32, 32, 4, 2, 256, 64, 64, 4
    FP8_CONFIG = {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128]}

    def config(self, sparse=True):
        config = {
            "architectures": ["Kolibri1_1ForCausalLM" if sparse else "Kolibri1ForCausalLM"],
            "model_type": "kolibri1_1" if sparse else "kolibri1",
            "vocab_size": self.N_VOCAB,
            "hidden_size": self.N_EMBD,
            "num_hidden_layers": 4,
            "num_attention_heads": self.N_HEAD,
            "num_key_value_heads": self.N_HEAD_KV,
            "head_dim": self.HEAD_DIM,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "num_experts": self.N_EXPERT,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": self.N_FF_EXP,
            "shared_expert_intermediate_size": self.N_FF_SHEXP,
            "norm_topk_prob": False,
            "sliding_window": 513,
            "layer_types": ["sliding_attention", "indexed_attention", "sliding_attention", "indexed_attention"],
            "quantization_config": self.FP8_CONFIG,
        }
        if sparse:
            config.update({
                "mlp_only_layers": [0],
                "mlp_only_intermediate_size": self.N_FF,
                "index_layers": [1],
                "kv_source_layers": [0, 1, 2, 1],
                "index_topk": 2048,
                "index_pool_size": 16384,
                "index_compression": 16,
                "index_num_local_tokens": 128,
                "index_num_sink_tokens": 8,
                "index_num_heads": self.N_HEAD // self.N_HEAD_KV,
                "kv_cache_qat": True,
            })
        return config

    def checkpoint(self, sparse=True):
        """Returns the checkpoint tensors and the FP8 reference values by source weight name."""
        generator = torch.Generator().manual_seed(0)
        tensors: dict[str, torch.Tensor] = {}
        fp8_reference: dict[str, np.ndarray] = {}

        def add(name, *shape, dtype=torch.bfloat16):
            tensors[name] = torch.randn(shape, generator=generator).to(dtype)

        def add_fp8(name, shape, scale):
            weight, scale_inv, reference = TestBailingMoeV3Conversion.fp8(shape, scale)
            tensors[name + ".weight"] = weight
            tensors[name + ".weight_scale_inv"] = scale_inv
            fp8_reference[name + ".weight"] = reference

        H, D = self.N_EMBD, self.HEAD_DIM
        add("model.embed_tokens.weight", self.N_VOCAB, H)
        add("model.norm.weight", H)
        add("lm_head.weight", self.N_VOCAB, H)
        for bid in range(4):
            p = f"model.layers.{bid}."
            for norm in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
                add(p + norm + ".weight", H)
            add(p + "self_attn.q_proj.weight", self.N_HEAD * D, H)
            add(p + "self_attn.k_proj.weight", self.N_HEAD_KV * D, H)
            add(p + "self_attn.v_proj.weight", self.N_HEAD_KV * D, H)
            add(p + "self_attn.o_proj.weight", H, self.N_HEAD * D)
            add(p + "self_attn.q_norm.weight", D)
            add(p + "self_attn.k_norm.weight", D)
            if sparse and bid == 0:
                add_fp8(p + "mlp.gate_proj", (self.N_FF, H), [[0.5], [2.0]])
                add_fp8(p + "mlp.up_proj", (self.N_FF, H), [[0.25], [4.0]])
                add_fp8(p + "mlp.down_proj", (H, self.N_FF), [[1.5, 0.125]])
                continue
            add(p + "mlp.gate.weight", self.N_EXPERT, H, dtype=torch.float32)
            add(p + "moe.router.expert_bias", self.N_EXPERT, dtype=torch.float32)
            for xid in range(self.N_EXPERT):
                add_fp8(p + f"mlp.experts.{xid}.gate_proj", (self.N_FF_EXP, H), [[0.5 + xid]])
                add_fp8(p + f"mlp.experts.{xid}.up_proj", (self.N_FF_EXP, H), [[0.25 * (xid + 1)]])
                add_fp8(p + f"mlp.experts.{xid}.down_proj", (H, self.N_FF_EXP), [[2.0 + xid]])
            add_fp8(p + "mlp.shared_experts.gate_proj", (self.N_FF_SHEXP, H), [[3.0]])
            add_fp8(p + "mlp.shared_experts.up_proj", (self.N_FF_SHEXP, H), [[0.5]])
            add_fp8(p + "mlp.shared_experts.down_proj", (H, self.N_FF_SHEXP), [[0.75]])
        return tensors, fp8_reference

    def expected_shapes(self, sparse=True):
        H, D = self.N_EMBD, self.HEAD_DIM
        shapes = {
            "token_embd.weight": (self.N_VOCAB, H),
            "output_norm.weight": (H,),
            "output.weight": (self.N_VOCAB, H),
        }
        for bid in range(4):
            shapes.update({
                f"blk.{bid}.attn_norm.weight": (H,),
                f"blk.{bid}.post_attention_norm.weight": (H,),
                f"blk.{bid}.ffn_norm.weight": (H,),
                f"blk.{bid}.post_ffw_norm.weight": (H,),
                f"blk.{bid}.attn_q.weight": (self.N_HEAD * D, H),
                f"blk.{bid}.attn_output.weight": (H, self.N_HEAD * D),
                f"blk.{bid}.attn_q_norm.weight": (D,),
            })
            # the reindexer reads the K/V of layer 1
            if not sparse or bid != 3:
                shapes.update({
                    f"blk.{bid}.attn_k.weight": (self.N_HEAD_KV * D, H),
                    f"blk.{bid}.attn_v.weight": (self.N_HEAD_KV * D, H),
                    f"blk.{bid}.attn_k_norm.weight": (D,),
                })
            if sparse and bid == 0:
                shapes.update({
                    "blk.0.ffn_gate.weight": (self.N_FF, H),
                    "blk.0.ffn_up.weight": (self.N_FF, H),
                    "blk.0.ffn_down.weight": (H, self.N_FF),
                })
                continue
            shapes.update({
                f"blk.{bid}.ffn_gate_inp.weight": (self.N_EXPERT, H),
                f"blk.{bid}.exp_probs_b.bias": (self.N_EXPERT,),
                f"blk.{bid}.ffn_gate_exps.weight": (self.N_EXPERT, self.N_FF_EXP, H),
                f"blk.{bid}.ffn_up_exps.weight": (self.N_EXPERT, self.N_FF_EXP, H),
                f"blk.{bid}.ffn_down_exps.weight": (self.N_EXPERT, H, self.N_FF_EXP),
                f"blk.{bid}.ffn_gate_shexp.weight": (self.N_FF_SHEXP, H),
                f"blk.{bid}.ffn_up_shexp.weight": (self.N_FF_SHEXP, H),
                f"blk.{bid}.ffn_down_shexp.weight": (H, self.N_FF_SHEXP),
            })
        return shapes

    def run_converter(self, model_class, config, tensors, ftype=gguf.LlamaFileType.ALL_F32, **kwargs):
        result, kv = self.convert(model_class, tensors, ftype=ftype, hparams_override=config, with_kv=True, **kwargs)
        prefix = "kolibri1."
        return result, {key[len(prefix):]: value for key, value in kv.items() if key.startswith(prefix)}

    def test_registered_classes(self):
        for config, expected in ((self.config(sparse=True), Kolibri11Model), (self.config(sparse=False), Kolibri1Model)):
            arch = get_model_architecture(config, ModelType.TEXT)
            self.assertIs(get_model_class(arch), expected)

    def test_kolibri11(self):
        tensors, fp8_reference = self.checkpoint()
        result, kv = self.run_converter(Kolibri11Model, self.config(), tensors, check_lazy=True)

        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, self.expected_shapes())
        for name, (dtype, _) in result.items():
            self.assertEqual(dtype, gguf.GGMLQuantizationType.F32, name)

        self.assertEqual(kv["block_count"], 4)
        self.assertEqual((kv["feed_forward_length"], kv["leading_dense_block_count"]), (self.N_FF, 1))
        self.assertEqual((kv["expert_count"], kv["expert_used_count"]), (self.N_EXPERT, 2))
        self.assertEqual((kv["expert_feed_forward_length"], kv["expert_shared_feed_forward_length"]), (self.N_FF_EXP, self.N_FF_SHEXP))
        self.assertEqual((kv["expert_shared_count"], kv["expert_weights_norm"]), (1, False))
        self.assertEqual(kv["expert_gating_func"], gguf.ExpertGatingFuncType.SIGMOID)
        self.assertEqual((kv["attention.head_count"], kv["attention.head_count_kv"]), (self.N_HEAD, self.N_HEAD_KV))
        self.assertEqual((kv["attention.key_length"], kv["attention.value_length"], kv["rope.dimension_count"]), (self.HEAD_DIM,) * 3)
        self.assertEqual((kv["attention.sliding_window"], kv["vocab_size"]), (513, self.N_VOCAB))
        self.assertEqual(list(kv["attention.sliding_window_pattern"]), [True, False, True, False])
        self.assertEqual(list(kv["attention.indexer.types"]), [False, True, False, False])
        self.assertEqual(list(kv["attention.kv_source_map"]), [0, 1, 2, 1])
        self.assertEqual(kv["attention.indexer.top_k"], 2048)
        self.assertEqual((kv["attention.indexer.candidate_block_size"], kv["attention.indexer.candidate_top_k_blocks"]), (16, 1024))

        def source(name):
            return tensors[name].float().numpy()

        # no rope row permutation
        np.testing.assert_array_equal(result["blk.1.attn_q.weight"][1], source("model.layers.1.self_attn.q_proj.weight"))
        np.testing.assert_array_equal(result["blk.1.attn_k.weight"][1], source("model.layers.1.self_attn.k_proj.weight"))
        # post_attention_layernorm is the norm before the FFN
        np.testing.assert_array_equal(result["blk.2.ffn_norm.weight"][1], source("model.layers.2.post_attention_layernorm.weight"))
        np.testing.assert_array_equal(result["blk.2.post_attention_norm.weight"][1], source("model.layers.2.post_attn_norm.weight"))
        np.testing.assert_array_equal(result["blk.2.post_ffw_norm.weight"][1], source("model.layers.2.post_ffn_norm.weight"))
        np.testing.assert_array_equal(result["blk.3.exp_probs_b.bias"][1], source("model.layers.3.moe.router.expert_bias"))

        for proj, target in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
            experts = np.stack([fp8_reference[f"model.layers.1.mlp.experts.{xid}.{proj}.weight"] for xid in range(self.N_EXPERT)])
            np.testing.assert_array_equal(result[f"blk.1.{target}.weight"][1], experts)
        for source_name, target in (
            ("model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
            ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
            ("model.layers.2.mlp.shared_experts.up_proj.weight", "blk.2.ffn_up_shexp.weight"),
        ):
            np.testing.assert_array_equal(result[target][1], fp8_reference[source_name], target)

    def test_kolibri1(self):
        """Kolibri-1 has no dense prefix and no sparse attention: every layer keeps its K/V, the FFN length is the expert length."""
        tensors, _ = self.checkpoint(sparse=False)
        result, kv = self.run_converter(Kolibri1Model, self.config(sparse=False), tensors)
        self.assertEqual({name: data.shape for name, (_, data) in result.items()}, self.expected_shapes(sparse=False))
        self.assertEqual(kv["feed_forward_length"], self.N_FF_EXP)
        for key in ("leading_dense_block_count", "attention.indexer.top_k", "attention.indexer.types", "attention.kv_source_map"):
            self.assertNotIn(key, kv)

    def test_unsupported_sparse_attention(self):
        tensors, _ = self.checkpoint()
        for key, value in (("index_num_sink_tokens", 4), ("index_num_heads", 1), ("kv_cache_qat", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.run_converter(Kolibri11Model, {**self.config(), key: value}, tensors)
        with self.assertRaisesRegex(ValueError, "Sliding-window layers"):
            self.run_converter(Kolibri11Model, {**self.config(), "index_layers": [0]}, tensors)
        with self.assertRaisesRegex(ValueError, "leading layers"):
            self.run_converter(Kolibri11Model, {**self.config(), "mlp_only_layers": [1]}, tensors)


if __name__ == "__main__":
    unittest.main()

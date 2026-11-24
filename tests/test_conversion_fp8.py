"""Test FP8 conversion with small synthetic checkpoints; no GPU or model downloads required.

Install dependencies and run from the repository root:
    python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python -m unittest discover -s tests -p 'test_conversion_fp8.py' -v

Use unittest discovery so the repository root is available for imports.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from conversion.base import gguf
from conversion.inkling2 import Inkling2Model, _E4M3, _E2M1


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


if __name__ == "__main__":
    unittest.main()

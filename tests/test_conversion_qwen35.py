"""Test Qwen 3.5 / 3.6 conversion with small synthetic checkpoints; no GPU or model downloads required.

Install dependencies and run from the repository root:
    python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
    python -m unittest discover -s tests -p 'test_conversion_qwen35.py' -v

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

from conversion import get_model_class
from conversion.base import gguf
from conversion.qwen import Qwen3_5MoeTextModel, Qwen3_5TextModel
from conversion.qwen3vl import Qwen3VLVisionModel


H, N_VOCAB = 32, 64
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
N_HEAD, N_KV, HEAD_DIM = 2, 1, 256
LK_HEADS, LV_HEADS, LK_DIM, LV_DIM, CONV_K = 2, 4, 8, 4, 4
CONV_DIM = 2 * LK_HEADS * LK_DIM + LV_HEADS * LV_DIM
N_FF, N_EXP, N_EXP_USED, N_FF_EXP, N_FF_SHEXP = 48, 4, 2, 16, 24
V_H, V_DEPTH, V_FF, V_PATCH, V_NPOS = 16, 2, 32, 4, 16


def text_config(moe: bool) -> dict:
    cfg = {
        "attn_output_gate": True, "full_attention_interval": 4, "head_dim": HEAD_DIM, "hidden_size": H,
        "layer_types": LAYER_TYPES, "linear_conv_kernel_dim": CONV_K, "linear_key_head_dim": LK_DIM,
        "linear_num_key_heads": LK_HEADS, "linear_num_value_heads": LV_HEADS, "linear_value_head_dim": LV_DIM,
        "max_position_embeddings": 262144, "mtp_num_hidden_layers": 1, "num_attention_heads": N_HEAD,
        "num_hidden_layers": len(LAYER_TYPES), "num_key_value_heads": N_KV, "rms_norm_eps": 1e-06, "vocab_size": N_VOCAB,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    }
    if moe:
        cfg.update({"num_experts": N_EXP, "num_experts_per_tok": N_EXP_USED, "moe_intermediate_size": N_FF_EXP, "shared_expert_intermediate_size": N_FF_SHEXP})
    else:
        cfg["intermediate_size"] = N_FF
    return cfg


def vision_config(deepstack: list[int]) -> dict:
    return {
        "deepstack_visual_indexes": deepstack, "depth": V_DEPTH, "hidden_size": V_H, "intermediate_size": V_FF, "num_heads": 2,
        "num_position_embeddings": V_NPOS, "out_hidden_size": H, "patch_size": V_PATCH, "spatial_merge_size": 2, "temporal_patch_size": 2,
    }


class Checkpoint:
    """Random tensors with the Qwen 3.5 checkpoint names (model.language_model.*, model.visual.*, mtp.*)."""

    def __init__(self, moe: bool = False, fused_experts: bool = True, deepstack: list[int] | None = None):
        self.gen = torch.Generator().manual_seed(0)
        self.moe = moe
        self.deepstack = deepstack or []
        p = "model.language_model."
        self.tensors = {p + "embed_tokens.weight": self.rnd(N_VOCAB, H), p + "norm.weight": self.rnd(H), "lm_head.weight": self.rnd(N_VOCAB, H)}
        for i, layer_type in enumerate(LAYER_TYPES):
            self.add_layer(f"{p}layers.{i}", layer_type, fused_experts)
        self.tensors["mtp.fc.weight"] = self.rnd(H, 2 * H)
        self.tensors["mtp.norm.weight"] = self.rnd(H)
        self.add_layer("mtp.layers.0", "full_attention", fused_experts)
        self.add_vision()

    def rnd(self, *shape):
        return torch.randn(*shape, generator=self.gen)

    def add_layer(self, p, layer_type, fused_experts):
        t = self.tensors
        t[f"{p}.input_layernorm.weight"] = self.rnd(H)
        t[f"{p}.post_attention_layernorm.weight"] = self.rnd(H)
        if layer_type == "linear_attention":
            t[f"{p}.linear_attn.A_log"] = self.rnd(LV_HEADS)
            t[f"{p}.linear_attn.conv1d.weight"] = self.rnd(CONV_DIM, 1, CONV_K)
            t[f"{p}.linear_attn.dt_bias"] = self.rnd(LV_HEADS)
            t[f"{p}.linear_attn.in_proj_a.weight"] = self.rnd(LV_HEADS, H)
            t[f"{p}.linear_attn.in_proj_b.weight"] = self.rnd(LV_HEADS, H)
            t[f"{p}.linear_attn.in_proj_qkv.weight"] = self.rnd(CONV_DIM, H)
            t[f"{p}.linear_attn.in_proj_z.weight"] = self.rnd(LV_HEADS * LV_DIM, H)
            t[f"{p}.linear_attn.norm.weight"] = self.rnd(LV_DIM)
            t[f"{p}.linear_attn.out_proj.weight"] = self.rnd(H, LV_HEADS * LV_DIM)
        else:
            t[f"{p}.self_attn.q_proj.weight"] = self.rnd(N_HEAD * HEAD_DIM * 2, H)
            t[f"{p}.self_attn.k_proj.weight"] = self.rnd(N_KV * HEAD_DIM, H)
            t[f"{p}.self_attn.v_proj.weight"] = self.rnd(N_KV * HEAD_DIM, H)
            t[f"{p}.self_attn.o_proj.weight"] = self.rnd(H, N_HEAD * HEAD_DIM)
            t[f"{p}.self_attn.q_norm.weight"] = self.rnd(HEAD_DIM)
            t[f"{p}.self_attn.k_norm.weight"] = self.rnd(HEAD_DIM)
        if not self.moe:
            t[f"{p}.mlp.gate_proj.weight"] = self.rnd(N_FF, H)
            t[f"{p}.mlp.up_proj.weight"] = self.rnd(N_FF, H)
            t[f"{p}.mlp.down_proj.weight"] = self.rnd(H, N_FF)
            return
        t[f"{p}.mlp.gate.weight"] = self.rnd(N_EXP, H)
        t[f"{p}.mlp.shared_expert.gate_proj.weight"] = self.rnd(N_FF_SHEXP, H)
        t[f"{p}.mlp.shared_expert.up_proj.weight"] = self.rnd(N_FF_SHEXP, H)
        t[f"{p}.mlp.shared_expert.down_proj.weight"] = self.rnd(H, N_FF_SHEXP)
        t[f"{p}.mlp.shared_expert_gate.weight"] = self.rnd(1, H)
        gate_up, down = self.rnd(N_EXP, 2 * N_FF_EXP, H), self.rnd(N_EXP, H, N_FF_EXP)
        if fused_experts:
            t[f"{p}.mlp.experts.gate_up_proj"] = gate_up
            t[f"{p}.mlp.experts.down_proj"] = down
        else:
            for e in range(N_EXP):
                t[f"{p}.mlp.experts.{e}.gate_proj.weight"] = gate_up[e, :N_FF_EXP]
                t[f"{p}.mlp.experts.{e}.up_proj.weight"] = gate_up[e, N_FF_EXP:]
                t[f"{p}.mlp.experts.{e}.down_proj.weight"] = down[e]

    def add_vision(self):
        p = "model.visual."
        t = self.tensors
        t[p + "patch_embed.proj.weight"] = self.rnd(V_H, 3, 2, V_PATCH, V_PATCH)
        t[p + "patch_embed.proj.bias"] = self.rnd(V_H)
        t[p + "pos_embed.weight"] = self.rnd(V_NPOS, V_H)
        for name, shape in (("norm", (V_H,)), ("linear_fc1", (4 * V_H, 4 * V_H)), ("linear_fc2", (H, 4 * V_H))):
            t[f"{p}merger.{name}.weight"] = self.rnd(*shape)
            t[f"{p}merger.{name}.bias"] = self.rnd(shape[0])
        for i in range(V_DEPTH):
            for name, shape in (("attn.qkv", (3 * V_H, V_H)), ("attn.proj", (V_H, V_H)), ("mlp.linear_fc1", (V_FF, V_H)),
                                ("mlp.linear_fc2", (V_H, V_FF)), ("norm1", (V_H,)), ("norm2", (V_H,))):
                t[f"{p}blocks.{i}.{name}.weight"] = self.rnd(*shape)
                t[f"{p}blocks.{i}.{name}.bias"] = self.rnd(shape[0])
        for j in range(len(self.deepstack)):
            for name, shape in (("norm", (4 * V_H,)), ("linear_fc1", (4 * V_H, 4 * V_H)), ("linear_fc2", (H, 4 * V_H))):
                t[f"{p}deepstack_merger_list.{j}.{name}.weight"] = self.rnd(*shape)
                t[f"{p}deepstack_merger_list.{j}.{name}.bias"] = self.rnd(shape[0])

    def config(self) -> dict:
        arch = "Qwen3_5MoeForConditionalGeneration" if self.moe else "Qwen3_5ForConditionalGeneration"
        return {"architectures": [arch], "text_config": text_config(self.moe), "vision_config": vision_config(self.deepstack), "tie_word_embeddings": False}


def tiled(t: torch.Tensor, head_dim: int, dim: int = 0) -> torch.Tensor:
    """V heads grouped by K head (k * r + j) -> tiled order (j * n_k + k), written as an explicit index list."""
    r = LV_HEADS // LK_HEADS
    order = [k * r + j for j in range(r) for k in range(LK_HEADS)]
    index = torch.tensor([h * head_dim + d for h in order for d in range(head_dim)])
    return t.index_select(dim, index)


class TestQwen35Conversion(unittest.TestCase):
    def convert(self, cls, ckpt: Checkpoint, tensors=None, config=None, ftype=gguf.LlamaFileType.ALL_F32):
        """Returns ({name: (dtype, array)}, {key: value}) for the converted model."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = config or ckpt.config()
            (path / "config.json").write_text(json.dumps(config))
            (path / "preprocessor_config.json").write_text(json.dumps({"image_mean": [0.5] * 3, "image_std": [0.5] * 3}))
            save_file({k: v.contiguous() for k, v in (tensors or ckpt.tensors).items()}, path / "model.safetensors")
            model = cls(path, ftype, path / "unused.gguf", hparams=json.loads(json.dumps(config)))
            model.prepare_tensors()
            model.set_gguf_parameters()
            # copy: the tensors are views of the mmapped checkpoint, which must be closed before the directory is removed
            out = {name: (info.dtype, gguf.LazyNumpyTensor.to_eager(info.tensor).copy()) for name, info in model.gguf_writer.tensors[0].items()}
            kv = {key: value.value for key, value in model.gguf_writer.kv_data[0].items()}
            return out, kv

    def assert_tensor(self, out, name, expected, dtype=gguf.GGMLQuantizationType.F32):
        self.assertIn(name, out)
        actual_dtype, actual = out[name]
        self.assertEqual(actual_dtype, dtype, name)
        np.testing.assert_array_equal(actual, expected.float().numpy(), name)

    def check_text_kv(self, kv, arch):
        expected = {
            "block_count": 4, "embedding_length": H, "attention.head_count": N_HEAD, "attention.head_count_kv": N_KV,
            "attention.key_length": HEAD_DIM, "attention.value_length": HEAD_DIM,
            "ssm.conv_kernel": CONV_K, "ssm.state_size": LK_DIM, "ssm.group_count": LK_HEADS,
            "ssm.time_step_rank": LV_HEADS, "ssm.inner_size": LV_HEADS * LV_DIM,
            "attention.recurrent_layers": [True, True, True, False], "full_attention_interval": 4,
            "rope.dimension_count": HEAD_DIM // 4, "rope.dimension_sections": [11, 11, 10, 0], "rope.freq_base": 10000000,
        }
        for key, value in expected.items():
            self.assertEqual(kv[f"{arch}.{key}"], value, key)
        # the MTP head is not exported
        self.assertNotIn(f"{arch}.nextn_predict_layers", kv)

    def check_linear_attention(self, out, ckpt, bid, merged):
        src = {k.removeprefix(f"model.language_model.layers.{bid}.linear_attn."): v for k, v in ckpt.tensors.items() if k.startswith(f"model.language_model.layers.{bid}.linear_attn.")}
        n_qk = 2 * LK_HEADS * LK_DIM
        qkv = torch.cat([src["in_proj_qkv.weight"][:n_qk], tiled(src["in_proj_qkv.weight"][n_qk:], LV_DIM)])
        z = tiled(src["in_proj_z.weight"], LV_DIM)
        beta, alpha = tiled(src["in_proj_b.weight"], 1), tiled(src["in_proj_a.weight"], 1)
        blk = f"blk.{bid}."
        if merged:
            self.assert_tensor(out, blk + "attn_qkv_gate.weight", torch.cat([qkv, z]))
            self.assert_tensor(out, blk + "ssm_beta_alpha.weight", torch.cat([beta, alpha]))
        else:
            self.assert_tensor(out, blk + "attn_qkv.weight", qkv)
            self.assert_tensor(out, blk + "attn_gate.weight", z)
            self.assert_tensor(out, blk + "ssm_beta.weight", beta)
            self.assert_tensor(out, blk + "ssm_alpha.weight", alpha)
        conv = src["conv1d.weight"].squeeze(1)
        self.assert_tensor(out, blk + "ssm_conv1d.weight", torch.cat([conv[:n_qk], tiled(conv[n_qk:], LV_DIM)]))
        self.assert_tensor(out, blk + "ssm_a", -torch.exp(tiled(src["A_log"], 1)))
        self.assert_tensor(out, blk + "ssm_dt.bias", tiled(src["dt_bias"], 1))
        self.assert_tensor(out, blk + "ssm_norm.weight", src["norm.weight"])
        self.assert_tensor(out, blk + "ssm_out.weight", tiled(src["out_proj.weight"], LV_DIM, dim=1))
        self.assert_tensor(out, blk + "attn_norm.weight", ckpt.tensors[f"model.language_model.layers.{bid}.input_layernorm.weight"] + 1)

    def test_registration(self):
        for name, cls in (
            ("Qwen3_5ForConditionalGeneration", Qwen3_5TextModel), ("Qwen3_5ForCausalLM", Qwen3_5TextModel), ("Qwen3_5TextModel", Qwen3_5TextModel),
            ("Qwen3_5MoeForConditionalGeneration", Qwen3_5MoeTextModel), ("Qwen3_5MoeForCausalLM", Qwen3_5MoeTextModel),
        ):
            self.assertIs(get_model_class(name), cls, name)
        for name in ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"):
            self.assertIs(get_model_class(name, mmproj=True), Qwen3VLVisionModel, name)

    def test_dense(self):
        ckpt = Checkpoint()
        out, kv = self.convert(Qwen3_5TextModel, ckpt)
        self.assertEqual(kv["general.architecture"], "qwen35")
        self.check_text_kv(kv, "qwen35")
        self.assertEqual(kv["qwen35.feed_forward_length"], N_FF)

        expected = {"token_embd.weight": (N_VOCAB, H), "output_norm.weight": (H,), "output.weight": (N_VOCAB, H)}
        for bid, layer_type in enumerate(LAYER_TYPES):
            blk = f"blk.{bid}."
            expected.update({blk + "attn_norm.weight": (H,), blk + "post_attention_norm.weight": (H,),
                             blk + "ffn_gate.weight": (N_FF, H), blk + "ffn_up.weight": (N_FF, H), blk + "ffn_down.weight": (H, N_FF)})
            if layer_type == "linear_attention":
                expected.update({
                    blk + "attn_qkv_gate.weight": (CONV_DIM + LV_HEADS * LV_DIM, H), blk + "ssm_beta_alpha.weight": (2 * LV_HEADS, H),
                    blk + "ssm_conv1d.weight": (CONV_DIM, CONV_K), blk + "ssm_a": (LV_HEADS,), blk + "ssm_dt.bias": (LV_HEADS,),
                    blk + "ssm_norm.weight": (LV_DIM,), blk + "ssm_out.weight": (H, LV_HEADS * LV_DIM),
                })
            else:
                expected.update({
                    blk + "attn_q.weight": (N_HEAD * HEAD_DIM * 2, H), blk + "attn_k.weight": (N_KV * HEAD_DIM, H),
                    blk + "attn_v.weight": (N_KV * HEAD_DIM, H), blk + "attn_output.weight": (H, N_HEAD * HEAD_DIM),
                    blk + "attn_q_norm.weight": (HEAD_DIM,), blk + "attn_k_norm.weight": (HEAD_DIM,),
                })
        self.assertEqual({name: value.shape for name, (_, value) in out.items()}, expected)

        for bid in range(3):
            self.check_linear_attention(out, ckpt, bid, merged=True)
        self.assert_tensor(out, "blk.3.attn_q_norm.weight", ckpt.tensors["model.language_model.layers.3.self_attn.q_norm.weight"] + 1)
        self.assert_tensor(out, "output_norm.weight", ckpt.tensors["model.language_model.norm.weight"] + 1)

    def test_dense_fp8_keeps_separate_projections(self):
        """A pre-quantized checkpoint is not merged: one merged weight cannot hold two per-tensor scales."""
        ckpt = Checkpoint()
        tensors, scales = dict(ckpt.tensors), {}
        for name in [n for n in tensors if ".linear_attn." in n and n.endswith(("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight", "out_proj.weight"))]:
            tensors[name] = tensors[name].to(torch.float8_e4m3fn)
            ckpt.tensors[name] = tensors[name].float()
            tensors[name + "_scale_inv"] = scales[name] = torch.tensor([1.0])
        config = ckpt.config()
        config["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3"}
        out, _ = self.convert(Qwen3_5TextModel, ckpt, tensors=tensors, config=config, ftype=gguf.LlamaFileType.GUESSED)
        for name in ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out"):
            self.assertEqual(out[f"blk.0.{name}.weight"][0], gguf.GGMLQuantizationType.F8_E4M3, name)
        self.assertNotIn("blk.0.attn_qkv_gate.weight", out)
        self.assertNotIn("blk.0.ssm_beta_alpha.weight", out)

    def test_moe(self):
        ckpt = Checkpoint(moe=True)
        out, kv = self.convert(Qwen3_5MoeTextModel, ckpt)
        self.assertEqual(kv["general.architecture"], "qwen35moe")
        self.check_text_kv(kv, "qwen35moe")
        for key, value in (("expert_count", N_EXP), ("expert_used_count", N_EXP_USED), ("expert_feed_forward_length", N_FF_EXP), ("expert_shared_feed_forward_length", N_FF_SHEXP)):
            self.assertEqual(kv[f"qwen35moe.{key}"], value, key)

        for bid in range(3):
            self.check_linear_attention(out, ckpt, bid, merged=False)
        p = "model.language_model.layers.0.mlp."
        gate_up = ckpt.tensors[p + "experts.gate_up_proj"]
        self.assert_tensor(out, "blk.0.ffn_gate_exps.weight", gate_up[:, :N_FF_EXP])
        self.assert_tensor(out, "blk.0.ffn_up_exps.weight", gate_up[:, N_FF_EXP:])
        self.assert_tensor(out, "blk.0.ffn_down_exps.weight", ckpt.tensors[p + "experts.down_proj"])
        self.assert_tensor(out, "blk.0.ffn_gate_inp.weight", ckpt.tensors[p + "gate.weight"])
        self.assert_tensor(out, "blk.0.ffn_gate_inp_shexp.weight", ckpt.tensors[p + "shared_expert_gate.weight"])
        for proj in ("gate", "up", "down"):
            self.assert_tensor(out, f"blk.0.ffn_{proj}_shexp.weight", ckpt.tensors[p + f"shared_expert.{proj}_proj.weight"])
        self.assertFalse([name for name in out if name.startswith("blk.4.") or "nextn" in name])

        # per-expert checkpoints give the same GGUF tensors
        per_expert, _ = self.convert(Qwen3_5MoeTextModel, Checkpoint(moe=True, fused_experts=False))
        self.assertEqual(per_expert.keys(), out.keys())
        for name in out:
            np.testing.assert_array_equal(per_expert[name][1], out[name][1], name)

    def test_nvfp4_linear_attention_is_rejected(self):
        model = object.__new__(Qwen3_5TextModel)
        model.hparams = text_config(False)
        weight, scale = torch.zeros(CONV_DIM, H // 2, dtype=torch.uint8), torch.zeros(CONV_DIM, H // 16)
        with self.assertRaises(NotImplementedError):
            model._repack_nvfp4("model.layers.0.linear_attn.in_proj_qkv.weight", weight, scale, torch.ones(1), torch.ones(1))

    def test_mmproj(self):
        for deepstack in ([], [1]):
            with self.subTest(deepstack=deepstack):
                ckpt = Checkpoint(moe=bool(deepstack), deepstack=deepstack)
                out, kv = self.convert(Qwen3VLVisionModel, ckpt)
                self.assertEqual(kv["clip.projector_type"], "qwen3vl_merger")
                self.assertEqual(kv["clip.vision.image_size"], 4 * V_PATCH)  # sqrt(num_position_embeddings) * patch_size
                self.assertEqual(kv["clip.vision.block_count"], V_DEPTH)
                self.assertEqual(kv["clip.vision.projection_dim"], H)
                self.assertEqual(kv["clip.vision.spatial_merge_size"], 2)
                self.assertEqual(kv["clip.vision.is_deepstack_layers"], [i in deepstack for i in range(V_DEPTH)])

                p = "model.visual."
                patch = ckpt.tensors[p + "patch_embed.proj.weight"]
                self.assert_tensor(out, "v.patch_embd.weight", patch[:, :, 0])
                self.assert_tensor(out, "v.patch_embd.weight.1", patch[:, :, 1])
                self.assert_tensor(out, "mm.0.weight", ckpt.tensors[p + "merger.linear_fc1.weight"], gguf.GGMLQuantizationType.F32)
                self.assert_tensor(out, "mm.2.weight", ckpt.tensors[p + "merger.linear_fc2.weight"], gguf.GGMLQuantizationType.F32)
                self.assert_tensor(out, "v.post_ln.weight", ckpt.tensors[p + "merger.norm.weight"])
                self.assert_tensor(out, "v.blk.1.attn_qkv.weight", ckpt.tensors[p + "blocks.1.attn.qkv.weight"])
                for idx in deepstack:
                    self.assert_tensor(out, f"v.deepstack.{idx}.fc1.weight", ckpt.tensors[p + "deepstack_merger_list.0.linear_fc1.weight"])
                n_expected = 4 + 2 * 3 + V_DEPTH * 12 + 6 * len(deepstack)
                self.assertEqual(len(out), n_expected)
                self.assertTrue(all(name.startswith(("v.", "mm.")) for name in out))


if __name__ == "__main__":
    unittest.main()

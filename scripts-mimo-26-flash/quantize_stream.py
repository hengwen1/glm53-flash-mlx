"""Quantize MiMo-V2.6-Flash one decoder layer at a time, never holding the 173 GB source.

Why per layer rather than per shard: the release spreads each layer's 256 routed experts
over all 64 expert-parallel shards, and the runtime's sanitize stacks them into one
`switch_mlp` tensor per projection -- it needs all 256 present at once. So the index is
walked by layer: the tensors of layer N are gathered lazily from whichever shards hold
them (the shard payload is read directly, one tensor at a time, nothing is memory-mapped
wholesale), sanitized as a group, quantized tensor by tensor, and written out.

Why the weights are not read with `mx.load` or `safetensors.safe_open(framework="mlx")`:
the release is stored as block-128 FP8 (`F8_E4M3` + `F32` `weight_scale_inv`) and MXFP4
(`U8` + `U8` `weight_scale`). The MLX binding of safetensors in this environment cannot
materialise either without going through NumPy, which has no bf16/fp8 dtype. A tiny
header reader (see `TensorStore`) reproduces exactly what `mx.load` returns -- FP8 as raw
`uint8` for `mx.from_fp8`, BF16 as `uint16` reinterpreted -- and reads only the keys asked
for, so a layer costs a few tens of MB instead of 2.6 GB per shard.

The source layout has two transforms the runtime's `sanitize` performs, done here instead
so the streaming path never needs the whole checkpoint:
  * `self_attn.qkv_proj` (FP8) is one fused tensor that was already tensor-parallel
    sharded, each shard's FP8 scale grid padded to a 128-row block individually. It is
    dequantized per shard and split into q/k/v (see `split_fused_qkv`); feeding the scale
    over the fused tensor as one grid crosses shard boundaries and silently corrupts every
    shard after the first.
  * routed experts (`U8` MXFP4 + `U8` scale) are stacked into `switch_mlp` tensors.

Which modules are quantized is *derived* from the runtime: a full-size `Model` is built
lazily (MLX allocates nothing) and every leaf that defines `to_quantized` -- the question
`nn.quantize` asks -- is recorded, then filtered by the recipe. If oMLX's MiMo patch is not
importable the module list is enumerated from the config instead, and the same recipe is
applied to it.

Recipe:
  * routed experts (`switch_mlp`, ~97% of the text parameters)   --bits / --expert-bits, group 64
    (the release ships them MXFP4: `--expert-bits mxfp4` keeps that bit-for-bit and skips
    the dequantize/requantize round-trip)
  * everything else quantizable                                   --bits / --other-bits, group 64
    (attention q/k/v/o, dense MLPs, embeddings, lm_head)
  * kept as stored: the MoE router (`mlp.gate.weight`, bf16) and its correction bias
    (fp32), attention sink biases, all norms, the vision tower, the audio encoder and the
    speech embeddings.
  * the built-in MTP head (`model.mtp.*`, 3 next-token predictors) follows the same family
    rules and is kept by default; pass --no-mtp to drop it.

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 8 --expert-bits mxfp4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4 --text-only

Every bits option takes an affine bit width (2-8) or a microscaling format, `mxfp4`/`mxfp8`
(`mxf4`/`mxf8` accepted): E2M1/E4M3 elements sharing one e8m0 scale per group of 32 and no
bias term, so the group size is pinned to 32 for those modules.

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits mxfp4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 8 --expert-bits mxfp4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

AUX_FILES = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "special_tokens_map.json", "chat_template.jinja", "preprocessor_config.json",
    "configuration_mimo_v2.py", "modeling_mimo_v2.py", "LICENSE",
)
AUX_DIRS = ("audio_tokenizer",)
# Router logits and the sigmoid correction bias stay in the stored dtype: quantization
# noise there changes which experts fire, not just the expert output.
GATING_BF16 = ("mlp.gate", "e_score_correction_bias")
NON_TEXT_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.")

_MXFP_RE = re.compile(r"^mx(?:f(?:p)?)?([48])$")

_FP8_BLOCK = 128
_TP_CANDIDATES = (1, 2, 4, 8, 16, 32)

# safetensors dtype -> NumPy little-endian dtype. BF16 and F8_* are special-cased: MLX wants
# them as raw uint16/uint8, and NumPy has no native bf16/fp8.
_NP_DTYPE = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2",
    "I8": "i1", "U8": "u1", "U16": "<u2", "U32": "<u4", "U64": "<u8", "BOOL": "?",
}


# --------------------------------------------------------------------------------------
# bits parsing
# --------------------------------------------------------------------------------------

def parse_bits(value) -> dict:
    """A --bits value -> {"bits", "mode"}: affine widths (2-8) or mxfp4/mxfp8 (mxf4/mxf8 accepted)."""
    text = str(value).strip().lower()
    if m := _MXFP_RE.match(text):
        return {"bits": int(m.group(1)), "mode": "mxfp" + m.group(1)}
    try:
        return {"bits": int(text), "mode": "affine"}
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid bits {value!r}: use 2-8, mxfp4 or mxfp8") from None


def quant_params(mode_bits: dict, group_size: int) -> dict:
    """Complete per-module params; the mx spec requires (and this pins) a group size of 32."""
    mode = mode_bits["mode"]
    out = {"group_size": 32 if mode != "affine" else group_size, "bits": mode_bits["bits"]}
    if mode != "affine":
        out["mode"] = mode
    return out


def is_identity_mxfp4(params: dict) -> bool:
    """Whether a module's recipe reproduces the release's own MXFP4 storage exactly."""
    return params.get("mode") == "mxfp4" and params["bits"] == 4 and params["group_size"] == 32


def fmt_bits(params: dict) -> str:
    return params["mode"] if params["mode"] != "affine" else f"{params['bits']}b"


# --------------------------------------------------------------------------------------
# recipe / quantizable modules
# --------------------------------------------------------------------------------------

def recipe(path: str, args) -> dict | None:
    if path.startswith(NON_TEXT_PREFIXES):
        return None
    if path.endswith("mlp.gate") or path.endswith(GATING_BF16):
        return None
    for sub, params in (args.override or []):
        if sub in path:
            return quant_params(params, args.group_size)
    if ".switch_mlp." in path:
        return quant_params(args.expert_params, args.group_size)
    return quant_params(args.other_params, args.group_size)


def runtime_model(raw_cfg: dict, with_mtp: bool):
    """A lazily-built MiMo Model (from oMLX's vendored mlx-lm patch), or None if unavailable."""
    try:
        from omlx.patches.mimo_v2 import apply_mimo_v2_patch
        apply_mimo_v2_patch()
        from omlx.patches.mlx_lm_mtp import set_mtp_active
        set_mtp_active(with_mtp)
        import mlx_lm.models.mimo_v2 as mimo
    except Exception as exc:  # noqa: BLE001 - the fallback is the point
        print(f"  (runtime model unavailable: {exc}); enumerating modules from config", flush=True)
        return None
    cfg = dict(raw_cfg)
    if not with_mtp:
        cfg["num_nextn_predict_layers"] = 0
    return mimo.Model(mimo.ModelArgs.from_dict(cfg))


def quantizable_paths(model, args) -> dict[str, dict]:
    """Every Linear/Embedding leaf the recipe admits, keyed by runtime module path."""
    out = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        params = recipe(path, args)
        if params is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if weight.shape[-1] % params["group_size"]:
            print(f"  skip {path}: in-dim {weight.shape[-1]} not divisible by {params['group_size']}")
            continue
        out[path] = params
    return out


def enumerate_paths(raw_cfg: dict, with_mtp: bool) -> list[str]:
    """Fallback module list when the runtime is not importable; mirrors the model tree."""
    paths = ["model.embed_tokens", "lm_head"]
    for i in range(raw_cfg["num_hidden_layers"]):
        paths += [f"model.layers.{i}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        mlp = "switch_mlp" if raw_cfg["moe_layer_freq"][i] else ""
        for p in ("gate_proj", "up_proj", "down_proj"):
            paths.append(f"model.layers.{i}.mlp.{mlp + '.' if mlp else ''}{p}")
    if with_mtp:
        for j in range(int(raw_cfg.get("num_nextn_predict_layers") or 0)):
            paths += [
                f"model.mtp.layers.{j}.eh_proj",
                f"model.mtp.layers.{j}.self_attn.q_proj",
                f"model.mtp.layers.{j}.self_attn.k_proj",
                f"model.mtp.layers.{j}.self_attn.v_proj",
                f"model.mtp.layers.{j}.self_attn.o_proj",
                f"model.mtp.layers.{j}.mlp.gate_proj",
                f"model.mtp.layers.{j}.mlp.up_proj",
                f"model.mtp.layers.{j}.mlp.down_proj",
            ]
    return paths


def fallback_quantizable_paths(raw_cfg: dict, args, with_mtp: bool) -> dict[str, dict]:
    out = {}
    for path in enumerate_paths(raw_cfg, with_mtp):
        params = recipe(path, args)
        if params is not None:
            out[path] = params
    return out


# --------------------------------------------------------------------------------------
# source reader: raw safetensors header walk, one tensor at a time
# --------------------------------------------------------------------------------------

def _to_mx(dtype: str, shape, buf: bytes) -> mx.array:
    if dtype == "BF16":
        # NumPy has no bf16; reinterpret raw little-endian uint16 as an MLX bf16 array.
        return mx.array(np.frombuffer(buf, dtype="<u2").reshape(shape)).view(mx.bfloat16)
    if dtype.startswith("F8"):
        # Same shape mx.load returns for fp8: raw bytes, consumed by mx.from_fp8.
        return mx.array(np.frombuffer(buf, dtype="u1").reshape(shape))
    return mx.array(np.frombuffer(buf, dtype=_NP_DTYPE[dtype]).reshape(shape))


class TensorStore:
    """Reads named tensors out of one sharded safetensors file via its header."""

    def __init__(self, path: Path):
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n).decode("utf-8"))
        self._data = 8 + n

    def shape(self, key: str):
        info = self.header.get(key)
        return None if info is None else tuple(info["shape"])

    def get(self, key: str) -> mx.array:
        info = self.header[key]
        start, end = info["data_offsets"]
        self._fh.seek(self._data + start)
        return _to_mx(info["dtype"], info["shape"], self._fh.read(end - start))

    def close(self) -> None:
        self._fh.close()


class SourceReader:
    def __init__(self, src: Path, index: dict):
        self._src, self._index = src, index
        self._files: dict[str, TensorStore] = {}

    def _store(self, shard: str) -> TensorStore:
        store = self._files.get(shard)
        if store is None:
            store = self._files[shard] = TensorStore(self._src / shard)
        return store

    def get(self, key: str) -> mx.array:
        return self._store(self._index[key]).get(key)

    def shape(self, key: str):
        shard = self._index.get(key)
        return None if shard is None else self._store(shard).shape(key)

    def close(self) -> None:
        for store in self._files.values():
            store.close()
        self._files.clear()


# --------------------------------------------------------------------------------------
# fused qkv: block-128 FP8, pre-sharded for tensor parallelism
# --------------------------------------------------------------------------------------

def head_geometry(raw_cfg: dict, layer_idx: int):
    if bool(raw_cfg["hybrid_layer_pattern"][layer_idx]):
        return (raw_cfg["swa_num_attention_heads"], raw_cfg["swa_num_key_value_heads"],
                raw_cfg["swa_head_dim"], raw_cfg["swa_v_head_dim"])
    return (raw_cfg["num_attention_heads"], raw_cfg["num_key_value_heads"],
            raw_cfg["head_dim"], raw_cfg["v_head_dim"])


def mtp_head_geometry(raw_cfg: dict):
    return (raw_cfg["swa_num_attention_heads"], raw_cfg["swa_num_key_value_heads"],
            raw_cfg["swa_head_dim"], raw_cfg["swa_v_head_dim"])


def _part_rows(n_h, n_kv, hd, vhd, tp):
    return (n_h // tp) * hd, (n_kv // tp) * hd, (n_kv // tp) * vhd


def _layout_matches(n_h, n_kv, hd, vhd, tp, qkv_rows, scale_rows):
    if n_h % tp or n_kv % tp:
        return False
    rows = sum(_part_rows(n_h, n_kv, hd, vhd, tp))
    padded = -(-rows // _FP8_BLOCK) * _FP8_BLOCK
    return rows * tp == qkv_rows and padded * tp == scale_rows * _FP8_BLOCK


def detect_fused_qkv_tp(raw_cfg: dict, shape_of) -> int:
    """Recover the tensor-parallel degree from the full-attention scale grids.

    Full attention is what pins it down: sliding-window geometry happens to be block-aligned,
    which leaves its degree ambiguous. Each shard's FP8 grid is padded on its own, so the
    padding shows up interleaved through the fused tensor.
    """
    for i in range(raw_cfg["num_hidden_layers"]):
        if bool(raw_cfg["hybrid_layer_pattern"][i]):
            continue
        qkv_key = f"model.layers.{i}.self_attn.qkv_proj.weight"
        scale_key = f"{qkv_key}_scale_inv"
        qkv_shape, scale_shape = shape_of(qkv_key), shape_of(scale_key)
        if qkv_shape is None or scale_shape is None:
            continue
        n_h, n_kv, hd, vhd = head_geometry(raw_cfg, i)
        for tp in _TP_CANDIDATES:
            if _layout_matches(n_h, n_kv, hd, vhd, tp, qkv_shape[0], scale_shape[0]):
                return tp
        raise ValueError(
            f"unable to determine fused-qkv TP layout from layer {i} "
            f"(weight rows={qkv_shape[0]}, scale rows={scale_shape[0]})"
        )
    return 1  # no fused qkv with a scale companion: the split path is inert


def split_fused_qkv(qkv_fp8: mx.array, scale_inv: mx.array, tp: int, n_h, n_kv, hd, vhd):
    """Dequantize a pre-sharded fused qkv FP8 tensor and split it into q/k/v (bf16)."""
    block = _FP8_BLOCK
    bf16 = mx.bfloat16
    q_pr, k_pr, v_pr = _part_rows(n_h, n_kv, hd, vhd, tp)
    actual = q_pr + k_pr + v_pr
    padded = -(-actual // block) * block
    n = qkv_fp8.shape[-1]

    qkv = mx.from_fp8(qkv_fp8, dtype=bf16).reshape(tp, actual, n)
    if padded > actual:
        qkv = mx.pad(qkv, ((0, 0), (0, padded - actual), (0, 0)))
    n_cols = scale_inv.shape[1]
    pad_side = block * n_cols - n
    if pad_side > 0:
        qkv = mx.pad(qkv, ((0, 0), (0, 0), (0, pad_side)))

    blocked = qkv.reshape(tp * padded // block, block, n_cols, block)
    qkv = (blocked * scale_inv[:, None, :, None]).reshape(tp, padded, n_cols * block)[:, :actual, :n]

    q = mx.contiguous(qkv[:, :q_pr, :]).reshape(tp * q_pr, n).astype(bf16)
    k = mx.contiguous(qkv[:, q_pr:q_pr + k_pr, :]).reshape(tp * k_pr, n).astype(bf16)
    v = mx.contiguous(qkv[:, q_pr + k_pr:, :]).reshape(tp * v_pr, n).astype(bf16)
    return q, k, v


def dequant_block_fp8(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Undo the plain block-128 FP8 storage of one dense weight."""
    block = _FP8_BLOCK
    w = mx.from_fp8(weight, dtype=mx.bfloat16)
    m, n = w.shape
    pad_m, pad_n = (-m) % block, (-n) % block
    if pad_m or pad_n:
        w = mx.pad(w, ((0, pad_m), (0, pad_n)))
    w = w.reshape((m + pad_m) // block, block, (n + pad_n) // block, block)
    w = (w * scale_inv[:, None, :, None]).reshape(m + pad_m, n + pad_n)
    return w[:m, :n].astype(mx.bfloat16)


# --------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------

def materialise(x):
    with mx.stream(mx.cpu):  # keep the read off the Metal command buffer
        mx.eval(x)
    return x


def quantize(w: mx.array, params: dict):
    # mxfp modes return (weight, scales); affine returns (weight, scales, biases)
    kw = dict(group_size=params["group_size"], bits=params["bits"], mode=params.get("mode", "affine"))
    try:
        out = mx.quantize(materialise(w), **kw); mx.eval(out); return out
    except RuntimeError as err:
        if "Timeout" not in str(err):
            raise
        with mx.stream(mx.cpu):
            out = mx.quantize(w, **kw); mx.eval(out); return out


def emit_experts(raw: dict, prefix: str, params: dict, n_experts: int, emit) -> int:
    """Stack one layer's routed experts into the runtime's `switch_mlp` layout.

    Each expert is dequantized (MXFP4 -> bf16), quantized to the recipe and appended, so a
    layer's full bf16 expert set (12.9 GB) is never resident at once. When the recipe is the
    release's own MXFP4 the packed bytes are stacked verbatim: no round-trip, no loss.
    """
    identity = is_identity_mxfp4(params)
    stacked = 0
    for proj in ("gate_proj", "up_proj", "down_proj"):
        if f"{prefix}.experts.0.{proj}.weight" not in raw:
            continue
        stacked += 1
        packed, scales, biases = [], [], []
        for e in range(n_experts):
            w = raw.pop(f"{prefix}.experts.{e}.{proj}.weight")
            s = raw.pop(f"{prefix}.experts.{e}.{proj}.weight_scale")
            w32 = w.view(mx.uint32)
            if identity:
                packed.append(w32); scales.append(s)
                continue
            dense = materialise(mx.dequantize(w32, s, group_size=32, bits=4, mode="mxfp4"))
            q = quantize(dense, params)
            packed.append(q[0]); scales.append(q[1])
            if len(q) == 3:
                biases.append(q[2])
        target = f"{prefix}.switch_mlp.{proj}"
        emit(target + ".weight", mx.stack(packed))
        emit(target + ".scales", mx.stack(scales))
        if biases:
            emit(target + ".biases", mx.stack(biases))
    return stacked


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF snapshot directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--bits", dest="bits_params", type=parse_bits, required=True, metavar="BITS",
                    help="affine bit width (2-8) or mxfp4/mxfp8; default for every family")
    ap.add_argument("--expert-bits", dest="expert_params", type=parse_bits, metavar="BITS",
                    help="routed experts (switch_mlp); defaults to --bits (mxfp4 keeps the release)")
    ap.add_argument("--other-bits", dest="other_params", type=parse_bits, metavar="BITS",
                    help="everything else quantizable; defaults to --bits")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-mtp", action="store_true",
                    help="drop the built-in MTP head tensors (model.mtp.*)")
    ap.add_argument("--text-only", action="store_true",
                    help="keep the vision tower / audio encoder / speech embeddings out of the output")
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="only convert the first N backbone layers (debugging)")
    ap.add_argument("--override", action="append", metavar="SUBSTRING=BITS",
                    help="bits for modules whose path contains SUBSTRING (ablations); may repeat")
    ap.add_argument("--max-context", type=int, default=None, metavar="TOKENS",
                    help="cap the built config's `max_position_embeddings`; keeps the source value by default")
    args = ap.parse_args()
    args.override = [(o.split("=")[0], parse_bits(o.split("=")[1])) for o in (args.override or [])]
    args.expert_params = args.expert_params or args.bits_params
    args.other_params = args.other_params or args.bits_params

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.loads((src / "config.json").read_text())
    if args.max_context is not None:
        cap = raw_cfg.get("max_position_embeddings")
        if cap and args.max_context > cap:
            sys.exit(f"--max-context {args.max_context} exceeds the model's max_position_embeddings {cap}")
        raw_cfg["max_position_embeddings"] = args.max_context

    expert_params = quant_params(args.expert_params, args.group_size)
    with_mtp = not args.no_mtp
    model = runtime_model(raw_cfg, with_mtp)
    if model is not None:
        qpaths = quantizable_paths(model, args)
    else:
        qpaths = fallback_quantizable_paths(raw_cfg, args, with_mtp)
    print(f"quantizable modules: {len(qpaths)} (experts {fmt_bits(args.expert_params)}, "
          f"other {fmt_bits(args.other_params)})", flush=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    reader = SourceReader(src, index)

    n_layers = raw_cfg["num_hidden_layers"]
    n_mtp = int(raw_cfg.get("num_nextn_predict_layers") or 0) if with_mtp else 0
    n_experts = int(raw_cfg["n_routed_experts"])

    groups: dict[str, list[str]] = defaultdict(list)
    for key in index:
        if key.startswith("model.mtp."):
            groups[f"mtp{int(re.search(r'model\.mtp\.layers\.(\d+)\.', key).group(1))}"].append(key)
        elif key.startswith("visual."):
            groups["vision"].append(key)
        elif key.startswith("audio_encoder."):
            groups["audio"].append(key)
        elif key.startswith("speech_embeddings."):
            groups["speech"].append(key)
        elif m := re.search(r"(?:^|\.)layers\.(\d+)\.", key):
            groups[f"layer{int(m.group(1)):03d}"].append(key)
        else:
            groups["top"].append(key)

    backbone = [f"layer{i:03d}" for i in range(n_layers)]
    if args.limit_layers:
        backbone = backbone[: args.limit_layers]
    order = ["top"] + backbone
    if n_mtp:
        order += [f"mtp{i}" for i in range(n_mtp)]
    if not args.text_only:
        order += ["vision", "audio", "speech"]
    dropped = sorted(g for g in groups if g not in order)
    print(f"groups: {len(order)}" + (f" (dropping {dropped})" if dropped else ""), flush=True)

    tp = detect_fused_qkv_tp(raw_cfg, reader.shape)
    print(f"fused-qkv tensor-parallel degree: {tp}", flush=True)

    target = args.shard_gb * 1e9
    out_index, pending = {}, {}
    pending_bytes = total_out = out_n = 0
    counts = {"quantized": 0, "as_stored": 0, "mtp": 0}
    started = time.time()

    def emit(key: str, value: mx.array) -> None:
        nonlocal pending_bytes
        pending[key] = value
        pending_bytes += value.nbytes
        if pending_bytes >= target:
            flush()

    def flush() -> None:
        nonlocal pending, pending_bytes, out_n, total_out
        if not pending:
            return
        out_n += 1
        name = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(str(dst / name), pending, metadata={"format": "mlx"})
        for key in pending:
            out_index[key] = name
        size = (dst / name).stat().st_size
        total_out += size
        print(f"  -> {name}  {len(pending)} tensors  {size/1e9:.2f} GB  "
              f"(total {total_out/1e9:.1f} GB, {time.time()-started:.0f}s)", flush=True)
        pending, pending_bytes = {}, 0

    for gi, g in enumerate(order, 1):
        keys = groups.get(g)
        if not keys:
            continue
        raw = {key: reader.get(key) for key in keys}

        # 1) fused qkv -> q/k/v (per-shard FP8 scale grids)
        if g.startswith("layer") or g.startswith("mtp"):
            layer_idx = int(g[5:]) if g.startswith("layer") else int(g[3:])
            prefix = (f"model.layers.{layer_idx}.self_attn" if g.startswith("layer")
                      else f"model.mtp.layers.{layer_idx}.self_attn")
            qkv_key = f"{prefix}.qkv_proj.weight"
            if qkv_key in raw:
                geom = head_geometry(raw_cfg, layer_idx) if g.startswith("layer") else mtp_head_geometry(raw_cfg)
                q, k, v = split_fused_qkv(raw.pop(qkv_key), raw.pop(f"{qkv_key}_scale_inv"), tp, *geom)
                raw[f"{prefix}.q_proj.weight"] = q
                raw[f"{prefix}.k_proj.weight"] = k
                raw[f"{prefix}.v_proj.weight"] = v

        # 2) remaining block-128 FP8 -> bf16
        for scale_key in [k for k in raw if k.endswith("_scale_inv")]:
            weight_key = scale_key[: -len("_scale_inv")]
            raw[weight_key] = dequant_block_fp8(raw.pop(weight_key), raw.pop(scale_key))

        # 3) routed experts -> quantized switch_mlp
        if g.startswith("layer") and raw_cfg["moe_layer_freq"][int(g[5:])]:
            prefix = f"model.layers.{int(g[5:])}.mlp"
            counts["quantized"] += emit_experts(raw, prefix, expert_params, n_experts, emit)

        # 4) quantize / pass through the rest
        for key, value in raw.items():
            module = key.rsplit(".", 1)[0]
            if "mtp." in key:
                counts["mtp"] += 1
            if key.endswith(".weight") and module in qpaths:
                q = quantize(value, qpaths[module])
                emit(module + ".weight", q[0])
                emit(module + ".scales", q[1])
                if len(q) == 3:
                    emit(module + ".biases", q[2])
                counts["quantized"] += 1
            else:
                emit(key, materialise(value))
                counts["as_stored"] += 1
        print(f"[{gi}/{len(order)}] {g}: {len(keys)} -> {len(raw)} tensors", flush=True)
        del raw
        mx.clear_cache()
    flush()
    reader.close()

    root_quant = quant_params(args.bits_params, args.group_size)
    quant = dict(root_quant)
    root_mode = root_quant.get("mode")
    for path, params in qpaths.items():
        # Compare against the pristine root, not the dict we are growing, or the first
        # override makes every later module look different and get a redundant entry.
        if params != root_quant:
            entry = dict(params)
            if root_mode is not None and "mode" not in entry:
                entry["mode"] = "affine"
            quant[path] = entry
    cfg_out = dict(raw_cfg)
    cfg_out.pop("quantization_config", None)
    cfg_out["quantization"] = quant
    cfg_out["quantization_config"] = quant
    if args.no_mtp:
        cfg_out["num_nextn_predict_layers"] = 0
        if "text_config" in cfg_out:
            cfg_out["text_config"]["num_nextn_predict_layers"] = 0
    json.dump(cfg_out, open(dst / "config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": total_out}, "weight_map": out_index},
              open(dst / "model.safetensors.index.json", "w"), indent=2)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    if not args.text_only:
        for name in AUX_DIRS:
            if (src / name).is_dir():
                shutil.copytree(src / name, dst / name, dirs_exist_ok=True)
    print(f"\n{out_n} shards, {total_out/1e9:.1f} GB, {(time.time()-started)/60:.1f} min; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

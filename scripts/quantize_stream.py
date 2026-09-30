"""Quantize GLM-5.3-Flash one decoder layer at a time, never holding the 643 GB source.

Why per layer rather than per shard: the release stores the 288 routed experts of a layer as
separate tensors, and the runtime's sanitize stacks them into one `switch_mlp` tensor per
projection — it needs all 288 present at once. So the index is walked by layer: the tensors of
layer N are gathered lazily from whichever shards hold them (MLX memory-maps, nothing is read
until evaluation), sanitized as a group, quantized tensor by tensor, and written out.

Which modules are quantized is *derived* from the runtime: a full-size `Model` is built lazily
and every leaf that defines `to_quantized` — the question `nn.quantize` asks — is recorded, then
filtered by the recipe. Output keys follow the runtime's own (mlx-vlm) naming, so the checkpoint
reloads through `mlx_vlm`-style loaders and through this package's `load.py` alike.

Recipe:
  * routed experts (`switch_mlp`, 97% of parameters)      --bits / --expert-bits, group 64
  * everything else quantizable                            --bits / --other-bits, group 64
    (MLA low-rank projections, absorbed kv_b, shared experts, dense MLPs,
    embeddings, lm_head).
  * KDA input projections: always 8-bit affine (needed for fused matmul kernel in stock omlx)
    or --kda-bits
  * lightning-indexer projections: always 8-bit (block selection errors compound; ~0.2%)
    or --indexer-bits
  * kept as stored: the MoE router and its correction bias, mHC arrays (fp32 `base`/`scale`),
    KDA `A_log`/`dt_bias` (fp32), convolutions, norms, the vision tower (bf16)
  * the built-in MTP head (`mtp.*`, layer 45) follows the same family rules and is kept by default;
    pass --no-mtp to drop it (earlier builds did).

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 4 --other-bits 8   # mixed

Every bits option takes an affine bit width (2-8) or a microscaling format, `mxfp4`/`mxfp8`
(`mxf4`/`mxf8` accepted): E2M1/E4M3 elements sharing one e8m0 scale per group of 32 and no bias
term, so the group size is pinned to 32 for those modules.

    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits mxfp4
    python scripts/quantize_stream.py --src <hf dir> --dst <out dir> --bits 8 --expert-bits mxfp4  # mixed
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

try:
    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_runtime_patch
    apply_mlx_vlm_mtp_runtime_patch("glm5_next")
    from mlx_vlm.models.glm5_next import Model, ModelConfig, TextConfig, VisionConfig
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from glm53_flash_mlx.glm5_next import Model, ModelConfig, TextConfig, VisionConfig

AUX_FILES = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "chat_template.jinja", "processor_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE",
)
GATING_BF16 = ("mlp.gate", "e_score_correction_bias")
KDA_FUSED_PROJS = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.forget_gate.f_a_proj",
    ".self_attn.g_a_proj",
    ".self_attn.b_proj",
)

_MXFP_RE = re.compile(r"^mx(?:f(?:p)?)?([48])$")


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


def fmt_bits(params: dict) -> str:
    return params["mode"] if params["mode"] != "affine" else f"{params['bits']}b"


def recipe(path: str, args) -> dict | None:
    if path.startswith("vision_model"):
        return None
    if path.endswith("mlp.gate") or path.endswith(GATING_BF16):
        return None
    for sub, params in (args.override or []):
        if sub in path:
            return quant_params(params, args.group_size)
    if any(path.endswith(proj) for proj in KDA_FUSED_PROJS):
        return quant_params(args.kda_params, args.group_size)
    if ".indexer." in path:
        return quant_params(args.indexer_params, args.group_size)
    if ".switch_mlp." in path:
        return quant_params(args.expert_params, args.group_size)
    return quant_params(args.other_params, args.group_size)


def quantizable_paths(cfg, args) -> dict[str, dict]:
    model = Model(cfg)  # lazy: nothing allocated
    out = {}
    for path, module in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if not hasattr(module, "to_quantized"):
            continue
        params = recipe(path, args)
        if params is None:
            continue
        if module.weight.shape[-1] % params["group_size"]:
            print(f"  skip {path}: in-dim {module.weight.shape[-1]} not divisible by {params['group_size']}")
            continue
        out[path] = params
    return out


def materialise(x):
    with mx.stream(mx.cpu):
        mx.eval(x)
    return x


def quantize(w, params):
    # mxfp modes return (weight, scales) — no bias term; affine returns (weight, scales, biases)
    kw = dict(group_size=params["group_size"], bits=params["bits"], mode=params.get("mode", "affine"))
    try:
        out = mx.quantize(materialise(w), **kw); mx.eval(out); return out
    except RuntimeError as err:
        if "Timeout" not in str(err):
            raise
        with mx.stream(mx.cpu):
            out = mx.quantize(w, **kw); mx.eval(out); return out


def make_config(raw_cfg: dict, with_mtp: bool = True) -> ModelConfig:
    cfg = ModelConfig.from_dict(raw_cfg)
    tc = dict(raw_cfg["text_config"])
    if not with_mtp:
        tc["num_nextn_predict_layers"] = 0
    cfg.text_config = TextConfig.from_dict(tc)
    cfg.vision_config = VisionConfig.from_dict(raw_cfg["vision_config"])
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", dest="bits_params", type=parse_bits, required=True, metavar="BITS",
                    help="affine bit width (2-8) or mxfp4/mxfp8; default for every family")
    ap.add_argument("--expert-bits", dest="expert_params", type=parse_bits, metavar="BITS",
                    help="routed experts (switch_mlp); defaults to --bits")
    ap.add_argument("--other-bits", dest="other_params", type=parse_bits, metavar="BITS",
                    help="everything else quantizable; defaults to --bits")
    ap.add_argument("--kda-bits", dest="kda_params", type=parse_bits, metavar="BITS",
                    help="linear attention (KDA) input projections; defaults to 8b affine")
    ap.add_argument("--indexer-bits", dest="indexer_params", type=parse_bits, metavar="BITS",
                    help="lightning-indexer projections; defaults to 8b affine")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-mtp", action="store_true",
                    help="drop the built-in MTP head tensors (layer 45)")
    ap.add_argument("--shard-gb", type=float, default=10.0)
    ap.add_argument("--limit-layers", type=int, default=0)
    ap.add_argument("--override", action="append", metavar="SUBSTRING=BITS",
                    help="bits for modules whose path contains SUBSTRING (ablations); may repeat")
    ap.add_argument("--max-context", type=int, default=None, metavar="TOKENS",
                    help="cap the built config's `max_position_embeddings`; keeps the source value by default")
    args = ap.parse_args()
    args.override = [(o.split("=")[0], parse_bits(o.split("=")[1])) for o in (args.override or [])]
    args.expert_params = args.expert_params or args.bits_params
    args.other_params = args.other_params or args.bits_params
    args.kda_params = args.kda_params or parse_bits(8)
    args.indexer_params = args.indexer_params or parse_bits(8)

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    raw_cfg = json.load(open(src / "config.json"))
    if args.max_context is not None:
        src_cap = raw_cfg.get("max_position_embeddings") or raw_cfg.get("text_config", {}).get("max_position_embeddings")
        if src_cap and args.max_context > src_cap:
            sys.exit(f"--max-context {args.max_context} exceeds the model's "
                     f"max_position_embeddings {src_cap}")
        raw_cfg["max_position_embeddings"] = args.max_context
        raw_cfg.setdefault("text_config", {})["max_position_embeddings"] = args.max_context

    cfg = make_config(raw_cfg, with_mtp=not args.no_mtp)
    model = Model(cfg)
    qpaths = quantizable_paths(cfg, args)
    print(f"quantizable modules: {len(qpaths)} (experts {fmt_bits(args.expert_params)}, "
          f"other {fmt_bits(args.other_params)}, "
          f"kda {fmt_bits(args.kda_params)}, "
          f"indexer {fmt_bits(args.indexer_params)})", flush=True)

    index = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    n_layers = cfg.text_config.num_hidden_layers
    n_mtp = getattr(cfg.text_config, "num_nextn_predict_layers", 0) if not args.no_mtp else 0
    groups: dict[str, list[str]] = defaultdict(list)
    for key in index:
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
        if m:
            groups[f"layer{int(m.group(1)):03d}"].append(key)
        elif key.startswith("model.visual."):
            groups["vision"].append(key)
        else:
            groups["top"].append(key)

    backbone = [f"layer{i:03d}" for i in range(n_layers)]
    if args.limit_layers:
        backbone = backbone[: args.limit_layers]
    order = ["top"] + backbone
    if n_mtp > 0 and not args.no_mtp:
        order += [f"layer{n_layers + i:03d}" for i in range(n_mtp)]
    order += ["vision"]
    dropped = sorted(g for g in groups if g not in order)
    if dropped:
        print(f"groups: {len(order)} (dropping {dropped})", flush=True)
    else:
        print(f"groups: {len(order)}", flush=True)

    opened: dict[str, dict] = {}
    def fetch(keys):
        out = {}
        for k in keys:
            shard = index[k]
            if shard not in opened:
                opened[shard] = mx.load(str(src / shard))
            out[k] = opened[shard][k]
        return out

    target = args.shard_gb * 1e9
    out_index, pending, pending_bytes = {}, {}, 0
    out_n = total_out = 0
    counts = {"quantized": 0, "as_stored": 0, "mtp": 0}
    started = time.time()

    def flush():
        nonlocal pending, pending_bytes, out_n, total_out
        if not pending:
            return
        out_n += 1
        name = f"model-{out_n:05d}.safetensors"
        mx.save_safetensors(str(dst / name), pending, metadata={"format": "mlx"})
        for key in pending:
            out_index[key] = name
        size = (dst / name).stat().st_size; total_out += size
        print(f"  -> {name}  {len(pending)} tensors  {size/1e9:.2f} GB  (total {total_out/1e9:.1f} GB, {time.time()-started:.0f}s)", flush=True)
        pending, pending_bytes = {}, 0

    for gi, g in enumerate(order, 1):
        if not groups[g]:
            continue
        raw = fetch(groups[g])
        sane = model.sanitize(raw)
        print(f"[{gi}/{len(order)}] {g}: {len(raw)} -> {len(sane)} tensors", flush=True)
        for key, value in sane.items():
            module = key.rsplit(".", 1)[0]
            if "mtp." in key:
                counts["mtp"] += 1
            if key.endswith(".weight") and module in qpaths:
                p = qpaths[module]
                q = quantize(value, p)
                emit = {module + ".weight": q[0], module + ".scales": q[1]}
                if len(q) == 3:
                    emit[module + ".biases"] = q[2]
                counts["quantized"] += 1
            else:
                emit = {key: materialise(value)}
                counts["as_stored"] += 1
            for k, v in emit.items():
                pending[k] = v; pending_bytes += v.nbytes
            if pending_bytes >= target:
                flush()
        del raw, sane
        opened.clear()
        mx.clear_cache()
    flush()

    quant = quant_params(args.bits_params, args.group_size)
    root_mode = quant.get("mode")
    for path, p in qpaths.items():
        if p != quant:
            entry = dict(p)
            if root_mode is not None and "mode" not in entry:
                entry["mode"] = "affine"
            quant[path] = entry
    cfg_out = dict(raw_cfg)
    cfg_out.pop("quantization_config", None)
    cfg_out["quantization"] = quant; cfg_out["quantization_config"] = quant
    if args.no_mtp:
        if "text_config" in cfg_out:
            cfg_out["text_config"]["num_nextn_predict_layers"] = 0
        cfg_out["num_nextn_predict_layers"] = 0
    if args.max_context is not None:
        cfg_out["max_position_embeddings"] = args.max_context
        if "text_config" in cfg_out:
            cfg_out["text_config"]["max_position_embeddings"] = args.max_context
    json.dump(cfg_out, open(dst / "config.json", "w"), indent=2)
    json.dump({"metadata": {"total_size": total_out}, "weight_map": out_index}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    for name in AUX_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    print(f"\n{out_n} shards, {total_out/1e9:.1f} GB, {(time.time()-started)/60:.1f} min; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

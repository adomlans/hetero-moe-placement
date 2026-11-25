#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-① 真权重装载器: 从【本机 HF 缓存】里的 SmolLM-135M 取真东西, 一个字节都不下载。

为什么用它能解开 P0-①: 台账里所有 relRMSE 都建立在"随机 std=1 的玩具权重 + 随机 std=1 的
激活"上。这不是"精度略悲观"的问题, 而是【误差的来源分布不对】:
  · 玩具权重 x~N(0,1) 的 amax/127 格点几乎各向同性 => int8 误差被系统性低估;
  · 玩具权重下 swiglu 中间量级 ~4e4~7e4, fp16 直接溢出 223 个 inf/nan(乙-2c),
    那是标度问题不是精度问题;
  · 命中分布是 seed 造出来的 A/C 两档, 不是真激活分布下的样子。

本机实测(HF_HUB_OFFLINE=1, 2026-11-21):
  · `model.safetensors` 272 张量全在, from_pretrained 1.4s 起来, 生成连贯英文
    ("... to recognize objects and their relationships. The model is trained on ...")
    => 权重可信, 不是残缺缓存。
  · 30 层 × {gate,up,down}_proj, D=576 / N=1536, 与 expert_kind="swiglu" 同族形状。

约定(每条都影响可复现性):
  1) 【一个真层的 MLP = 一个专家】。E=8 个专家 = 8 个不同真层的 MLP。
     这不是"8 个专家共享一份真权重", 而是刻意让每个专家拿到【不同层统计特性】的真矩阵。
  2) 布局一律转成【引擎布局】: g/u = (D,N), d = (N,D)。真权重存的是 (N,D)/(N,D)/(D,N)。
  3) 路由用真模型自己的东西, 不用 bias 造热度: `router_rows()` 默认取 lm_head 里
     均匀间隔的 E 行(真训练出来的 D 维向量)。同时也提供 mode="randn" 做对照 ——
     对照那一档如果命中接近均匀, 就说明"偏斜来自真投影方向", 不是我们把 bias 拧出来的。
  4) 激活也是真的: `capture()` 在真模型第 L 层的 MLP 入口挂 hook, 抓 post-attention-LN
     之后的隐藏态当 x。文本来自本仓库自己的 readme.md(真英文散文, 不是随机数)。
"""
import os
import glob
import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub")
REPO = "models--HuggingFaceTB--SmolLM-135M"
# 本仓库根目录(往上找 readme.md): 真英文文本的来源
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))


def snapshot() -> str:
    """本机快照目录。找不到就明确报错, 绝不去联网。"""
    hits = sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))
    hits = [h for h in hits if os.path.exists(os.path.join(h, "model.safetensors"))]
    if not hits:
        raise FileNotFoundError(
            f"本机缓存里没有完整的 SmolLM-135M 快照({HUB}/{REPO})。"
            f"🚨 这台机器 DNS->127.0.0.1, 不要指望 from_pretrained 去下载。")
    return hits[0]


def state_dict():
    """直接读 safetensors(不经过 transformers, 少一次 500MB 的拷贝)。"""
    from safetensors.torch import load_file
    return load_file(os.path.join(snapshot(), "model.safetensors"))


def dims(sd=None):
    sd = sd or state_dict()
    w = sd["model.layers.0.mlp.gate_proj.weight"]
    n, d = w.shape                       # llama 的 gate_proj 是 (intermediate, hidden)
    nl = sum(1 for k in sd if k.endswith("input_layernorm.weight"))
    return dict(D=int(d), N=int(n), layers=int(nl))


def expert_mats(layer_ids, sd=None):
    """取这些真层的 MLP 当专家, 返回【引擎布局】的 g/u/d。"""
    sd = sd or state_dict()
    out = []
    for L in layer_ids:
        p = f"model.layers.{L}.mlp."
        g = sd[p + "gate_proj.weight"].float().t().contiguous()   # (N,D)->(D,N)
        u = sd[p + "up_proj.weight"].float().t().contiguous()     # (N,D)->(D,N)
        d = sd[p + "down_proj.weight"].float().t().contiguous()   # (D,N)->(N,D)
        out.append({"g": g, "u": u, "d": d, "src": f"SmolLM-135M/L{L}"})
    return out


def real_text(n_chars=24000):
    """真英文文本: 本仓库各章 readme.md 的拼接(带 markdown 噪声, 但那是真分布不是高斯)。"""
    files = sorted(glob.glob(os.path.join(PROJ_ROOT, "**", "readme.md"), recursive=True))
    buf = []
    for f in files:
        try:
            buf.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            continue
        if sum(map(len, buf)) > n_chars:
            break
    if not buf:
        raise FileNotFoundError(f"在 {PROJ_ROOT} 下没找到 readme.md => 没有真文本可用")
    return "\n".join(buf)[:n_chars]


def capture(layer=8, bs=64, n_chars=24000, dtype=torch.float32):
    """真激活: 在第 `layer` 层的 MLP 入口挂 hook, 抓它看到的隐藏态(post-attn-LN 之后)。
    返回 (>=bs, D) 的行, 调用方自己切前 bs 行。"""
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 强制离线, 别让 import 去联网
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        m = AutoModelForCausalLM.from_pretrained(snap)
        m.eval().to(dtype)
        box = {}

        def _hook(_mod, inp, _out):
            box["x"] = inp[0].detach()

        h = m.model.layers[layer].mlp.register_forward_hook(_hook)
        ids = tok(real_text(n_chars), return_tensors="pt").input_ids
        if ids.shape[1] < bs:
            raise RuntimeError(f"真文本只有 {ids.shape[1]} 个 token, 不够 bs={bs}")
        with torch.no_grad():
            m(ids[:, :bs])
        h.remove()
        return box["x"][0].float().contiguous(), m.config.hidden_size
    finally:
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


def router_rows(E, D, mode="embed", seed=42):
    """路由矩阵 (E,D)。
    mode="embed"  : 取真 lm_head 里均匀间隔的 E 行 —— 真训练出来的 D 维方向, 不同行的
                    "偏好"差别很大 => 命中偏斜是【真激活 x 真方向】碰出来的, 不是 bias 拧的。
    mode="randn"  : 标准初始化 N(0,1/sqrt(D)) 的随机路由, 作对照。
    """
    if mode == "randn":
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(E, D, generator=g) / (D ** 0.5)).contiguous()
    sd = state_dict()
    lm = sd["lm_head.weight"] if "lm_head.weight" in sd else sd["model.embed_tokens.weight"]
    idx = torch.linspace(0, lm.shape[0] - 1, E).round().long()
    return lm[idx].float().contiguous()


def real_src(layer_ids, E=None, gate_mode="embed", seed=42):
    """给 HeteroMoE(..., real_src=) 用的整包。"""
    ex = expert_mats(layer_ids)
    D = ex[0]["g"].shape[0]
    return dict(experts=ex, gate_w=router_rows(len(ex), D, gate_mode, seed),
                gate_b=torch.zeros(len(ex)),
                note=f"SmolLM-135M layers={list(layer_ids)} gate={gate_mode}")


def weight_stats(ex=None, layer_ids=(0, 15, 29)):
    """真权重 vs 玩具权重的统计对照(int8 格点看的就是 amax 与 std 的比值)。"""
    ex = ex or expert_mats(layer_ids)
    rows = []
    for e in ex:
        for k in ("g", "u", "d"):
            w = e[k].float()
            rows.append((e.get("src", "toy"), k, w.std().item(), w.abs().max().item(),
                         (w.abs().max() / 127.0).item()))
    return rows


if __name__ == "__main__":
    sd = state_dict()
    print("dims:", dims(sd))
    ex = expert_mats((0, 15, 29))
    print("专家矩阵张数:", len(ex), "首个 src:", ex[0]["src"])
    for r in weight_stats(ex):
        print(f"  {r[0]:<18} {r[1]}  std={r[2]:.4f} max|w|={r[3]:.3f} int8步长={r[4]:.2e}")
    x, d = capture(layer=8, bs=16, n_chars=4000)
    print(f"真激活: {tuple(x.shape)} std={x.std():.3f} max|x|={x.abs().max():.3f} "
          f"行范数 min/mean/max={x.norm(dim=1).min():.2f}/{x.norm(dim=1).mean():.2f}/{x.norm(dim=1).max():.2f}")
    for gm in ("embed", "randn"):
        W = router_rows(8, d, gm)
        sc = x @ W.t()
        tk = sc.topk(2, dim=1).indices
        cnt = torch.bincount(tk.flatten(), minlength=8).float()
        print(f"gate={gm:<6} 16 行 top-2 命中分布={[int(v) for v in cnt]} 命中专家={sorted(set(tk.flatten().tolist()))}")

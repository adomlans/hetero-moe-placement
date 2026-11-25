#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-2 判决表: 专家从【单矩阵 x@W】换成真形状 SwiGLU 三矩阵, 延迟/精度/两堵墙各付多少。

【为什么这一刀是主线, 不是练手】
  现在专家是一个无激活的 x@W。真 MoE(DEEPSeek/Qwen/Mixtral 全家)是
      g = x@Wg,  u = x@Wu,  h = silu(g)*u,  out = h@Wd
  KTransformers 那条路线的卖点是【冷专家的权重字节必须从 DRAM 搬进 CPU】, 而这个卖点
  的价钱直接等于【一个专家几张矩阵】: 1 张 2MB -> 3 张 6MB。不换成真形状, 前面所有
  "字节墙 49us/专家" 的结论都只是在一个玩具形状上成立。

【这一刀要买三样东西】
  ① 延迟: gemm -> swiglu 的比值。字节 bound 的期望是 ~3x, 但 h 那一趟有标量 expf,
     bs=1 时阶段2 只有 <=top_k 个 slot 可并行 => 可能更差; 明显高于 3x 就是有隐藏串行。
  ② 精度: 多出来的那笔【h 再量化】。拆成 W量化 / x量化 / h再量化 三笔分别定价,
     而且全部在【只含 CPU 腿】的口径上比 —— 混进 GPU 热专家那半边就看不出是谁的误差。
  ③ 两堵墙重标: 字节墙 49us/专家 -> 147us/专家, 与算力墙(7us/slot)的交叉点不变,
     但【放置规则】吃的是 expert_bytes(), 所以会选到不同的 gc。

【口径】同进程轮转 + 逐轮配对 + p50; 🚨 两个 kind 必须钉在【同一个 gc】上比
  (OPTLOG 方法论第 9 条: 不钉住就是把放置规则的变化算成专家形状的收益)。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 5))
NT_ROUNDS = int(os.environ.get("NT_ROUNDS", 3))
D, E, N, K = 1024, 8, 2048, 2
BS_LIST = [int(v) for v in os.environ.get("BS", "1,4,32,128").split(",")]
CFG = [("A", [0, 1]), ("C", [5, 6])]
HIT_BS = int(os.environ.get("HIT_BS", 512))
NTS = [int(v) for v in os.environ.get("NTS", "1,2,3,6").split(",")]
FIXGC = int(os.environ.get("FIXGC", "0")) or None
# 全开档 = 前 14 刀的终点。这一刀只换【专家形状】, 别的都不动
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
HITS = {}


def hits_cum(hot, kind):
    torch.manual_seed(42)
    probe = H.HeteroMoE(D, E, N, 2, K, hot_experts=hot, cpu_delay_ms=0.0,
                        expert_kind=kind).cuda()
    hits = H.measure_hit_rates(probe, torch.randn(HIT_BS, D).cuda())
    del probe
    return hits


def gc_for(kind, hits):
    """放置规则吃的字节必须跟着 n_mat 走(方法论第 8 条)。"""
    n_mat = 3 if kind == "swiglu" else 1
    return H.plan_gpu_capacity(hits, N * D * n_mat, 4 * N * D * n_mat, gpu_expert_count=2)


def wall_line(kind, label, gc, bs):
    """两堵墙在【真形状】下的重标。字节墙按有流量的专家数, 算力墙按 slot 数。"""
    n_mat = 3 if kind == "swiglu" else 1
    hits = HITS[label]
    slots = [h / 100 * bs * K for h in hits]
    cpu_slots = [round(v, 2) for v in slots[gc:]]
    n_traffic = sum(1 for v in cpu_slots if v > 0)
    mx = max(cpu_slots) if cpu_slots else 0.0
    wall = "算力墙" if mx >= 5 else "字节墙"
    return (f"    {kind:<7} 每专家 {N*D*n_mat>>20}MB(={n_mat}张) 字节墙≈{49*n_mat}us/有流量专家"
            f" | 本帧流量专家 {n_traffic} 个 slot={cpu_slots} 最大 {mx:.2f} → 【{wall}】")


# ============================================================
# CPU 腿的四种口径 —— 全部【只含 CPU 侧专家】, 和 (L2输出 - 热专家半边) 同口径
#   ref : fp32 真权重(不量化)
#   s_w : 只有 W 量化(x 用 fp32, h 不量化)
#   s_wx: W + x 量化(h 不量化)
#   s_all: W + x + h 再量化  <- 这一条应当逐位等于 kernel 的整数域行为
# 🚨 用 float64 而不是 int64: 这些点积最大 2048*127*127=3.3e7 < 2^53, fp64 里就是
#    精确整数运算, 不必依赖 cublas 的 int64 支持。
# ============================================================
@torch.no_grad()
def cpu_legs(m, x, ts, ti, w):
    bs = x.shape[0]
    ref = torch.zeros(bs, m.out_dim, device=x.device, dtype=torch.float64)
    sw_ = torch.zeros_like(ref)
    swx = torch.zeros_like(ref)
    sall = torch.zeros_like(ref)
    x_q, x_sc = m._quantize_act(x)
    xd = x.double()
    xqd = x_q.double()
    for e in range(len(m.cpu_experts_t)):
        mask = (ti == m.gpu_expert_count + e)
        b, k = mask.nonzero(as_tuple=True)
        if b.numel() == 0:
            continue          # 零流量专家: 三张权重一个字都不读
        wg_d, wu_d, wd_d = w[f"g{e}"], w[f"u{e}"], w[f"d{e}"]        # fp32 真权重 (D,N)/(D,N)/(N,D)
        wg_q, wu_q, wd_q = w[f"gq{e}"], w[f"uq{e}"], w[f"dq{e}"]      # int8 格点(已 dequant 前)
        sg, su, sd = float(m.cpu_experts_scale[e]), float(m.cpu_up_scale[e]), float(m.cpu_down_scale[e])
        sc = ts[b, k].double().unsqueeze(1)
        # ---- ① 真权重
        h = torch.nn.functional.silu(xd[b] @ wg_d) * (xd[b] @ wu_d)
        ref.index_add_(0, b, (h @ wd_d) * sc)
        # ---- ② 只有 W 量化
        # 🚨 gq 是【裸 int8 格点】, gqd 是"gq*scale"。这里要的是 dequant 后的 W = gq*sg,
        #    写成 (xd @ gqd) * sg 等于把 scale 乘了两遍(sg≈2e-4 => 结果缩小 5000 倍,
        #    relRMSE 会显示成 99.8% 这种"看着像正交、其实是漏了量级"的假数)。
        g = (xd[b] @ w[f"gq{e}"]) * sg
        u = (xd[b] @ w[f"uq{e}"]) * su
        h = torch.nn.functional.silu(g) * u
        sw_.index_add_(0, b, (h @ w[f"dqd{e}"]) * sc)
        # ---- ③ W + x 量化(h 仍是 fp32)
        g = (xqd[b] @ w[f"gq{e}"]) * (x_sc[b].double() * sg).unsqueeze(1)
        u = (xqd[b] @ w[f"uq{e}"]) * (x_sc[b].double() * su).unsqueeze(1)
        h = torch.nn.functional.silu(g) * u
        swx.index_add_(0, b, (h @ w[f"dqd{e}"]) * sc)
        # ---- ④ 再加 h 再量化(kernel 的算法: 每 slot 一个 amax/127, 半偶舍入)
        hs = (h.abs().amax(1, keepdim=True).clamp(min=1e-8)) / 127.0
        hq = (h / hs).round().clamp(-127, 127)
        y = (hq @ w[f"dq{e}"]) * (hs * sd)
        sall.index_add_(0, b, y * sc)
    return dict(ref=ref, s_w=sw_, s_wx=swx, s_all=sall)


def prep_w(m):
    """把 CPU 侧三张矩阵(真权重 + 量化格点)一次搬上 GPU, 缓存成 dict。"""
    w = {}
    for e in range(len(m.cpu_experts_t)):
        wg = m.cpu_experts_t[e].t().contiguous()          # (D,N)
        wu = m.cpu_up_t[e].t().contiguous()               # (D,N)
        wd = m.cpu_down_t[e].t().contiguous()             # (N,D)
        w[f"g{e}"] = wg.double().cuda(); w[f"u{e}"] = wu.double().cuda(); w[f"d{e}"] = wd.double().cuda()
        gq = m.cpu_w_g[e].t().contiguous().cuda()          # (D,N) int8
        uq = m.cpu_w_u[e].t().contiguous().cuda()
        dq = m.cpu_w_d[e].t().contiguous().cuda()           # 🚨 wq_d 存的是 (D,N)(行=输出j,长N),
        #    喂 torch 的 hq@Wd 要 (N,D) => 必须 .t()。C++ 里不转置是因为它就是按 GEMV 连续存的
        w[f"gq{e}"] = gq.double(); w[f"uq{e}"] = uq.double(); w[f"dq{e}"] = dq.double()
        w[f"gqd{e}"] = gq.double() * float(m.cpu_experts_scale[e])
        w[f"uqd{e}"] = uq.double() * float(m.cpu_up_scale[e])
        w[f"dqd{e}"] = dq.double() * float(m.cpu_down_scale[e])
    return w


def rel(a, ref):
    den = ref.pow(2).mean().sqrt().item()
    return float("nan") if den == 0 else (a - ref).pow(2).mean().sqrt().item() / den * 100


def one(bs, label, hot):
    hits = HITS[label]
    gc_sw, pred_sw = gc_for("swiglu", hits)
    gc_g, pred_g = gc_for("gemm", hits)
    gc = FIXGC or gc_sw
    print("\n" + "=" * 78)
    print(f"配置 {label} | bs={bs:>4} | 放置钉死 gc={gc} (swiglu 规则要 {gc_sw}/{pred_sw:.0f}us, "
          f"gemm 规则要 {gc_g}/{pred_g:.0f}us)"
          f"  hot={hot} {'← 放对' if set(hot) < set(range(gc)) else '← 故意放错(打 CPU 腿)'}")
    print(wall_line("gemm", label, gc, bs))
    print(wall_line("swiglu", label, gc, bs))
    print("=" * 78)
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    models = {}
    for kind in ("gemm", "swiglu"):
        torch.manual_seed(42)     # 🚨 同 seed: gemm 那张 W 就是 Wg, 两档路由必须逐位相同
        models[kind] = H.HeteroMoE(D, E, N, gc, K, hot_experts=hot, cpu_delay_ms=0.0,
                                   expert_kind=kind, **FULL).cuda()
    with torch.no_grad():
        # ---- 闸门 A/B: 两个 kind 的三模式互差都必须逐位 0
        for kind in ("gemm", "swiglu"):
            m = models[kind]
            ts_, ti_ = m.route(x)
            ref = m.fp32_reference(x, ts_, ti_)
            outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
            worst = max((outs[a] - outs[b]).abs().max().item()
                        for i, a in enumerate(outs) for b in list(outs)[i + 1:])
            _, _, rel_all = H.err_metrics(outs["L2_零CUDA+pinned"], ref)
            print(f"  闸门 {kind:<7} 三模式互差={worst:.9f}  整条 L2 vs 本实例 fp32 参考 "
                  f"relRMSE={rel_all:.4f}%  max_abs={H.err_metrics(outs['L2_零CUDA+pinned'], ref)[0]:.5f}"
                  f"  | 本帧 CPU 腿 slot={int((ti_ >= m.gpu_expert_count).sum())}")
        # ---- 精度三笔(只量 CPU 腿): 只在 swiglu 档做
        m = models["swiglu"]
        ts_, ti_ = m.route(x)
        n_slot = int((ti_ >= m.gpu_expert_count).sum())
        hot = m._hot_compute(x, ts_, ti_).double()
        leg_kernel = (m(x, async_mode=True, mode="L2_零CUDA+pinned").double() - hot).contiguous()
        w = prep_w(m)
        L = cpu_legs(m, x, ts_, ti_, w)
        if n_slot == 0:
            # 🚨 方法论第 10 条: CPU 腿一个 slot 都没摊到 => 整条输出的 relRMSE 只反映
            #    GPU 那半边的 fp16, 拿它当"量化代价"读就是把没参与的那条腿算成免费的。
            print(f"  精度四笔: 🚨跳过 —— 本帧 CPU 腿 0 个 slot(bs=1 + 热专家接走全部流量), "
                  f"上面那行 relRMSE 不是量化的代价")
        else:
            rms = L["ref"].pow(2).mean().sqrt().item()
            r1, r2, r3 = (rel(L["s_w"], L["ref"]), rel(L["s_wx"], L["ref"]), rel(L["s_all"], L["ref"]))
            print(f"  精度四笔(只量 CPU 腿 {n_slot} 个 slot, 分母=本实例 CPU 腿 fp32 真值 rms={rms:.3g}):")
            print(f"    ①只有 W 量化      relRMSE={r1:.4f}%")
            print(f"    ②W+x 量化        relRMSE={r2:.4f}%   (+x 这一笔 {r2-r1:+.4f}pp)")
            print(f"    ③W+x+h 再量化     relRMSE={r3:.4f}%   (+h 这一笔 {r3-r2:+.4f}pp  ← 真形状新增的那笔)")
            d = (leg_kernel - L["s_all"]).abs().max().item()
            scale = L["s_all"].abs().max().item()
            ok = "OK(≤1 LSB 级)" if scale and d / scale < 1e-4 else "🚨 对不上, kernel 有 bug"
            print(f"    ④kernel 实测      relRMSE={rel(leg_kernel, L['ref']):.4f}%   "
                  f"④vs③ 绝对={d:.3g} (③量级 {scale:.3g} => 相对 {d/scale if scale else float('nan'):.2g}) {ok}")
        del w
    # ---- 延迟: gemm vs swiglu 轮转
    samples = {k: [] for k in models}
    for _ in range(ROUNDS):
        for kind in ("gemm", "swiglu"):
            samples[kind].append(H.bench(models[kind], x, True, "L2_零CUDA+pinned")[0])
    pg = statistics.median(samples["gemm"]); ps = statistics.median(samples["swiglu"])
    print(f"{'口径':<26}{'p50':>9}{'σ%':>6}{'min':>9}   轮间分布")
    for kind in ("gemm", "swiglu"):
        v = samples[kind]
        sd = statistics.stdev(v) / min(v) * 100 if len(v) > 1 else 0.0
        print(f"{kind:<26}{statistics.median(v):>9.3f}{sd:>6.0f}{min(v):>9.3f}   "
              + " ".join(f"{t:.3f}" for t in sorted(v)))
    print(f"  → swiglu / gemm = {ps/pg:.2f}x  (字节 bound 的期望约 3x)")
    del models
    torch.cuda.empty_cache()
    return gc


def nt_scan(bs, label, hot, gc):
    """swiglu 全开档的 n_threads 扫描: bs=1 时阶段2 只有 <=top_k 个 slot 可并行,
       阶段1/3 按【行】切线程还有 2048 行可切 => 最优 nt 会不会漂?"""
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    out = {}
    for nt in NTS:
        torch.manual_seed(42)
        kw = dict(FULL); kw.pop("n_threads"); kw.pop("tok_block")
        m = H.HeteroMoE(D, E, N, gc, K, hot_experts=hot, cpu_delay_ms=0.0,
                        expert_kind="swiglu", n_threads=nt, tok_block=8, **kw).cuda()
        out[nt] = [H.bench(m, x, True, "L2_零CUDA+pinned")[0] for _ in range(NT_ROUNDS)]
        del m
    print(f"  swiglu 全开档 n_threads 扫描 (配置{label} bs={bs} gc={gc}):", end="")
    for nt in NTS:
        v = out[nt]
        print(f"\n    nt={nt:<3} p50={statistics.median(v):.3f}  min={min(v):.3f}  "
              f"轮间 {sorted(round(t,3) for t in v)}", end="")
    best = min(NTS, key=lambda nt: statistics.median(out[nt]))
    print(f"\n  → 最优线程数 = {best}   (主表 bs=32 写死的是 6)")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"[机器状态] {H.machine_state()}")
    print(f"[被测] D={D} E={E} expert={N} top_k={K} | bs={BS_LIST} | 轮转 {ROUNDS} 轮 | "
          f"命中探针 {HIT_BS} 行 = 累计命中")
    for lb, hot in CFG:
        HITS[lb] = hits_cum(hot, "gemm")
        sw = hits_cum(hot, "swiglu")
        same = (HITS[lb] == sw)
        print(f"  累计命中率 {lb}: {HITS[lb]}   swiglu 档 {sw} "
              f"{'← 两档逐位相同(独立 Generator 的约定成立, 路由可比)' if same else '🚨 两档路由不同, 延迟表不可比!'}")
    for bs in BS_LIST:
        gcs = {}
        for label, hot in CFG:
            gcs[label] = one(bs, label, hot)
        for label, hot in CFG:
            nt_scan(bs, label, hot, gcs[label])
    print("\n判读提示: ①比值≈3x = 这刀只是如实付钱; 明显 >3x 才有隐藏串行(先看阶段2 的标量 expf);\n"
          "        ② ③-② 那一列就是【h 再量化】的全部代价, 它决定要不要给 h 上更细的格点(每128个一组等);\n"
          "        ③ 两堵墙那两行: swiglu 把字节墙抬到 147us/专家 => 同一个 slot 数下更容易站字节墙, "
          "放置规则该重跑(T15)。")

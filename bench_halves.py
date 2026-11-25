#!/usr/bin/env python
"""T13 分账探针: 放置(gc)这件事, GPU 半边和 CPU 半边各自花多少。

为什么需要它: T9 的放置规则只看了 CPU 半边(被摸字节 <= 0.5xL3 就停手), 完全没给
GPU 半边定价。T11/T12 把主机侧和 GPU 侧都削下去之后,"多给 GPU 一个专家"的相对
价格变了 —— 主表 T9 段里配置A gc=3 反超规则选的 gc=2 有 12.2%(11 轮配对 10/11)。
这说明规则缺的不是系数, 是【另一半的成本】。

这里把两边分别量出来(同一进程轮转, 各自 5 取中位):
  GPU半边 = pre图 + D2H图 + 热专家图 连着跑完并同步(没有 CPU worker 参与)
  CPU半边 = 只调 avx kernel(_into), 输入用真实那一帧的 pinned 数据
然后检验: argmin max(GPU, CPU) 是不是 == argmin 端到端。
成立 => 规则可以升级成"两条腿的 roofline", 不再需要 magic 系数。

乙-2b 追加(2026-11-20): 加 EKIND / BS 两个 env, 于是同一张表能回答另一个问题 ——
  swiglu 档端到端比 gemm 多付的那 ~0.3ms, 到底在哪半边?
  关键控制组: bs=1 配置A 的 CPU 腿是【0 个 slot】(kernel 一个字节都不读),
  可 bench_swiglu 实测那一格 0.231 -> 0.530(+0.299ms)。字节墙解释不了 0 字节,
  所以这笔只能是固定成本(GPU 半边 or 主机入队)。两半边一拆就知道是谁。
  EKIND=gemm,swiglu 时两种形状在【同一进程里轮转】, 不做跨进程比较。
"""
import os, sys, time, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

GCS = [int(v) for v in os.environ.get("GCS", "2,3,4,5").split(",")]
KINDS = [k for k in os.environ.get("EKIND", "gemm").split(",") if k]
assert set(KINDS) <= {"gemm", "swiglu"}, KINDS
BS = int(os.environ.get("BS", "32"))
HS, NE, ES, TOPK, NT, TB = 1024, 8, 2048, 2, 6, 8
CFG = [("A", [0, 1]), ("C", [5, 6])]
MODE = "L2_零CUDA+pinned"
torch.manual_seed(42); x = torch.randn(BS, HS).cuda()
L3 = H.l3_bytes()
ROUNDS = int(os.environ.get("ROUNDS", 9))


def gpu_half(m, xx, ts, ti, iters=300):
    """GPU 半边单独跑完(pre 图 + D2H 图 + 热专家图), 无 worker。取 5 组中位。"""
    def once():
        s_ts, s_ti, _, _ = m._prepare(xx)
        m._hot_half(xx, s_ts, s_ti)
        torch.cuda.current_stream().synchronize()
    for _ in range(30):
        once()
    vals = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            once()
        vals.append((time.perf_counter() - t0) * 1e3 / iters)
    return statistics.median(vals)


def cpu_half(m, xx, iters=200):
    """CPU 半边单独跑: 只调 kernel, 输入是真实那一帧的 pinned 整数数据。
    🚨 按 kind 分派入口(swiglu 是三矩阵那条), 别拿 gemm 入口去量 swiglu 的腿。"""
    m(xx, async_mode=True, mode=MODE)          # 把 pins 填成这一帧的真数据
    p = m._pin
    if m.kind == "swiglu":
        def once():
            m._cpu_swiglu_into(p["xq"], p["sc"], p["idx"], p["scr"], p["out"], False)
    else:
        def once():
            H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                p["xq"], p["sc"], p["idx"], p["scr"], m.cpu_w, m.cpu_experts_scale,
                p["out"], m.gpu_expert_count, True, NT, TB)
    for _ in range(20):
        once()
    vals = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            once()
        vals.append((time.perf_counter() - t0) * 1e3 / iters)
    return statistics.median(vals)


print(f"L3={L3 >> 20}MB bs={BS} 端到端轮数={ROUNDS} 被测形状={KINDS} gc={GCS}"
      f"  (每专家 gemm=2MB / swiglu=6MB)")
for lb, hot in CFG:
    torch.manual_seed(42)
    probe = H.HeteroMoE(HS, NE, ES, 2, TOPK, hot_experts=hot).cuda()
    hits = H.measure_hit_rates(probe, x); del probe      # 两档路由逐位相同, 探针只建一次
    KEYS = [(kind, gc) for kind in KINDS for gc in GCS]
    M = {}
    for key in KEYS:
        kind, gc = key
        torch.manual_seed(42)
        M[key] = H.HeteroMoE(HS, NE, ES, gc, TOPK, hot_experts=hot, n_threads=NT,
                             tok_block=TB, gpu_impl="cat", graph_scope="full",
                             expert_kind=kind).cuda()
        for _ in range(30):
            M[key](x, async_mode=True, mode=MODE)
    torch.cuda.synchronize()
    S = {k: [] for k in KEYS}
    for _ in range(ROUNDS):
        for k in KEYS:                    # 轮转: 形状之间也逐轮配对, 不跨进程比
            S[k].append(H.bench(M[k], x, True, MODE)[0])
    ts, ti = M[KEYS[0]].route(x)
    n_slot = int((ti >= M[KEYS[0]].gpu_expert_count).sum())
    print(f"\n{'=' * 90}\n配置{lb} 命中率 {hits} 本帧 CPU 腿 slot={n_slot}  loadavg="
          f"{open('/proc/loadavg').read().split()[0]}\n{'=' * 90}")
    print(f"{'形状':>7}{'gc':>4}{'CPU被摸MB':>10}{'CPU常驻MB':>10}{'GPU半边':>9}{'CPU半边':>9}"
          f"{'max(半,半)':>11}{'端到端':>9}{'预测=实测?':>11}")
    g, c = {}, {}
    for k in KEYS:
        for _ in range(2):                     # 两边交替测, 抗漂移
            g[k] = gpu_half(M[k], x, ts, ti)
            c[k] = cpu_half(M[k], x)
    for kind in KINDS:                  # 每种形状各自检验"两半边 max() 能否预测最优 gc"
        sub = [k for k in KEYS if k[0] == kind]
        med = {k: statistics.median(S[k]) for k in sub}
        pred = {k: max(g[k], c[k]) for k in sub}
        bg, bp = min(med, key=med.get), min(pred, key=pred.get)
        print(f"  [{kind}] 实测最优 gc={bg[1]} | 只看两半边的 max() 会选 gc={bp[1]} "
              f"{'→ 模型成立, 规则可以给 GPU 半边定价' if bg == bp else '→ 模型不够, 还得加项'}")
    for k in KEYS:
        kind, gc = k
        eb = M[k].expert_bytes() >> 20
        traf = sum(1 for e in range(gc, NE) if hits[e] > 0) * eb
        print(f"{kind:>7}{gc:>4}{traf:>10}{(NE - gc) * eb:>10}{g[k]:>9.3f}{c[k]:>9.3f}"
              f"{max(g[k], c[k]):>11.3f}{statistics.median(S[k]):>9.3f}")
    if len(KINDS) == 2:                 # ← 乙-2b 要看的就是这一段
        print(f"  {'Δ(swiglu-gemm)':>14}{'':>14}"
              + "".join(f"{x_:>11}" for x_ in ("GPU半边", "CPU半边", "", "端到端")))
        for gc in GCS:
            kg, ks = ("gemm", gc), ("swiglu", gc)
            print(f"  {'gc=' + str(gc):>12}{'':>14}"
                  f"{'':>11}{g[ks] - g[kg]:>+9.3f}{c[ks] - c[kg]:>+9.3f}{'':>11}"
                  f"{statistics.median(S[ks]) - statistics.median(S[kg]):>+9.3f}")
    M.clear(); torch.cuda.empty_cache()

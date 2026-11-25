#!/usr/bin/env python
"""判决表 T11: 前段实现 torch(12 张 aten) vs fused(2 张 triton) —— 同进程轮转 A/B。

为什么又是一张判决表而不是直接看主表: 主表一次 4 分钟, 而这一刀的预期收益只有
30~40us(前段图 replay 41.8us -> 12.2us), 在 loadavg≈1 的机器上主表的轮间抖动
就有 ±2%~3%(0.39ms 的 2% = 8us, 看着够, 但配置 A 的 delay=3 那一档抖 20% 也发生过)。
所以照 T8/T10 的规矩: 每个变体【每一轮都跑一遍】, 报中位数 + 逐轮配对(谁在多少轮
里更慢), 而不是 min-of-N —— min 会奖励方差大的变体, 这个坑 T8-② 踩过。

口径: 配置 A(热专家在GPU)/C(冷专家在GPU) x delay {0, 3}ms
      x 变体 {torch, fused} x {sync, L1_松GIL, L2_零CUDA+pinned}
正确性: 逐位闸门先过(idx / x_q 与 torch 版必须完全相同)才允许看延迟。
用法: ROUNDS=11 python bench_t11_ab.py
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 11))
VARIANTS = ["torch", "fused"]
KEYS = [("A", "torch"), ("A", "fused"), ("C", "torch"), ("C", "fused")]
MODELS = {}
torch.manual_seed(42); x0 = torch.randn(32, 1024).cuda()
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    for v in VARIANTS:
        for d in (0.0, 3.0):
            torch.manual_seed(42)
            MODELS[(label, v, d)] = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                                                cpu_delay_ms=d, graph_scope="full",
                                                pre_impl=v).cuda()

# ---------- 0. 先过正确性闸门(逐位), 再谈延迟 ----------
print("=== 正确性闸门: fused 前段 vs torch 前段(同 seed 同输入) ===")
bad = 0
for label in ("A", "C"):
    torch.manual_seed(7); xe = torch.randn(32, 1024).cuda()
    mt, mf = MODELS[(label, "torch", 0.0)], MODELS[(label, "fused", 0.0)]
    st, it, qt, sc = mt._prepare(xe)
    sf, if_, qf, sfc = mf._prepare(xe)
    di = int((it != if_).sum()); dq = int((qt != qf).sum())
    ds = (st - sf).abs().max().item() / st.abs().max().item()
    ot = mt(xe, async_mode=False); of = mf(xe, async_mode=False)
    do = (ot - of).abs().max().item() / ot.abs().max().item()
    ok = di == 0 and dq == 0 and ds < 1e-6
    bad += not ok
    print(f"  {label}: topk_indices 不同={di}  x_q 不同={dq}  "
          f"scores 相对差={ds:.2e}  整条输出相对差={do:.2e}  {'OK' if ok else '🚨FAIL'}")
if bad:
    print("🚨 前段替换改了路由人选 —— 停止, 不许看延迟"); sys.exit(1)

# ---------- 1. 轮转测延迟 ----------
res = {(l, v, d, mo): [] for l in ("A", "C") for v in VARIANTS
       for d in (0.0, 3.0) for mo in ("sync", "L1", "L2")}
for r in range(ROUNDS):
    for label in ("A", "C"):
        for d in (0.0, 3.0):
            torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
            for v in VARIANTS:
                m = MODELS[(label, v, d)]
                res[(label, v, d, "sync")].append(H.bench(m, x, False, None)[0])
                res[(label, v, d, "L1")].append(H.bench(m, x, True, "L1_松GIL")[0])
                res[(label, v, d, "L2")].append(H.bench(m, x, True, "L2_零CUDA+pinned")[0])

la = open('/proc/loadavg').read().split()
gov = open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
mhz = open('/proc/cpuinfo').read().split('cpu MHz')[1].split(':')[1].strip()[:4]
print(f"\ngovernor={gov} MHz={mhz} loadavg={la[0]}/{la[1]} rounds={ROUNDS}  "
      f"(单位 ms, 中位数; 括号=min)")
print(f"{'配置':<5}{'delay':>6}" + "".join(f"{mo:>22}" for mo in ("sync", "L1_松GIL", "L2_pinned")))
for label in ("A", "C"):
    for d in (0.0, 3.0):
        cells = []
        for mo in ("sync", "L1", "L2"):
            a = res[(label, "torch", d, mo)]; b = res[(label, "fused", d, mo)]
            cells.append(f"{statistics.median(a):.3f}/{min(a):.3f} → "
                         f"{statistics.median(b):.3f}/{min(b):.3f}")
        print(f"{label:<5}{d:>6.1f}" + "".join(f"{c:>22}" for c in cells))

print("\n=== L2: fused 相对 torch 的增减(负=更快) + 逐轮配对 ===")
print(f"{'配置':<5}{'delay':>6}{'p50 前':>9}{'p50 后':>9}{'Δp50':>9}"
      f"{'min 前':>9}{'min 后':>9}{'Δmin':>9}{'fused更慢的轮数':>18}")
for label in ("A", "C"):
    for d in (0.0, 3.0):
        a = res[(label, "torch", d, "L2")]; b = res[(label, "fused", d, "L2")]
        worse = sum(1 for u, v in zip(a, b) if v > u)
        p50a, p50b = statistics.median(a), statistics.median(b)
        mna, mnb = min(a), min(b)
        print(f"{label:<5}{d:>6.1f}{p50a:>9.3f}{p50b:>9.3f}{(p50b/p50a-1)*100:>8.1f}%"
              f"{mna:>9.3f}{mnb:>9.3f}{(mnb/mna-1)*100:>8.1f}%{worse:>12}/{len(a)}")

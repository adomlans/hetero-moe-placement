#!/usr/bin/env python
"""判决表: 同进程内【轮转】A/B, 用来在脏机器上分辨真假收益。

为什么必须轮转: 本机跑着桌面(微信/Edge/code/codex, loadavg≈2), 6 线程的 CPU
kernel 会被随机抢核拖慢 30%+; 同一份代码两次量到 0.288 / 0.231 都发生过。
"先跑完 A 再跑完 B"的表在这种机器上是噪声, 只有【每一轮都把所有变体各跑一遍】
再取 min, 漂移才对每个变体同等作用。表头自动打印 governor/MHz/loadavg,
loadavg>1.5 会打 🚨 —— 那一次的绝对值只能看相对值。
用法: ROUNDS=7 python bench_scope_ab.py
口径: 配置 A(热专家在GPU)/C(冷专家在GPU)
      x 变体 {off, hot, full, full+ww(T8-② 把等 D2H 挪进 worker, 已被 p50 否证)}
      x {sync, L1_松GIL, L2_零CUDA+pinned}, 另加 T_kernel(纯 CPU kernel, 不含任何 GPU)。
"""
import os, sys, statistics, time, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 9))
# 🚨 判决用【中位数】, min 只当参考。踩过的坑: T8-② 那个变体方差是别家的 4 倍,
#    min-of-7 让它"赢"了 16%, p50 一看其实 +1%。min 会奖励方差大的变体。
CFG = [("A", [0, 1]), ("C", [5, 6])]
# (graph_scope, d2h_wait, 显示名) —— T8-② 只加了一个变体, 其它保持原样好对比
VARIANTS = [("off", "main", "off"), ("hot", "main", "hot"),
            ("full", "main", "full"), ("full", "worker", "full+ww")]
MODELS = {}
torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
for label, hot in CFG:
    for sc, dw, name in VARIANTS:
        torch.manual_seed(42)
        MODELS[(label, name)] = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                                            cpu_delay_ms=0.0, graph_scope=sc,
                                            d2h_wait=dw).cuda()
KEYS = [(l, n, mo) for l, _ in CFG for _, _, n in VARIANTS for mo in ("sync", "L1", "L2")]
res = {k: [] for k in KEYS}
tk = {l: [] for l, _ in CFG}
for r in range(ROUNDS):
    for label, hot in CFG:
        for _, _, name in VARIANTS:
            m = MODELS[(label, name)]
            res[(label, name, "sync")].append(H.bench(m, x, False, None)[0])
            res[(label, name, "L1")].append(H.bench(m, x, True, "L1_松GIL")[0])
            res[(label, name, "L2")].append(H.bench(m, x, True, "L2_零CUDA+pinned")[0])
        # T_kernel: 用真实路由, 只跑 CPU kernel(不含 GPU / 不含 DMA)
        m = MODELS[(label, "off")]
        ts, ti = m.route(x); xq, xs = m._quantize_act(x)
        px, ps, pi, psc = (xq.cpu().pin_memory(), xs.cpu().pin_memory(),
                           ti.cpu().pin_memory(), ts.cpu().pin_memory())
        po = torch.empty(32, 2048).pin_memory()
        best = 1e9
        for _ in range(5):
            for _ in range(10):
                H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                    px, ps, pi, psc, m.cpu_experts_q, m.cpu_experts_scale, po,
                    m.gpu_expert_count, True, m.n_threads, m.tok_block)
            t0 = time.perf_counter()
            for _ in range(300):
                H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                    px, ps, pi, psc, m.cpu_experts_q, m.cpu_experts_scale, po,
                    m.gpu_expert_count, True, m.n_threads, m.tok_block)
            best = min(best, (time.perf_counter() - t0) / 300 * 1000)
        tk[label].append(best)
print(f"governor={open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()}"
      f"  MHz={open('/proc/cpuinfo').read().split('cpu MHz')[1].split(':')[1].strip()[:4]}"
      f"  loadavg={open('/proc/loadavg').read().split()[0]}  rounds={ROUNDS}")
print(f"{'配置':<5}{'变体':<12}{'sync':>16}{'L1_松GIL':>16}{'L2_pinned':>16}{'T_kernel':>10}")
for label, _ in CFG:
    for _, _, name in VARIANTS:
        cells = []
        for mo in ("sync", "L1", "L2"):
            v = res[(label, name, mo)]
            cells.append(f"{statistics.median(v):.3f}({statistics.stdev(v)/min(v)*100:.0f}%)")
        print(f"{label:<5}{name:<12}" + "".join(f"{c:>16}" for c in cells)
              + f"{statistics.median(tk[label]):>10.3f}")
print("\nL2 中位数相对 full 的增减(负数=更快); 括号里是 min 口径, 用来对照【方差陷阱】:")
for label, _ in CFG:
    bv = res[(label, "full", "L2")]
    base50, basemin = statistics.median(bv), min(bv)
    for _, _, name in VARIANTS:
        v = res[(label, name, "L2")]
        print(f"  {label} {name:<12} p50={statistics.median(v):.3f} "
              f"{(statistics.median(v)/base50-1)*100:+6.1f}%   "
              f"min={min(v):.3f} {(min(v)/basemin-1)*100:+6.1f}%")

#!/usr/bin/env python
"""T10 定价: 把 CPU 权重从 biased uint8 换成 signed int8(修 maddubs int16 饱和 bug),
   端到端要付多少钱?  同进程【轮转】+ 逐轮配对, 两种卡尺(min / 中位数)。

口径说明:
  * 两个模型用同一个 manual_seed(42) 建 => 权重完全相同, 唯一差别是 w_pack。
    kernel 侧按张量 dtype 自动选路径(int8=>signed, uint8=>biased), 不可能配错。
  * T_kernel 用【真实路由】(不是构造的槽位), 所以它就是配置里 CPU 半边真正的那一段。
  * 逐轮配对 = 同一轮里 signed/biased-1, 机器漂移对两者同等作用; "几轮同向"就是符号检验。
用法: ROUNDS=9 python bench_pack_ab.py
"""
import os, sys, statistics, time, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 9))
CFG = [("A", [0, 1]), ("C", [5, 6])]
PACKS = ("biased", "signed")
MODELS = {}
torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
for label, hot in CFG:
    for pk in PACKS:
        torch.manual_seed(42)
        MODELS[(label, pk)] = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                                          cpu_delay_ms=0.0, graph_scope="full",
                                          w_pack=pk).cuda()

def kernel_only(m, px, ps, pi, psc, po, iters=300):
    best = 1e9
    for _ in range(4):
        for _ in range(10):
            H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                px, ps, pi, psc, m.cpu_w, m.cpu_experts_scale, po,
                m.gpu_expert_count, True, m.n_threads, m.tok_block)
        t0 = time.perf_counter()
        for _ in range(iters):
            H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                px, ps, pi, psc, m.cpu_w, m.cpu_experts_scale, po,
                m.gpu_expert_count, True, m.n_threads, m.tok_block)
        best = min(best, (time.perf_counter() - t0) / iters * 1000)
    return best

KEYS = [(l, pk, mo) for l, _ in CFG for pk in PACKS for mo in ("sync", "L2", "Tk")]
res = {k: [] for k in KEYS}
for r in range(ROUNDS):
    for label, _ in CFG:
        for pk in PACKS:
            m = MODELS[(label, pk)]
            res[(label, pk, "sync")].append(H.bench(m, x, False, None)[0])
            res[(label, pk, "L2")].append(H.bench(m, x, True, "L2_零CUDA+pinned")[0])
            ts, ti = m.route(x); xq, xs = m._quantize_act(x)
            px, ps, pi, psc = (xq.cpu().pin_memory(), xs.cpu().pin_memory(),
                               ti.cpu().pin_memory(), ts.cpu().pin_memory())
            po = torch.empty(32, 2048).pin_memory()
            res[(label, pk, "Tk")].append(kernel_only(m, px, ps, pi, psc, po))

la = open('/proc/loadavg').read().split()
print(f"governor={open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()}"
      f"  MHz={open('/proc/cpuinfo').read().split('cpu MHz')[1].split(':')[1].strip()[:4]}"
      f"  loadavg={la[0]}/{la[1]}  rounds={ROUNDS}  (loadavg>1.5 只看相对值)")
print(f"{'配置':<4}{'口径':<10}" + "".join(f"{p:>18}" for p in PACKS)
      + f"{'中位口径':>10}{'min口径':>10}  逐轮配对")
for label, _ in CFG:
    for mo in ("Tk", "L2", "sync"):
        b = res[(label, "biased", mo)]; s = res[(label, "signed", mo)]
        med = statistics.median(s) / statistics.median(b) - 1
        mn = min(s) / min(b) - 1
        pairs = [ss / bb - 1 for ss, bb in zip(s, b)]
        slower = sum(1 for p in pairs if p > 0)
        print(f"{label:<4}{mo:<10}"
              + "".join(f"{statistics.median(v):>11.3f}({min(v):.3f})"
                        for v in (b, s))
              + f"{med*100:>9.1f}%{mn*100:>9.1f}%  signed更慢 {slower}/{ROUNDS} 轮"
                f"  配对中位 {statistics.median(pairs)*100:+.1f}%")
print("\n读法: Tk = 纯 CPU kernel(真实路由, 300 次取 min), L2 = 端到端, sync = 基线。")
print("      括号里是 min; '中位口径'是两堆中位数之比, '配对中位'是逐轮比值的中位数。")

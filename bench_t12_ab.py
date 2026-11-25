#!/usr/bin/env python
"""判决表 T12: GPU 半边的主机收尾 —— 二流+Event vs 单流 vs 单流+尾图。
同进程轮转 + 逐轮配对 + 中位数(理由同 bench_t11_ab: min 会奖励方差大的变体)。

三个变体:
  sep        = T11 原状。每帧 new torch.cuda.Event + record + stream ctx + wait_event,
               热专家图丢到 gpu_stream 上。
  single     = 全回默认流。热专家 replay 与它的源(pre图/d2h图)被默认流顺序天然隔开,
               那个 Event 只是把两条流【完全串行化】了一遍, 并行从没发生过。
  single+tailG = 再把"H2D + add"折成第四张图(1 次 replay)。
               🚨 它把返回张量变成图内静态缓冲 => 下一帧覆写上一帧, 接口语义变了。
                  所以本表最后单独把这条代价打出来(连续两帧, 看第一帧有没有被改动)。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 11))
VARIANTS = [("sep", False), ("single", False), ("single", True)]
NAMES = ["sep", "single", "single+tailG"]
MODELS = {}
torch.manual_seed(42); x0 = torch.randn(32, 1024).cuda()
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    for d in (0.0, 3.0):
        for (st, tg), nm in zip(VARIANTS, NAMES):
            torch.manual_seed(42)
            MODELS[(label, d, nm)] = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                                                 cpu_delay_ms=d, graph_scope="full",
                                                 l2_stream=st, tail_graph=tg).cuda()

# ---------- 0. 正确性: 三个变体的输出必须逐位相同 ----------
print("=== 正确性闸门(同 seed 同输入, 输出逐位) ===")
bad = 0
for label in ("A", "C"):
    torch.manual_seed(7); xe = torch.randn(32, 1024).cuda()
    base = MODELS[(label, 0.0, "sep")](xe, async_mode=True, mode="L2_零CUDA+pinned").clone()
    torch.cuda.synchronize()
    for nm in NAMES[1:]:
        got = MODELS[(label, 0.0, nm)](xe, async_mode=True, mode="L2_零CUDA+pinned")
        torch.cuda.synchronize()
        d = (got - base).abs().max().item()
        bad += d != 0.0
        print(f"  {label} {nm:<14} 与 sep 的差 = {d:.6f}  {'OK' if d == 0 else '🚨FAIL'}")
if bad:
    print("🚨 改了流/图就把输出改错了 —— 停止, 不许看延迟"); sys.exit(1)

# ---------- 1. 覆写代价显式打印 ----------
print("\n=== tail_graph 的接口语义代价: 连续两帧, 第一帧的输出有没有被第二帧改掉 ===")
for label in ("A", "C"):
    torch.manual_seed(7); xa = torch.randn(32, 1024).cuda(); xb = torch.randn(32, 1024).cuda()
    for nm in ("single", "single+tailG"):
        m = MODELS[(label, 0.0, nm)]
        o1 = m(xa, async_mode=True, mode="L2_零CUDA+pinned"); torch.cuda.synchronize()
        o1_hold = o1.clone()
        m(xb, async_mode=True, mode="L2_零CUDA+pinned"); torch.cuda.synchronize()
        over = (o1 - o1_hold).abs().max().item()
        print(f"  {label} {nm:<14} 第一帧张量在第二帧之后被改写了 {over:.6f} "
              f"{'(=> 必须 .clone() 才能留住)' if over else '(不受影响)'}")

# ---------- 2. 轮转延迟 ----------
res = {(l, d, nm, mo): [] for l in ("A", "C") for d in (0.0, 3.0)
       for nm in NAMES for mo in ("sync", "L1", "L2")}
for r in range(ROUNDS):
    for label in ("A", "C"):
        for d in (0.0, 3.0):
            torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
            for nm in NAMES:
                m = MODELS[(label, d, nm)]
                res[(label, d, nm, "sync")].append(H.bench(m, x, False, None)[0])
                res[(label, d, nm, "L1")].append(H.bench(m, x, True, "L1_松GIL")[0])
                res[(label, d, nm, "L2")].append(H.bench(m, x, True, "L2_零CUDA+pinned")[0])
la = open('/proc/loadavg').read().split()
gov = open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()
mhz = open('/proc/cpuinfo').read().split('cpu MHz')[1].split(':')[1].strip()[:4]
print(f"\ngovernor={gov} MHz={mhz} loadavg={la[0]}/{la[1]} rounds={ROUNDS}")
print(f"{'配置':<5}{'delay':>6}{'变体':<14}{'sync p50':>10}{'L1 p50':>10}{'L2 p50':>10}"
      f"{'L2 min':>10}{'vs sep(p50)':>13}{'更慢轮数':>10}")
for label in ("A", "C"):
    for d in (0.0, 3.0):
        base = res[(label, d, "sep", "L2")]
        for nm in NAMES:
            v = res[(label, d, nm, "L2")]
            worse = sum(1 for u, w in zip(base, v) if w > u)
            print(f"{label:<5}{d:>6.1f}{nm:<14}"
                  f"{statistics.median(res[(label,d,nm,'sync')]):>10.3f}"
                  f"{statistics.median(res[(label,d,nm,'L1')]):>10.3f}"
                  f"{statistics.median(v):>10.3f}{min(v):>10.3f}"
                  f"{(statistics.median(v)/statistics.median(base)-1)*100:>12.1f}%"
                  f"{worse:>8}/{len(v)}")

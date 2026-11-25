#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-1 判决表: 把 batch_size 变成变量 —— 真 decode 场景下前 14 刀还剩多少。

【为什么这是回主线的第一刀】
  main() 里 bs=32 从 T1 写死到现在, 所以我们量的其实是【一次 32 个 token 的小批 prefill】。
  而 KTransformers / Ostinato 那条路线的正题不是它: 它们解决的是【单机跑大 MoE 做 decode】,
  那时每帧只有 1 个 token、只摸到 top_k=2 个专家、CPU 侧每个专家只摊到 1 个 slot。
  🚨 这正好把两条已经下过的结论挪出适用范围:
     · T5 的 tok_block=8 靠"同一张权重被多个 slot 复用"赚钱 —— bs=1 时每专家只有 1 个 slot, 该退化成中性
     · T14 的 guided 靠"很多行分给 6 条线程时抗掉队"赚钱 —— bs=1 时行数和 slot 数都变了
  所以这一刀不是新优化, 是【给已下的结论画适用范围】, 同时回答"这套东西能不能落地"。

【口径】同进程轮转 + 逐轮配对 + p50(和主表一致); 每个 bs 各建各的模型(CUDA Graph 按形状录)。
【放置规则的输入】按生产协议办 = 【累计命中】(这里用 512 行探针)。
   🚨 不能用单帧: decode 一帧只有 top_k=2 个 slot, 拿单帧当分布就是把噪声当统计量。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 7))
NT_ROUNDS = int(os.environ.get("NT_ROUNDS", 5))
D, E, N, K = 1024, 8, 2048, 2
BS_LIST = [int(v) for v in os.environ.get("BS", "1,4,32,128").split(",")]
CFG = [("A", [0, 1]), ("C", [5, 6])]
HIT_BS = int(os.environ.get("HIT_BS", 512))     # 累计命中探针的批大小
# FIXGC=N: 让三档用【同一个】gc。默认按主表惯例(T4/T5 档 gc=2, 全开档由规则选) =>
# 那样"全开 vs T4基线"里混着【放置规则】的变化。要干净地量"主机+kernel 这 12 刀还剩多少",
# 就 FIXGC=<全开的 gc> 再跑一遍(两次跑各 2 分钟, 都值得留)。
FIXGC = int(os.environ.get("FIXGC", "0")) or None

# 三档 = 前 14 刀的锚点。🚨 除最后一档外全部钉死历史默认值
# (OPTLOG.md 方法论第 3 条: ctor 默认值会变, 不钉住就是把新刀的收益算到旧刀头上)
PIN = dict(pre_impl="torch", l2_stream="sep", tail_graph=False, cpu_sched=0, d2h_wait="main")
RUNGS = [("T4 基线(sparse,nt1,tb1)", dict(gpu_impl="sparse", graph_scope="off", n_threads=1, tok_block=1)),
         ("+T5 线程6+tb8",           dict(gpu_impl="sparse", graph_scope="off", n_threads=6, tok_block=8)),
         ("全开(T6~T14)",            dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
                                          pre_impl="fused", l2_stream="single", auto="v2", cpu_sched=1))]
NAMES = [r[0] for r in RUNGS]
NTS = [int(v) for v in os.environ.get("NTS", "1,2,3,6").split(",")]
HITS = {}


def rung_kwargs(kw):
    """档位描述 -> (ctor 形参, 放置规则版本)。auto 不是 ctor 形参, 必须摘掉。"""
    kw = dict(PIN, **kw)
    return kw, kw.pop("auto", None)


def hits_cum(hot):
    """累计命中(= 生产里的命中计数器 EMA)。用 512 行而不是 bs 行, 理由见文件头。"""
    torch.manual_seed(42)
    probe = H.HeteroMoE(D, E, N, 2, K, hot_experts=hot, cpu_delay_ms=0.0).cuda()
    hits = H.measure_hit_rates(probe, torch.randn(HIT_BS, D).cuda())
    del probe
    return hits


def one(bs, label, hot, delay=0.0):
    """一个 (bs, 配置) 的一整张表: 三档轮转 + 先过精度闸门。"""
    hits = HITS[label]
    slots = [h / 100 * bs * K for h in hits]
    gc, pred = H.plan_gpu_capacity(hits, N * D, 4 * N * D, gpu_expert_count=2)
    cpu_slots = [round(v, 2) for v in slots[gc:]]
    n_traffic = sum(1 for v in cpu_slots if v > 0)
    mx = max(cpu_slots) if cpu_slots else 0.0
    wall = "算力墙" if mx >= 5 else "字节墙"
    print("\n" + "=" * 78)
    print(f"配置 {label} | bs={bs:>4} | delay={delay}ms | gc(v2)={gc} 预测{pred:.0f}us"
          f" | CPU 侧驻 {E - gc} 个专家, 本帧有流量 {n_traffic} 个, slot(期望)={cpu_slots}")
    print(f"每专家最大 slot={mx:.2f} → 站【{wall}】(两堵墙交叉点≈5)  "
          f"GPU 常驻 id [0,{gc})  hot={hot} {'← 放对' if set(hot) < set(range(gc)) else '← 故意放错(打 CPU 腿)'}")
    print("=" * 78)
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    models = {}
    for name, kw in RUNGS:
        kw2, auto = rung_kwargs(kw)
        gcv = FIXGC or (H.plan_gpu_capacity(hits, N * D, 4 * N * D, gpu_expert_count=2)[0]
                        if auto else 2)
        torch.manual_seed(42)      # 🚨 每个模型都重新 seed: 否则各档权重不同, 精度列不能横比
        models[name] = H.HeteroMoE(D, E, N, gcv, K, hot_experts=hot,
                                   cpu_delay_ms=delay, **kw2).cuda()
    # 闸门先于延迟。🚨 每个模型必须和【它自己】的 fp32 参考比: 各档 gc 不同 => 权重抽法不同,
    #    拿某一档的参考去比别的档 = 在比两个不同的模型(第一版就报了 12~15% 的假误差)
    with torch.no_grad():
        for name in NAMES:
            m = models[name]
            ts_, ti_ = m.route(x)
            ref = m.fp32_reference(x, ts_, ti_)
            outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
            worst = max((outs[a] - outs[b]).abs().max().item()
                        for i, a in enumerate(outs) for b in list(outs)[i + 1:])
            _, _, rel = H.err_metrics(outs["L2_零CUDA+pinned"], ref)
            print(f"  闸门 {name:<24} gc={FIXGC or '(见上)'} 三模式互差={worst:.9f}  "
                  f"L2 vs 本档 fp32 参考 relRMSE = {rel:.4f}%")
    samples = {k: [] for k in models}
    for _ in range(ROUNDS):
        for name in NAMES:                      # 轮转, 不按档位成块
            samples[name].append(H.bench(models[name], x, True, "L2_零CUDA+pinned")[0])
    base = statistics.median(samples[NAMES[0]])
    print(f"{'档位':<26}{'p50':>9}{'σ%':>6}{'min':>9}{'vs T4基线':>12}   轮间分布")
    for name in NAMES:
        v = samples[name]
        pv = statistics.median(v)
        sd = statistics.stdev(v) / min(v) * 100 if len(v) > 1 else 0.0
        print(f"{name:<26}{pv:>9.3f}{sd:>6.0f}{min(v):>9.3f}{(pv / base - 1) * 100:>11.1f}%   "
              + " ".join(f"{t:.3f}" for t in sorted(v)))
    del models
    torch.cuda.empty_cache()


def nt_scan(bs, label, hot):
    """全开档的 n_threads 扫描。bs=1 时这问的是【几个核在搬那 2MB 权重】= GEMV 的带宽问题。"""
    gc, _ = H.plan_gpu_capacity(HITS[label], N * D, 4 * N * D, gpu_expert_count=2)
    base, _ = rung_kwargs(RUNGS[-1][1])          # 全开档(已带 fused / single / cpu_sched=1)
    base.pop("n_threads"); base.pop("tok_block")  # 🚨 这两个由下面的扫描变量显式传, 留着会撞 kwargs
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    out = {}
    for nt in NTS:
        torch.manual_seed(42)
        m = H.HeteroMoE(D, E, N, gc, K, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=nt, tok_block=8, **base).cuda()
        out[nt] = [H.bench(m, x, True, "L2_零CUDA+pinned")[0] for _ in range(NT_ROUNDS)]
        del m
    print(f"  全开档 n_threads 扫描 (配置{label} bs={bs} gc={gc}; 主表 bs=32 那一档规则选的是 gc=3"
          f" —— 本表用 512 行累计命中, 尾部专家不再是 0, 所以规则把 gc 抬到 {gc}):", end="")
    for nt in NTS:
        v = out[nt]
        sd = statistics.stdev(v) / min(v) * 100 if len(v) > 1 else 0.0
        print(f"\n    nt={nt:<3} p50={statistics.median(v):.3f}  min={min(v):.3f}  σ{sd:.0f}%  "
              f"轮间 {sorted(round(t, 3) for t in v)}", end="")
    best = min(NTS, key=lambda nt: statistics.median(out[nt]))
    print(f"\n  → 最优线程数 = {best}   (主表在 bs=32 上写死的常数是 6)")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"[机器状态] {H.machine_state()}")
    print(f"[被测] D={D} E={E} expert={N} top_k={K} | bs={BS_LIST} | 轮转 {ROUNDS} 轮 | "
          f"命中探针 {HIT_BS} 行 = 累计命中(不是单帧)")
    HITS = {lb: hits_cum(hot) for lb, hot in CFG}
    for lb, hot in CFG:
        print(f"  累计命中率 {lb}: {HITS[lb]}   (主表 bs=32 单帧量的是 A[40.6,46.9,3.1,0,0,0,1.6,7.8] "
              f"C[1.6,4.7,1.6,0,0,45.3,40.6,6.2] —— 形状该一致, 只是更平滑)")
    for bs in BS_LIST:
        for label, hot in CFG:
            one(bs, label, hot)
        for label, hot in CFG:
            nt_scan(bs, label, hot)
    print("\n判读提示: ①『全开 vs T4基线』若随 bs 变小而缩进噪声 → 前 14 刀主要在打 GEMM 维度的浪费, "
          "decode 要另开一刀;\n"
          "        ② 最优线程数随 bs 漂 → 主表那组(6 线程, tb=8)是 prefill 专属常数, 不能写死;\n"
          "        ③ bs=1 时 CPU 侧本帧有流量的专家必然 <=top_k 个 → 放置规则该看【带宽×专家数】而不是【字节×slot】。")

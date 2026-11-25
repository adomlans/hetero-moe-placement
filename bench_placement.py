"""T9-① 放置探针: GPU 容量(gc) vs CPU 侧【被摸到的】权重字节。

背景(微基准 + 端到端实测):
  常驻 <=8MB(4 专家) 111GB/s(在 L3) / 12MB(6 专家) 45GB/s(DRAM 带宽 bound)。
  🚨 但"常驻几个"不是预算, "这一帧会被摸到几张权重"才是: 一个 CPU 专家只要摊到
  1 个槽位, 它那 2MB 就得整趟读完; 0 流量的专家连读都不读, 完全免费。

T9 原本的三件事:
  (1) 把"被摸到的字节"压回预算, 端到端能拿回多少? (机制: 减 DRAM 字节)
  (2) 代价: GPU 半边 dense GEMM 随 gc 线性变贵 —— gc=3 与 gc=4 的 CPU 侧逐位
      等价(专家 3 零流量), 它们的差就是"多给 GPU 一个专家"的标价。
  (3) 预算系数取 0.5xL3 还是 0.75xL3? (空机器微基准 8MB 还行, 端到端 8MB 已慢 13.8%)

✅ T13 追加: 规则的标定【会过期】。T9 定 budget_frac=0.5 时 GPU 半边还很贵;
   T11(前段融合)+T12(单流)把主机侧/GPU 侧都削下去之后, "多给 GPU 一个专家"的
   相对价格变了 —— 主表的 T9 段里配置A gc=3 反超规则选的 gc=2 有 12.2%。
   所以这里加了两个东西:
     a) 逐轮【配对】差值(vs 规则选的那档) + 同向轮数 —— 光看两条中位数会被
        桌面漂移骗(T8 的教训); "N 轮里几轮同向"才是判决依据。
     b) 一次运行同时打印 budget_frac ∈ {0.75,0.5,0.25} 各自会选哪档,
        于是"该取哪个系数"直接从同一份测量里读, 不用来回跑。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 9))
GCS = [int(v) for v in os.environ.get("GCS", "2,3,4").split(",")]
FRAC = float(os.environ.get("FRAC", 0.5))
HS, NE, ES, TOPK = 1024, 8, 2048, 2
CFG = [("A", [0, 1]), ("C", [5, 6])]          # A=热专家在GPU, C=热专家在CPU
MODE = "L2_零CUDA+pinned"
torch.manual_seed(42); x = torch.randn(32, HS).cuda()
L3 = H.l3_bytes()
print(f"L3={L3 >> 20}MB 每专家={ES * HS >> 20}MB "
      f"预算({FRAC}xL3)={int(L3 * FRAC) >> 20}MB "
      f"= 最多 {int(L3 * FRAC) // (ES * HS)} 个【有流量】的 CPU 专家")

for delay in (0.0, 3.0):
    hits, M = {}, {}
    for lb, hot in CFG:
        torch.manual_seed(42)
        probe = H.HeteroMoE(HS, NE, ES, 2, TOPK, hot_experts=hot, cpu_delay_ms=delay).cuda()
        hits[lb] = H.measure_hit_rates(probe, x)
        del probe
        for gc in GCS:
            torch.manual_seed(42)
            M[(lb, gc)] = H.HeteroMoE(HS, NE, ES, gc, TOPK, hot_experts=hot,
                                      cpu_delay_ms=delay, n_threads=6, tok_block=8,
                                      gpu_impl="cat", graph_scope="full").cuda()
            for _ in range(30):
                M[(lb, gc)](x, async_mode=True, mode=MODE)
    torch.cuda.synchronize()
    S = {k: [] for k in M}; SY = {k: [] for k in M}
    for _ in range(ROUNDS):                       # 🚨 轮转: 桌面漂移对每档同等作用
        for k in M:
            S[k].append(H.bench(M[k], x, True, MODE)[0])
            SY[k].append(H.bench(M[k], x, False, None)[0])
    print(f"\n{'=' * 92}\nT9-① 放置 delay={delay}ms loadavg="
          f"{open('/proc/loadavg').read().split()[0]} ROUNDS={ROUNDS}\n{'=' * 92}")
    for lb, hot in CFG:
        rec, tra = H.plan_cpu_residency(hits[lb], ES * HS, gpu_expert_count=min(GCS),
                                        budget_frac=FRAC)
        picks = {f: H.plan_cpu_residency(hits[lb], ES * HS,
                                         gpu_expert_count=min(GCS), budget_frac=f)[0]
                 for f in (0.75, 0.5, 0.25)}
        gc2, pred2 = H.plan_gpu_capacity(hits[lb], ES * HS, 4 * ES * HS,
                                         gpu_expert_count=min(GCS))
        meds = {gc: statistics.median(S[(lb, gc)]) for gc in GCS}
        best = min(meds, key=meds.get)
        # 逐轮配对: 每一轮都用【同一轮】的两个数相减, 桌面漂移两边同时发生 => 抵消
        # d>0 = 这一档比最优慢(差值占自身的百分比)
        paired = {gc: [a - b for a, b in zip(S[(lb, gc)], S[(lb, best)])] for gc in GCS}
        print(f"配置{lb} 命中率 {hits[lb]}\n  规则v1(T9)给出 gc={rec} (被摸权重 {tra >> 20}MB)"
              f" | 换系数: " + "  ".join(f"{f}xL3→gc={g}" for f, g in picks.items())
              + f"\n  规则v2(T13 两条腿)给出 gc={gc2} (预测 {pred2:.0f}us)"
              f"{'✅选中实测最优' if gc2 == best else '🚨与实测最优不符'}"
              f" | 实测最优 gc={best} (v1 {'✅' if rec == best else '🚨差 ' + format((meds[rec] / meds[best] - 1) * 100, '.1f') + '%'})")
        print(f"{'GPU专家gc':>10}{'CPU有流量专家':>14}{'被摸MB':>8}{'常驻MB':>8}"
              f"{'L2 med':>9}{'σ%':>5}{'sync':>9}{'vs 首档':>9}{'配对vs最优':>11}{'更快轮':>8}")
        base = statistics.median(S[(lb, GCS[0])])
        for gc in GCS:
            tr = sum(ES * HS for e in range(gc, NE) if hits[lb][e] > 0)
            ntr = sum(1 for e in range(gc, NE) if hits[lb][e] > 0)
            v = S[(lb, gc)]
            d = paired[gc]
            w = sum(1 for z in d if z < 0)         # 该档【赢过】最优的轮数(应≈0)
            ptxt = "        —" if gc == best else \
                f"{statistics.median(d) / statistics.median(v) * 100:>+8.1f}%"
            print(f"{gc:>10}{ntr:>14}{tr >> 20:>8}{(NE - gc) * ES * HS >> 20:>8}"
                  f"{statistics.median(v):>9.3f}"
                  f"{statistics.stdev(v) / statistics.mean(v) * 100:>5.0f}"
                  f"{statistics.median(SY[(lb, gc)]):>9.3f}"
                  f"{(statistics.median(v) / base - 1) * 100:>+8.1f}%"
                  f"{ptxt:>11}{w:>4}/{len(d)}"
                  + ("  ← 规则选的" if gc == rec else "")
                  + ("  ← 实测最优" if gc == best else ""))
    M.clear(); torch.cuda.empty_cache()

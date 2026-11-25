"""T10-B 外推: 放置规则(gc = 让 CPU 侧被摸到的权重 <= 0.5xL3 的最小 GPU 容量)
在 bs / 命中率形状变了之后, 还能不能选中实测最优那一档?

为什么要做: 规则是在"bs=32, 8 专家, top_k=2, 两个特定命中率形状"上长出来的。
如果不换个规模再验一次就写进代码注释, 它跟过拟合没区别。
这里把 bs 从 1 扫到 32 —— bs 小的时候一帧只摊到 2 个专家, 规则应当自动说
"别动 GPU"(因为被摸到的字节本来就 <= 预算); bs 大时多个专家被摸到, 规则应当
开始把 gc 往上抬。规则的选择必须落在实测 argmin 上(或只付 <= 一个噪声带的代价)。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

BSS = [int(v) for v in os.environ.get("BSS", "1,4,16,32").split(",")]
GCS = [int(v) for v in os.environ.get("GCS", "2,3,4").split(",")]
FRAC = float(os.environ.get("FRAC", 0.5))
ROUNDS = int(os.environ.get("ROUNDS", 7))
HS, NE, ES, TOPK = 1024, 8, 2048, 2
CFG = [("A", [0, 1]), ("C", [5, 6])]
MODE = "L2_零CUDA+pinned"
L3 = H.l3_bytes()
print(f"L3={L3 >> 20}MB 每专家={ES * HS >> 20}MB 预算={int(L3 * FRAC) >> 20}MB "
      f"| bs={BSS} gc={GCS} ROUNDS={ROUNDS} FRAC={FRAC}")

for bs in BSS:
    torch.manual_seed(42); x = torch.randn(bs, HS).cuda()
    print(f"\n{'=' * 96}\nbs={bs}  loadavg={open('/proc/loadavg').read().split()[0]}\n{'=' * 96}")
    print(f"{'cfg':<4}{'gc':>4}{'被摸专家':>9}{'被摸MB':>8}{'L2 med':>10}{'σ%':>5}"
          f"{'vs 最优':>9}{'规则选?':>13}{'最优?':>8}")
    for lb, hot in CFG:
        torch.manual_seed(42)
        probe = H.HeteroMoE(HS, NE, ES, 2, TOPK, hot_experts=hot).cuda()
        hits = H.measure_hit_rates(probe, x)
        del probe
        rec, _ = H.plan_cpu_residency(hits, ES * HS, gpu_expert_count=min(GCS),
                                      budget_frac=FRAC)
        rec2 = H.plan_gpu_capacity(hits, ES * HS, 4 * ES * HS,
                                   gpu_expert_count=min(GCS))[0]
        M = {}
        for gc in GCS:
            torch.manual_seed(42)
            M[gc] = H.HeteroMoE(HS, NE, ES, gc, TOPK, hot_experts=hot, n_threads=6,
                               tok_block=8, gpu_impl="cat",
                               graph_scope="full").cuda()
            for _ in range(20):
                M[gc](x, async_mode=True, mode=MODE)
        torch.cuda.synchronize()
        S = {gc: [] for gc in GCS}
        for _ in range(ROUNDS):
            for gc in GCS:
                S[gc].append(H.bench(M[gc], x, True, MODE)[0])
        med = {gc: statistics.median(S[gc]) for gc in GCS}
        best = min(med, key=med.get)
        for gc in GCS:
            ntr = sum(1 for e in range(gc, NE) if hits[e] > 0)
            mb = sum(ES * HS for e in range(gc, NE) if hits[e] > 0) >> 20
            v = S[gc]
            print(f"{lb:<4}{gc:>4}{ntr:>9}{mb:>8}{med[gc]:>10.3f}"
                  f"{statistics.stdev(v) / statistics.mean(v) * 100:>5.0f}"
                  f"{(med[gc] / med[best] - 1) * 100:>+8.1f}%"
                  f"{('←规则' if gc == rec else ''):>7}"
                  f"{('←v2' if gc == rec2 else ''):>6}{('✓' if gc == best else ''):>8}")
        def ag(r):
            return "一致" if best == r else (f"不一致(选{r}/实测{best}, "
                                             f"代价{(med[r]/med[best]-1)*100:+.1f}%)"
                                             if r in med else f"选{r}(没测)")
        print(f"  v1 gc={rec} → {ag(rec)}   v2 gc={rec2} → {ag(rec2)}   命中率 {hits}")
        M.clear(); torch.cuda.empty_cache()

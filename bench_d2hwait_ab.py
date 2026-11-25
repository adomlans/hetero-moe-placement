"""T8-② 终审: L2 里"等 D2H 落地"放在主线程还是 worker —— 两种卡尺 + 逐轮配对。

为什么要这个探针: 同一个改动, 主表口径(min-of-3-means) 说 -2~-5%, 逐帧 p50 口径说
+1~+2%。两个口径不能都算数。原因是【高方差变体的 min 天然好看】: worker_block 的
σ=14%, 主表 -6.9% 全来自尾巴, p50 反而 +0.6% —— 那是方差假收益, 不是延迟收益。
所以必须: ① 同进程轮转(消漂移) ② 逐轮配对(消绝对值漂移) ③ 主表与 p50 并列 + 符号检验。

结论(写进 hetero_moe_int8.d2h_wait 注释): 否证, 默认保持 main。
  delay=0 A: 主表 -2.7%(9/13 同向) / p50 -1.5% ; worker_block 主表 -6.9% 但 p50 +0.6%
  delay=0 C: 主表 -1.6% / p50 +0.7% → 打平
  delay=3 A/C: 全部 ±0.3% 以内, 符号 7/15~10/15 = 抛硬币
省掉的只有"线程被唤醒"那 ~0.017ms; budget 表里的 cur_sync=0.038 是【GPU 还没造出
数据】, 换谁等 DMA 都是同一时刻完成, 调度学不掉。
"""
import os, sys, time, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 15))
NPD    = int(os.environ.get("NPD", 200))
DELAYS = [float(v) for v in os.environ.get("DELAYS", "0.0,3.0").split(",")]
VAR = ["main", "worker", "worker_block"]
CFG = [("A", [0, 1]), ("C", [5, 6])]
MODE = "L2_零CUDA+pinned"
torch.manual_seed(42); x = torch.randn(32, 1024).cuda()


def p50(t):
    s = sorted(t); return s[len(s) // 2]


def both(m):
    """同一轮里先给主表口径(min-of-3-means), 再给逐帧 p50"""
    a = H.bench(m, x, True, MODE)[0]
    f = []
    for _ in range(NPD):
        t0 = time.perf_counter(); m(x, async_mode=True, mode=MODE)
        f.append((time.perf_counter() - t0) * 1000)
    return a, p50(f)


for delay in DELAYS:
    M = {}
    for lb, hot in CFG:
        for v in VAR:
            torch.manual_seed(42)
            M[(lb, v)] = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                                     cpu_delay_ms=delay, n_threads=6, tok_block=8,
                                     gpu_impl="cat", graph_scope="full",
                                     d2h_wait=v).cuda()
            for _ in range(30):
                M[(lb, v)](x, async_mode=True, mode=MODE)
    torch.cuda.synchronize()
    S = {k: ([], []) for k in M}
    for _ in range(ROUNDS):
        for k in M:
            a, b = both(M[k]); S[k][0].append(a); S[k][1].append(b)
    print(f"\n{'=' * 78}\nT8-② 等D2H口径终审 delay={delay}ms  loadavg="
          f"{open('/proc/loadavg').read().split()[0]}  ROUNDS={ROUNDS} NPD={NPD}\n{'=' * 78}")
    print(f"{'cfg':<4}{'等法':<14}{'主表口径med':>12}{'σ%':>6}{'p50口径med':>12}{'σ%':>6}"
          f"{'配对Δ主表':>11}{'配对Δp50':>11}{'符号(主表/p50)':>16}")
    sd = lambda t: statistics.stdev(t) / statistics.mean(t) * 100
    for lb, _ in CFG:
        bm = [S[(lb, 'main')][0][r] for r in range(ROUNDS)]
        bp = [S[(lb, 'main')][1][r] for r in range(ROUNDS)]
        for v in VAR:
            am, ap = S[(lb, v)]
            dm = [am[r] / bm[r] - 1 for r in range(ROUNDS)]
            dp = [ap[r] / bp[r] - 1 for r in range(ROUNDS)]
            sm = sum(1 for d in dm if d < 0); sp = sum(1 for d in dp if d < 0)
            print(f"{lb:<4}{v:<14}{statistics.median(am):>12.3f}{sd(am):>6.0f}"
                  f"{statistics.median(ap):>12.3f}{sd(ap):>6.0f}"
                  f"{statistics.median(dm) * 100:>+10.1f}%{statistics.median(dp) * 100:>+10.1f}%"
                  f"{f'{sm}/{len(dm)} vs {sp}/{len(dp)}':>16}")
    M.clear(); torch.cuda.empty_cache()

"""分布级 A/B: 抓"min 好看但 p50 没变"的假收益。

起因: T8-②(把"等 D2H 落地"从主线程挪进 CPU worker)用 min-of-7 量到 -16%,
高兴了一整轮; 换成 p50 重测只有 -0.7%(A) / +1.1%(C), 而且 σ 从 28% 涨到 36%
—— 因为自旋等 event 多占一个核。原因是方法论级的: **min-of-N 系统性奖励方差大
的那个变体**, 在脏机器上尤其致命。所以任何新刀必须先过这个口径再说收益。
顺带的物理结论: worker 本来就要等 DMA 落地才能开工, 换谁等都是一样的时刻,
主线程那次 sync 里真正能省掉的只有"线程被唤醒"的十几微秒。
用法: N=300 python bench_dist_ab.py     # 报 min/p50/p90/p99/max/σ + 相对 main 的增减
"""
import os, sys, time, statistics, torch
sys.path.insert(0,'.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
N = int(os.environ.get("N", 400))
VAR = ["main", "worker", "worker_block"]
CFG = [("A",[0,1]), ("C",[5,6])]
M={}
torch.manual_seed(42); x=torch.randn(32,1024).cuda()
for lb,hot in CFG:
    for v in VAR:
        torch.manual_seed(42)
        M[(lb,v)] = H.HeteroMoE(1024,8,2048,2,2,hot_experts=hot,cpu_delay_ms=0.0,
                                graph_scope="full", d2h_wait=v).cuda()
        for _ in range(30): M[(lb,v)](x, async_mode=True, mode="L2_零CUDA+pinned")
torch.cuda.synchronize()
S={k:[] for k in M}
def one(m):
    t0=time.perf_counter(); m(x, async_mode=True, mode="L2_零CUDA+pinned")
    return (time.perf_counter()-t0)*1000
for r in range(N):
    for k in M: S[k].append(one(M[k]))
print(f"governor={open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()} "
      f"loadavg={open('/proc/loadavg').read().split()[0]} N={N}")
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p*len(v)))]
print(f"{'配置':<5}{'等法':<14}{'min':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}{'σ%':>7}")
for lb,_ in CFG:
    base=[ ]
    for v in VAR:
        t=S[(lb,v)]
        print(f"{lb:<5}{v:<14}{min(t):>8.3f}{q(t,.5):>8.3f}{q(t,.9):>8.3f}{q(t,.99):>8.3f}"
              f"{max(t):>8.3f}{statistics.stdev(t)/statistics.mean(t)*100:>7.0f}")
    b50=q(S[(lb,'main')],.5)
    for v in VAR:
        print(f"    {v:<12} p50 vs main = {(q(S[(lb,v)],.5)/b50-1)*100:+6.1f}%   "
              f"min vs main = {(min(S[(lb,v)])/min(S[(lb,'main')])-1)*100:+6.1f}%")

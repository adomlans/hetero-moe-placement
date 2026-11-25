"""
CPU 亲和性卡尺(不需要 root): 本机 i5-10400 物理核 0-5, 超线程兄弟是 N 和 N+6。
libgomp 不绑核时由 Linux 调度器随机放 —— 6 个 worker + 1 个主线程 = 7 条 runnable
抢 6 个物理核, 主线程被挤到兄弟线程上, 每次 aten dispatch 都要付频率爬坡的税。
用法(每个变体一个新进程, 因为 libgomp 只在进程启动时读环境变量):
    GOMP_CPU_AFFINITY=1-5 python bench_affinity.py 5
"""
import os, sys, time, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

nt = int(sys.argv[1]) if len(sys.argv) > 1 else 6
tag = sys.argv[2] if len(sys.argv) > 2 else ""
out = []
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                    n_threads=nt, tok_block=8).cuda()
    tsy = H.bench(m, x, False, None)[0]
    t1 = H.bench(m, x, True, "L1_松GIL")[0]
    t2 = H.bench(m, x, True, "L2_零CUDA+pinned")[0]
    out.append(f"{label}:sync={tsy:.3f}/L1={t1:.3f}/L2={t2:.3f}")
print(f"nt={nt:<3} {tag:<34} {'  '.join(out)}", flush=True)

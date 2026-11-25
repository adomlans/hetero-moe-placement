"""
✅ T5 配套卡尺: 只量 CPU kernel, 把 GPU/传输/Python 全部摘掉。
用法: 用它量出 T_kernel, 再用 hetero_moe_int8.py 的 wall time 减一下,
      剩下的就是"固定成本"(launch + H2D/D2H + Python + 那次 stream sync)。
命中率直接抄 hetero_moe_int8.py 实测的两张表, 所以这里的 T_kernel 能和你
主表上的毫秒数对上号。
"""
import sys, time, torch
from torch.utils.cpp_extension import load
sys.setswitchinterval(1e-5)

ext = load(name="avx_gemv_int8", sources=["avx_gemv_int8.cpp"],
           extra_cflags=["-O3", "-mavx2", "-mfma", "-fopenmp"],
           extra_ldflags=["-fopenmp"], verbose=False)

H, E, NE, GPU_CNT = 1024, 2048, 8, 2      # hidden, expert_size, num_experts, gpu 专家数
BS, TOPK = 32, 2
# 实测命中率(%槽位), 抄自 hetero_moe_int8.py delay=0 那两张表
HIT = {"A": [40.6, 46.9, 3.1, 0.0, 0.0, 0.0, 1.6, 7.8],
       "C": [1.6, 4.7, 1.6, 0.0, 0.0, 45.3, 40.6, 6.2]}

def case(hit):
    g = torch.Generator().manual_seed(42)
    p = torch.tensor(hit, dtype=torch.float64); p = p / p.sum()
    idx = torch.multinomial(p, BS * TOPK, replacement=True, generator=g).view(BS, TOPK).to(torch.int64)
    return (torch.randint(-100, 100, (BS, H), generator=g, dtype=torch.int8),
            torch.rand(BS, generator=g) * 0.02, idx,
            torch.rand(BS, TOPK, generator=g),
            torch.randint(0, 256, (NE - GPU_CNT, E, H), generator=g, dtype=torch.uint8),
            torch.rand(NE - GPU_CNT, generator=g) * 0.02)

def bench(fn, iters=100):
    for _ in range(10): fn()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        best = min(best, (time.perf_counter() - t0) / iters * 1000)
    return best

for key, name in (("A", "配置 A (热专家在GPU)"), ("C", "配置 C (冷专家在GPU)")):
    x, xs, idx, sc, w, ws = case(HIT[key])
    n_slot = int((idx >= GPU_CNT).sum())
    out = torch.empty(BS, E, dtype=torch.float32).pin_memory()
    print(f"\n===== {name} : 落到 CPU 的槽位 {n_slot}/{BS*TOPK} =====")
    print(f"{'tb  nt':>7}" + "".join(f"{nt:>9}" for nt in (1, 2, 4, 6, 8, 12)) + "   (单位 ms = T_kernel)")
    for tb in (1, 4, 8):
        row = [bench(lambda: ext.avx2_moe_cpu_experts_int8_into(
                x, xs, idx, sc, w, ws, out, GPU_CNT, True, nt, tb))
               for nt in (1, 2, 4, 6, 8, 12)]
        print(f"{tb:>7}" + "".join(f"{v:>9.3f}" for v in row))
    t = bench(lambda: ext.avx2_moe_cpu_experts_int8_into(x, xs, idx, sc, w, ws, out, GPU_CNT, True, 6, 8))
    print(f"  tb=8 nt=6: T_kernel={t:.3f}ms  {t*1000/max(n_slot,1):.1f}us/slot  "
          f"权重等效带宽 {max(n_slot,1)*E*H/(t/1000)/1e9:.1f} GB/s")

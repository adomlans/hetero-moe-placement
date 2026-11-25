#!/usr/bin/env python
"""纯 CPU kernel 微基准: 故意构造路由分布, 回答"kernel 到底卡在哪"。

绕开整个 MoE/GPU, 直接手搓 int8 权重 + 指定"每个 CPU 专家分到几个 token 槽位",
扫描 分布 x tok_block x OMP线程。它推翻过两个想当然的结论:
  * 【不是权重带宽】1专家x1槽 = 0.020ms, 折算 100GB/s(还在 L3 里);
    1专家x8槽算术量 x8 只慢 2.85x => 是 maddubs 端口 throughput 瓶颈,
    所以"把权重补齐到 2 的幂"这类省带宽的招一律无效(已实测无收益)。
  * 【L3 悬崖在 4~6MB】4专家x1槽 0.070ms vs 6专家x1槽 0.340ms —— 工作集超过
    i5-10400 的 12MB L3 就掉到内存, 这才是"槽位散在多专家"贵的真原因。
用法: python bench_kernel_micro.py
"""
import sys, time, os, torch
from torch.utils.cpp_extension import load
sys.path.insert(0,'.'); sys.setswitchinterval(1e-5)
ext = load(name="avx_gemv_int8", sources=["avx_gemv_int8.cpp"],
           extra_cflags=["-O3","-mavx2","-mfma","-fopenmp"], extra_ldflags=["-fopenmp"], verbose=False)
H, E, NE, GC = 1024, 2048, 8, 2
BS, TOPK = 32, 2
g = torch.Generator().manual_seed(1)
XQ  = torch.randint(-100,100,(BS,H),generator=g,dtype=torch.int8)
XS  = torch.rand(BS,generator=g)*0.02
WQ  = torch.randint(0,256,(NE-GC,E,H),generator=g,dtype=torch.uint8)
WS  = torch.rand(NE-GC,generator=g)*0.02
def make(slots_per_exp):          # dict: cpu专家e -> 槽位数
    idx = torch.zeros(BS, TOPK, dtype=torch.int64)
    sc  = torch.rand(BS, TOPK, generator=g)
    flat = []
    for e, n in slots_per_exp.items(): flat += [GC+e]*n
    flat += [0]*(BS*TOPK-len(flat))       # 其余槽位塞给 GPU 专家(被 kernel 跳过)
    perm = torch.randperm(BS*TOPK, generator=g)
    idx = torch.tensor(flat, dtype=torch.int64)[perm].view(BS, TOPK)
    return idx.pin_memory(), sc.pin_memory()
def wall(fn, iters=200):
    for _ in range(20): fn()
    best=1e9
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(iters): fn()
        best=min(best,(time.perf_counter()-t0)/iters*1000)
    return best
OUT = torch.zeros(BS,E).pin_memory()
CASES = [("1专家x1槽", {0:1}), ("1专家x4槽", {0:4}), ("1专家x8槽", {0:8}), ("1专家x16槽",{0:16}),
         ("4专家x1槽", {0:1,1:1,2:1,3:1}), ("4专家x2槽", {0:2,1:2,2:2,3:2}),
         ("4专家x4槽", {0:4,1:4,2:4,3:4}), ("4专家x8槽", {0:8,1:8,2:8,3:8}),
         ("6专家x1槽", {e:1 for e in range(6)}), ("6专家x4槽", {e:4 for e in range(6)})]
# ✅ T8-①: 专家数扫描 —— 每专家固定 4 槽(算术量随专家数线性增),
#    若时间随【权重总量】而不是随【槽数】跳, 就是带宽 bound。
for e in range(1, 7):
    nm = f"{e}专家x4槽"
    if not any(c[0] == nm for c in CASES):
        CASES.append((nm, {k: 4 for k in range(e)}))
L3 = H.l3_bytes() if hasattr(H, "l3_bytes") else (12 << 20)
print(f"L3={L3>>20}MB  每专家权重={WQ.shape[1]*WQ.shape[2]>>20}MB  "
      f"(有效带宽按【单趟权重字节数】算, 多槽时权重会被重扫, 所以是下界)")
print(f"{'分布':<12}{'槽':>4}{'tb':>4}{'工作集':>8}" + "".join(f"{'nt='+str(n):>10}" for n in (1,6))
      + f"{'GB/s(nt6)':>11}  在L3?")
for tb in (1, 8):
    for name, sp in CASES:
        tot = sum(sp.values())
        ne = len(sp)
        ws = ne * WQ.shape[1] * WQ.shape[2]
        idx, sc = make(sp); row=[]
        for nt in (1,6):
            row.append(wall(lambda nt=nt: ext.avx2_moe_cpu_experts_int8_into(
                XQ,XS,idx,sc,WQ,WS,OUT,GC,True,nt,tb)))
        gbs = ws / row[1] / 1e6
        print(f"{name:<12}{tot:>4}{tb:>4}{ws>>20:>7}MB" + "".join(f"{v:>10.3f}" for v in row)
              + f"{gbs:>11.0f}  {'是' if ws <= L3*0.75 else '🚨漏'}")

# ============================================================
# ✅ T10: signed 打包(默认)比 biased 多两条 uop(abs + sign_epi8), 买不买得起?
#    WQ_S8 直接从 WQ 减 128 派生 => 两边是【同一批整数】, 唯一差别就是 kernel 路径,
#    所以这个 A/B 里没有量化格点/数据分布的混淆项。
#    预期: DRAM bound 区(6专家12MB)几乎免费(带宽不缺 uop);
#          L3 区(4专家8MB, 111GB/s)算术开始显形, 会看到百分数上的代价。
# ============================================================
WQ_S8 = (WQ.to(torch.int16) - 128).to(torch.int8)
print("\n=== T10 定价: biased(maddubs, 有饱和 bug) vs signed(可证明精确) ===")
print(f"{'分布':<12}{'tb':>3}{'nt':>3}{'biased':>9}{'signed':>9}{'代价':>8}   同一批整数?")
for name, sp in [("4专家x4槽", {0:4,1:4,2:4,3:4}), ("6专家x4槽", {e:4 for e in range(6)}),
                 ("1专家x8槽", {0:8}), ("6专家x1槽", {e:1 for e in range(6)})]:
    idx, sc = make(sp)
    for tb in (1, 8):
        for nt in (1, 6):
            b = wall(lambda: ext.avx2_moe_cpu_experts_int8_into(
                XQ, XS, idx, sc, WQ, WS, OUT, GC, True, nt, tb, signed_w=0))
            s_ = wall(lambda: ext.avx2_moe_cpu_experts_int8_into(
                XQ, XS, idx, sc, WQ_S8, WS, OUT, GC, True, nt, tb))
            print(f"{name:<12}{tb:>3}{nt:>3}{b:>9.3f}{s_:>9.3f}{(s_/b-1)*100:>7.1f}%"
                  f"   {'✅' if tb==1 and nt==1 else ''}")

#!/usr/bin/env python
"""T11 候选 C: 把前段(门控 GEMM -> softmax -> topk -> 逐行 amax -> 量化)融成 2 张 triton kernel。

为什么要动它(probe_pre.py 实测, bs=32):
    前段图 12 张 kernel, GPU 真忙 45.7us, 墙钟 45.7us(完全 GPU bound)
      ampere_sgemm_32x32_...  19.4us  <- 32x1024 @ 1024x8 的小 GEMM, 只有 27 GFLOP/s
      gatherTopK + bitonicSort 9.1us  <- 给 (32 行 x 8 列) 跑一整套 bitonic 排序
      reduce + 量化链          ~12us
    理论活儿 = 读 x 128KB + W 32KB ≈ 0.5us。
🚨 它在关键路径【最前面】: 前段不完事, D2H 就不能发, CPU worker 就开不了工。
   所以这 45us 是配置 A/C 共有、且所有 batch 都要付的固定成本 —— 比 tail 那 15us 大得多。

正确性闸门(先过闸门再谈延迟, 顺序不能反):
    topk_indices 必须与 torch 逐位相同(整数, 没有"近似"可言)
    x_q          必须与 torch 逐位相同(int8 格点; 差 1 个 LSB 就会改 kernel 输出)
    scores/scale 与 torch 差 < 1e-6 / 1e-7
    x/scale 的 round 必须同为【半数取偶】: torch.round 是 banker's, triton 用 rint
    GEMM 用 input_precision="ieee" —— 🚨 默认 tf32 会改 logits, 近分的 token 会跳到别的专家,
    那不是"精度略降"而是【路由结果变了】, 整个专家分配都会跟着变。
用法: python bench_pre_fuse.py
"""
import sys, time, torch, triton
import triton.language as tl
from triton.language.extra import libdevice   # rint = 半数取偶, 和 torch.round 同一个舍入模式
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H


from pre_fused import route_quant_fused as fused_route_quant, cap_graph, gate_wt, alloc_pre_out


def wall(fn, n=500):
    for _ in range(30): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best


print(f"[机器状态] {H.machine_state()}")
print("=" * 78)
print("正确性闸门: 融合前段 vs torch 前段(逐位)     延迟: 12张kernel -> 2张kernel")
print("=" * 78)
print(f"{'bs':>4}  {'idx逐位':>8} {'scores差':>10} {'x_q逐位':>8} {'scale差':>10}"
      f"  {'torch前段':>10} {'融合前段':>9} {'省':>7}")
allok = True
for bs in (1, 4, 32, 128):
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[0, 1]).cuda()
    x = torch.randn(bs, 1024).cuda()
    ts_t, ti_t, xq_t, xs_t = m._route_quant(x)
    m._gate_wt = gate_wt(m)
    work = alloc_pre_out(m, bs, 'cuda')
    ts_f, ti_f, xq_f, xs_f = fused_route_quant(m, x, work)
    idx_ok = bool((ti_f == ti_t).all())
    xq_ok = bool((xq_f == xq_t).all())
    sd = (ts_f - ts_t).abs().max().item()
    cd = (xs_f - xs_t).abs().max().item()
    ok = idx_ok and xq_ok and sd < 1e-6 and cd < 1e-7
    allok &= ok
    t_old = wall(lambda: m._route_quant(x))
    t_new = wall(lambda: fused_route_quant(m, x, work))
    print(f"{bs:>4}  {str(idx_ok):>8} {sd:>10.2e} {str(xq_ok):>8} {cd:>10.2e}"
          f"  {t_old:>8.4f}ms {t_new:>8.4f}ms {t_old / t_new:>6.2f}x   {'PASS ✅' if ok else 'FAIL 🚨'}")
print("\n" + ("ALL PASS ✅ 可以接进模型" if allok else "🚨 有失败项, 不许接线"))

# ------------------------------------------------------------------
# 上面那列延迟【不能拿来决策】: eager 下 triton 的 Python 派发费 ~25us/次,
# 会把 GPU 时间整个盖住。真路径里前段录在 CUDA Graph 内 => 派发费=0,
# 所以必须在"图 replay"口径下重比一次。这才是会写进总账的数。
# ------------------------------------------------------------------
print("\n" + ("ALL PASS ✅ 可以接进模型" if allok else "🚨 有失败项, 不许接线"))

# ------------------------------------------------------------------
# 上面那列延迟【不能拿来决策】: eager 下 triton 的 Python 派发费 ~25us/次,
# 会把 GPU 时间整个盖住。真路径里前段录在 CUDA Graph 内 => 派发费=0,
# 所以必须在"图 replay"口径下重比一次。这才是会写进总账的数。
# ------------------------------------------------------------------
def cap_graph(fn):
    cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(cap); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g


print("\n" + "=" * 78)
print("图 replay 口径(= 真路径口径): 前段图 12 张 kernel vs 2 张 kernel")
print("=" * 78)
print(f"{'bs':>4}  {'torch录图':>10} {'融合录图':>10} {'省':>8} {'省下的us':>9}")
for bs in (1, 4, 32, 128):
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[0, 1]).cuda()
    x = torch.randn(bs, 1024).cuda()
    m._gate_wt = gate_wt(m)
    work = alloc_pre_out(m, bs, 'cuda')
    g_old = cap_graph(lambda: m._route_quant(x))
    g_new = cap_graph(lambda: fused_route_quant(m, x, work))
    t_old, t_new = wall(g_old.replay), wall(g_new.replay)
    print(f"{bs:>4}  {t_old:>8.4f}ms {t_new:>8.4f}ms {t_old / t_new:>7.2f}x "
          f"{(t_old - t_new) * 1000:>8.1f}us")
    del m, g_old, g_new
sys.exit(0 if allok else 1)

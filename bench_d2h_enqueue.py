#!/usr/bin/env python
"""主机【入队】成本探针: 只量 cuda 操作提交到流的开销(不 sync)。

当 GPU 侧真的很快时, 剩下的延迟不在 GPU 也不在 CPU 算力, 而在"主线程一条条
往流里塞命令"的 aten dispatch + driver 开销。这脚本把每一段单独量出来:
  ① 4x copy_ (D2H)  ② _foreach_copy_  ③ 把 4 次 D2H 录成第三张图 replay
  ④ .to()+add(现状收尾)  ⑤ 预分配 copy_+add  ⑥ add(out=)  ⑦⑧ 拆半对照
实测结论: ③ 0.0218 -> 0.0064ms(所以 L2 用了第三张 D2H 图, 且只有 graph_scope=full
才合法 —— 源地址必须是 _g_pre 的静态输出), ⑤ 0.0208 -> 0.0158ms,
② _foreach_copy_ 跨设备不比 4x copy_ 快(已否证, 别再做)。
用法: python bench_d2h_enqueue.py
"""
import sys, time, torch
sys.path.insert(0,'.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
torch.manual_seed(42); x = torch.randn(32,1024).cuda(); torch.manual_seed(42)
m = H.HeteroMoE(1024,8,2048,2,2,hot_experts=[5,6],cpu_delay_ms=0.0,graph_scope="full").cuda()
m._ensure_graphs(x)
m._prepare(x); ts,ti,xq,xs = m._prepare(x)
bs = 32
m._ensure_pins(bs, x.device); p = m._pin
srcs = [xq, xs, ti, ts]; dsts = [p["xq"],p["sc"],p["idx"],p["scr"]]
def enq(fn, iters=500, reps=7):
    """只量【主机入队】耗时(不 sync) —— 这就是固定成本里的那一段"""
    for _ in range(20): fn(); torch.cuda.current_stream().synchronize()
    best=1e9
    for _ in range(reps):
        torch.cuda.current_stream().synchronize(); t0=time.perf_counter()
        for _ in range(iters): fn()
        dt=(time.perf_counter()-t0)/iters*1000
        torch.cuda.current_stream().synchronize(); best=min(best,dt)
    return best
print(f"{'方案':<26}{'入队 ms':>9}")
print(f"{'① 现状 4x copy_':<26}{enq(lambda: [d.copy_(s, non_blocking=True) for d,s in zip(dsts,srcs)]):>9.4f}")
try:
    print(f"{'② _foreach_copy_':<26}{enq(lambda: torch._foreach_copy_(dsts, srcs, non_blocking=True)):>9.4f}")
except Exception as e: print("② _foreach_copy_ 失败:", type(e).__name__, str(e)[:90])
# ③ 第三张图: 把 4 次 D2H 录成 graph, 主机只剩 1 次 replay
try:
    g3 = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        for _ in range(3):
            for d,s in zip(dsts,srcs): d.copy_(s, non_blocking=True)
    torch.cuda.current_stream().wait_stream(cap); torch.cuda.synchronize()
    with torch.cuda.graph(g3):
        for d,s in zip(dsts,srcs): d.copy_(s, non_blocking=True)
    print(f"{'③ D2H 录进第三张图':<26}{enq(lambda: g3.replay()):>9.4f}   (1 次 replay)")
except Exception as e:
    print("③ D2H 图失败:", type(e).__name__, str(e)[:140])
# ---- 收尾那一刀: pinned -> GPU + add ----
hout = torch.empty(bs, 2048, device=x.device, dtype=torch.float32)
gpu_out = torch.empty(bs, 2048, device=x.device, dtype=torch.float32)
res = torch.empty(bs, 2048, device=x.device, dtype=torch.float32)
print(f"{'④ 现状 .to()+add':<26}{enq(lambda: torch.add(gpu_out, p['out'].to(device=x.device, dtype=x.dtype, non_blocking=True))):>9.4f}")
print(f"{'⑤ 预分配 copy_+add':<26}{enq(lambda: torch.add(gpu_out, hout.copy_(p['out'], non_blocking=True))):>9.4f}")
print(f"{'⑥ 预分配 copy_+add(out=)':<26}{enq(lambda: torch.add(gpu_out, hout.copy_(p['out'], non_blocking=True), out=res)):>9.4f}")
print(f"{'⑦ 只 copy_ 不 add':<26}{enq(lambda: hout.copy_(p['out'], non_blocking=True)):>9.4f}")
print(f"{'⑧ 只 add':<26}{enq(lambda: torch.add(gpu_out, hout, out=res)):>9.4f}")

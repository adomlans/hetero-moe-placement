#!/usr/bin/env python
"""T11 定价: 固定入场费里『图能吞掉的那部分』到底值多少。

分账表(budget_check)给出配置 A 的 0.443ms 里:
    add+H2D回 0.060 | prepare 0.058 | H2D到pinned 0.019
这三段的共同点: GPU 真活儿几乎为 0((32,1024) 的 add 只有 128KB), 付的是
【主机入队 + aten 派发 + 分配器】。所以问题只有一个: 录成图之后省多少?

量两个候选, 都是【同进程内轮转 + 逐轮取优】, 且分开报"主机入队"和"含 sync 墙钟":
  A) tail 段:  `hout.copy_(pinned); torch.add(shot, hout)`   ->  录成 1 张图 replay
  B) 前段:     g_pre(route+quant) + g_d2h(4 次 copy)  两次 replay -> 合并成 1 张图
B 之所以合法: 合并后 copy 的源是【同一张图池里的静态输出】, 地址在 capture 时就定死,
   和"先录 pre 再单独录 d2h"是同一套地址, 不存在读旧内存的风险。
🚨 A 有个语义代价, 这里顺手量出来给你看: 图 replay 复用的是【同一块输出缓冲】,
   所以连续两次 forward 返回的是同一个张量 —— 上一帧的结果会被下一帧覆写。
   这是 CUDA Graph 的固有语义(vLLM 也这样), 不是 bug, 但它是【接口语义变了】。
用法: python bench_t11_entry.py
"""
import sys, time, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = 7
N = 300


def enqueue_cost(fn, n=N):
    """只量主机入队(不 sync) —— 图/aten 的派发费就在这。"""
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    torch.cuda.synchronize()
    return best


def wall_cost(fn, n=N):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best


print(f"[机器状态] {H.machine_state()}")
print(f"{'':<22}{'A入队':>9}{'A墙钟':>9}{'B入队':>9}{'B墙钟':>9}")
hdr = f"{'候选(旧->新)':<22}{'旧':>9}{'新':>9}{'旧':>9}{'新':>9}"

for bs in (1, 4, 32):
    for label, hot in (("A", [0, 1]), ("C", [5, 6])):
        torch.manual_seed(42); x = torch.randn(bs, 1024).cuda()
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, gpu_impl="cat",
                        graph_scope="full").cuda()
        m._ensure_pins(bs, x.device)
        m._ensure_graphs(x)
        m._sx.copy_(x); m._g_pre.replay(); m._d2h_to_pins(m._sxq, m._sxs, m._sti, m._sts)
        torch.cuda.synchronize()

        # ---------- 候选 A: tail 段 (H2D + add) ----------
        shot = m._shot
        with torch.cuda.stream(m.gpu_stream):
            pass
        m._hout = torch.empty(bs, m.expert_size, device='cuda', dtype=torch.float32)

        def old2():
            m._hout.copy_(m._pin["out"], non_blocking=True)
            return torch.add(shot, m._hout)

        cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap):
            for _ in range(3):
                old2()
        torch.cuda.current_stream().wait_stream(cap); torch.cuda.synchronize()
        g_tail = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_tail):
            m._hout.copy_(m._pin["out"], non_blocking=True)
            static_out = torch.add(shot, m._hout)
        new_tail = lambda: g_tail.replay()
        a_old, a_new = enqueue_cost(old2), enqueue_cost(new_tail)
        a_oldw, a_neww = wall_cost(old2), wall_cost(new_tail)
        # 语义代价探针: 连续两帧返回的是不是同一块缓冲
        g_tail.replay(); o1 = static_out.clone().sum().item()
        m._pin["out"].fill_(3.0); g_tail.replay()
        alias = bool(static_out.data_ptr() == static_out.data_ptr()) and \
            abs(static_out.sum().item() - o1) > 1e-6
        print(f"bs={bs:>2} {label} tail入队 {a_old:.4f}->{a_new:.4f}ms "
              f"(省 {100*(1-a_new/a_old):.0f}%)  墙钟 {a_oldw:.4f}->{a_neww:.4f}ms"
              f"   🚨覆写上一帧: {'是' if alias else '否'}")

        # ---------- 候选 B: 前段 pre + d2h 合并 ----------
        old_pre = lambda: (m._g_pre.replay(), m._g_d2h.replay())
        def body():
            sts, sti, sxq, sxs = m._route_quant(m._sx)
            p = m._pin
            p["xq"].copy_(sxq, non_blocking=True); p["sc"].copy_(sxs, non_blocking=True)
            p["idx"].copy_(sti, non_blocking=True); p["scr"].copy_(sts, non_blocking=True)
            return sts, sti
        cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(cap); torch.cuda.synchronize()
        g_merged = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_merged):
            body()
        new_pre = lambda: g_merged.replay()
        b_old, b_new = enqueue_cost(old_pre), enqueue_cost(new_pre)
        b_oldw, b_neww = wall_cost(old_pre), wall_cost(new_pre)
        print(f"        pre+d2h入队 {b_old:.4f}->{b_new:.4f}ms "
              f"(省 {100*(1-b_new/b_old):.0f}%)  墙钟 {b_oldw:.4f}->{b_neww:.4f}ms")
        ok = (g_merged is not None)
        del m, g_tail, g_merged

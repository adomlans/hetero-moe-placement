"""
分账卡尺 (作业 6-A/6-B 的自动版): 把一条 forward 拆成三笔
    墙钟 = 固定成本 + max(GPU 半边, T_kernel)
    固定成本 = launch + 主机往返 + Python + 那次 stream sync
GPU 半边顺手数 kernel 张数和主机往返次数(作业 6-B)。
"""
import sys, time, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

def wall(fn, warm=20, iters=200, reps=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t0) / iters * 1000)
    return best

def gpu_half(m, x, ts, ti, st):
    """GPU 半边: 墙钟 / kernel 张数 / 主机往返次数 / GPU 真忙碌"""
    t = wall(lambda: m._gpu_hot_experts(x, ts, ti, st))
    try:
        for _ in range(10): m._gpu_hot_experts(x, ts, ti, st)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(20): m._gpu_hot_experts(x, ts, ti, st)
            torch.cuda.synchronize()
        ka = p.key_averages()
        nk = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cudaGraphLaunch")) / 20
        nr = sum(e.count for e in ka if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                                                  "cudaMemcpyAsync", "cudaMemcpy")) / 20
        busy = sum(e.device_time_total for e in ka if e.device_time_total > 0) / 20 / 1000
    except Exception:
        nk = nr = busy = float("nan")
    return t, nk, nr, busy

print(f"[机器状态] {H.machine_state()}")
print(f"{'':<14}{'墙钟L2':>9}{'GPU半边':>9}{'kernel':>8}{'往返':>7}{'真忙':>8}"
      f"{'T_kernel':>10}{'固定成本':>10}   ({'配置'})")
for label, hot in (("配置 A", [0, 1]), ("配置 C", [5, 6])):
    for impl, gr in (("sparse", "off"), ("cat", "off"),
                     ("cat", "hot"), ("cat", "full")):      # ✅ T7: 加 full 档
        torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, gpu_impl=impl, graph_scope=gr).cuda()
        ts, ti = m.route(x)
        st = torch.cuda.Stream()
        # --- T_kernel: 用这个模型自己的路由/量化结果, 只跑 CPU kernel ---
        x_q, x_s = m._quantize_act(x)
        px = x_q.cpu().pin_memory(); ps = x_s.cpu().pin_memory()
        pi = ti.cpu().pin_memory(); psc = ts.cpu().pin_memory()
        po = torch.empty(32, 2048, dtype=torch.float32).pin_memory()
        tk = wall(lambda: H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
            px, ps, pi, psc, m.cpu_experts_q, m.cpu_experts_scale, po,
            m.gpu_expert_count, True, m.n_threads, m.tok_block), iters=300)
        tg, nk, nr, busy = gpu_half(m, x, ts, ti, st)
        tl2 = H.bench(m, x, True, "L2_零CUDA+pinned")[0]
        fixed = tl2 - max(tg, tk)
        tag = f"{label} {impl}/{gr}"
        print(f"{tag:<14}{tl2:>9.3f}{tg:>9.3f}{nk:>8.0f}{nr:>7.1f}{busy:>8.3f}"
              f"{tk:>10.3f}{fixed:>10.3f}")

# ============================================================
# 固定成本明细: 手抄一份和 _forward_async_L2 逐行同构的序列, 分段掐表
#   (主机侧耗时, 不是 GPU 耗时; 累加 ≈ 墙钟里扣掉 GPU/CPU 真干活的那段)
#   ✅ T7: 手抄版改成调 _prepare/_hot_half —— 否则 full 档的收益在明细里是隐形的
# ============================================================
N = 300
# 明细表按档位分开跑, 直接对比同一段代码在两种 graph_scope 下的主机开销:
#     python budget_check.py off    # = T6a (eager 路由+量化)
#     python budget_check.py full   # = T7  (路由+量化在图里)
DETAIL_SCOPE = sys.argv[1] if len(sys.argv) > 1 else "off"
assert DETAIL_SCOPE in ("off", "hot", "full"), DETAIL_SCOPE
print(f"\n{'='*74}\n固定成本明细 (L2, 主机侧 ms/次, graph_scope={DETAIL_SCOPE})\n{'='*74}")
KEYS = ("prepare(路由+量化)", "H2D到pinned", "cur_sync", "submit",
        "GPU热专家", "fut.result", "add+H2D回", "stream_sync")
for label, hot in (("配置 A", [0, 1]), ("配置 C", [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                    n_threads=6, tok_block=8, graph_scope=DETAIL_SCOPE).cuda()
    if DETAIL_SCOPE != "off":
        m._ensure_graphs(x)          # 🚨 录图里有 synchronize, 不能在 stream ctx 里做
    acc = {k: 0.0 for k in KEYS}
    tick = time.perf_counter
    for _ in range(20):
        m(x, async_mode=True, mode="L2_零CUDA+pinned")
    torch.cuda.synchronize()
    t_all = tick()
    for _ in range(N):
        t0 = tick()
        ts, ti, xq, xs = m._prepare(x); t1 = tick()
        m._ensure_pins(x.shape[0], x.device); p = m._pin
        m._d2h_to_pins(xq, xs, ti, ts); t2 = tick()      # ✅ 和真路径同构(full 档走第三张图)
        torch.cuda.current_stream().synchronize(); t3 = tick()
        fut = m.cpu_pool.submit(lambda: H.avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
            p["xq"], p["sc"], p["idx"], p["scr"], m.cpu_experts_q, m.cpu_experts_scale,
            p["out"], m.gpu_expert_count, True, m.n_threads, m.tok_block) or p["out"]); t4 = tick()
        ev = torch.cuda.Event(); ev.record()
        with torch.cuda.stream(m.gpu_stream):
            m.gpu_stream.wait_event(ev)
            g = m._hot_half(x, ts, ti); t5 = tick()
            r = fut.result(); t6 = tick()
            out = m._h2d_result(g); t7 = tick()          # ✅ 和真路径同构
            m.gpu_stream.synchronize()
        t8 = tick()
        for k, dt in zip(KEYS, (t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t8-t7)):
            acc[k] += dt
    tot = (time.perf_counter() - t_all) / N * 1000
    print(f"\n{label}  整条墙钟(手抄版) = {tot:.3f} ms")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<18}{v/N*1000:>8.3f} ms   {v/N*1000/tot*100:>5.1f}%")

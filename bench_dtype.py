#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-2c 判决表: 把 GPU 半边的热专家权重从 fp32 降到 bf16/fp16, 每帧少读一半字节, 值多少。

【这笔账从哪来(乙-2b 的归因)】
  bs=1 配置A 的 CPU 腿是【0 个 slot】(热专家接走全部流量), 可端到端还是 0.23ms;
  换成真形状 swiglu 就变 0.53ms。+0.3ms 与 slot 无关, bench_halves 把它钉死在 GPU 半边
  (ΔGPU +0.294 / ΔCPU +0.162 被 max() 重叠藏掉 / Δ端到端 +0.298)。
  再往下拆: GPU 半边读的是 fp32 权重 ——
      gemm  cat 档读 _wcat                 = D*gc*N     = 41.9MB @ 0.177ms = 237GB/s
      swiglu cat 档读 _wcat_gu + _wstack_d = D*gc*3N    = 125.8MB @ 0.470ms = 268GB/s
  两个数都贴着这块卡 skinny GEMM 的标定带宽(250GB/s) => 这一半边是【字节 bound】。
  字节 bound 的东西只能靠【少读字节】减价, 两个相乘的旋钮:
      ① 精度: fp32 -> fp16/bf16 (/2) -> int8 (/4)      ← 本刀
      ② 流量: bs=1 只有 top_k=2 个热专家真有流量, 却固定算了 gc 个  ← 下一刀 乙-2d

【这一刀要买三样东西, 缺一样就是白改】
  ① 延迟: GPU 半边和端到端各降多少。字节 bound 的期望是 GPU 半边约 /2;
     端到端降得比 GPU 半边少是【应该的】—— 那笔固定串行主机开销不随 dtype 变。
  ② 正确性三道闸, 一道都不能花:
     闸门1 三模式互差逐位 0(这条专防"返回半精 => _hout 用 gpu_out.dtype 建 =>
           CPU 腿那条 fp32 结果被相加截成半精");
     闸门2 路由 idx 必须和同 seed 的 fp32 模型【逐位相同】(dtype 只该动 GPU 那半边的
           乘加, 动到路由就是把乙-1 否证过的 tf32 前段又请回来);
     闸门3 整条 L2 vs 本实例 fp32 参考的 relRMSE —— 这才是降精真正要付的钱。
  ③ 放置规则的价格要跟着改: plan_gpu_capacity 的 gpu_weight_bytes 一直按 fp32 记,
     它自己的注释就写着"哪天 GPU 侧上了 fp16 权重, 两个字节数都要跟着改"。
     表末列出每个 dtype 下规则重选的 gc —— 降精会让 GPU 半边变便宜 => 规则该敢给更大 gc。

【已知会红的地方, 红得有道理才算数】
  swiglu 的中间量级在【随机 std=1 权重】下是 ~4e4~7e4(乙-2 量到 ③量级 6.95e+04),
  而 fp16 上限是 65504 => 这一档【预期溢出】。bf16 有位宽没精度(尾数 8bit =>
  相对 ~4e-3)。真 checkpoint 都按 1/sqrt(fan_in) 初始化、不会顶到这个量级,
  所以这张表的 fp16 行读的是"标度没管好会发生什么", 不是"fp16 不能用"。
"""
import os, sys, math, statistics, time, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ROUNDS = int(os.environ.get("ROUNDS", 5))
D, E, N, K = 1024, 8, 2048, 2
BS_LIST = [int(v) for v in os.environ.get("BS", "1").split(",")]
KINDS = [k for k in os.environ.get("EKIND", "gemm,swiglu").split(",") if k]
DTS = [d for d in os.environ.get("DTYPES", "fp32,bf16,fp16").split(",") if d]
GCS = [int(v) for v in os.environ.get("GC", "5").split(",")]
CFG = [("A", [0, 1]), ("C", [5, 6])]
MODE = "L2_零CUDA+pinned"
HIT_BS = int(os.environ.get("HIT_BS", 512))
assert set(KINDS) <= {"gemm", "swiglu"}, KINDS
assert set(DTS) <= {"fp32", "fp16", "bf16"}, DTS
# 全开档 = 前 14 刀的终点。本刀只换【GPU 侧权重精度】, 别的都不动
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
ELS = {"fp32": 4, "fp16": 2, "bf16": 2}
HITS = {}


def hits_cum(hot, kind):
    torch.manual_seed(42)
    probe = H.HeteroMoE(D, E, N, 2, K, hot_experts=hot, cpu_delay_ms=0.0,
                        expert_kind=kind).cuda()
    h = H.measure_hit_rates(probe, torch.randn(HIT_BS, D).cuda())
    del probe
    return h


def gpu_bytes_frame(m):
    """这一档【每帧真被读掉的】GPU 权重字节 = cat 档实际用到的那几个缓存。
    🚨 别把 _wstack/_wstack_gu 也计进来: 那是 bmm 档才读的, cat 档根本不碰,
       计进去就等于凭空给 cat 档加了一笔它没付的钱(和乙-2 初版算岔字节墙同类)。"""
    names = ("_wcat",) if m.kind == "gemm" else ("_wcat_gu", "_wstack_d")
    assert all(isinstance(nm, str) and nm.startswith("_") and len(nm) > 2 for nm in names)
    tot = 0
    for nm in names:
        t = getattr(m, nm, None)
        assert t is not None, f"{nm} 还没建 —— 先跑一次 forward"
        tot += t.numel() * t.element_size()
    return tot


def gpu_half(m, xx, ts, ti, iters=300):
    """GPU 半边(pre图 + D2H图 + 热专家图)单独跑完, 无 worker。取 5 组中位。"""
    def once():
        s_ts, s_ti, _, _ = m._prepare(xx)
        m._hot_half(xx, s_ts, s_ti)
        torch.cuda.current_stream().synchronize()
    for _ in range(30):
        once()
    vals = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            once()
        vals.append((time.perf_counter() - t0) * 1e3 / iters)
    return statistics.median(vals)


print(f"[被测] D={D} E={E} expert={N} top_k={K} | bs={BS_LIST} | 形状={KINDS} "
      f"| dtype={DTS} | gc={GCS} | 轮转 {ROUNDS} 轮")
print(f"[口径] 同进程轮转 + 逐轮配对 + p50; 🚨 同 kind/gc 下【只换 dtype】, "
      f"权重与路由由同一 seed 保证逐位可比")

for bs in BS_LIST:
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    for lb, hot in CFG:
        HITS[lb] = {k: hits_cum(hot, k) for k in KINDS}
        keys = [(kd, dt, gc) for kd in KINDS for dt in DTS for gc in GCS]
        M, GATE = {}, {}
        for (kd, dt, gc) in keys:
            torch.manual_seed(42)                       # 🚨 同 seed: 权重/路由逐位相同
            M[(kd, dt, gc)] = H.HeteroMoE(D, E, N, gc, K, hot_experts=hot,
                                          cpu_delay_ms=0.0, expert_kind=kd,
                                          gpu_dtype=dt, **FULL).cuda()
            M[(kd, dt, gc)](x, async_mode=True, mode=MODE)   # 建缓存 + 热身
        # ---- 闸门: 路由必须与 fp32 档逐位相同; 三模式互差必须逐位 0
        for (kd, dt, gc) in keys:
            m = M[(kd, dt, gc)]
            ts_, ti_ = m.route(x)
            outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
            worst = max((outs[a] - outs[b]).abs().max().item()
                        for i, a in enumerate(outs) for b in list(outs)[i + 1:])
            ref = m.fp32_reference(x, ts_, ti_)
            out = outs[MODE]
            max_abs, _, rel = H.err_metrics(out, ref)
            nbad = int(torch.isinf(out).sum() + torch.isnan(out).sum())
            ts0, ti0 = M[(kd, "fp32", gc)].route(x)
            idx_d = int((ti_ != ti0).sum())
            # 🚨 max_abs 是【绝对】数, 而 swiglu 那两档输出量级 ~5e4 —— 不印参考 rms,
            #    283 这种数字会被读成"误差爆炸", 其实它只有 0.59%。
            GATE[(kd, dt, gc)] = (worst, idx_d, rel, max_abs, nbad,
                                  int((ti_ >= m.gpu_expert_count).sum()),
                                  ref.pow(2).mean().sqrt().item())
            del outs, ref, out
        # ---- 轮转配对
        S = {k: [] for k in keys}
        for _ in range(ROUNDS):
            for k in keys:
                S[k].append(H.bench(M[k], x, True, MODE)[0])
        G = {k: gpu_half(M[k], x, *M[k].route(x)) for k in keys}
        for kd in KINDS:
            for gc in GCS:
                hits = HITS[lb][kd]
                print(f"\n{'=' * 96}")
                ncpu = GATE[(kd, DTS[0], gc)][5]
                print(f"配置 {lb} | bs={bs:>4} | {kd} | gc={gc} "
                      f"hot={hot} {'← 放对' if set(hot) < set(range(gc)) else '← 故意放错(打 CPU 腿)'}"
                      f" | 本帧 GPU 腿 slot={bs * K - ncpu} / CPU 腿 slot={ncpu}  "
                      f"loadavg={open('/proc/loadavg').read().split()[0]}")
                print('=' * 110)
                if bs * K - ncpu == 0:
                    print("  🚨 本帧 GPU 腿【0 个 slot】: 稠密写法照样把 gc 个专家全算了一遍 => "
                          "下面 GPU字节/GPU半边 全是【为 0 流量付的钱】(= 乙-2d 要打的对象); "
                          "同理 relRMSE 与 dtype 无关, 因为整条输出里根本没有 GPU 的成分。")
                print(f"{'dtype':>6}{'GPU字节/帧MB':>13}{'GPU半边':>9}{'GB/s':>7}{'端到端p50':>10}"
                      f"{'σ%':>5}{'min':>8}{'vs fp32':>10}{'三模式互差':>12}{'idx差':>7}"
                      f"{'relRMSE':>10}{'max_abs':>10}{'参考rms':>10}{'规则选gc':>9}")
                base = statistics.median(S[(kd, "fp32", gc)]) if "fp32" in DTS else None
                eb = N * D * (3 if kd == "swiglu" else 1)
                for dt in DTS:
                    k = (kd, dt, gc)
                    med = statistics.median(S[k])
                    sd = statistics.pstdev(S[k]) / med * 100
                    w, idx_d, rel, ma, nbad, _, rms = GATE[k]
                    gbs = gpu_bytes_frame(M[k]) / G[k] / 1e9 * 1e3
                    # 放置规则重跑: CPU 侧仍是 int8(eb 字节), GPU 侧换成这个 dtype 的价
                    gc_r, _ = H.plan_gpu_capacity(hits, eb, eb * ELS[dt],
                                                  gpu_expert_count=2)
                    flag = f"  🚨inf/nan={nbad}" if nbad else ""
                    print(f"{dt:>6}{gpu_bytes_frame(M[k]) / 2**20:>13.1f}{G[k]:>9.3f}"
                          f"{gbs:>7.0f}{med:>10.3f}{sd:>5.0f}{min(S[k]):>8.3f}"
                          f"{(med / base - 1) * 100:>+9.1f}%"
                          f"{w:>12.9f}{idx_d:>7}{rel:>9.4f}%{ma:>10.5f}{rms:>10.3g}{gc_r:>7}{flag}")
                    if dt == "fp16" and nbad:
                        print(f"       ↑ fp16 在【随机 std=1 权重】的 swiglu 中间量级"
                              f"(~4e4~7e4)下溢出(上限 65504)。真 checkpoint 按 "
                              f"1/sqrt(fan_in) 初始化不会顶到这个量级 —— 这行读作"
                              f"【标度没管好会怎样】, 不读作 fp16 不能用。")
                if "fp32" in DTS and len(DTS) > 1:
                    for dt in DTS:
                        if dt == "fp32":
                            continue
                        kg, kx = (kd, "fp32", gc), (kd, dt, gc)
                        d_e2e = (statistics.median(S[kx]) / statistics.median(S[kg]) - 1) * 100
                        d_gpu = (G[kx] / G[kg] - 1) * 100
                        win = sum(1 for a, b in zip(S[kx], S[kg]) if a < b)
                        nbad = GATE[kx][4]
                        print(f"  配对 {dt} vs fp32: GPU半边 {d_gpu:+.1f}%"
                              f"  字节 {(ELS[dt] / 4 - 1) * 100:+.0f}%"
                              f"  有效带宽 {(G[kx] / G[kg] * ELS[dt] / 4 - 1) * 100:+.0f}%"
                              f"(>0 = 半精 kernel 跑得比按字节折算的还快)"
                              f"  端到端 {d_e2e:+.1f}%  更快轮数 {win}/{ROUNDS}"
                              + ("   🚨 这一档 nan/inf: 数值已经废了, 它的计时没有参考价值"
                                 if nbad else
                                 "   🚨 端到端缩得比 GPU 半边少是应该的(固定串行主机开销不随 dtype 变)"))
        M.clear(); G.clear(); S.clear()
        torch.cuda.empty_cache()

print("""
判读提示: ⓪ 关键看 GB/s 那一列: 字节减半而时间【没】减半 => 半精 kernel 的等效带宽掉了。
           这块卡 skinny GEMV 在 fp32 下 ~230-290GB/s(贴着纯读上限), bf16 若只拿到
           -20% 而不是 -50% 的时间, 说明 cuBLAS 在 M=1..4 上没给它挑到满带宽的 kernel。
           这决定"降精"这一刀到底值多少 —— 别照抄"fp16 一定快 2 倍"。
        ① GPU 半边应接近 -50%(字节 bound); 端到端缩得更少 = 剩下的那截是固定串行主机开销,
           它不属于 GPU 半边, 降精治不了它(那是 乙-2d 和主机侧的活)。
        ② 闸门2 idx差必须全 0 —— 不为 0 就是 dtype 漏进了前段, 整张表作废(乙-1 的老坑)。
        ③ relRMSE 是【整条 L2】的, 里面混着 CPU 腿的 int8。要单独给 GPU 腿定价,
           看 配置A + bs=1(那一格 CPU 腿 0 slot, 整条输出就是 GPU 半边)。
        ④ 规则选gc 那一列: 降精让 GPU 半边变便宜 => 规则应当敢给更大 gc。若不动,
           说明 plan_gpu_capacity 的 gpu_weight_bytes 没真的接上 dtype(见它注释第 4 条)。""")

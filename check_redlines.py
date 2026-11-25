#!/usr/bin/env python
"""红线校验器: 改完任何异步/图/绑核代码后【先跑这个】, 它比主表重要。

为什么需要它: 这一路加的每一刀(CUDA Graph / pinned 缓冲 / 松 GIL / 零 CUDA API)
都是"复用同一块内存"的把戏, 而这类把戏的失败模式是【静默出错数】——
主表照样变快, 只是算的是上一帧的数据。所以这里专门测三种"踩内存"症状:
  1) 三模式(L0/L1/L2)与 sync 必须逐位相同 —— 互差必须 == 0.0, 不是"接近 0"
  2) 换一批输入后图必须读新数据: 前段图(路由+量化)/后段图(热专家GEMM)/
     全链路 vs 纯 eager 三条口径分别对拍, 都必须 == 0
  3) 反复 replay 200 次后仍与首帧互差 0 —— 静态缓冲被踩的典型症状
精度基线是钉死的常量 EXPECT(改了就不该动), 但注意它属于【哪套打包】:
  biased uint8(历史, 有 maddubs int16 饱和 bug): A=0.2855%/0.98708  C=1.4660%/3.00317
  signed int8(T10 起默认, 与理想 int32 精确累加逐位一致): 见下面 EXPECT
旧基线里嵌着 0.48pp 的算术误差(饱和截断), 不是量化误差 —— 见 bench_int8_exactness.py。
切换打包: env MOE_W_PACK=biased|signed, 或 HeteroMoE(w_pack=...)。
探针清单(每条都对应一次真实翻车或一次真实赌注):
  A. 三模式/scope x 换输入 x replay200         —— 静态缓冲踩内存
  B. d2h_wait=worker x 换 64 批输入 x2000       —— worker 读到上一帧 pinned 内存
  C. gc != top_k 的 _hot_gate                  —— 历史越界写(静默错数)
  D. T12 单流/尾图                              —— 去 Event 后流序 / 尾图覆写(定价)
  E. T11 融合前段 x bs∈{1,7,32,128}             —— mask 尾巴漏列(triton_moe_kernel 真案例)
  F. CPU kernel (n_threads,tok_block) 不变性    —— 整数运算, 任何非零差都是越界/漏算
  G. T14 OMP 行调度 guided / token块换序      —— 只改"谁算哪一行", 必须逐位不动
  H. 融合前段 x D 几何边界(不整除 BLOCK_D)     —— 真权重 D=576 的越界读(藏了十几轮才炸)
  I. 乙-2d traffic 档(GPU 半边按真流量读权重)  —— 与 cat 互差在一个量级 + 不整除几何 + 覆写
  J. P1-② 动态放置(计数器->滑窗->决策->就地迁移) —— 迁移后与按新序重建逐位同 + 图不重录 + 可逆
用法: python check_redlines.py     (全过打印 ALL RED LINES PASS, 否则 exit 1)
已用它判过: "torch.add(out=) 预分配落点" —— 精度看着对, 其实每轮返回同一块缓冲,
调用方存下的结果被下一轮覆写, 第 3 条红线直接把这种假正确打下来。
"""
import sys, time, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

# (max_abs, relRMSE%) —— 按打包方式分开钉。signed 是 T10 起的默认:
#   max_abs 从 3.00317 掉到 1.09505(配置C, 2.7x), 因为那 0.48% 饱和误差集中砸在少数
#   最坏元素上, 所以它对 max_abs 的伤害远大于对 relRMSE 的伤害。
EXPECT_ALL = {"signed": {"A": (0.54276, 0.2690), "C": (1.09505, 1.3843)},   # = 理想 int32
              "biased": {"A": (0.98708, 0.2855), "C": (3.00317, 1.4660)}}   # 历史(带饱和)
EXPECT = EXPECT_ALL[H.W_PACK]
print(f"精度基线打包 = {H.W_PACK}: {EXPECT}")
fails = []
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    for scope in ("off", "hot", "full"):
        torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, graph_scope=scope).cuda()
        if scope != "off":
            m._ensure_graphs(x)
        ts, ti = m.route(x); ref = m.fp32_reference(x, ts, ti)
        outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
        worst = max((outs[a]-outs[b]).abs().max().item()
                    for i,a in enumerate(outs) for b in list(outs)[i+1:])
        ma, snr, rel = H.err_metrics(outs[H.MODES[0]], ref)
        ema, esnr, erel = H.err_metrics(m(x, async_mode=False), ref)
        t_sync = m(x, async_mode=False)
        ws = (t_sync - outs[H.MODES[0]]).abs().max().item()
        # --- 🚨 T7 专属: 换一批输入, 图必须读新数据(不能吃到上一轮的静态缓冲) ---
        torch.manual_seed(7); x2 = torch.randn(32, 1024).cuda()
        o2 = m(x2, async_mode=True, mode="L2_零CUDA+pinned")
        ts2, ti2, xq2, xs2 = m._prepare(x2)          # 走同一条(可能是图的)路
        ts2e, ti2e = m.route(x2)
        # ✅ T11: 对拍口径改成"同一份 scores 的 eager vs 图"。
        #    原写法拿【eager 路由的 scores】喂 eager GEMM, 去和【图里融合前段算出的
        #    scores】喂的图 GEMM 比 —— 两份 scores 本来就差 ~3e-7 相对值
        #    (tl.dot ieee 与 cublas sgemm 归约顺序不同, 见 pre_fused.py 文件头),
        #    于是"后段图有没有读到旧数据"这个探针会被前段的 ULP 差污染成假红。
        #    这条探针要测的是【staleness】, 与"两家 GEMM 是否同归约序"无关 =>
        #    两边必须吃同一份 scores; 两家的 ULP 差改由下面新增的路由一致性探针单独管。
        eager_hot = m._hot_compute(x2, ts2, ti2)
        # 全链路口径: 图算出来的整条 L2 输出 vs 纯 eager 的 热专家+冷专家
        eq2, es2 = m._quantize_act(x2)
        eager_all = eager_hot + H.avx_gemv_ext.avx2_moe_cpu_experts_int8(
            eq2, es2, ti2, ts2, m.cpu_w, m.cpu_experts_scale,   # ✅ T11: 同上, 同一份 scores   # ✅ T10: 必须和模型用同一套打包
            m.gpu_expert_count, False, 6, 8)
        d_full = (o2 - eager_all).abs().max().item()
        d_hot = (m._hot_half(x2, ts2e, ti2e) - eager_hot).abs().max().item()
        ts_g, ti_g, xq_g, xs_g = m._prepare(x2)
        eq, es = m._quantize_act(x2)
        # --- ✅ T11 新增: 融合前段 vs eager 前段 (原 d_pre 是自我对拍, 恒等于 0, 没用) ---
        #    逐位要求: topk_indices 与 x_q 必须【完全相同】(选的人 + 量化整数);
        #    scores 只要求 <1e-6 相对值 —— 归约顺序不同, 不可能逐位。
        #    🚨 idx 一旦不一致 = 近分 token 换了专家 => FAIL, 不许放宽。
        d_idx = int((ti2 != ti2e).sum().item())
        d_xq = int((xq_g != eq).sum().item())        # 图内量化结果 vs eager, int8 逐位
        d_pre = (ts2 - ts2e).abs().max().item() / max(ts2.abs().max().item(), 1e-9)
        d_pre_s = (ts_g - ts2).abs().max().item()
        ok = (worst == 0.0 and abs(rel-EXPECT[label][1]) < 1e-3
              and abs(ma-EXPECT[label][0]) < 1e-4 and abs(erel-EXPECT[label][1]) < 1e-3
              and ws == 0.0 and d_hot == 0.0 and d_full == 0.0
              and d_idx == 0 and d_xq == 0 and d_pre < 1e-6)
        if not ok: fails.append((label, scope, worst, rel, ma, erel, ws, d_hot, d_full,
                                 f"idx={d_idx} xq={d_xq} scores={d_pre:.2e}"))
        print(f"{label} scope={scope:<5} 三模式互差={worst:.6f} relRMSE={rel:.4f}% "
              f"(sync版={erel:.4f}%) max_abs={ma:.5f} sync-L2互差={ws:.6f} "
              f"| 换输入: 前段图{d_pre_s:.3e} 后段图{d_hot:.3e} 全链路vs eager{d_full:.3e}"
              f"\n            融合前段 vs eager: idx差={d_idx} x_q差={d_xq} "
              f"scores相对差={d_pre:.2e}")
        # 反复 replay 200 次后仍然正确? (静态缓冲被踩的典型症状)
        for _ in range(200): m(x, async_mode=True, mode="L2_零CUDA+pinned")
        torch.cuda.synchronize()
        again = (m(x, async_mode=True, mode="L2_零CUDA+pinned") - outs["L2_零CUDA+pinned"]).abs().max().item()
        if again != 0.0: fails.append((label, scope, "drift", again))
        print(f"    replay200 后与首帧互差 = {again:.6f}")
# ============================================================
# ✅ T8-② 专属红线: 把"等 D2H 落地"从主线程挪进 CPU worker 之后,
#    worker 只靠一个 cuda event 保证看到的是【本帧】数据。
#    赌注是错的就会静默读到上一帧的 pinned 内存 —— 主表照样变快, 数是旧的。
#    所以这里用 64 个互不相同的 batch 轮流打, 每次 x 都换 => 上一帧的结果
#    和本帧必然差很多, 一旦读错立刻暴露。
# ============================================================
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                    n_threads=6, tok_block=8, graph_scope="full",
                    d2h_wait="worker").cuda()
    xs = [torch.randn(32, 1024).cuda() for _ in range(64)]
    for xx in xs[:4]:
        m(xx, async_mode=True, mode="L2_零CUDA+pinned")   # 预热 + 录图
    worst2 = 0.0
    for i in range(2000):
        xx = xs[i % 64]
        got = m(xx, async_mode=True, mode="L2_零CUDA+pinned")
        if i % 50 == 0:
            ts_, ti_ = m.route(xx)
            want = m(xx, async_mode=False)
            worst2 = max(worst2, (got - want).abs().max().item())
    print(f"{label} d2h_wait=worker  换 64 批输入 x2000 次, 与 sync 最大互差 = {worst2:.6f}")
    if worst2 != 0.0:
        fails.append((label, "worker-wait 读到旧数据", worst2))

# ============================================================
# ✅ T9 专属红线: gpu_expert_count != top_k 时热专家门控必须对。
#    历史真 bug: _hot_gate 的 scatter 累加目标写成 zeros_like(contrib),
#    形状是 (bs, top_k) 而不是 (bs, gpu_expert_count)。历史上 gc 恒等于 top_k=2
#    所以恰好同形没暴露; gc>2 时 scatter_add_ 拿 clamp(max=gc-1) 的索引【越界写
#    显存】(release 不做边界检查), 且 xin 只有 top_k*hidden 列 -> matmul 形状报错。
#    下面的用例专门让"命中的热专家落在 gate 的第 top_k 列之后"(ncols>2 就是见证),
#    老写法要么直接崩要么静默错。gc != top_k 时 relRMSE 基线会变, 所以只断言
#    互差==0 + 三种 GPU 实现(sparse/cat/bmm)互差在一个量级内。
# ============================================================
print("\n--- gc != top_k 回归 (T9 修的越界写) ---")
for gc, hot in ((3, [1, 2]), (4, [2, 3]), (6, [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, gc, 2, hot_experts=hot, cpu_delay_ms=0.0,
                    n_threads=6, tok_block=8, gpu_impl="cat",
                    graph_scope="full").cuda()
    m._ensure_graphs(x)
    ts, ti = m.route(x); ref = m.fp32_reference(x, ts, ti)
    gate = m._hot_gate(ts, ti)
    ncols = int((gate.abs().sum(0) > 0).sum())        # 用到几列 => 老代码的越界证据
    outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
    worst = max((outs[a]-outs[b]).abs().max().item()
                for i,a in enumerate(outs) for b in list(outs)[i+1:])
    ws = (m(x, async_mode=False) - outs[H.MODES[0]]).abs().max().item()
    rel = H.err_metrics(outs[H.MODES[0]], ref)[2]
    imp = {}
    for impl in ("sparse", "cat", "bmm"):
        m.gpu_impl = impl
        imp[impl] = m._hot_compute(x, ts, ti)
    scale = max(v.abs().max().item() for v in imp.values())
    dimpl = max((imp[a]-imp[b]).abs().max().item()
                for i,a in enumerate(imp) for b in list(imp)[i+1:])
    m.gpu_impl = "cat"
    ok = worst == 0.0 and ws == 0.0 and rel < 3.0 and dimpl < 1e-2 * scale
    if not ok:
        fails.append((f"gc={gc}", worst, ws, rel, dimpl, scale))
    print(f"gc={gc} hot={hot} 用到{ncols}列(>2 即老代码越界) 三模式互差={worst:.6f} "
          f"sync互差={ws:.6f} relRMSE={rel:.4f}% 三实现互差={dimpl:.3e}/scale{scale:.0f}"
          f"{' 不合格' if not ok else ''}")

# ============================================================
# ✅ T12 专属红线: 单流(l2_stream="single") + 可选尾图(tail_graph=True)。
#    这两刀都在"复用同一块显存"的危险区里:
#      single: 热专家 replay 回默认流 => 少了 Event 跨流同步, 赌的是"默认流顺序
#              天然隔开 pre/d2h/hot"。赌错 = 热专家读到上一帧的 _sx。
#      tailG : 返回的是【图内静态缓冲】, 下一帧 replay 必然覆写它。
#    🚨 tailG 的覆写【不是 bug 是定价】—— 这里把它打印出来逼调用方看清,
#       真正的断言是"单帧内必须与 sync 逐位相同" + "要留结果就得 .clone()"。
# ============================================================
print("\n--- T12 流/尾图 (single 与 tail_graph) ---")
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    base = None
    for st, tg in (("sep", False), ("single", False), ("single", True)):
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, graph_scope="full",
                        pre_impl="fused", l2_stream=st, tail_graph=tg).cuda()
        o = m(x, async_mode=True, mode="L2_零CUDA+pinned")   # 首帧: 录图
        o = m(x, async_mode=True, mode="L2_零CUDA+pinned")   # 次帧: replay
        ts, ti = m.route(x); ref = m.fp32_reference(x, ts, ti)
        ws = (m(x, async_mode=False) - o).abs().max().item()
        d = 0.0 if base is None else (o - base).abs().max().item()
        rel = H.err_metrics(o, ref)[2]
        ovw = 0.0
        if tg:
            # 🚨 必须"覆写前快照 vs 覆写后再读同一块缓冲"。写成 (keep - o) 是自我
            #    对拍 —— 两个名字指向同一个张量, 恒等于 0, 抓不到任何东西。
            keep, snap = o, o.clone()
            torch.manual_seed(99); xn = torch.randn(32, 1024).cuda()
            m(xn, async_mode=True, mode="L2_零CUDA+pinned")
            ovw = (keep - snap).abs().max().item()
        ok = ws == 0.0 and d == 0.0 and abs(rel - EXPECT[label][1]) < 1e-3
        if not ok: fails.append((label, f"stream={st}/tail={tg}", ws, d, rel))
        print(f"{label} 流={st:<6} tail_graph={str(tg):<5} 与sync互差={ws:.6f} "
              f"vs基准互差={d:.6f} relRMSE={rel:.4f}%"
              + (f"  ← 覆写代价: 下一帧把返回张量改了 {ovw:.4f} (必须 .clone())" if tg else ""))
        if base is None: base = o.clone()             # 🚨 clone: 否则基准自己也被覆写
    del m, base

# ============================================================
# ✅ T11 补漏: 融合前段在【bs 不是 8 的整数倍 / bs=1】时也必须逐位等价。
#    主表只跑 bs=32 —— 边界(mask 尾巴、单行)最容易让 tl 的 mask 写错而静默丢列,
#    而"丢列"这种 bug 在 bs=32 上恰好被整除掩盖(见 triton_moe_kernel.py 那个真案例)。
# ============================================================
print("\n--- T11 融合前段 x batch 边界 (逐位对拍) ---")
for bs in (1, 7, 32, 128):
    torch.manual_seed(42); x = torch.randn(bs, 1024).cuda()
    outs = {}
    for pre in ("torch", "fused"):
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[0, 1], cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, graph_scope="full",
                        pre_impl=pre).cuda()
        m._ensure_graphs(x)
        sts, sti, xq, xs = m._prepare(x)
        te, ti = m.route(x); eq, es = m._quantize_act(x)
        d_idx = int((sti != ti).sum().item())
        d_xq = int((xq != eq).sum().item())
        d_sc = (sts - te).abs().max().item() / max(te.abs().max().item(), 1e-9)
        o = m(x, async_mode=True, mode="L2_零CUDA+pinned")
        outs[pre] = (o.clone(), d_idx, d_xq, d_sc)
        del m
    o_t, di, dx, dsc = outs["torch"][0], *outs["fused"][1:]
    d_full = (outs["torch"][0] - outs["fused"][0]).abs().max().item()
    scale = o_t.abs().max().item()
    # 🚨 逐位要求只能加在【离散量】上: idx(选了谁) 与 x_q(量化整数) 必须完全相同。
    #    整条输出【不可能】逐位 —— 两家 GEMM(cublas sgemm vs tl.dot ieee)归约顺序
    #    不同 => scores 差 ~4e-7 相对值, 顺流而下就是 ~5e-5 绝对差(输出尺度 ~107)。
    #    阈值取输出尺度的 2e-6(实测 5e-7, 留 4x), 它比任何真 bug 窄 5 个数量级:
    #    staleness/漏列那种症状是 O(1), 不是 O(1e-5)。
    rel_full = d_full / max(scale, 1e-9)
    ok = rel_full < 2e-6 and di == 0 and dx == 0 and dsc < 1e-6
    if not ok: fails.append((f"pre bs={bs}", rel_full, di, dx, dsc))
    print(f"bs={bs:>4} idx差={di} x_q差={dx} scores相对差={dsc:.2e} "
          f"整条L2输出 torch-vs-fused: 绝对{d_full:.3e} / 相对{rel_full:.2e} "
          f"(阈值2e-6, 逐位只要求 idx+x_q)")

# ============================================================
# ✅ H. 融合前段 x D 的【几何边界】—— D 不是 BLOCK_D(=256) 整数倍时尾巴块必须掩码。
#    真事: 真权重 SmolLM-135M 的 D=576 = 2x256+64。老代码的 D 循环只掩了行没掩列,
#    最后一块多算 64 行 => x 越界读进邻居张量, gate_wt(只有 D*EP 个元素)越界读到
#    ~1.4 倍处 => 脏比特进 tl.dot => acc=NaN => softmax 全 NaN => tl.argmax 落进
#    为 2 的幂补齐的 15 号假专家道(EP=16 而 E=8) => 一路传到 C++ 抛
#    "expert id 15 out of range"。玩具 D=1024 = 4x256 正好整除, 所以这刀在
#    T11~T14 十几轮和全部老红线里【完全隐形】—— 老红线只测了整除的几何。
#    🚨 手法要点: 光比"和 torch 对不对"抓不住越界读 —— 吃到零就看不出差别(实测过:
#    同一地址、输入干净、输出是垃圾、再 replay 一次就对)。所以把 x 和 gate_wt 都放
#    进【尾巴上紧挨着 NaN 的缓冲】: 只要越界, 必吃到 NaN, 当场显形。
#    D=1024 那格是【对照组】(整除 => 无尾巴块 => 毒读必然为 0); 顺带做"牙检":
#    不整除的那几格若对照也读不到毒, 说明这条红线自己失效了(也算失败)。
# ============================================================
print("\n--- H 融合前段 x D 几何边界 (D 不整除 BLOCK_D=256 的尾巴块, 毒邻居) ---")
def _poisoned(shape, src, pad=4096):
    """把张量放进一块【尾部紧挨 NaN】的缓冲, 返回 1-D 视图 —— 越界读必吃到 NaN。"""
    n = int(torch.tensor(shape).prod().item())
    buf = torch.full((n + pad,), float('nan'), dtype=src.dtype, device=src.device)
    buf[:n].copy_(src.reshape(-1))
    return buf, buf[:n]

from pre_fused import route_quant_fused, alloc_pre_out, route_fused
for D, divisible in ((100, False), (576, False), (640, False), (1024, True)):
    bs, E, EP = 32, 8, 16
    torch.manual_seed(42)
    xbuf, x1 = _poisoned((bs, D), torch.randn(bs, D).cuda()); x = x1.view(bs, D)
    torch.manual_seed(42)
    m = H.HeteroMoE(D, E, 1536, 2, 2, hot_experts=[0, 1], cpu_delay_ms=0.0,
                    n_threads=6, tok_block=8, pre_impl="fused").cuda()
    gbuf, gw = _poisoned((D, EP), m._gate_wt_dev(x.device).clone().contiguous())
    te, ti = m.route(x); eq, es = m._quantize_act(x)          # torch 裁判(与实现无关)
    ts, tix, xq, xs = route_quant_fused(m, x, alloc_pre_out(m, bs, x.device), gate_w=gw)
    di = int((tix != ti).sum().item()); dx = int((xq != eq).sum().item())
    dsc = (ts - te).abs().max().item() / max(te.abs().max().item(), 1e-9)
    oob = int((tix >= E).sum().item()); nn = int(torch.isnan(ts).sum().item())
    # 对照 = 把 TAIL 强关(= 修复前的漏掩码), 用来证明这条红线自己还有牙
    ts2, ti2, xq2, xs2 = alloc_pre_out(m, bs, x.device)
    route_fused[((bs + 15) // 16, )](x, gw, m.gate.bias, ts2, ti2, bs, D, E,
                                     K=m.top_k, BLOCK_B=16, BLOCK_D=256, EP=EP,
                                     TAIL=False, num_warps=4)
    teeth = int((ti2 != ti).sum().item()) + int(torch.isnan(ts2).sum().item())
    ok = di == 0 and dx == 0 and dsc < 1e-6 and oob == 0 and nn == 0
    if not ok:
        fails.append((f"D={D} 尾巴块", di, dx, dsc, oob, nn))
    if teeth == 0 and not divisible:
        fails.append((f"D={D} 红线失效(不整除却读不到毒)",))
    if teeth > 0 and divisible:
        fails.append((f"D={D} 对照组不该有毒读",))
    print(f"D={D:>4} 整除256={divisible}  修复后 idx差={di} x_q差={dx} scores相对差={dsc:.2e} "
          f"越界id={oob} NaN={nn}  {'OK' if ok else '🚨RED'}"
          f" | 对照(强关TAIL=老代码)坏值={teeth} {'← 红线有牙' if (teeth > 0) == (not divisible) else '🚨红线失效'}")
    del m, xbuf, gbuf, x, gw
    torch.cuda.empty_cache()

# 端到端补一刀: D=576 全图(真权重几何 + graph_scope=full) —— 崩点就在这条路上
torch.manual_seed(42); x = torch.randn(32, 576).cuda()
torch.manual_seed(42)
m = H.HeteroMoE(576, 8, 1536, 2, 2, hot_experts=[0, 1], cpu_delay_ms=0.0,
                n_threads=6, tok_block=8, graph_scope="full", pre_impl="fused").cuda()
m._ensure_graphs(x)
ts, ti, xq, xs = m._prepare(x)
te, ti_e = m.route(x)
o = m(x, async_mode=True, mode="L2_零CUDA+pinned")
di = int((ti != ti_e).sum().item()); oob = int((ti >= 8).sum().item())
fin = bool(torch.isfinite(o).all().item()) and int(torch.isnan(ts).sum().item()) == 0
rel = (o - m.fp32_reference(x, te, ti_e)).abs().max().item() / max(
      m.fp32_reference(x, te, ti_e).abs().max().item(), 1e-9)
ok = di == 0 and oob == 0 and fin
if not ok: fails.append(("D=576 全图端到端", di, oob, fin))
print(f"D= 576 全图(full+fused) 端到端: idx差={di} 越界id={oob} 输出有限={fin} "
      f"vs fp32 参考相对差={rel:.2e}  {'OK' if ok else '🚨RED'}  ← 真权重崩的那条路")
del m; torch.cuda.empty_cache()

# ============================================================
# ✅ CPU kernel 的 (n_threads, tok_block) 不变性 —— 且专挑【bs 除不尽 tok_block】。
#    每个 token 的点积在 kernel 里是固定顺序累加(256 一块 int32), 与线程数/分块无关,
#    所以任何不一致都是【尾巴块越界或漏算】的证据, 不是"浮点噪声"(它是整数运算)。
#    这一条同时守住 T4/T5 那两个调参: 调线程数、调块大小都不许改变任何一个数。
# ============================================================
print("\n--- CPU kernel (n_threads,tok_block) 不变性 + 尾巴块 ---")
torch.manual_seed(42)
mref = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[0, 1], cpu_delay_ms=0.0,
                   n_threads=6, tok_block=8, graph_scope="off").cuda()
for bs in (1, 7, 31, 33):
    torch.manual_seed(bs); x = torch.randn(bs, 1024).cuda()
    ts, ti = mref.route(x)
    q, sc = mref._quantize_act(x)
    ref = H.avx_gemv_ext.avx2_moe_cpu_experts_int8(
        q, sc, ti, ts, mref.cpu_w, mref.cpu_experts_scale,
        mref.gpu_expert_count, False, 1, 1)
    worst_k = 0.0
    for nt, tb in ((1, 8), (6, 8), (4, 4), (8, 8), (6, 1)):
        got = H.avx_gemv_ext.avx2_moe_cpu_experts_int8(
            q, sc, ti, ts, mref.cpu_w, mref.cpu_experts_scale,
            mref.gpu_expert_count, False, nt, tb)
        worst_k = max(worst_k, (got - ref).abs().max().item())
    # 端到端也走一遍(tb=8 > bs 是最容易出事的组合)
    worst_e = 0.0
    for nt, tb in ((6, 8), (1, 1), (4, 4)):
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[0, 1], cpu_delay_ms=0.0,
                        n_threads=nt, tok_block=tb, graph_scope="full",
                        pre_impl="fused").cuda()
        o = m(x, async_mode=True, mode="L2_零CUDA+pinned")
        if nt == 6 and tb == 8:
            o_ref = o.clone()
        else:
            worst_e = max(worst_e, (o - o_ref).abs().max().item())
        del m
    if worst_k != 0.0 or worst_e != 0.0:
        fails.append((f"kernel bs={bs}", worst_k, worst_e))
    print(f"bs={bs:>4} kernel 互差={worst_k:.6f} 端到端互差={worst_e:.6f}")
del mref

# ============================================================
# G. ✅ T14: OMP 行调度(sched) 与 token块换序(loop) 的不变性
#    这两个开关只决定【哪条线程算哪一行】和【哪些 token 共用一次权重载入】,
#    每个 (行 i, token t) 的点积、以及同一 (b,i) 上多个 slot 的累加顺序都没变
#    => 必须逐位 0。T14 判决表敢把"static,32 -> guided"说成纯性能改动, 前提就是这条。
#    🚨 一旦哪天上了 work-stealing 之外的分摊(比如按 token 再切并行), 这条会先响。
# ============================================================
print("\n--- T14 OMP 行调度/换序 不变性 (kernel 级 + 端到端) ---")
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    torch.manual_seed(42)
    m0 = H.HeteroMoE(1024, 8, 2048, 3, 2, hot_experts=hot, cpu_delay_ms=0.0,
                     n_threads=6, tok_block=8, graph_scope="off").cuda()
    ts, ti = m0.route(x)
    q, sc = m0._quantize_act(x)
    a = (q, sc, ti, ts, m0.cpu_w, m0.cpu_experts_scale)
    ref = H.avx_gemv_ext.avx2_moe_cpu_experts_int8(*a, m0.gpu_expert_count, False, 6, 8, -1, 0, 0)
    worst_k = 0.0
    for lp in (0, 1):
        for scp in (0, 1):
            got = H.avx_gemv_ext.avx2_moe_cpu_experts_int8(
                *a, m0.gpu_expert_count, False, 6, 8, -1, lp, scp)
            worst_k = max(worst_k, (got - ref).abs().max().item())
    worst_e, o_ref = 0.0, None
    for lp, scp in ((0, 0), (0, 1), (1, 1)):
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 3, 2, hot_experts=hot, cpu_delay_ms=0.0,
                        n_threads=6, tok_block=8, graph_scope="full",
                        pre_impl="fused", cpu_loop=lp, cpu_sched=scp).cuda()
        o = m(x, async_mode=True, mode="L2_零CUDA+pinned")
        torch.cuda.synchronize()
        if (lp, scp) == (0, 0):
            o_ref = o.clone()
        else:
            worst_e = max(worst_e, (o - o_ref).abs().max().item())
        del m
    del m0
    if worst_k != 0.0 or worst_e != 0.0:
        fails.append((f"T14 sched/loop {label}", worst_k, worst_e))
    print(f"{label} loop×sched kernel 互差={worst_k:.9f} 端到端互差={worst_e:.9f}")

# ============================================================
# ✅ I. 乙-2d traffic 档: GPU 半边按【本帧真流量】读权重(triton 间接寻址)。
#    它和 cat 档读的是同一组权重(_wcat_gu/_wstack_d), 区别只在"读几次、按什么顺序加":
#    => 路由 idx 必须【逐位相同】(路由代码一行没动);
#    => y 互差必须是【归约序噪声底】而不是"精度差别"(fp32 1e-5 / bf16 2e-2 相对量级),
#       超了就是少读的字节其实是别人该参与的那一笔(比如冷 slot 的门控/零没写对)。
#    🚨 三条专属红线, 缺一条就抓不到这一刀的经典翻车方式:
#      (a) 不整除 BLOCK 的几何(D=570/N=1530): traffic 的两个 kernel 靠 mask 收尾,
#          这正是 T11 那条"D=576 越界读 => NaN => 路由跳假专家"的老路, 必须自己测;
#      (b) 连续两帧 + 换回第一帧: _tH/_tPART 是【每引擎复用的 scratch】, 冷 slot 必须写零,
#          留旧值 = 上一帧的数混进本帧(静默错数, 主表照样变快);
#      (c) gc=0 与"本帧 0 个 slot 落 GPU": 这两档 traffic 一个字节都不读, 必须与
#          纯 CPU 腿逐位相同 —— 它是【零流量正确性】, 也顺带证明稠密税在哪。
# ============================================================
print("\n--- I 乙-2d traffic(cat vs traffic, 不整除几何 + 两帧覆写 + 零流量) ---")
def _t_build(D, N, gc, impl, dt="fp32", scope="full"):
    torch.manual_seed(42)
    m = H.HeteroMoE(D, 8, N, gpu_expert_count=gc, top_k=2, expert_kind="swiglu",
                    cpu_delay_ms=0.0, n_threads=6, tok_block=8, gpu_impl=impl,
                    gpu_dtype=dt, graph_scope=scope, pre_impl="fused",
                    l2_stream="single", cpu_sched=1, d2h_wait="main").cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode="L2_零CUDA+pinned")
    return m

for D, N, divisible in ((576, 1536, True), (570, 1530, False)):
    for dt in ("fp32", "bf16"):
        for gc in (1, 3, 8):
            torch.manual_seed(7); x = torch.randn(3, D).cuda()
            mc, mt = _t_build(D, N, gc, "cat", dt), _t_build(D, N, gc, "traffic", dt)
            for _ in range(3):                       # 暖机(triton JIT + 录图)
                mc(x, async_mode=True, mode="L2_零CUDA+pinned")
                mt(x, async_mode=True, mode="L2_零CUDA+pinned")
            yc = mc(x, async_mode=True, mode="L2_零CUDA+pinned")
            yt = mt(x, async_mode=True, mode="L2_零CUDA+pinned")
            _, ic = mc.route(x); _, it = mt.route(x)
            same_idx = bool((ic == it).all().item())
            nn = int(torch.isnan(yt).sum() + torch.isinf(yt).sum())
            rel = (yt.float() - yc.float()).norm().item() / max(
                  yc.float().norm().item(), 1e-30)
            ref = mc.fp32_reference(x, ic, it)
            rt = (yt.float() - ref).norm().item() / max(ref.norm().item(), 1e-30)
            thr = 2e-6 if dt == "fp32" else 2e-2
            # (b) 两帧覆写: 换 x2 再换回 x1, 第三帧必须与第一帧【逐位】相同
            torch.manual_seed(99); x2 = torch.randn(3, D).cuda()
            y2 = mt(x2, async_mode=True, mode="L2_零CUDA+pinned")
            y1b = mt(x, async_mode=True, mode="L2_零CUDA+pinned")
            stale = (y1b - yt).abs().max().item()
            ov2 = (y2 - mt(x2, async_mode=True, mode="L2_零CUDA+pinned")).abs().max().item()
            ok = same_idx and nn == 0 and rel < thr and stale == 0.0 and ov2 == 0.0
            if not ok:
                fails.append((f"traffic D={D} gc={gc} {dt}", same_idx, nn, rel, stale, ov2))
            print(f"D={D} 整除64={divisible} {dt} gc={gc}: idx同={same_idx} nan/inf={nn} "
                  f"y相对互差={rel:.2e}(阈{thr:g}) traffic对参考={rt:.2e} "
                  f"两帧后回读第一帧互差={stale:.1e} 帧2重放互差={ov2:.1e} "
                  f"{'OK' if ok else '🚨RED'}")
            # (c) 零流量: gc=0 => 两档都等于纯 CPU 腿; 且 traffic 不碰 GPU 权重
            m0c, m0t = _t_build(D, N, 0, "cat", dt), _t_build(D, N, 0, "traffic", dt)
            for _ in range(3):
                m0c(x, async_mode=True, mode="L2_零CUDA+pinned")
                m0t(x, async_mode=True, mode="L2_零CUDA+pinned")
            z = (m0t(x, async_mode=True, mode="L2_零CUDA+pinned")
                 - m0c(x, async_mode=True, mode="L2_零CUDA+pinned")).abs().max().item()
            if z != 0.0:
                fails.append((f"traffic gc=0 D={D} {dt}", z))
            print(f"    gc=0(零流量) {dt}: cat 与 traffic 互差={z:.1e} (必须 0) "
                  f"{'OK' if z == 0.0 else '🚨RED'}")
            del mc, mt, m0c, m0t
            torch.cuda.empty_cache()

# gemm 玩具档必须【明确拒绝】traffic —— 否则哪天有人在玩具形状上把它当通用加速
try:
    torch.manual_seed(42)
    H.HeteroMoE(1024, 8, 2048, 3, 2, expert_kind="gemm", cpu_delay_ms=0.0,
                n_threads=6, tok_block=8, gpu_impl="traffic").cuda()
    fails.append(("traffic 未拒绝 gemm 玩具档",))
    print("traffic x gemm: 🚨没抛错(应当抛)")
except (AssertionError, NotImplementedError) as e:
    print(f"traffic x gemm 玩具档: 按设计抛 {type(e).__name__} OK")


# ============================================================
# J. P1-② 动态放置(计数器 -> 滑窗 -> 决策 -> 就地迁移)。
#    这一刀的失败模式【全是静默错数】, 而且比 I 段更阴: 迁移【只改内容不改地址】,
#    所以图照样 replay、主表照样变快, 只是五处布局里漏改的那一处还在读旧专家 ——
#    路由分数指向 A 的权重、GPU 腿算的是 B 的权重。逐条钉死:
#      (a) DYN 关闭(enabled=False)时引擎【逐字不变】—— 新功能必须纯 opt-in,
#          这是【默认行为不变】那条硬约束唯一可执行的形态;
#      (b) 计数器只读 pinned 缓冲, 不进数值路径 —— 读 8 次再跑, 输出逐位同;
#      (c) 计数器必须 == route(x) 的 bincount —— 拿假直方图做决策 = 整张判决表作废;
#      (d) 迁移后与【按新序重建的引擎】输出 + fp32_reference 都逐位同 —— P1-② 的命
#          (bench_dyn 闸门② 同口径, 这里升级为常驻红线; 换内容不换地址能不能成立全看它);
#      (e) 图不重录: id(_g_pre)/id(_g_hot)/_gkey/_wver 从暖机后到末次迁移全程不变;
#      (f) 记账(slot2eid)与实际内容一致: check_content() 空表 + _gate_wt 列 == gate.weight 行;
#      (g) 迁移可逆: 反着再换两次回去, 输出与初值【逐位】相同 —— 换出=丢弃的写法在这里红。
#    恒等序重建那一发(d 的前置)先自证 real_src 这条路本身无损, 否则后面的逐位对比没意义。
# ============================================================
print("\n--- J P1-② 动态放置(计数器不改数值 / 迁移后与按新序重建逐位同 / 图不重录 / 可逆) ---")
import dyn_placement as DP

_J_KW = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
             pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
_J_E, _J_D, _J_N = 8, 576, 1536
_J_MODE = "L2_零CUDA+pinned"


def _j_fwd(m, x):
    return m(x, async_mode=True, mode=_J_MODE).float().cpu().clone()


def _j_build(gc, dt="fp32", src=None, order=None):
    """src=None => 引擎自己的随机初值权重; 给了 src(未置换基表) => 按 order[槽位]=eid 建。"""
    rs = None
    if src is not None:
        ex, gw, gb = src
        # 🚨 gate.bias 与 gate.weight 一样是【每槽位一格】: ExpertSwapper.swap 同时换两者的
        #    第 i/j 格, 所以按新序重建时 bias 也必须同序换。第一次写这段时 bias 写的 clone()
        #    (不换) => 重建引擎自己内部错位, 于是 route 的 idx 对不上、y 差出 341(= 假红在
        #    对照身上, 不是迁移错了)。真源那条路 gate_b 恒 0 掩盖了它, 玩具权重不掩盖。
        rs = dict(experts=[ex[e] for e in order],
                  gate_w=gw[list(order)].contiguous(),
                  gate_b=gb[list(order)].contiguous(), note="redJ")
    torch.manual_seed(42)
    m = H.HeteroMoE(_J_D, _J_E, _J_N, gpu_expert_count=gc, top_k=2, expert_kind="swiglu",
                    real_src=rs, cpu_delay_ms=0.0, gpu_dtype=dt, **_J_KW).cuda()
    m(torch.zeros(1, _J_D, device="cuda"), async_mode=True, mode=_J_MODE)   # 建缓冲+录图
    return m


def _j_capture(m):
    """从刚建好的引擎抓【未置换基表】(eid 空间; 此刻 slot2eid 还是恒等序)。
    GPU 腿读存储权重(fp32 存储与 gpu_dtype 无关), CPU 腿读 cpu_*_t 的转置。"""
    gc = m.gpu_expert_count
    ex = []
    for s in range(m.num_experts):
        if s < gc:
            ex.append(dict(g=m.gpu_experts[s].data.cpu().clone(),
                           u=m.gpu_up[s].data.cpu().clone(),
                           d=m.gpu_down[s].data.cpu().clone()))
        else:
            ex.append(dict(g=m.cpu_experts_t[s - gc].t().cpu().clone(),
                           u=m.cpu_up_t[s - gc].t().cpu().clone(),
                           d=m.cpu_down_t[s - gc].t().cpu().clone()))
    return ex, m.gate.weight.data.cpu().clone(), m.gate.bias.data.cpu().clone()


for dt, gc in (("fp32", 3), ("bf16", 3), ("fp32", 1)):
    torch.manual_seed(11)
    x = torch.randn(8, _J_D).cuda()
    m = _j_build(gc, dt)
    src = _j_capture(m)
    assert float(src[2].abs().max()) > 0, "🚨 玩具 bias 全 0 => J 段这条牙是空的"
    for _ in range(3):
        y0 = _j_fwd(m, x)
    gk0 = (id(m._g_pre), id(m._g_hot), m._gkey, tuple(m._wver or ()))
    # (d 前置) 恒等序重建 == 原引擎
    y_id = _j_fwd(_j_build(gc, dt, src=src, order=list(range(_J_E))), x)
    d_id = (y_id - y0).abs().max().item()
    # (a) DYN 关闭: 建 DynPlacer + 跑 12 个检查点, 输出必须逐字不动(且一次都不该迁移)
    dyn = DP.DynPlacer(m, src[0], gc, policy=DP.policy_from_name("static_A", gc),
                       window=4, check_every=1, warm=1, order=list(range(_J_E)))
    dyn.sw.src_gate = src[1]                    # 挂上门控基表, 闸门⑥ 才会核门控行/列
    for f in range(12):
        _j_fwd(m, x)
        dyn.step(f)
    y_a = _j_fwd(m, x)
    off_d = (y_a - y0).abs().max().item()
    # 🚨 在这一刻就把 n_swap 抄下来: 下面 (d)(g) 还要手工迁移 4 次, 到 print 时读
    #    dyn.sw.n_swap 会读到 4, 看起来像 DYN 关着也偷跑 —— 第一版就被这样误读过。
    off_n = dyn.sw.n_swap
    off_bad = dyn.sw.check_content()
    off_ok = off_d == 0.0 and off_n == 0 and not off_bad
    # (b)(c) 计数器
    cs = [DP.counts_from_engine(m, _J_E) for _ in range(8)]
    y_b = _j_fwd(m, x)
    cnt_d = (y_b - y0).abs().max().item()
    _, ti = m.route(x)
    pin_d = int((cs[0] - DP.slot_counts(ti.detach().cpu(), _J_E)).abs().max().item())
    cnt_ok = cnt_d == 0.0 and all(torch.equal(c, cs[0]) for c in cs) and pin_d == 0
    # (d)(e)(f) 真迁移两次
    sw = dyn.sw
    i2 = min(1, gc - 1)
    sw.swap(0, gc)
    sw.swap(i2, gc + 1)
    order_new = list(sw.slot2eid)
    m_new = _j_build(gc, dt, src=src, order=order_new)
    y_s, y_r = _j_fwd(m, x), _j_fwd(m_new, x)
    swap_d = (y_s - y_r).abs().max().item()
    sa, ia = m.route(x)
    sb, ib = m_new.route(x)
    idx_same = bool((ia == ib).all().item())
    ref_d = (m.fp32_reference(x, sa, ia).float().cpu()
             - m_new.fp32_reference(x, sb, ib).float().cpu()).abs().max().item()
    gk1 = (id(m._g_pre), id(m._g_hot), m._gkey, tuple(m._wver or ()))
    graph_ok = gk1 == gk0
    bad6 = sw.check_content()
    gate_ok = all(torch.equal(m._gate_wt[:, s].cpu().reshape(-1),
                              m.gate.weight.data[s].cpu()) for s in range(_J_E))
    # (g) 反着换回去
    sw.swap(i2, gc + 1)
    sw.swap(0, gc)
    y_c = _j_fwd(m, x)
    rev_d = (y_c - y0).abs().max().item()
    rev_ok = rev_d == 0.0 and list(sw.slot2eid) == list(range(_J_E)) and not sw.check_content()
    ok = (d_id == 0.0 and off_ok and cnt_ok and swap_d == 0.0 and ref_d == 0.0
          and idx_same and graph_ok and not bad6 and gate_ok and rev_ok)
    if not ok:
        fails.append(("dyn", dt, gc, d_id, off_d, cnt_d, pin_d, swap_d, ref_d,
                      idx_same, graph_ok, bad6[:3], gate_ok, rev_d))
    print(f"{dt} gc={gc}: 恒等重建互差={d_id:.1e} DYN关互差={off_d:.1e}(跑12帧后n_swap={off_n}) "
          f"计数器互差={cnt_d:.1e}/与route差={pin_d} 迁移后vs重建={swap_d:.1e} "
          f"参考也逐位同={ref_d:.1e}(idx同={idx_same}) 图不重录={graph_ok} "
          f"记账=内容={not bad6}{bad6[:2]} 两份门控布局同序={gate_ok} 可逆={rev_d:.1e} "
          f"{'OK' if ok else '🚨RED'}   [新序={order_new}]")
    del m, m_new, dyn, sw
    torch.cuda.empty_cache()


print("\nALL RED LINES PASS" if not fails else f"\n🚨 FAIL: {fails}")
sys.exit(0 if not fails else 1)

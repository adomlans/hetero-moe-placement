# -*- coding: utf-8 -*-
"""T11: 前段(门控 GEMM -> softmax -> topk -> 逐行 amax -> 激活量化)的融合实现。

为什么值得动(probe 实测, bs=32, 前段录在 CUDA Graph 里):
    torch 版 = 12 张 kernel, 图 replay 墙钟 41.7us / device 45.7us
      ampere_sgemm_32x32_...   19.4us  <- 32x1024 @ 1024x8 的小 GEMM, 27 GFLOP/s
      gatherTopK+bitonicSort    9.1us  <- 给 (32 行 x 8 列) 跑整套 bitonic 排序
      reduce + 量化链          ~12us
    理论活儿 = 读 x 128KB + W 32KB ≈ 0.5us。
    而且它在关键路径【最前面】: 前段不完事 => D2H 发不出去 => CPU worker 开不了工。

正确性口径(先过闸门再谈延迟, 顺序不能反; 见 bench_pre_fuse.py):
    topk_indices 与 torch 逐位相同 / x_q 与 torch 逐位相同 / scores<1e-6 / scale 精确
    两条实现约束:
      1) tl.dot(input_precision="ieee") —— 🚨 默认 tf32 会改 logits, 近分的 token 会
         跳到别的专家。那不是"精度略降", 是【路由分配变了】, 整个专家组合都会跟着变。
      2) 舍入用 libdevice.rint(半数取偶) —— torch.round 就是 banker's rounding,
         用 round-half-away 会让 x_q 在 .5 处差 1 LSB, 逐位闸门当场就红。
      3) topk 用迭代 argmax(tie_break_left) 复现 torch.topk 的【降序 + 同值取小下标】。
      4) 🚨【几何不变性】D 不是 BLOCK_D(=256) 整数倍时, D 循环的最后一块必须掩码。
         真权重 SmolLM-135M 的 D=576 = 2x256+64 -> 漏掩码会多读 64 行 -> acc=NaN ->
         argmax 落进补齐的 15 号假专家道 -> 下游 TORCH_CHECK(eid < gc+n_cpu) 当场抛。
         玩具 D=1024 正好整除, 所以这个越界读藏了十几轮 + 全部红线。红线里必须同时
         测【整除 D】和【不整除 D】两种几何, 见 check_redlines.py 的 T11 x D 边界。

【为什么 GEMM 从 24.5us 掉到 9.6us —— 而且和 ieee/tf32 无关】
  第一版直接拿 gate.weight (E, D) 在 kernel 里跨步读成 (BLOCK_D, EP):
      wv = tl.load(gw + offs_e[None,:]*D + offs_d[:,None])
  dot 的 B 操作数按列取, 每个元素一次独立访存, 完全不能合并 => 24.5us。
  预转置成 (D, EP) 连续(gate_wt)后 9.6us。权重是静态的, 转置只在建模型时做一次。
  对照表(bs=32, device 时间): ieee+跨步 24.5 / ieee+预转置 9.6 / tf32+跨步 6.5 /
  torch sgemm 19.1。🚨 tf32 再快 3us 但会改 logits => 近分的 token 换专家, 直接否掉。
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice   # rint = 半数取偶, 与 torch.round 同模式


@triton.jit
def route_fused(x_ptr, gw_ptr, gb_ptr, sc_ptr, ix_ptr,
                B, D, E,
                K: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_D: tl.constexpr, EP: tl.constexpr,
                TAIL: tl.constexpr = False):
    """一个 program 处理 BLOCK_B 行: GEMM -> +bias -> softmax -> 迭代 argmax 取 top-K。
    K 是 constexpr, 所以 top-K 展开成 K 轮 max/argmax, 完全不需要排序网络。"""
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mb = offs_b < B
    offs_e = tl.arange(0, EP)
    acc = tl.zeros((BLOCK_B, EP), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        # 🚨🚨 TAIL: D 不是 BLOCK_D 整数倍时(真权重 D=576, BLOCK_D=256 => 2 块 + 64 尾巴),
        #    最后一块会【多读 64 行】: x 越界读到相邻张量, gate_wt 只有 D*EP 个元素、
        #    越界读到 ~1.4 倍处。xv 里的脏比特进 dot => acc 变 NaN => softmax 全 NaN =>
        #    tl.argmax 落进为 2 的幂补齐的假专家道(EP=16 而 E=8 => id 15), 越界一路传到 C++。
        #    玩具 D=1024 = 4x256 正好整除, 所以这刀在十几轮优化和全部红线里都是隐形的。
        #    偶数 D 走 else(与修复前逐位相同, 计时不变), 奇数 D 才付掩码的代价。
        if TAIL:
            md = offs_d < D
            xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :],
                         mask=mb[:, None] & md[None, :], other=0.0)
            wv = tl.load(gw_ptr + offs_d[:, None] * EP + offs_e[None, :],
                         mask=(offs_e[None, :] < E) & md[:, None], other=0.0)
        else:
            xv = tl.load(x_ptr + offs_b[:, None] * D + offs_d[None, :], mask=mb[:, None], other=0.0)
            # 🚨 gw_ptr 必须是【预转置好的 (D, EP) 连续】权重(见文件头)。
            #    这里按 (BLOCK_D, EP) 连续读; 换成跨步读 gate.weight 会让这张卡从 9.6us 涨到 24.5us。
            wv = tl.load(gw_ptr + offs_d[:, None] * EP + offs_e[None, :],
                         mask=(offs_e[None, :] < E), other=0.0)
        acc += tl.dot(xv, wv, input_precision="ieee")
    logits = acc + tl.load(gb_ptr + offs_e, mask=offs_e < E, other=0.0)[None, :]
    logits = tl.where(offs_e[None, :] < E, logits, float('-inf'))
    ex = tl.exp(logits - tl.max(logits, 1)[:, None])
    sm = ex / tl.sum(ex, 1)[:, None]
    vals = sm
    for j in tl.static_range(K):
        mx = tl.max(vals, 1)
        am = tl.argmax(vals, 1)            # tie_break_left => 与 torch.topk 同序
        tl.store(sc_ptr + offs_b * K + j, mx, mask=mb)
        tl.store(ix_ptr + offs_b * K + j, am.to(tl.int64), mask=mb)
        vals = tl.where(offs_e[None, :] == am[:, None], float('-inf'), vals)


@triton.jit
def quant_fused(x_ptr, q_ptr, s_ptr, D, BLOCK_D: tl.constexpr):
    """逐行 amax -> scale -> int8 量化。一 program 一行, 整行进寄存器 => amax 与量化
    在同一处做完: 省掉 reduce_kernel 那张卡, 也省掉 x 的第二趟往返。"""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    msk = offs < D
    xv = tl.load(x_ptr + b * D + offs, mask=msk, other=0.0)
    s = tl.maximum(tl.max(tl.abs(xv), 0) / 127.0, 1e-8)
    q = tl.clamp(libdevice.rint(xv / s), -127.0, 127.0).to(tl.int8)
    tl.store(q_ptr + b * D + offs, q, mask=msk)
    tl.store(s_ptr + b, s)


def gate_wt(m, EP=None):
    """把 gate.weight (E, D) 预转置成 (D, EP) 连续。只在建模型时做一次。
    EP 补齐到 2 的幂且 >=16(tl.dot 的 N 维下限), 补出来的列在 kernel 里被 mask 成 0,
    且 softmax/topk 用 `offs_e < E` 把它们当 -inf 处理, 所以不会污染结果。"""
    EP = EP or triton.next_power_of_2(max(m.num_experts, 16))
    w = m.gate.weight.detach()                      # (E, D)
    wt = w.t().contiguous()                          # (D, E)
    if wt.shape[1] < EP:
        wt = torch.cat([wt, wt.new_zeros(wt.shape[0], EP - wt.shape[1])], dim=1)
    return wt


def route_quant_fused(m, x, out, gate_w=None):
    """替身版 _route_quant: 12 张 aten kernel -> 2 张 triton kernel。
    out = (topk_scores, topk_indices, x_q, x_scales) 由调用方预分配 ——
    🚨 必须预分配: 图内每次分配会换地址, 而 CPU worker 那边认的是固定地址。"""
    ts, ti, xq, xs = out
    B, D = x.shape
    gate_w = m._gate_wt if gate_w is None else gate_w
    route_fused[(triton.cdiv(B, 16),)](
        x, gate_w, m.gate.bias, ts, ti, B, D, m.num_experts,
        K=m.top_k, BLOCK_B=16, BLOCK_D=256, EP=triton.next_power_of_2(max(m.num_experts, 16)),
        # 🚨 D 不整除 BLOCK_D 时必须开尾巴块掩码(真权重 D=576)。写死 False 就是当年那个 bug。
        TAIL=(D % 256) != 0, num_warps=4)
    quant_fused[(B,)](x, xq, xs, D, BLOCK_D=triton.next_power_of_2(D), num_warps=8)
    return ts, ti, xq, xs


def alloc_pre_out(m, bs, device):
    return (torch.empty(bs, m.top_k, device=device, dtype=torch.float32),
            torch.empty(bs, m.top_k, device=device, dtype=torch.int64),
            torch.empty(bs, m.hidden_size, device=device, dtype=torch.int8),
            torch.empty(bs, device=device, dtype=torch.float32))


def cap_graph(fn):
    cap = torch.cuda.Stream(); cap.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(cap):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(cap); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()                     # 🚨 返回 g, 不是 fn() 的返回值
    return g

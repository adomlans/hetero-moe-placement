# -*- coding: utf-8 -*-
"""乙-2d: GPU 半边的 traffic-sparse 实现 —— 按【本帧真流量】读权重, 不按 gc。

【要还的债(全部是已入库的账, 不是新猜的)】
  乙-2b/乙-2c 归因: GPU 半边是【字节 bound】—— gemm/cat 档每帧读 gc*D*N、
    swiglu/cat 档每帧读 gc*D*2N + gc*N*D = gc 个专家的三张矩阵, 实测 237~268GB/s
    贴着 RTX 3060 的 skinny GEMM 标定带宽。
  P0-④a(bs=1 放对)钉死了单价: gc=3 -> gc=8 两档【真流量完全相同】(CPU腿slot=0,
    top_k=2 => GPU 腿最多 2 个专家有 token), 但 p50 0.257 -> 0.434ms(fp32) /
    0.215 -> 0.317ms(bf16) => 每个零流量专家白付 35us(fp32)/20us(bf16)。
  乙-2c 的配对段又量到: 配置C bs=1 GPU 腿【0 个 slot】也照付 0.18ms(降精能省掉
    => 它是稠密写法的固定成本, 不是数值成本)。
  => 结论: 只要【读字节的次数】从 gc 变成"本帧真命中的 (token,slot) 数",
     bs 小的时候这一半边就能按比例掉。这一刀不改精度、不改路由、不改放置, 只少读字节。

【为什么稠密写法(cuBLAS)省不了这笔钱】
  稠密把"哪个 token 命中哪个热专家"从控制流降级成数据(冷专家门控记 0) => 换来
  零主机往返 + 形状静态 + 能整段录 CUDA Graph(T6 的真收益, 必须保住), 代价是
  形状里带 gc => B 操作数每帧被完整读 gc 张矩阵, 门控是 0 也照样读回来再乘掉。

【这一版怎么把字节省掉, 同时不丢上面三条】
  网格开在【流量】上而不是专家上: grid = (bs*top_k, 分块数)。
  第 3 维根本不存在 —— 一个 program 领一个 (token,slot):
    1) 从 device 内存里读 eid = idx[slot](🚨不是从主机读 => 零同步照旧);
    2) eid >= gc(这个 slot 归 CPU 腿)=> 直接不碰权重: 门控/存零, 字节 = 0;
    3) 否则按 eid 间接寻址读 W[eid], 算完这个 slot 的部分和。
  形状只跟 bs 和 top_k 有关(都与 gc 无关) => 照样静态、照样录图、照样零主机往返。
  省下来的就是"没命中的那 (gc - 本帧命中数) 个专家的字节"。

【归约序变了 —— 口径必须先说清】
  稠密档: 一趟大 GEMM 把 gc 个专家的 g/u 一起算(cuBLAS 内部分块), down 是 bmm,
    最后在 gc 维上加和; traffic 档: 每个 (token,slot) 自己按 D / N 分块累加,
    然后在 top_k 维上加和。两者【不是同一个归约序】=> 逐位不可能相同。
  所以 traffic 的闸门不是"逐位 = 0", 而是:
    ① 路由 idx 逐位相同(路由代码一行没动)  ② y 互差 <= 归约序噪声底
    ③ KL/top-1 与稠密档同量级(比值进台账, 绝对值不报)
  fp32 下这是 1e-6 级的事; bf16 下 traffic 的中间量在 fp32 里累加、只在存 h 时
  截半精, 而稠密档的 g/u 先被 cuBLAS 截成 bf16 再算 silu => traffic 更准一点,
  这条差异要如实写进判决表, 不许读成"降精变好了"。

【什么时候该用(算式, 不是感觉)】
  稠密每帧读 gc 个专家; traffic 每帧读"本帧命中 GPU 的 (token,slot) 数"个专家,
  上界 = bs*top_k => 省字节的条件 = bs*top_k < gc。
  🚨 bs 大了这一条必反(bs=32, K=2 => 64 个 slot 去抢 8 个专家, 每个专家被重读 8 遍,
     而稠密只读一遍并把 token 摊销掉) => 那一档稠密才是对的算法。
  v1 只做 swiglu(真形状那一档, 也是 P0 全部判决表用的那一档); gemm 玩具档直接抛错,
  免得哪天有人在玩具形状上拿它当"通用加速"。
"""
import os
import torch
import triton
import triton.language as tl

# 分块大小。默认 64 是按【真形状 + 流量小】挑的: D=576=9x64, N=1536=24x64 都整除,
# bs=1 时 gu 网格 = 2 slot x 24 块 = 48 个 program(3060 有 28 个 SM)刚好铺满一轮。
BN = int(os.environ.get("TRAFFIC_BN", 64))     # gu 档的 N 分块 / down 档的 N 分块
BD = int(os.environ.get("TRAFFIC_BD", 64))     # gu 档的 D 分块 / down 档的 D 分块


@triton.jit
def _traffic_gu(x_ptr, w_ptr, idx_ptr, h_ptr,
                D, N, GC, K: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    """h[slot, n块] = silu(x_t · Wg[eid][:,块]) * (x_t · Wu[eid][:,块])
    w_ptr = _wcat_gu, 形状 (D, GC*2N): 专家 eid 的 gate 在第 eid*2N 列起, up 再往后 N 列。"""
    slot = tl.program_id(0)
    eid = tl.load(idx_ptr + slot)
    if eid < GC:                                  # 🚨 冷专家: 一个字节都不读
        t = slot // K
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)
        col0 = eid * (2 * N) + offs_n
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            xv = tl.load(x_ptr + t * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            # 🚨 两处 mask 都不能省: 真权重 D=576 但玩具档/别的 D 不一定整除 BLOCK_D。
            #    T11 那条"奇数 D 越界读 => NaN => 路由跳假专家"的教训就是这么来的。
            base = offs_d[:, None] * (2 * N * GC) + col0[None, :]
            mk = mask_d[:, None] & mask_n[None, :]
            acc_g += tl.sum(xv[:, None] * tl.load(w_ptr + base, mask=mk, other=0.0
                                                  ).to(tl.float32), 0)
            acc_u += tl.sum(xv[:, None] * tl.load(w_ptr + base + N, mask=mk, other=0.0
                                                  ).to(tl.float32), 0)
        h = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(h_ptr + slot * N + offs_n, h.to(h_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _traffic_down(h_ptr, wd_ptr, idx_ptr, sc_ptr, part_ptr,
                  D, N, GC, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    """part[slot, d块] = score[slot] * (h_slot · Wd[eid][:,d块]);  冷 slot 存【零】。
    冷 slot 也要写: part 缓冲被 part.sum(1) 全量吃, 留旧值 = 上一帧的数混进本帧。"""
    slot = tl.program_id(0)
    eid = tl.load(idx_ptr + slot)
    offs_d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if eid < GC:
        score = tl.load(sc_ptr + slot).to(tl.float32)
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            hv = tl.load(h_ptr + slot * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += tl.sum(hv[:, None] * tl.load(
                wd_ptr + eid * (N * D) + offs_n[:, None] * D + offs_d[None, :],
                mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32), 0)
        acc = acc * score
    tl.store(part_ptr + slot * D + offs_d, acc, mask=mask_d)


def traffic_hot(m, x, topk_scores, topk_indices, h_buf, part_buf, bn=None, bd=None):
    """GPU 半边 traffic 档入口。返回 (bs, out_dim) fp32 —— 与稠密档同一个契约:
    _hot_compute 负责出图前抬回 fp32, 这里只在 dt 上算中间量。"""
    D, N, K = m.hidden_size, m.expert_size, m.top_k
    gc, dt = m.gpu_expert_count, x.dtype
    bs = x.shape[0]
    slots = bs * K
    bn = BN if bn is None else bn
    bd = BD if bd is None else bd
    _traffic_gu[(slots, triton.cdiv(N, bn))](
        x, m._wcat_gu, topk_indices, h_buf, D, N, gc, K,
        BLOCK_N=bn, BLOCK_D=bd, num_warps=4)
    _traffic_down[(slots, triton.cdiv(D, bd))](
        h_buf, m._wstack_d, topk_indices, topk_scores, part_buf, D, N, gc,
        BLOCK_D=bd, BLOCK_N=bn, num_warps=4)
    # 最后一加放在 fp32: part 缓冲本来就是 fp32。稠密档是在 dt(bf16)上加完再抬,
    # 这一笔差异写进判决表, 不当"traffic 自带精度收益"。
    return part_buf.view(bs, K, D).sum(1)

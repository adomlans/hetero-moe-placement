#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-2c 归因探针: 把热专家 GEMM 的三个真实形状【裸跑 mm】(不套 CUDA Graph、不套重叠),
量 fp32 / bf16 / fp16 各自的等效带宽。

它回答的是判决表 `bench_dtype.py` 解释不了的那一半:
  字节减半了, GPU 半边【没】减半 —— 到底是 cuBLAS 的半精 kernel 不给力,
  还是"GPU 半边"里本来就有一截不随 dtype 变的固定成本(pre 图 + D2H 图 ≈ 0.05ms)?
  把 mm 从图里拎出来单独量, 两个原因就能分开。

读法:
  ① 同 shape 下 bf16 时间 ≈ fp32 的一半 => kernel 没问题, 差距在别处;
  ② 末尾两行 torch.sum 是这块卡的【纯读上限】(实测 ~300-310GB/s), 是所有的分母。
"""
import torch, time, statistics
def t(fn, iters=400):
    for _ in range(50): fn()
    torch.cuda.synchronize(); v=[]
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); v.append((time.perf_counter()-t0)*1e3/iters)
    return statistics.median(v)
D,N,GC=1024,2048,5
SHAPES={"gu":((1,D),(D,GC*2*N)),"cat":((1,GC*D),(GC*D,N)),"down":((1,N),(N,D))}
print("== 裸 mm(不套 graph) 的等效带宽: 半精到底能不能到满带宽 ==")
for M in (1,4,32):
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        nm=str(dt).split('.')[-1]
        for tag,(xs,ws) in SHAPES.items():
            x=torch.randn(M,xs[1],device='cuda',dtype=dt)
            w=torch.randn(*ws,device='cuda',dtype=dt)
            us=t(lambda: x@w); mb=ws[0]*ws[1]*w.element_size()/2**20
            print(f"M={M:>3} {nm:<9} {tag:<5} {mb:6.1f}MB {us:7.3f}ms {mb/1024/(us/1e3):6.0f}GB/s")
        del x, w
print("== 纯读上限对照(torch.sum) ==")
for dt in (torch.bfloat16, torch.float32):
    n=GC*2*N*D
    w=torch.randn(n,device='cuda',dtype=dt)
    us=t(lambda: torch.sum(w))
    print(f"{str(dt).split('.')[-1]:<9} {w.numel()*w.element_size()/2**20:6.1f}MB {us:7.3f}ms {w.numel()*w.element_size()/2**30/(us/1e3):6.0f}GB/s")
    del w

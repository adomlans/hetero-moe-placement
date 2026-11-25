#!/usr/bin/env python
"""T10 收尾定价: 「免费的不饱和方案」到底值不值。

背景: signed 打包精确但本机要付 7~10% 延迟(bench_pack_ab.py 实测), 因为 AVX2 的
maddubs 只吃 uint8 x int8, 想让权重变真 signed 就得先取绝对值再补符号(多 2 条指令)。
另一条路是【不动 kernel, 只把激活降到 7 位】:
    biased 权重 |w|∈[0,255], 激活 |x|<=63  =>  单对之和 <= 2*255*63 = 32130 < 32767
=> 天然不会触发 int16 饱和, 额外指令 = 0。
代价全在精度: 激活少 1 位 => 激活量化误差翻倍。

这里用 signed kernel(已证与理想 int32 逐位一致)去跑 7 位激活, 得到的就是
"biased+7位" 那套方案该有的精确结果 —— 因为两者的差别只剩 kernel 的指令数, 算术相同。
判据: 若精度代价(相对误差涨幅)明显贵于 7~10% 延迟, 就维持 signed 默认。
用法: python bench_act_bits.py
"""
import sys, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ORIG = H.HeteroMoE._quantize_act


def make(div):
    def q(self, x):
        s = torch.clamp(x.abs().max(dim=-1).values / div, min=1e-8)
        qv = (x / s.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        return qv, s
    return q


def pairmax(m, xq, div):
    """验证 7 位方案的『可证明不饱和』前提在实测数据上确实成立(用 biased 字节)。"""
    W = m.cpu_experts_q.cuda().to(torch.float64)
    x64 = xq.cuda().to(torch.float64)[:8]
    Hd = x64.shape[1]
    xv = x64.reshape(8, Hd // 2, 2)
    mx = 0.0
    for e0 in range(0, W.shape[0], 256):
        wv = W[e0:e0 + 256].reshape(-1, Hd // 2, 2)
        mx = max(mx, torch.einsum('tpc,epc->tep', xv, wv).abs().max().item())
    return mx


print("=" * 78)
print("激活位宽定价: 8位(现状) vs 7位(免费避开 maddubs 饱和) —— signed kernel, 对 fp32 参考")
print("=" * 78)
print(f"{'配置':<8} {'激活':<10} {'relRMSE%':>10} {'max_abs':>10} {'SNR dB':>8}  {'代价':>10}")
base = {}
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    for div, name in ((127.0, "8位(/127)"), (63.0, "7位(/63)")):
        H.HeteroMoE._quantize_act = make(div)
        torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
        torch.manual_seed(42)
        m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                        n_threads=6, tok_block=8, w_pack="signed").cuda()
        ts, ti = m.route(x)
        ref = m.fp32_reference(x, ts, ti)
        ma, snr, rel = H.err_metrics(m(x, async_mode=False), ref)
        base[(label, div)] = (rel, ma)
        tag = "" if div == 127.0 else f"{rel / base[(label, 127.0)][0]:.2f}x"
        print(f"{label:<8} {name:<10} {rel:>9.4f}% {ma:>10.5f} {snr:>8.2f}  {tag:>10}")
        if div == 63.0:
            pm = pairmax(m, m._prepare(x)[2], 63)
            print(f"         {'':<8} 7位时 biased 单对之和实测最大 {pm:.0f} "
                  f"< 32767 {'✅ 确实免费安全' if pm < 32767 else '🚨 仍会饱和'}")
H.HeteroMoE._quantize_act = ORIG
print("\n对照: signed 打包(精确 kernel, 8位激活) = 现状; biased+8位 = 老的错算术")
print("      延迟代价 bench_pack_ab.py: signed 比 biased 慢 ~7~10% (本机 i5-10400)")

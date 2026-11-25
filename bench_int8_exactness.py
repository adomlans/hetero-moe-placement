#!/usr/bin/env python
"""T10 判决探针: CPU INT8 kernel 到底有没有【算错】, 错多少, 是谁的锅。

背景(为什么值得单独写一个探针):
  主表只报"和 fp32 参考的相对误差", 里面混着两种东西 ——
  (a) 量化本身的信息损失(int8 每 tensor)  (b) kernel 实现引入的算术误差。
  之前 1.4660% 一直被认为是 (a)。把 (a) 单独算出来(理想 int32 精确累加, 和 kernel
  该算的东西一字不差, 只是用 float64 精确求和)只有 1.3844% => 有 0.48pp 是 (b)。

根因(已钉死, 别重测):
  _mm256_maddubs_epi16 把【相邻两个乘积之和】压进 int16。历史 biased 打包把权重存成
  +128 偏置的 uint8 [0,255], |x|<=127 => 单对乘积 <=32385, 一对之和最坏 2*32385=64770
  > 32767 => 【静默饱和截断】。本探针的"饱和仿真"(逐对相加后 clamp 到 int16)与真 kernel
  逐位吻合(max_abs~1e-5, relRMSE~1e-6%), 而"不饱和仿真"差 0.48% —— 锅就是这个指令。

修法与验证口径:
  signed 打包(权重存真·对称 int8, kernel 用 maddubs(|w|, x*sign(w))) => 每对之和
  |s| <= 2*127*127 = 32258 < 32767, 可证明不饱和 => 判据:
      signed 的 kernel 输出 必须与 理想 int32 逐位一致(max_abs ~1e-5 = fp32 舍入),
      且 relRMSE 从 1.4660% 降到理想的 1.3844%。
  两种打包量化到【同一批整数】(w/s 本来就落在 [-127,127]), 所以两边差异里没有任何
  量化格点的混淆项, 纯粹是那 0.48% 算术误差。
用法: python bench_int8_exactness.py
"""
import sys, torch
sys.path.insert(0, '.')
sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H

ext = H.avx_gemv_ext
EC = 256          # 每次处理的输出神经元块(控制 pair 张量显存)


def dots_pair(x64, w64, sat):
    """按 kernel 的【相邻两元素成对】顺序算点积。
    x64 (T,H) w64 (E,H) -> (T,E)。sat=True 时先把每对之和饱和到 int16(复现 maddubs)。"""
    T, Hd = x64.shape
    xv = x64.reshape(T, Hd // 2, 2)
    out = torch.empty(T, w64.shape[0], dtype=torch.float64, device=x64.device)
    for e0 in range(0, w64.shape[0], EC):
        wv = w64[e0:e0 + EC]
        if sat:
            s = torch.einsum('tpc,epc->tep', xv, wv.reshape(-1, Hd // 2, 2))
            out[:, e0:e0 + wv.shape[0]] = s.clamp(-32768, 32767).sum(-1)
        else:
            out[:, e0:e0 + wv.shape[0]] = torch.einsum('th,eh->te', x64, wv)
    return out


def cpu_half_ideal(m, xq, xs, ti, ts, wq, saturate):
    """CPU 冷专家半边的两种'应然'口径: saturate=False 是理想 int32 精确累加,
    True 是把 maddubs 的 int16 饱和也仿真进去。biased 打包才需要减 128*sum_x。"""
    signed = wq.dtype == torch.int8
    W = wq.cuda().to(torch.float64)
    x64 = xq.cuda().to(torch.float64)
    out = torch.zeros(xq.shape[0], m.expert_size, device='cuda', dtype=torch.float32)
    for eid in range(m.gpu_expert_count, m.num_experts):
        mask = (ti == eid)
        if not mask.any():
            continue
        tok, slot = mask.nonzero(as_tuple=True)
        e = eid - m.gpu_expert_count
        d = dots_pair(x64[tok], W[e], saturate)
        corr = 0.0 if signed else 128 * x64[tok].sum(-1, keepdim=True)
        val = (d - corr).float() * (xs[tok].unsqueeze(1) * m.cpu_experts_scale[e])
        out.index_add_(0, tok, val * ts[tok, slot].unsqueeze(1))
    return out


def sat_pair_rate(m, xq, wq, n_tok=8):
    """直接数一下: 真实数据里有多少「相邻乘积对之和」越过了 int16。"""
    W = wq.cuda().to(torch.float64)
    x64 = xq.cuda().to(torch.float64)[:n_tok]
    Hd = x64.shape[1]
    xv = x64.reshape(n_tok, Hd // 2, 2)
    over = tot = 0
    mx = 0.0
    for e0 in range(0, W.shape[0], EC):
        wv = W[e0:e0 + EC].reshape(-1, Hd // 2, 2)
        s = torch.einsum('tpc,epc->tep', xv, wv)
        over += int((s.abs() > 32767).sum()); tot += s.numel()
        mx = max(mx, s.abs().max().item())
    return over, tot, mx


print("=" * 78)
print("kernel vs 理想 int32 vs 饱和仿真 —— 两种权重打包")
print("=" * 78)
hdr = f"{'配置':<3} {'打包':<8} {'kernel relRMSE':>14} {'理想int32':>10} {'kernel-理想':>26} {'饱和仿真 vs kernel':>18}"
print(hdr)
for label, hot in (("A", [0, 1]), ("C", [5, 6])):
    torch.manual_seed(42); x = torch.randn(32, 1024).cuda()
    torch.manual_seed(42)
    m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=hot,
                    n_threads=6, tok_block=8).cuda()
    ts, ti, xq, xs = m._prepare(x)
    hot_out = m._hot_compute(x, ts, ti)
    ts_e, ti_e = m.route(x)
    ref = m.fp32_reference(x, ts_e, ti_e)
    for nm, wq in (("biased", m.cpu_experts_q), ("signed", m.cpu_experts_q_s8)):
        kern = ext.avx2_moe_cpu_experts_int8(
            xq, xs, ti, ts, wq, m.cpu_experts_scale,
            m.gpu_expert_count, False, 6, 8) + hot_out
        ideal = cpu_half_ideal(m, xq, xs, ti, ts, wq, False) + hot_out
        sim = cpu_half_ideal(m, xq, xs, ti, ts, wq, True) + hot_out
        krel = H.err_metrics(kern, ref)[2]
        irel = H.err_metrics(ideal, ref)[2]
        d = kern - ideal
        s = kern - sim
        print(f"{label:<3} {nm:<8} {krel:>13.4f}% {irel:>9.4f}% "
              f"{'max_abs=%.5f rel=%.4f%%' % (d.abs().max().item(), H.err_metrics(kern, ideal)[2]):>26} "
              f"{'max_abs=%.5f' % s.abs().max().item():>18}")
        if nm == "biased":
            over, tot, mx = sat_pair_rate(m, xq, wq)
            print(f"    {'':<6} 饱和证据: 单对之和最大 {mx:.0f} (>32767 即截断), "
                  f"抽样 {over}/{tot} 对越界 = {over / tot * 100:.5f}%")
            bias_rel = H.err_metrics(kern, ref)[2]
        else:
            gain = bias_rel - H.err_metrics(kern, ref)[2]
            exact_ok = (kern - ideal).abs().max().item() < 1e-3
            print(f"    {'':<6} signed 修好的相对误差 = {gain:+.4f}pp   "
                  f"与理想 int32 逐位一致: {'✅' if exact_ok else '🚨 不是!'}")

# 量化格点一致性: 两种打包必须量化到同一批整数, 否则上面的 A/B 不干净
print("\n格点自检(两种打包必须是同一批整数):")
torch.manual_seed(42)
m = H.HeteroMoE(1024, 8, 2048, 2, 2, hot_experts=[5, 6])
same = int((m.cpu_experts_q.to(torch.int16) - 128 ==
            m.cpu_experts_q_s8.to(torch.int16)).sum())
print(f"  {same}/{m.cpu_experts_q.numel()} 个字节相同 "
      f"({'✅ 同一格点' if same == m.cpu_experts_q.numel() else '🚨 格点不同, A/B 不干净'})")

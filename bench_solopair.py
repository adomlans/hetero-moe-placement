#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-1e · solo 口径的 swiglu/gemm 比值: 同一格内把【两种量法】各测一遍并现算比值 —— 比值会不会被量法带着走。

【为什么还要这一刀】
  已发布的 swiglu/gemm 比值全部来自 bench_swiglu.one() = 【两个 kind 同时活着 + 逐轮轮转】。
  乙-1c 定稿: 轮转口径对【被测量档】收一笔税, C 列 ΔP = +9%~+15%(重的那一档 swiglu 在付)。
  乙-1d 镜像(被测=gemm): ΔP 0/16 过星标 ⇒ 轻的那一档(gemm 池 6MB < L3)在轮转里读不出可记账的税。
  ⇒ 两条合起来的推论: 已发布的 C 段比值 ≈ 真比值 x (1 + 税_swiglu) ⇒ 偏高 ~10%。
  🚨 但这条推论是【跨进程跨表拼出来的】(乙-1c 的 swiglu solo ÷ 乙-1d 的 gemm solo),
     而"跨段/跨进程拼出来的差不算数"正是方法论 37 自己立的规矩 —— 所以必须补一次同进程实测。

【这一刀只问一件事: solo 口径的比值是多少】同进程同格子内跑两种量法:
  量法 solo  一次只让一个 kind 活着: 建→测→删。两个 kind 各占一次 pos1(先建先测)与 pos2(后建后测):
               r1 = sw@pos1 / ge@pos2 = r真/φ,   r2 = sw@pos2 / ge@pos1 = r真·φ
               => 两序【几何均值】在一阶上精确消掉位置效应 φ, 【两序差】≈ 2xφ = 这条比值自己的噪声底
             SOLO_PASSES 遍(相邻两遍交换两个 order 的先后) => 每个 (pos,kind)  pooled 中位, 热漂移摊平
  量法 pair  两档都活着 + 逐轮轮转(= one() 的处境, 已发布数值就是这个口径), 两个方向都走;
             整个 pair 段跑两遍(P1 夹在 solo 段前 / P2 在其后) => 两遍之差 = 轮转口径自己的底
  对账列     已发布比值(5088fec0 主表, 轮转口径)硬编码在 PUBLISHED, 只作对账不作分母。

【预设判决树】(先写死, 跑完不许改; 双峰格按 BIM_THR 判【不可判】, 不参与判决)
  A) Δ(轮转/solo) >= +6% 且过星标, 且集中在 C bs>=4
     => 轮转口径把 C 段比值抬高了: 已发布值要按 solo 口径更正; 乙-2「CPU 腿边际成本 ≈3x = 如实付钱」
        在低 bs 段改写(solo 比值 2.2~2.4 = 比字节比例【便宜】, 不是贵)。
  B) |Δ(轮转/solo)| 在带内 且 solo 比值与已发布值差 <= 3%
     => 比值对量法免疫成立(税同比例打在分子与分母上), 乙-1 系列结项, 旧表一张都不用动。
  C) solo 比值在 C 段也 ≈3x、轮转反而读 2.2~2.4
     => 🚨反号: 乙-1d 那条"已发布 C 段偏高 ~10%"的跨表推论作废, 回头查 gemm 侧被低估的原因。
  D) 位置底 > 10% 且没有格过星标 => 本口径分辨力不够(比值底 = 2x 位置效应), 要么加 SOLO_PASSES/ROUNDS
     要么放弃这条口径 —— 🚨 不许拿"带内"当"免疫"的证据, 那是把没测出来读成没差异。
  E) 只有 A 段在带内 / 只有 C 段过门槛 => 税记在【CPU 腿字节】头上(第三次现身, 与 乙-2b 同一笔钱);
     A 段也过门槛 => 与 CPU 腿无关, 记在【GPU 半边 + 分配器】头上, 乙-1c 判读第④条要改。
【新预测(可证伪)】solo 比值: bs=1 与 bs=4 ≈ 2.2~2.4, 只有 bs=128 逼近 3x。
  机制假设(假设不是结论): 低 bs 段 gemm 站的是【延迟地板】而不是字节墙, 分母没随字节数一起缩,
  所以比值显不出 3x; 高 bs 双方都进吞吐 bound, 字节比例才浮出来。兑现或打脸都记账。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
# 🚨 全部复用 pairtax / swiglu 那两份常数与工具 —— 本表要和已入库的表逐位可比
from bench_pairtax import build, arm_time, arm_rot, stat, bimodal, TAX_MIN, BIM_THR, MODE, NMAT
from bench_swiglu import D, E, N, K, CFG, BS_LIST, FIXGC, gc_for, hits_cum

ROUNDS = int(os.environ.get("ROUNDS", 7))               # 每臂几发(与 pairtax 判决轮同发数)
SOLO_PASSES = int(os.environ.get("SOLO_PASSES", 2))     # solo 段跑几遍(相邻两遍交换 order 先后)
KINDS = ("swiglu", "gemm")                              # 比值 = swiglu / gemm, 顺序固定不许换
# 对账用: 已发布比值 = bench_swiglu.one() 轮转口径, 归档 5088fec0
# (results/2026-11-22_乙-1b_nt扫描_swiglu_bs1-128.txt 的 8 行 "swiglu / gemm = ?x")
PUBLISHED = {("A", 1): 2.29, ("A", 4): 2.23, ("A", 32): 2.91, ("A", 128): 2.39,
             ("C", 1): 2.28, ("C", 4): 2.70, ("C", 32): 2.82, ("C", 128): 3.13}
HITS = {}


def pair_rot(m_sw, m_ge, x, which):
    """轮转口径 = one() 的处境: 另一档每轮先跑一整发, 再量 which 那一档。"""
    other, meas = (m_sw, m_ge) if which == "gemm" else (m_ge, m_sw)
    return arm_rot(other, meas, x)


def cell(bs, label, hot, gc):
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    refs, pools = {}, {}
    for kind in KINDS:                          # 先各建一次: 取闸门参考输出 + 量 CPU 池字节
        m = build(kind, gc, hot)
        refs[kind] = m(x, async_mode=True, mode=MODE).clone()
        pools[kind] = (E - gc) * m.expert_bytes()
        del m; torch.cuda.empty_cache()
    m = build("gemm", gc, hot)                  # 只为印"每专家几 MB"
    eb_g = m.expert_bytes(); del m; torch.cuda.empty_cache()
    res, bim = {}, {}

    def rec(tag, arm, v):
        bi = bimodal(v)
        if bi:
            bim[tag] = bi
        p, sd, mn = stat(v)
        dr = (statistics.mean(v[-2:]) / p - 1) * 100
        note = (f"\n        🚨双峰(阈值 {BIM_THR:.0f}%): 低簇 {bi[1]} 发≈{bi[0]:.3f} / "
                f"高簇 {bi[3]} 发≈{bi[2]:.3f} 间隔 {bi[4]:.0f}% => p50 落哪个簇是抛硬币" if bi else "")
        print(f"      {arm:<16} p50={p:.3f} σ={sd:.0f}% min={mn:.3f} 漂={dr:+.0f}% 轮间 "
              + " ".join(f"{t:.3f}" for t in sorted(round(t, 3) for t in v)) + note)
        res[tag] = p
        return p

    print("\n" + "=" * 104)
    sw_mb = eb_g * NMAT["swiglu"] // NMAT["gemm"]
    tail = '放对(CPU 腿几乎没活)' if set(hot) < set(range(gc)) else '故意放错(打 CPU 腿)'
    print(f"配置 {label} | bs={bs:>4} | gc={gc} | 比值=swiglu/gemm | 每专家 gemm {eb_g / 2**20:.2f}MB / "
          f"swiglu {sw_mb / 2**20:.2f}MB | solo 池 sw {pools['swiglu'] / 2**20:.1f}MB "
          f"gemm {pools['gemm'] / 2**20:.1f}MB  轮转池 {(pools['swiglu'] + pools['gemm']) / 2**20:.1f}MB "
          f"(L3 {H.L3 >> 20}MB) | solo {SOLO_PASSES} 遍 x 两序 + 轮转两遍 | hot={hot} ← {tail}")
    print("=" * 104)

    # ---------- 轮转口径 P1(先跑, 与 P2 一起夹住 solo 段) ----------
    ms = {k: build(k, gc, hot) for k in KINDS}
    for k in KINDS:
        rec(("pair1", k), "轮转 " + k, pair_rot(ms["swiglu"], ms["gemm"], x, k))
    del ms; torch.cuda.empty_cache()

    # ---------- solo 口径: SOLO_PASSES 遍, 每遍两个 order(谁占 pos1 换一次) ----------
    shots = {}
    for ip in range(SOLO_PASSES):
        firsts = list(KINDS) if ip % 2 == 0 else list(KINDS)[::-1]   # 🚨 交换 order 先后 => 摊平热漂移
        for first in firsts:
            other = KINDS[1 - KINDS.index(first)]
            for slot, kind in ((1, first), (2, other)):
                m = build(kind, gc, hot)
                v = arm_time(m, x)
                del m; torch.cuda.empty_cache()
                rec((f"solo{ip + 1}", kind), f"solo {kind}@pos{slot}", v)
                shots.setdefault((slot, kind), []).extend(v)
    med = {k: statistics.median(v) for k, v in shots.items()}
    n_shot = len(shots[(1, "swiglu")])
    print(f"  solo pooled 中位({n_shot} 发/格): sw@pos1={med[(1, 'swiglu')]:.3f} "
          f"ge@pos2={med[(2, 'gemm')]:.3f}  |  ge@pos1={med[(1, 'gemm')]:.3f} "
          f"sw@pos2={med[(2, 'swiglu')]:.3f}")

    # ---------- 轮转口径 P2(后跑) ----------
    ms = {k: build(k, gc, hot) for k in KINDS}
    for k in KINDS:
        rec(("pair2", k), "轮转 " + k, pair_rot(ms["swiglu"], ms["gemm"], x, k))
    d_sw = (ms["swiglu"](x, async_mode=True, mode=MODE) - refs["swiglu"]).abs().max().item()
    d_ge = (ms["gemm"](x, async_mode=True, mode=MODE) - refs["gemm"]).abs().max().item()
    del ms; torch.cuda.empty_cache()
    ok = "OK" if d_sw == 0 and d_ge == 0 else "🚨 建/删历史改写了数值, 本格全部不可比"
    print(f"  闸门 两档建/删 x {SOLO_PASSES} 遍两序 solo + 两轮转段 之后: swiglu 逐位差={d_sw:.9f} "
          f"gemm 逐位差={d_ge:.9f} {ok}")

    # ---------- 两个口径的比值与各自的底 ----------
    r1 = med[(1, "swiglu")] / med[(2, "gemm")]        # swiglu 占 pos1
    r2 = med[(2, "swiglu")] / med[(1, "gemm")]        # gemm 占 pos1
    geo = (r1 * r2) ** 0.5
    pos_floor = abs(r1 / r2 - 1) * 100
    rp1 = res[("pair1", "swiglu")] / res[("pair1", "gemm")]
    rp2 = res[("pair2", "swiglu")] / res[("pair2", "gemm")]
    pair_geo = (rp1 * rp2) ** 0.5
    pair_floor = abs(rp1 / rp2 - 1) * 100
    d_pair = (pair_geo / geo - 1) * 100
    thr = max(2 * max(pos_floor, pair_floor), TAX_MIN)
    print(f"  solo 比值(两序) = {r1:.3f} / {r2:.3f}  几何均值={geo:.3f}  🚨位置底(两序差)={pos_floor:.1f}%")
    print(f"  轮转比值(两遍) = {rp1:.3f} / {rp2:.3f}  几何均值={pair_geo:.3f}  轮转底(两遍差)={pair_floor:.1f}%")
    pub = PUBLISHED.get((label, bs))
    tail2 = f"   对账 已发布(轮转口径 5088fec0)={pub:.2f}  Δ(已发布/solo)={(pub / geo - 1) * 100:+.1f}%" if pub else ""
    print(f"  Δ(轮转/solo) = {d_pair:+.1f}%  门槛={thr:.1f}%{tail2}")
    if bim:
        who = " ".join(f"{t[0]}/{t[1]}" for t in sorted(bim))
        print(f"  🚨 本格双峰臂: {who} —— 双峰格的比值读【不可判】(两序底看不见双峰)")
    del refs
    return dict(solo1=r1, solo2=r2, geo=geo, pos_floor=pos_floor, pair1=rp1, pair2=rp2,
                pair_geo=pair_geo, pair_floor=pair_floor, pub=pub, bim=bool(bim), gate=(d_sw == 0 and d_ge == 0))


if __name__ == "__main__":
    print(f"[机器状态] {H.machine_state()}")
    print(f"[乙-1e solo 口径比值探针] D={D} E={E} N={N} top_k={K} | bs={BS_LIST} | 比值=swiglu/gemm | "
          f"每臂 {ROUNDS} 发 x (5 暖机 + 30 计时 x 3 reps) | solo {SOLO_PASSES} 遍 x 两序 + 轮转两遍 | "
          f"门槛 TAX_MIN={TAX_MIN}% 双峰阈值 BIM_THR={BIM_THR}%")
    for lb, hot in CFG:
        HITS[lb] = hits_cum(hot, "swiglu")
        print(f"  累计命中率 {lb}(swiglu): {HITS[lb]}")
    out = {}
    for bs in BS_LIST:
        for label, hot in CFG:
            gc = FIXGC or gc_for("swiglu", HITS[label])[0]    # 🚨 与已发布表同一根 gc 桩(钉 swiglu 规则)
            out[(label, bs)] = (gc, cell(bs, label, hot, gc))
    print("\n" + "=" * 118)
    print("汇总: solo 口径比值(两序几何均值) vs 轮转口径比值(两遍几何均值) vs 已发布(轮转口径)")
    print(f"      星标 * = |Δ(轮转/solo)| > max(2 x max(位置底, 轮转底), TAX_MIN={TAX_MIN}%);  ✗ = 本格有双峰臂 => 不可判")
    print("      判读: Δ≈0(带内) => 比值对量法免疫; Δ 显著为正 => 已发布值偏高, 要按 solo 口径更正")
    print("=" * 118)
    h1, h2 = "solo两序", "轮转两遍"
    print(f"{'配置':<6}{'bs':>5}{'gc':>4}{h1:>16}{'位置底':>8}{h2:>16}{'轮转底':>8}"
          f"{'solo比':>8}{'轮转比':>8}{'Δ轮转/solo':>13}{'已发布':>9}{'Δ已发布/solo':>14}{'闸门':>6}")
    for (label, bs), (gc, c) in sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        thr = max(2 * max(c["pos_floor"], c["pair_floor"]), TAX_MIN)
        d_pair = (c["pair_geo"] / c["geo"] - 1) * 100
        sig = "✗" if c["bim"] else ("*" if abs(d_pair) > thr else " ")
        solo两序 = f"{c['solo1']:.3f}/{c['solo2']:.3f}"
        轮转两遍 = f"{c['pair1']:.3f}/{c['pair2']:.3f}"
        line = (f"{label:<6}{bs:>5}{gc:>4}{solo两序:>16}{c['pos_floor']:>8.1f}{轮转两遍:>16}{c['pair_floor']:>8.1f}"
                f"{c['geo']:>8.3f}{c['pair_geo']:>8.3f}{d_pair:>+11.1f}%{sig:>2}")
        line += f"{c['pub']:>9.2f}{(c['pub'] / c['geo'] - 1) * 100:>+13.1f}%" if c["pub"] else f"{'-':>9}{'-':>14}"
        line += f"{'OK' if c['gate'] else '🚨':>6}"
        print(line)
    print("\n判读提示: ① 位置底 = 这条比值自己的底(两序换的是谁占 pos1); 底大 = 建/删顺序本身就影响比值,\n"
          "           此时分辨力不够, 按判决树 D 加 SOLO_PASSES/ROUNDS 重跑, 别把带内读成免疫;\n"
          "        ② Δ(轮转/solo) 才是本表要的那个数: 正 = 轮转口径(=已发布口径)把比值抬高; 负 = 压低;\n"
          "        ③ 只看带星标的格; ✗(双峰)一律不可判 —— 两序底看不见双峰, 见 乙-1d 的 gemm C bs=32;\n"
          "        ④ 已发布列只作对账(不同进程不同批次), 不当分母; 它落在带内 = 旧表不用动。")

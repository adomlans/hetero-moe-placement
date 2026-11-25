#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-1c 配对税探针: 同一格里【单档驻留】和【两档在场】到底差多少, 差额能不能记账。

【为什么要这一刀】
  乙-1b 顺手量出一条【同配置跨量法水位差】(方法论 37): 同一份 bench_swiglu 表里,
  `one()`(gemm+swiglu 两个模型同时活着 + 逐轮轮转)与 `nt_scan()`(每个 nt 建一个再 del,
  只活一个)给出的 swiglu p50, 在 C 配置 x bs>=4 上差 -9.8% / -15.6% / -10.0%, 在 A 配置只有
  +-2.6%。这条差不是策略差、不是负载差、不是 bs 差 —— 它是【量法】差。
  影响面: 以往所有 C 列 p50 与 swiglu/gemm 比值都可能系统性偏高(乙-2 整条主线用的就是轮转配对)。

【这一刀只问一件事: 差额是谁造成的】三臂同进程同格子, 只换【另一档在不在场/跑不跑】:
  臂 S  solo      只建被测量档 => CPU 权重池只有一份                (== nt_scan 的处境)
  臂 Q  pair-idle 两档都建(另一档的池 + 第二张 CUDA 图都在场), 但另一档【一发都不跑】
  臂 P  pair-rot  两档都建, 另一档每轮先跑一整发 => 就是 one() 的处境(已入库的那些表)
  臂 R  blob      只建被测量档 + 另外按【另一档的字节数】申请一块静态 CPU 字节(从不读)
读法(判决树):
  dP 显著 且 dQ 在带内  => 税 = 【跑另一档】把它的热集打掉了(配对本身引入的干扰, 与池大小无关)
                          => 以往轮转表的 C 列是【被配对动作自己】压慢的, 该重测而非打折
  dQ ~= dP 显著          => 税 = 【另一档在场】就够(字节驻留 / 分配器 / 第二张图) => 容量级问题
  dR ~= dQ               => 再窄一步: 只要【字节在场】就够, 不需要模型对象或第二张图
  三个 d 全在带内        => 水位差是【跨段/跨进程】假象 => 方法论 37 改判, 以往数值不用动
🚨 两种 order 都要跑: 臂 S 在 pair_last 里是【最后】跑的(前面已经建/删过另一档),
   在 solo_first 里是【最先】跑的(全新进程状态)。两序不一致 = 差额的成分是【分配历史】,
   不是驻留 —— 那比"有税"更要紧, 必须当场说出来。

【口径】同进程 / 同 seed / 同 gc / 同 x / 同一发数; p50 取 H.bench 的 min-of-3reps 再取中位;
  🚨 不跨臂轮转(臂的定义就是不同的模型集, 轮转会把"删掉另一档"变成不可能) —— 用两种 order
  代替轮转, 用【同一臂两序之差】当噪声底; 星标门槛 = |Δ| > max(2x底, TAX_MIN=2%) ——
  🚨 绝对下限是判决轮补的: A 列底常 0.2%(门槛只要 0.4%), 1% 级的分配历史副作用也能拿星标。
  🚨 双峰判据(BIM_THR, 乙-1d 判决轮补): 两序底看不见【双峰】—— 某一臂的 7 发分成两簇时,
     两序的 S 臂可以都落在同一簇, 于是底很小而 Δ 全是簇间距离的假信号。
     排序后最大相邻间隔(两侧各须 >=2 发, 单发尾巴算热漂移不算双峰)> BIM_THR=8% => 该臂双峰,
     它的 Δ 星标作废(印 ✗), 该格读【不可判】而不是读"有益/有害"。
"""
import os, sys, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
# 🚨 形状/常数/放置规则全部复用乙-2 那一份 => 本表的水位差可以直接和已入库的 swiglu 表对账
from bench_swiglu import D, E, N, K, CFG, FULL, BS_LIST, FIXGC, gc_for, hits_cum

ROUNDS = int(os.environ.get("ROUNDS", 5))          # 每臂几发(每发 = H.bench: 5 暖机 + 30 计时 x 3 reps)
# order 名必须归一: 之前 ORDER=pairlast 会走到 else 分支(先量 S 再建另一档)却印成 "pairlast" => 标签与行为不符
_ALIAS = {"pair_last": "pair_last", "pairlast": "pair_last",
          "solo_first": "solo_first", "solofirst": "solo_first", "both": "both"}
ORDER = os.environ.get("ORDER", "both").strip().lower()
assert ORDER in _ALIAS, f"ORDER 只能是 pair_last|solo_first|both, 收到 {ORDER}"
ORDER = _ALIAS[ORDER]
MEAS = os.environ.get("MEAS", "swiglu")            # 被测量档(主线真形状 = swiglu 3 矩阵)
CTRL = os.environ.get("CTRL", "gemm")              # 另一档 = 干扰源(它的 CPU 权重池 = 被怀疑的那个池)
ARMS = [a for a in os.environ.get("ARMS", "S,Q,P,R").split(",") if a.strip()]
# 绝对下限: 噪声底可以很小(A 列常 0.1%~0.2%), 那时 2x底 只要求 |Δ|>0.4% => 1% 级的分配历史副作用
# 也能拿到星标。判决轮就是这样让 A 列冒出 -1.6% 的"负税"的 => 星标门槛 = max(2x底, TAX_MIN)。
TAX_MIN = float(os.environ.get("TAX_MIN", 2.0))
# 双峰判据: 排序后最大相邻间隔 / 下簇心 > 该值 => 该臂两簇。乙-1d 实测 gemm C 列两簇心
# 0.66us 与 0.83us(间隔 25%), 正常格的最大间隔 <=5%, 8% 落在两者之间。
BIM_THR = float(os.environ.get("BIM_THR", 8.0))
MODE = "L2_零CUDA+pinned"
NMAT = {"gemm": 1, "swiglu": 3}              # 每专家几张矩阵 —— 与 hetero_moe_int8.n_mat 同义
L3 = H.L3
HITS = {}


def build(kind, gc, hot):
    torch.manual_seed(42)                          # 🚨 同 seed: 两档路由逐位相同(乙-2 的约定)
    return H.HeteroMoE(D, E, N, gc, K, hot_experts=hot, cpu_delay_ms=0.0,
                       expert_kind=kind, **FULL).cuda()


def arm_time(m, x):
    return [H.bench(m, x, True, MODE)[0] for _ in range(ROUNDS)]


def arm_rot(m_other, m_meas, x):
    """P 臂 = one() 的处境: 另一档每轮先跑一整发, 再量被测量档。"""
    out = []
    for _ in range(ROUNDS):
        H.bench(m_other, x, True, MODE)
        out.append(H.bench(m_meas, x, True, MODE)[0])
    return out


def stat(v):
    return (statistics.median(v), statistics.stdev(v) / min(v) * 100 if len(v) > 1 else 0.0, min(v))


def bimodal(v):
    """两序底的盲区: 双峰。返回 (下簇心, 下簇发数, 上簇心, 上簇发数, 间隔%) 或 None。
    只认【两侧各 >=2 发】的分叉 —— 单发尾巴是发数内的热漂移(漂= 列已经在管它), 不算双峰。"""
    s = sorted(round(float(t), 3) for t in v)
    if len(s) < 6 or s[0] <= 0:
        return None
    cand = [i for i in range(len(s) - 1) if 2 <= i + 1 <= len(s) - 2]
    if not cand:
        return None
    i = max(cand, key=lambda k: s[k + 1] / s[k])
    gap = (s[i + 1] / s[i] - 1) * 100
    if gap <= BIM_THR:
        return None
    lo, hi = s[:i + 1], s[i + 1:]
    return (statistics.median(lo), len(lo), statistics.median(hi), len(hi), gap)


def cell(bs, label, hot, gc):
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    m_s = build(MEAS, gc, hot)
    eb = m_s.expert_bytes()
    pool_s = (E - gc) * eb
    # 🚨 按 kind 的 n_mat 现算, 不是写死除数 3: 干扰档换形状时这一项必须跟着变
    pool_other = (E - gc) * (eb // NMAT[MEAS]) * NMAT[CTRL]
    ref0 = m_s(x, async_mode=True, mode=MODE).clone()
    orders = ["pair_last", "solo_first"] if ORDER == "both" else [ORDER]
    res = {o: {} for o in orders}
    bim = {}                                        # (order, 臂) -> 双峰簇心/发数
    print("\n" + "=" * 100)
    print(f"配置 {label} | bs={bs:>4} | gc={gc} | 被测量={MEAS} 每专家 {eb / 2**20:.2f}MB "
          f"| solo CPU 池 {pool_s / 2**20:.1f}MB  pair CPU 池 {(pool_s + pool_other) / 2**20:.1f}MB "
          f"(L3 {L3 >> 20}MB)  hot={hot} {'← 放对(CPU 腿几乎没活)' if set(hot) < set(range(gc)) else '← 故意放错(打 CPU 腿)'}")
    print("=" * 100)
    for order in orders:
        m_g = blob = None
        if order == "pair_last":
            m_g = build(CTRL, gc, hot)
            if "Q" in ARMS: res[order]["Q"] = arm_time(m_s, x)
            if "P" in ARMS: res[order]["P"] = arm_rot(m_g, m_s, x)
            del m_g; m_g = None; torch.cuda.empty_cache()
            if "S" in ARMS: res[order]["S"] = arm_time(m_s, x)
        else:
            if "S" in ARMS: res[order]["S"] = arm_time(m_s, x)
            m_g = build(CTRL, gc, hot)
            if "Q" in ARMS: res[order]["Q"] = arm_time(m_s, x)
            if "P" in ARMS: res[order]["P"] = arm_rot(m_g, m_s, x)
            del m_g; m_g = None; torch.cuda.empty_cache()
        if "R" in ARMS:
            blob = torch.empty(pool_other, dtype=torch.uint8)
            blob.fill_(7)                                      # 摸一次 => 页真的在场
            res[order]["R"] = arm_time(m_s, x)
            del blob; blob = None
        d = (m_s(x, async_mode=True, mode=MODE) - ref0).abs().max().item()
        print(f"  [{order:<10}] 闸门 被测档输出在【建/删另一档 + 全部臂跑完】后与开头逐位差={d:.9f} "
              f"{'OK' if d == 0 else '🚨 被测档被改写了, 本臂不可比'}")
        for a in ("S", "Q", "P", "R"):
            if a in res[order]:
                p, sd, mn = stat(res[order][a])
                # 漂移 = 末两发均值 / p50 - 1。判决轮每臂轮间普遍右偏(最慢的发总在最后 1~2 发),
                # 它是【发数内的热漂移】不是臂间差 —— p50 已吸收它, min 列被它污染 => 只看不可用。
                dr = (statistics.mean(res[order][a][-2:]) / p - 1) * 100
                bi = bimodal(res[order][a])
                if bi:
                    bim[(order, a)] = bi
                print(f"    臂{a} p50={p:.3f} σ={sd:.0f}% min={mn:.3f} 漂={dr:+.0f}% 轮间 "
                      + " ".join(f"{t:.3f}" for t in sorted(round(t, 3) for t in res[order][a]))
                      + (f"\n        🚨双峰(阈值 {BIM_THR:.0f}%): 低簇 {bi[1]} 发≈{bi[0]:.3f} / "
                         f"高簇 {bi[3]} 发≈{bi[2]:.3f} 间隔 {bi[4]:.0f}% => 本臂 p50 落哪个簇是抛硬币, "
                         f"Δ 不可记账" if bi else ""))
    # ---- 水位差(以同一 order 内的 S 臂为分母) + 噪声底 = 同一臂在两序里的差
    rows = []
    for order in orders:
        base = res[order].get("S")
        if not base:
            continue
        pb = stat(base)[0]
        dd = {a: (stat(res[order][a])[0] / pb - 1) * 100 for a in ("Q", "P", "R") if a in res[order]}
        rows.append((order, pb, dd))
        print(f"  [{order:<10}] Δ=S 基准 {pb:.3f} | " +
              "  ".join(f"Δ{a}={dd[a]:+.1f}%" for a in dd))
    noise = {}
    if len(rows) == 2:
        for a in ("S", "Q", "P", "R"):
            if a in res["pair_last"] and a in res["solo_first"]:
                n = abs(stat(res["pair_last"][a])[0] / stat(res["solo_first"][a])[0] - 1) * 100
                noise[a] = n
                print(f"  噪声底(同臂跨 order) {a}: {n:.1f}%")
        print(f"  🚨 显著性门槛: |Δ| > max(2 x max(该臂底, S 臂底), TAX_MIN={TAX_MIN:.1f}%) 才打星标记账"
              " —— 底小不等于 1% 级的差可记账(绝对下限就是为此存在)")
        if bim:
            who = " ".join(f"{o}/{a}" for o, a in sorted(bim))
            print(f"  🚨 本格双峰臂: {who} —— 双峰格的【两序底】被簇间距离污染(S 臂两序可以都落在同一簇), "
                  "它的 Δ 一律读不可判, 汇总表里星标作废(印 ✗)")
    else:
        print("  🚨 只跑了一序 => 没有噪声底, 下面的 Δ 一律标 ? 不能记账(ORDER=both 才有底)")
    del m_s
    torch.cuda.empty_cache()
    return rows, (pool_s, pool_other), noise, bim


if __name__ == "__main__":
    print(f"[机器状态] {H.machine_state()}")
    print(f"[乙-1c 配对税探针] D={D} E={E} N={N} top_k={K} | bs={BS_LIST} | 被测量={MEAS} 干扰档={CTRL} | "
          f"每臂 {ROUNDS} 发 x (5 暖机 + 30 计时 x 3 reps) | 臂={''.join(ARMS)} | order={ORDER}")
    for lb, hot in CFG:
        HITS[lb] = hits_cum(hot, MEAS)
        print(f"  累计命中率 {lb}({MEAS}): {HITS[lb]}")
    if "S" not in ARMS:
        print("🚨 ARMS 里没有 S(solo) => Δ 没有分母, 汇总表会空 —— 本探针的 Δ 定义就是相对 solo")
    all_rows = {}
    for bs in BS_LIST:
        for label, hot in CFG:
            # gc_for 返回 (gc, 预测关键路径us) —— 取第一元。钉在【被测量档】的规则上,
            # 两档同 gc(乙-2 的约定: 不钉住就是把放置规则的变化算成配对税)
            gc = FIXGC or gc_for(MEAS, HITS[label])[0]
            rows, pools, noise, bim = cell(bs, label, hot, gc)
            all_rows[(label, bs)] = (rows, pools, gc, noise, bim)
    print("\n" + "=" * 112)
    print("汇总 = Δ(臂)/p50(S)  (正 = 该臂比 solo 慢 = 配对税; 负 = solo 反而慢 => 🚨假设反了)")
    print(f"      星标 * = |Δ| > max(2 x 噪声底, TAX_MIN={TAX_MIN:.1f}%)(可记账);   空格 = 带内读 0;   ? = 单序无底;"
          f" ✗ = 双峰污染(阈值 {BIM_THR:.0f}%) 星标作废")
    print("=" * 112)
    arms = [a for a in ("Q", "P", "R") if a in ARMS]
    print(f"{'配置':<6}{'bs':>5}{'gc':>4}{'soloMB':>8}{'pairMB':>8}{'order':>12}{'S基准':>9}"
          + "".join(f"{('Δ' + a):>11}" for a in arms)
          + "".join(f"{('底' + a):>8}" for a in arms) + f"{'底S':>8}")
    for (label, bs), (rows, pools, gc, noise, bim) in sorted(all_rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for order, pb, dd in rows:
            s_bim = ("S", order) in bim
            line = (f"{label:<6}{bs:>5}{gc:>4}{pools[0] / 2**20:>8.1f}{(pools[0] + pools[1]) / 2**20:>8.1f}"
                    f"{order:>12}{pb:>9.3f}")
            for a in arms:
                if a not in dd:
                    continue
                if not noise:
                    sig = "?"
                elif s_bim or (a, order) in bim:
                    sig = "✗"                                  # 簇间距离污染, 不可记账
                else:
                    thr = max(2 * max(noise.get(a, 0.0), noise.get("S", 0.0)), TAX_MIN)
                    sig = "*" if abs(dd[a]) > thr else " "
                line += f"{f'{dd[a]:+.1f}%' + sig:>11}"
            line += "".join(f"{noise.get(a, float('nan')):>8.1f}" for a in arms)
            line += f"{noise.get('S', float('nan')):>8.1f}" if noise else "       -"
            who = sorted({a for (oo, a) in bim if oo == order})
            if who:
                line += f"   🚨双峰:{''.join(who)}"
            print(line)
    print("\n  底列单位 = %(同臂跨 order 之差); Δ 列单位 = %(相对同序 S 臂)。"
          "读法: 先看星标, 没星标的 Δ 与 0 不可区分。")
    print("\n判读提示: ① 先看【噪声底(同臂跨 order)】再看 Δ —— 小于噪声底的 Δ 一律读成 0;\n"
          "        ② ΔP 大而 ΔQ 在带内 = 税在【跑另一档】(它把你热集打掉了) => 轮转配对这套口径要改,\n"
          "           不是给旧数打折; ΔQ≈ΔP = 税在【另一档在场】(字节驻留/第二张图) => 容量级问题;\n"
          "           ΔR≈ΔQ = 只要字节在场就够(连模型对象和第二张图都不需要);\n"
          "        ③ 全在带内 => 乙-1b 那条水位差是【跨段/跨进程】假象, 方法论 37 改判, 旧表不用动;\n"
          "        ④ A 配置(CPU 腿几乎没活)是天然对照: 若只有 C 有税, 税就记在【CPU 腿字节】头上,\n"
          "           与 GPU 半边无关 —— 那正好是乙-2b 定过的那笔钱, 同一笔钱第二次现身;\n"
          "        ⑤ 双峰(臂行 🚨双峰 / 汇总 ✗)= 该臂 7 发分成两簇, 两序底【看不见】它: 两序的 S 臂可以\n"
          "           都落在同一簇 => 底很小而 Δ 全是簇间距离的假信号(乙-1d 的 gemm C bs=32 就这样造出\n"
          "           -16.7%*/-23.6%* 两个【配对有益】的星标)。双峰格读【不可判】; 唯一能提前看见它的是\n"
          "           漂= 列与轮间列表 —— p50/σ/底 三个都会骗人, min 更是早就被污染不可用。")

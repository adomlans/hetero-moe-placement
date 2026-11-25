#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§5.2i附3 B1 换装置轮的 K‴ 组合器: 读 NP 个进程各自 dump 的 JSON => 跨进程聚合 => 印 K‴ 树落点。

【这个文件是什么 / 不是什么】
  是: 把【进程】当成新的重复层, 对已在各进程内算好的标量做跨进程聚合(中位数/几何均值/计数),
      然后按跑前写死的 K‴ 判据(a‴~e‴ + 三条问句)印落点 + 事前预测兑现/打脸 + 臂身份诊断。
  不是: 裁判。配对/星标/★2/门槛/追踪率全部由 bench_drifttrack.analyze() 在【各进程内】算,
      本文件只吃它 dump 出来的标量 => 「不碰裁判」这条硬约束在这里也守得住。
  🚨 判据数值与问句 = bench_drifttrack.py docstring 的 K‴ 段 + OPTLOG §5.2i附3 跑前写死版,
      跑完不许改: TARGET / T_LO / N_MIN / NP_MIN / TAX_MIN / T_K 取中位、dec 取均值 都在那边定死。

  【修⑱ = r5 泄露登记的直接后果, 2026-11-24】跑前纪律写的是「前置不过 => 三条问句不许读」, 但 r5 之前
      代码层没拦住: 组合器在 c‴ 已红时照样把跨进程 T 表(决策/发、T、延迟、净收益)整个印了出来, 于是
      K/K'/K''/K''' 四棵树全部被自己的打印消耗过。修⑱ = 把这条纪律落到代码: 前置(a'''~c''')任一红
      => 跨进程 T 表只印表头 + ██, 三条问句不参与求值(落点仍按跑前写死的路径判 M0)。
      🚨 本修只动【打印与求值顺序】, 不动任何判据数值/装置/裁判: TARGET/T_LO/N_MIN/NP_MIN/TAX_MIN/BLK
      /主尺/★2 口径/配对算法 一字未改。

【臂身份诊断(旁证, 不进判据)为什么值得单独印】
  r4 的真因假设 = 「每台引擎实例在建臂/分配时抽到一个固定偏移」。这个假设自带一个可证伪的推论:
  同一对同配置引擎(A2 vs A1)的 Δ% 只该取决于【谁先建臂】这件事的随机实现, 而不该跟着身份走。
  => 把每进程的 (A2/A1, A3/A1) 与「它比 A1 早建/晚建」并排列出来, 就能直接看:
     先建组与后建组的平均 Δ% 若分得开 => 偏移确实吃建臂位置(B1 修法有效);
     分不开(两组都 ~0 附近散开) => 偏移是实例固有、与顺序无关 => c‴ 必红 = 装置定案不可用。
"""
import os, sys, json, glob, math, statistics

W = 158


def geo(v):
    v = [float(q) for q in v if q == q and q > 0]
    return math.exp(sum(math.log(q) for q in v) / len(v)) if v else float("nan")


def med(v):
    v = [float(q) for q in v if q == q]
    return statistics.median(v) if v else float("nan")


def mean(v):
    v = [float(q) for q in v if q == q]
    return sum(v) / len(v) if v else float("nan")


def rngi(v):
    v = [int(q) for q in v if q == q]
    return f"[{min(v)}~{max(v)}]" if v else "[nan]"


def rng(v):
    v = [float(q) for q in v if q == q]
    return f"[{min(v):.2f}~{max(v):.2f}]" if v else "[nan]"


def f2(x, w=7, d=2):
    return f"{x:{w}.{d}f}" if x == x else " " * (w - 3) + "nan"


args = [a for a in sys.argv[1:]] or ["/tmp/dt_r5"]
paths = []
for a in args:
    paths += sorted(glob.glob(os.path.join(a, "*.json"))) if os.path.isdir(a) else sorted(glob.glob(a))
P = []
for p in paths:
    with open(p, encoding="utf-8") as fh:
        P.append((os.path.basename(p), json.load(fh)))

print("=" * W)
print(f"§5.2i附3 B1 换装置轮 K‴ 组合器(判据跑前写死在 bench_drifttrack.py docstring 与 OPTLOG §5.2i附3,"
      f" 跑完不许改) —— 求值序 M0‴ -> K1‴ -> K2‴ -> K3‴")
print("=" * W)
if len(P) < 1:
    print(f"🚨 没读到任何 dump JSON(参数={args}) => K‴ 无法求值, 本刀作废")
    sys.exit(2)

cfg = P[0][1]["cfg"]
ARMS = cfg["ARMS"]; KIND = cfg["KIND"]; POL = cfg["POL"]
nP = int(cfg["NBLK"]); BLK = int(cfg["BLK"])
TARGET = float(cfg["TARGET"]); T_LO = float(cfg["T_LO"]); N_MIN = int(cfg["N_MIN"])
NP_MIN = int(cfg["NP_MIN"]); TAX_MIN = float(cfg["TAX_MIN"]); PRED = cfg.get("PREDICT4", "M0")
NULL = [a for a in ARMS if KIND[a] == "null"]; DYN = [a for a in ARMS if KIND[a] == "dyn"]
REAL = [a for a in ARMS if KIND[a] != "null"]
NP = len(P)
M = [d["main"] for _, d in P]; G = [d["gates"] for _, d in P]; ALT = [d["alt"] for _, d in P]

diff = [nm for nm, d in P if {k: d["cfg"].get(k) for k in ("TARGET", "T_LO", "N_MIN", "NP_MIN",
                                                            "RULER", "BLK", "NBLK", "BS", "GC")} !=
        {k: cfg.get(k) for k in ("TARGET", "T_LO", "N_MIN", "NP_MIN", "RULER", "BLK", "NBLK", "BS", "GC")}]
print(f"      读到 NP={NP} 个进程 dump(NP_MIN={NP_MIN}) | 主尺={cfg['RULER']}(一格 {cfg['MS_PER_UNIT']})"
      f" | 装置 {len(ARMS)} 臂 x {nP} 块 x {BLK} 帧 | 判据 TARGET={TARGET}% T_LO={T_LO} N_MIN={N_MIN}"
      f" TAX_MIN={TAX_MIN}%")
print(f"      🚨 每格发数仍是 n=48 => 本刀【不是】第五次加发数; 新加的是【进程】这个独立重复层 + 建臂顺序随机化")
if diff:
    print(f"      🚨 这些进程的 cfg 与第一个不一致(装置被中途改过?): {diff}")
print(f"      🚨 三棵已消耗的树(K r1 c299b2b9 / K′ r3 96ecadf1 / K″ r4 48499173)在跨进程层【不重算也不复读】"
      f" => 唯一新落点 = 下面这棵 K‴")

# ---------------------------------------------------------------- 逐进程一览
print(f"\n{'进程':<6}{'seed':>5}  {'建臂序(只随机这个)':<32}{'n':>4}{'底_pair%':>9}{'底_序%':>8}{'门槛%':>8}"
      f"{'闸门':>6}{'C1★2':>6}{'O1★2':>6}{'头载':>6}{'尾载':>6}{'墙钟s':>7}")
for i, (nm, d) in enumerate(P):
    m = M[i]; g = G[i]
    st2 = d["main"]["STAR2"]
    print(f"{d['proc']:<6}{(d['seed'] or '-')!s:>5}  {','.join(d['build_order']):<32}"
          f"{m['nn']:>4}{f2(m['bp'], 9)}{f2(m['bs'], 8)}{f2(m['thr'], 8)}"
          f"{('全绿' if g['ok'] else '有红'):>6}{sum(st2.get('C1', [])):>4}/{nP}{sum(st2.get('O1', [])):>4}/{nP}"
          f"{d['head']['loadavg']:>6}{d['loadavg_end']:>6}{d['wall_s']:>7.0f}")
bp = [m["bp"] for m in M]; bs = [m["bs"] for m in M]; nn = [int(m["nn"]) for m in M]
mbp, mbs = med(bp), med(bs)
_fs = [v for v in (mbp, mbs) if v == v]
thr3 = max((2.0 * max(_fs)) if _fs else 0.0, TAX_MIN)
print(f"\n  跨进程 底_pair = {rng(bp)} 中位 {mbp:.2f}%   底_序 = {rng(bs)} 中位 {mbs:.2f}%   "
      f"=> 门槛‴ = max(2xmax(中位底_pair,中位底_序), TAX_MIN) = {thr3:.1f}%   每进程 n={rngi(nn)}")

# ---------------------------------------------------------------- 跨进程配对(同格几何均值)
XP = {a: [geo([m["PAL"][a][k] for m in M]) for k in range(nP)] for a in ARMS}
print(f"\n  跨进程配对(同格各进程配对的几何均值; Δ% = 相对 A1) 主尺={cfg['RULER']}")
print("  " + f"{'臂/策略':<14}" + "".join(f"{'blk' + str(k):>16}" for k in range(nP)))
for a in ARMS:
    print(f"  {a + '=' + POL[a]:<14}" + "".join(f"{XP[a][k]:>9.4f}({100 * (XP[a][k] - 1):+5.1f}%)" for k in range(nP)))


def _maj(a, k=None):
    """有多少个进程在【自己进程内】判该臂(块)过 ★2 且配对<1(k=None 时要求三块全过)。"""
    c = 0
    for m in M:
        s2 = m["STAR2"].get(a, [])
        pl = m["PAL"].get(a, [])
        if k is None:
            ok = len(s2) == nP and all(s2)
        else:
            ok = bool(k < len(s2) and s2[k] and k < len(pl) and pl[k] < 1.0)
        c += 1 if ok else 0
    return c


M_maj = math.ceil(2 * NP / 3)
tops = [g["tops"] for g in G]
a3 = all(g["ok"] for g in G)
b3 = all(g["blocks_diff"] for g in G) and len({tuple(tuple(t) for t in tp) for tp in tops}) == 1
c3 = (NP >= NP_MIN) and all(v >= N_MIN for v in nn) and mbp == mbp and mbp <= TARGET
d3 = ("C1" in ARMS) and all(abs(XP["C1"][k] - 1) * 100 > thr3 for k in range(nP)) and _maj("C1") >= M_maj
e3 = ("O1" in ARMS) and any(XP["O1"][k] < 1.0 and abs(XP["O1"][k] - 1) * 100 > thr3 and _maj("O1", k) >= M_maj
                     for k in range(nP))
K3 = [("a‴ 每个进程闸门全绿", a3, f"全绿进程={sum(1 for g in G if g['ok'])}/{NP}"
       f" 红闸门进程={[d[1]['proc'] for d, gg in zip(P, G) if not gg['ok']] or '无'}"
       f" | O1逐块放对={[bool(g['ock_all_ok']) for g in G]}"),
      ("b‴ 每进程上帝成员两两不同", b3, f"各进程上帝成员={tops[0]}(逐进程同) 进程数={NP}"),
      (f"c‴ 中位底_pair<=TARGET 且 每进程 n>=N_MIN", c3,
       f"中位底_pair={mbp:.2f}% TARGET={TARGET}% NP={NP}/{NP_MIN} n={rngi(nn)}/{N_MIN}"),
      ("d‴ C1 跨进程每块过 门槛‴ 且多数进程 ★2", d3,
       f"C1 跨进程Δ%={[round(100 * (XP['C1'][k] - 1), 1) for k in range(nP)]} 门槛‴={thr3:.1f}% "
       f"三块全过★2的进程={_maj('C1')}/{NP}(需>={M_maj})"),
      ("e‴ O1 跨进程至少一块<1 且过 门槛‴", e3,
       f"O1 跨进程配对={[round(XP['O1'][k], 4) for k in range(nP)]} "
       f"逐块满足『跨进程+{M_maj}进程★2』={[bool(XP['O1'][k] < 1.0 and abs(XP['O1'][k] - 1) * 100 > thr3 and _maj('O1', k) >= M_maj) for k in range(nP)]}")]
print()
for nm, ok, note in K3:
    print(f"  {'OK ' if ok else '🚨 '} {nm:<38} {note}")
# 修⑱: 前置 = a‴/b‴/c‴(跑前写死的三条前置); 任一红 => 后三条问句不许读、也不许被印出来
PRE = a3 and b3 and c3
print(f"  门槛: TARGET={TARGET}% T_LO={T_LO} N_MIN={N_MIN} NP_MIN={NP_MIN} TAX_MIN={TAX_MIN}%"
      f" 块长 BLK={BLK} 主尺={cfg['RULER']} NP={NP}")

# ---------------------------------------------------------------- 跨进程 T 与账
Tm = {a: med([m["T_R"].get(a, float("nan")) for m in M]) for a in ARMS}
Tmean = {a: mean([m["T_R"].get(a, float("nan")) for m in M]) for a in ARMS}
Dm = {a: mean([m["DEC"].get(a, 0.0) for m in M]) for a in ARMS}
print(f"\n  跨进程 T(T 取中位, 决策/发 取均值; 🚨 判读只用 blk>=1, blk0 的 T 是噪声放大)")
_HD = "  " + f"{'臂':<6}{'决策/发':>9}{'T 中位':>9}{'T 均值':>9} {'逐进程 T':<34}{'延迟中位(帧)':>14}{'净收益ms/发':>12}"
if not PRE:
    print("  🚨【修⑱ 遮蔽】前置 a‴~c‴ 未过 => 三条问句不许读: 本表的 决策/发、T、反应延迟、净收益 数值"
          f"一律不印(先例: K r1 `c299b2b9` / K′ r3 `96ecadf1` / K″ r4 `48499173` 与本刀 K‴ 全被自己的打印消耗过)")
    print(_HD)
    for a in ARMS:
        print(f"  {a:<6}" + "██".rjust(9) + "██".rjust(9) + "██".rjust(9) + " " + "██".ljust(34)
              + "██".rjust(14) + "██".rjust(12))
else:
    print(_HD)
    for a in ARMS:
        per = " ".join(f"{m['T_R'].get(a, float('nan')):.2f}" for m in M)
        print(f"  {a:<6}{Dm[a]:>9.2f}{f2(Tm[a], 9)}{f2(Tmean[a], 9)}{per:<34}"
              f"{f2(mean([m['DLBmed'].get(a, float('nan')) for m in M]), 14, 0)}"
              f"{f2(mean([m['NETG'].get(a, float('nan')) for m in M]), 12, 1)}")


def _k123(prf="K‴"):
    tracked = [a for a in DYN if Dm[a] >= 1.0 and Tm[a] == Tm[a] and Tm[a] >= T_LO]
    if tracked:
        a = max(tracked, key=lambda q: Tm[q])
        return prf + "1", (f"{a} 决策/发={Dm[a]:.2f} >=1 且 跨进程 T={Tm[a]:.2f} >= T_LO={T_LO}"
                           f"(逐进程 T={[round(m['T_R'].get(a, float('nan')), 2) for m in M]})"
                           f" => 自适应真能追块内漂移、吃到上限 >= {T_LO:.0%} => dyn 的「自适应」第一次可记账")
    firing = [a for a in DYN if Dm[a] > 0]
    if firing:
        a = max(firing, key=lambda q: Tm[q] if Tm[q] == Tm[q] else -9.0)
        return prf + "2", ("开火的 dyn 臂=" + str(firing) + "(决策/发 " + "/".join(f"{Dm[q]:.2f}" for q in firing)
                           + f") 但无一臂跨进程 T >= {T_LO}(最好 {a}: T="
                           + (f"{Tm[a]:.2f}" if Tm[a] == Tm[a] else "nan") + ") => 会开火但追不上"
                           + " => 判据参数[cooldown/max_swaps/window/BLK]要重标定, 那是另一刀")
    return prf + "3", ("块内真漂移下所有 dyn 臂跨进程决策数 = 0 => 「自适应」主张作废,"
                       "全部旧证据重登记为「开局一次性放置」=> 判据要重造不是调参")


V, why = _k123() if PRE else ("M0‴", "前置不过 => 挂账【第四次】, 三条问句不许读(修⑱: 问句未参与求值)")
bad = [f"{n} => {note}" for n, ok, note in K3 if not ok]
if bad:
    V, why = "M0‴", "前置不过 => 挂账【第四次】, 三条问句不许读: " + " ; ".join(bad)
print("\n  => 落点: " + V + "    理由: " + why)
print(f"     事前预测(跑前写死) = {PRED}(机制假设 = 偏移属于【第 k 次分配出来的那块内存】而非【A2 这个身份】:"
      f" 建臂序在 r2/r3/r4 三轮逐字相同而 底_pair = 0.97%/4.18%/8.41% 三轮三个数 => 洗牌只把偏移在身份间"
      f" 重分配, 而 底_pair = max(|A2/A1-1|,|A3/A1-1|) 读的是三个 iid 偏移的极差, 洗牌不改它的分布"
      f" => 中位底_pair 仍 ~5~8% => c‴ 红; 竞争假设 = K1‴/K2‴【偏移其实与建臂位置相关 => 跨进程平均掉"
      f" => c‴ 绿 => 三条问句第一次有资格被读, 那时 r2/r3/r4 的 H1 T=0.86→0.84→0.64 跨轮稳 => K1‴ 概率略高】"
      f"; 本轮 NP={NP} n={rng(nn)} BLK={BLK} 帧) => " + ("兑现" if V.replace("‴", "") == PRED else "打脸(如实记账)"))

# ---------------------------------------------------------------- 臂身份诊断(旁证, 不进判据)
print("\n" + "=" * W)
print("臂身份诊断【旁证, 不进判据】= r4 定性结论的直接可证伪检验: Δ% 到底跟【身份】还是跟【建臂位置】走")
print("=" * W)
print("  🚨 下面甲/乙/丙 = 【组合器诊断】三块, 与 r4 定稿的甲/乙/丙/丁【四证据】不是同一套标号, 别混。\n")
print(f"  甲 同配置三台(A1/A2/A3 static_A)的两两极差 —— 与「谁是分母」无关的一把尺:")
print("      " + f"{'进程':<6}{'A2/A1%':>9}{'A3/A1%':>9}{'A2/A3%':>9}{'两极最大%':>11}   逐块 A2/A1 与 A3/A1")
for i, (nm, d) in enumerate(P):
    m = M[i]
    d2 = mean([100 * (m["PAL"]["A2"][k] - 1) for k in range(nP)])
    d3v = mean([100 * (m["PAL"]["A3"][k] - 1) for k in range(nP)])
    d23 = mean([100 * abs(m["PAL"]["A2"][k] / m["PAL"]["A3"][k] - 1) for k in range(nP)])
    mx = max(abs(d2), abs(d3v), d23)
    per = " ".join(f"{k}:{100 * (m['PAL']['A2'][k] - 1):+.1f}/{100 * (m['PAL']['A3'][k] - 1):+.1f}" for k in range(nP))
    print("      " + f"{d['proc']:<6}{d2:>+9.1f}{d3v:>+9.1f}{d23:>9.1f}{mx:>11.1f}   {per}")
allv = []
for i, (nm, d) in enumerate(P):
    for a in ("A2", "A3"):
        if a in M[i]["PAL"]:
            allv.append((d["proc"], a, mean([100 * (M[i]["PAL"][a][k] - 1) for k in range(nP)]),
                         d["build_order"].index(a) - d["build_order"].index("A1")))
grp = {"先于A1建臂": [v for _, _, v, dp in allv if dp < 0], "后于A1建臂": [v for _, _, v, dp in allv if dp > 0]}
print(f"  乙 「谁先建臂」分组检验(n={len(allv)} = 2 对 x {NP} 进程; 每格 = 跨块平均 Δ%, 单位 %):")
for kk, vv in grp.items():
    print(f"      {kk:<12} n={len(vv):<3} 平均 {mean(vv):+6.2f}%  极差 {rng(vv)}   逐样本="
          + " ".join(f"{v:+.1f}" for v in vv))
print("      逐样本明细(进程/臂/Δ%/与 A1 的建臂位差): " + " ".join(f"{p}{a}({dp:+d})={v:+.1f}"
                                                               for p, a, v, dp in allv))
_sp = [abs(v) for _, _, v, _ in allv]
print(f"  丙 |Δ%| 的跨进程分布 = 平均 {mean(_sp):.2f}% 中位 {med(_sp):.2f}% 最大 {max(_sp):.2f}%"
      f"(n={len(_sp)}) —— 这就是本机「同配置实例」的偏移量级; 中位底_pair={mbp:.2f}% 与它同量级 =>"
      f" 底 是【实例偏移】而不是【抽样】的结论是否成立, 看这一行与甲表")
print("""
K‴ 判读提示: ① c‴ 绿 = 本机这套尺【可用】定案 => K1‴/K2‴/K3‴ 的落点第一次可以进正文;
              c‴ 红 = 偏移不吃建臂序 => 本机「7 臂同进程轮转 + BLK=24 + TARGET=5%」装置定案【不可用】
              (挂账第四次) => 下一刀只能换物理装置/语料(真漂移语料/换机器/memtest)或走 C 兜底收窄主张;
           ② 🚨 不许第五次加发数、不许下调 TARGET/T_LO/N_MIN/NP_MIN/thr_rel、不许改 ★/★2 口径、旧表一张不动;
           ③ 甲乙丙三块诊断是本刀真正产出, 但【不进判据】: 判据只有 a‴~e‴ 与三条问句;
           ④ 净收益/真净收益仍只作旁证(修⑩/修⑪ 单位口径不变, 主尺=sum => 一格 ms/块);
           ⑤ swapper 默认仍是 "sync"(升默认 = §5.2h附3 要用户点头), thr_rel=0.05 未动, 裁判未动。""")

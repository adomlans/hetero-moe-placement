#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""乙-1f · 双峰归因探针: 同一臂的 N 发分成两簇时, 高簇那几发到底是【桌面在抢核】还是【两种驻留稳态】。

【为什么还要这一刀】
  乙-1d 立了一条口径缺陷: 两序底看不见【双峰】—— 某一臂的 N 发分成两簇时 p50 落哪个簇是抛硬币,
  于是 乙-1c 的 C bs=4 两个星标作废、乙-1e 的 C bs=4(+13.0%)与 C bs=32 被判【不可判】。
  ⇒ 整个 乙-1 系列现在只挡在这一件事上: 双峰是什么。
  三次观测(乙-1c C bs=4 pair_last、乙-1d gemm C bs=32、乙-1e C bs=4/32/128)形态一致:
  【少数发站高簇、两簇心间隔 8%~23%】, 且只在配置 C(CPU 驻字节 > 0.75xL3)出现, A 段一次没有。
  🚨 但现在连"高簇发生在第几发"都看不见 —— 旧表的【轮间】列打印的是 sorted() 之后的发, 时序丢了。

【这一刀只问一件事: 高簇那几发, 机器环境到底一不一样】每一发【前后】各取一次环境快照:
  busy%   = /proc/stat 的 cpu 行在窗口内的 busy/(busy+idle) 差 —— 全机 12 核有多少时间在干活
  平均核数 = busy 增量(us) / 墙上时间(us) —— 这一发期间平均占几个核(6 线程满载 + GPU 轮询 ≈ 6~7)
  Δctxt   = /proc/stat 的 ctxt 行差 —— 全机上下文切换次数(千次)
  Δnv     = /proc/self/status 的 nonvoluntary_ctxt_switches 差 —— 【我们自己】被抢下核的次数
  🚨 本机【没有】/sys/devices/system/cpu/cpu0/cpufreq/time_in_state(实测该文件不存在), 所以只能用
     busy 差算驻留, 不能引 time_in_state; 单次快照约 100us, 每发只读 2 次, 对 ms 级的发无影响。
  跑完用 bimodal()(阈值 BIM_THR, 与 乙-1c/乙-1d/乙-1e 同一把尺)分簇, 再比两簇的三项中位数与重叠性。

【预设判决树】(先写死, 跑完不许改; 只在【双峰臂】上评估)
  A) 环境: 任一双峰臂的 高簇/低簇 三项中位之比 >= A_RATIO=1.5, 或两簇的【平均核数】分布不重叠
     => 高簇 = 桌面/后台在抢核, 不是硬件稳态。判读规则从 p50 改成【取众数簇】;
        乙-1c/乙-1d/乙-1e 三张归档按众数簇重读, 并给 C bs=4 定值;
        🚨 代价远大于 B: 意味着以前所有 C 段结论都被桌面污染过。
  B) 机制: 全部双峰臂的三项比值都落在 [0.8, 1.25], 且双峰只出现在 CPU 驻字节 > 0.75xL3 的配置 C 格
     => 高簇 = 两种缓存/放置稳态, 与桌面无关。写进边界【这类格的延迟是双峰的, p50 不可用】,
        放置规则按【高簇】(保守侧)付钱; 旧表数值不动, 只加双峰标注。
  C) 短序列假象: ROUNDS 提到 21 后【一个双峰臂都没有】(高簇 0 发)
     => 之前那几次双峰是 7 发太短撞上的发数内噪声。只要提高 ROUNDS, 代码与旧表都不动。
  D) 都不成立 => 挂账, 乙-1 系列按【比值稳、绝对值带 4%~6% 双峰不确定度】结项。
【事前预测(可证伪, 跑前写死)】押 B。
  理由(假设不是结论): 双峰三次都只在配置 C 出现(A 段 0/12), 且总是少数发站高簇 —— 形态像两种驻留
  稳态而不是随机抢核; P0-④a附 那次【轮内 σ=1% / 跨进程差 28.6%】是同形态。
  若 A 成立代价远大于 B(以前所有 C 段结论都要重读), 所以判据必须在跑之前就写死。
【交付边界】产出只有三种: ①每发时序 + 同发环境量(补上丢掉的时序); ②双峰臂两簇对照(三项中位 +
  是否重叠); ③按预设树自动印出的 A/B/C/D 落点 + 【众数簇 p50 vs p50】表。不改任何旧表与生产代码。
"""
import os, sys, time, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
# 🚨 分簇的尺子与 build/stat 全部复用 乙-1c 那一份 => 本探针切出来的簇与已入库表是同一刀
from bench_pairtax import build, stat, bimodal, TAX_MIN, BIM_THR, MODE, NMAT
from bench_swiglu import D, E, N, K, CFG, BS_LIST, FIXGC, gc_for, hits_cum

ROUNDS = int(os.environ.get("ROUNDS", 21))                 # 每臂几发; 7 发是 C 判据反例, 默认 21
CELL_BS = [int(v) for v in os.environ.get("BIM_BS", "4,32,128").split(",") if v.strip()]
A_RATIO = float(os.environ.get("A_RATIO", 1.5))            # 判决 A: 高簇/低簇 中位之比门槛
B_LO, B_HI = 0.8, 1.25                                     # 判决 B: 三项比值都须在带内
KINDS = ("swiglu", "gemm")
CLK = os.sysconf("SC_CLK_TCK")                             # /proc/stat 单位 = 1/CLK 秒(本机 100)
HITS = {}


def snap():
    """一次环境快照: /proc/stat(cpu 行 + ctxt + procs_running) + 本进程 ctxt 计数 + cpu0 当前倍频。"""
    txt = open("/proc/stat").read().split("\n")
    f = [int(t) for t in txt[0].split()[1:]]
    busy = (sum(f[:8]) - f[3] - f[4]) * 1000000 // CLK      # user+nice+sys+irq+sirq+steal, 单位 us
    idle = (f[3] + f[4]) * 1000000 // CLK                   # idle+iowait
    ctxt = prc = 0
    for ln in txt[1:]:
        if ln.startswith("ctxt "):
            ctxt = int(ln.split()[1])
        elif ln.startswith("procs_running"):
            prc = int(ln.split()[1])
        elif ctxt and prc:
            break
    vc = nv = 0
    for tid in os.listdir("/proc/self/task"):          # 🚨 必须逐线程求和: /proc/self/status 只报主线程,
        try:                                           #    OMP 那 12 个工作线程被抢占几次它看不见(nv 恒 0)
            for ln in open("/proc/self/task/" + tid + "/status"):
                if ln.startswith("voluntary_ctxt"):
                    vc += int(ln.rsplit("\t", 1)[1])
                elif ln.startswith("nonvoluntary_ctxt"):
                    nv += int(ln.rsplit("\t", 1)[1])
        except OSError:
            pass                                       # 线程正在退出 —— 少算一个不影响差值
    try:
        mhz = float(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) / 1000.0
    except OSError:
        mhz = 0.0
    return dict(t=time.perf_counter(), busy=busy, idle=idle, ctxt=ctxt, prc=prc, vc=vc, nv=nv, mhz=mhz)


def sig(a, b):
    """两发快照之间 = 一发的窗口 => 三个可与双峰簇对照的环境量。"""
    wall = max((b["t"] - a["t"]) * 1e6, 1.0)
    db, di = b["busy"] - a["busy"], b["idle"] - a["idle"]
    return dict(pct=100.0 * db / max(db + di, 1), cores=db / wall,
                ctxt=(b["ctxt"] - a["ctxt"]) / 1000.0, nv=b["nv"] - a["nv"],
                vc=b["vc"] - a["vc"], mhz=b["mhz"], wall=wall / 1000.0, prc=b["prc"])


def split(v):
    """与 bimodal() 同一把尺, 但返回【发序号】级别的归属(旧表只有 sorted 后的值, 归不了因)。"""
    s = sorted((round(float(t), 3), i) for i, t in enumerate(v))
    if len(s) < 6 or s[0][0] <= 0:
        return None
    cand = [i for i in range(len(s) - 1) if 2 <= i + 1 <= len(s) - 2]
    if not cand:
        return None
    i = max(cand, key=lambda k: s[k + 1][0] / s[k][0])
    gap = (s[i + 1][0] / s[i][0] - 1) * 100
    if gap <= BIM_THR:
        return None
    lo, hi = s[:i + 1], s[i + 1:]
    ref = bimodal(v)
    assert ref is not None and abs(ref[4] - gap) < 1e-6, "本探针的分簇与 bimodal() 不是同一把尺"
    return dict(lo=[j for _, j in lo], hi=[j for _, j in hi],
                lo_p=statistics.median([t for t, _ in lo]), hi_p=statistics.median([t for t, _ in hi]),
                gap=gap)


def timed_solo(m, x):
    vs, ss = [], []
    for _ in range(ROUNDS):
        a = snap()
        vs.append(H.bench(m, x, True, MODE)[0])
        ss.append(sig(a, snap()))
    return vs, ss


def timed_rot(m_other, m_meas, x):
    """轮转臂 = one() 的处境(另一档每发先跑一整遍), 快照窗口覆盖【另一档 + 被测量】整发。"""
    vs, ss = [], []
    for _ in range(ROUNDS):
        a = snap()
        H.bench(m_other, x, True, MODE)
        vs.append(H.bench(m_meas, x, True, MODE)[0])
        ss.append(sig(a, snap()))
    return vs, ss


SIGS = (("平均核数", "cores"), ("busy%", "pct"), ("Δctxt", "ctxt"))


def med(v):
    return statistics.median(v)


def cell(bs, label, hot, gc, rows):
    torch.manual_seed(42)
    x = torch.randn(bs, D).cuda()
    refs, pools = {}, {}
    for kind in KINDS:                              # 各建一次: 闸门参考输出 + CPU 池字节
        m = build(kind, gc, hot)
        refs[kind] = m(x, async_mode=True, mode=MODE).clone()
        pools[kind] = (E - gc) * m.expert_bytes()
        del m; torch.cuda.empty_cache()
    m = build("gemm", gc, hot); eb = m.expert_bytes(); del m; torch.cuda.empty_cache()
    sw_mb = eb * NMAT["swiglu"] // NMAT["gemm"]
    pair_mb = (pools["swiglu"] + pools["gemm"]) / 2**20
    tail = "放对(CPU 腿几乎没活)" if set(hot) < set(range(gc)) else "故意放错(打 CPU 腿)"
    print("\n" + "=" * 104)
    print(f"配置 {label} | bs={bs:>4} | gc={gc} | 每臂 {ROUNDS} 发 | 每专家 gemm {eb / 2**20:.2f}MB / "
          f"swiglu {sw_mb / 2**20:.2f}MB | 轮转 CPU 池 {pair_mb:.1f}MB  solo 池 sw "
          f"{pools['swiglu'] / 2**20:.1f}MB ge {pools['gemm'] / 2**20:.1f}MB "
          f"(L3 {H.L3 >> 20}MB, 0.75xL3={0.75 * H.L3 / 2**20:.1f}MB) | hot={hot} ← {tail}")
    print("=" * 104)

    def rec(tag, name, v, s, pool_mb, gate=""):
        """发序时序 + 两簇对照。🚨 这里必须按【发序】打印 —— 旧表 sorted() 把时序丢了, 双峰无法归因。"""
        p, sd, mn = stat(v)
        sp = split(v)
        print(f"      {name:<14} p50={p:.3f} σ={sd:.0f}% min={mn:.3f}"
              + (f" 双峰: 低{len(sp['lo'])} 发≈{sp['lo_p']:.3f} / 高{len(sp['hi'])} 发≈{sp['hi_p']:.3f} "
                 f"间隔 {sp['gap']:.0f}% 高簇占比 {len(sp['hi'])}/{ROUNDS} 众数簇 p50="
                 f"{(sp['hi_p'] if len(sp['hi']) > len(sp['lo']) else sp['lo_p']):.3f}" if sp else " 无双峰"))
        for i0 in range(0, ROUNDS, 7):
            print("        时序 " + "  ".join(
                f"#{j + 1} {v[j]:.3f}[{s[j]['pct']:.0f}%|{s[j]['cores']:.1f}核|{s[j]['ctxt']:.1f}k"
                f"|nv{s[j]['nv']:+d}|{s[j]['mhz'] / 1000:.2f}G]"
                for j in range(i0, min(i0 + 7, ROUNDS))))
        r = dict(label=label, bs=bs, tag=tag, p=p, bi=None, pool=pool_mb)
        if sp:
            lo, hi = sp["lo"], sp["hi"]
            ratios, txt, qs = [], [], {}
            for nm, key in SIGS:
                a, b = med([s[j][key] for j in lo]), med([s[j][key] for j in hi])
                q = (b / a) if a > 0 else (99.0 if b > 0 else 1.0)
                ratios.append(q); qs["q_" + key] = q
                txt.append(f"{nm} {a:.2f} vs {b:.2f} ({q:.2f}x)")
            lo_c = [s[j]["cores"] for j in lo]; hi_c = [s[j]["cores"] for j in hi]
            ov = min(hi_c) <= max(lo_c)
            print("        两簇环境(低 vs 高)  " + " | ".join(txt))
            print(f"        旁证 Δnv(我们全进程被抢占次数) 低={med([s[j]['nv'] for j in lo]):+.0f} "
                  f"高={med([s[j]['nv'] for j in hi]):+.0f}  Δvc 低={med([s[j]['vc'] for j in lo]):+.0f} "
                  f"高={med([s[j]['vc'] for j in hi]):+.0f}  cpu0 倍频 低={med([s[j]['mhz'] for j in lo]):.0f}MHz "
                  f"高={med([s[j]['mhz'] for j in hi]):.0f}MHz   "
                  f"平均核数分布 低[{min(lo_c):.2f},{max(lo_c):.2f}] 高[{min(hi_c):.2f},{max(hi_c):.2f}] "
                  f"=> {'重叠' if ov else '🚨不重叠'}   帧长 {med([s[j]['wall'] for j in lo]):.0f}ms vs "
                  f"{med([s[j]['wall'] for j in hi]):.0f}ms")
            r["bi"] = dict(lo=len(lo), hi=len(hi), gap=sp["gap"], lo_p=sp["lo_p"], hi_p=sp["hi_p"],
                           rmin=min(ratios), rmax=max(ratios), overlap=ov, **qs)
        rows.append(r)
        return r

    ms = {k: build(k, gc, hot) for k in KINDS}               # 轮转段(两档都在场)
    rec(("pair", "swiglu"), "轮转 swiglu", *timed_rot(ms["gemm"], ms["swiglu"], x), pair_mb)
    rec(("pair", "gemm"), "轮转 gemm", *timed_rot(ms["swiglu"], ms["gemm"], x), pair_mb)
    del ms; torch.cuda.empty_cache()
    for kind in KINDS:                                            # solo 段(一次只一档活着)
        m = build(kind, gc, hot)
        rec(("solo", kind), f"solo {kind}", *timed_solo(m, x), pools[kind] / 2**20)
        del m; torch.cuda.empty_cache()
    ms = {k: build(k, gc, hot) for k in KINDS}
    d = {k: (ms[k](x, async_mode=True, mode=MODE) - refs[k]).abs().max().item() for k in KINDS}
    del ms; torch.cuda.empty_cache()
    ok = "OK" if max(d.values()) == 0 else "🚨 建/删历史改写了数值, 本格全部不可比"
    print(f"  闸门 轮转段 + solo 段 + 重建档 之后: swiglu 逐位差={d['swiglu']:.9f} "
          f"gemm 逐位差={d['gemm']:.9f} {ok}")
    return ok == "OK"

if __name__ == "__main__":
    print(f"[机器状态] {H.machine_state()}")
    print(f"[乙-1f 双峰归因探针] D={D} E={E} N={N} top_k={K} | bs={CELL_BS} | 臂=轮转 swiglu/gemm + "
          f"solo swiglu/gemm | 每臂 {ROUNDS} 发 x (5 暖机 + 30 计时 x 3 reps), 每发前后各取一次 /proc 快照 | "
          f"分簇尺 BIM_THR={BIM_THR}%(与 乙-1c/乙-1d/乙-1e 同一把) | 判决门槛 A_RATIO={A_RATIO} B带=[{B_LO},{B_HI}]")
    for lb, hot in CFG:
        HITS[lb] = hits_cum(hot, "swiglu")
        print(f"  累计命中率 {lb}(swiglu): {HITS[lb]}")
    rows, gates = [], []
    for bs in CELL_BS:
        for label, hot in CFG:
            gc = FIXGC or gc_for("swiglu", HITS[label])[0]        # 🚨 与已发布表同一根 gc 桩(钉 swiglu 规则)
            gates.append(cell(bs, label, hot, gc, rows))

    print("\n" + "=" * 118)
    print("汇总: 每个臂的双峰归属 + 两簇环境之比(高/低)。三项 = 平均核数 / busy% / Δctxt; nv(被抢占)只作旁证")
    print("      众数簇 = 发数多的那一簇的心; 高簇占比 = 未来【取众数簇】判读要用的数")
    print("=" * 118)
    print(f"{'配置':<5}{'bs':>5}{'臂':>14}{'发':>5}{'p50':>8}{'众数簇':>8}{'高簇心':>8}{'高簇':>7}"
          f"{'间隔':>8}{'核数比':>8}{'busy比':>8}{'ctxt比':>8}{'核数重叠':>10}{'池MB':>7}")
    for r in rows:
        b = r["bi"]
        cols = [f"{r['label']:<5}{r['bs']:>5}{r['tag'][0] + ' ' + r['tag'][1]:>14}{ROUNDS:>5}{r['p']:>8.3f}"]
        if b:
            mode_p = b["hi_p"] if b["hi"] > b["lo"] else b["lo_p"]
            cols.append(f"{mode_p:>8.3f}{b['hi_p']:>8.3f}{('%d/%d' % (b['hi'], ROUNDS)):>7}")
            cols.append(f"{('%.0f%%' % b['gap']):>8}")
            for key in ("cores", "pct", "ctxt"):
                q = b["q_" + key]
                cols.append(f"{('%.2fx' % q):>8}")
            cols.append(f"{('重叠' if b['overlap'] else '不重叠🚨'):>10}")
        else:
            cols.append(f"{r['p']:>8.3f}{0.0:>8.3f}{'-':>7}{'-':>8}")
            cols.append(f"{'-':>8}{'-':>8}{'-':>8}{'-':>10}")
        cols.append(f"{r['pool']:>7.1f}")
        print("".join(cols))

    bim = [r for r in rows if r["bi"]]
    l3mb = 0.75 * H.L3 / 2**20
    a_arms = [r for r in bim if r["bi"]["rmax"] >= A_RATIO or not r["bi"]["overlap"]]
    b_pass = bool(bim) and all(B_LO <= r["bi"]["rmin"] and r["bi"]["rmax"] <= B_HI for r in bim)
    b_place = bool(bim) and all(r["label"] == "C" and r["pool"] > l3mb for r in bim)
    worst = max((r["bi"]["rmax"] for r in bim), default=0.0)
    nro = sum(1 for r in bim if not r["bi"]["overlap"])
    print("\n" + "=" * 118)
    print("预设判决树落点(跑前写死在 docstring, 跑完不许改)")
    print("=" * 118)
    thr_a = "三项最大比 >= %.2f 或 平均核数不重叠" % A_RATIO
    print(f"  A 环境   : {'命中' if a_arms else '未命中'} —— 双峰臂 {len(bim)} 个中【{thr_a}】的有 "
          f"{len(a_arms)} 个; 全场最大比 = {worst:.2f}x, 不重叠臂数 = {nro}")
    for r in a_arms:
        print(f"      🚨 {r['label']} bs={r['bs']} {r['tag'][0]}/{r['tag'][1]} 最大比 {r['bi']['rmax']:.2f}x "
              f"核数重叠={r['bi']['overlap']} 高簇 {r['bi']['hi']}/{ROUNDS} 发")
    print(f"  B 机制   : {'命中' if (b_pass and b_place) else '未命中'} —— 三项比值全在带内[{B_LO},{B_HI}] = {b_pass}"
          f"(最大 {worst:.2f}x); 双峰是否全落在 池>{l3mb:.1f}MB 的配置 C 格 = {b_place}")
    print(f"  C 短序列 : {'命中' if not bim else '未命中'} —— ROUNDS={ROUNDS} 下双峰臂数 = {len(bim)}"
          f"(命中 = 之前那几次双峰是 7 发太短的假象)")
    verdict = "A" if a_arms else ("B" if (b_pass and b_place) else ("C" if not bim else "D"))
    print(f"  D 挂账   : {'命中' if verdict == 'D' else '未命中'}")
    print(f"=> 落点: {verdict}    事前预测 = B => {'兑现' if verdict == 'B' else '打脸(如实记账)'}")
    print(f"   闸门 {sum(gates)}/{len(gates)} 格逐位差=0" + ("" if all(gates) else "  🚨 有格闸门失败"))
    print("\n判读提示: ① 只有【双峰臂】参与判决, 无双峰的臂只是背景; ② 落 A => 判读规则改【取众数簇】, "
          "乙-1c/乙-1d/乙-1e 按众数簇重读;\n"
          "        落 B => 旧表数值不动, 但边界要写【池>0.75xL3 的 C 格延迟双峰, p50 不可用, 放置按高簇付钱】;\n"
          "        落 C => 只提 ROUNDS; 落 D => 挂账。③ 帧长(wall)与 nv 用来排除【快照自己变慢】这种假象。")

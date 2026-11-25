#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B2 预检探针 · 真漂移语料【零判决成本 · 不进记录】

【它只回答那一句】
  r1~r5 的「漂移」是靠【挑行】造出来的(blk1/blk2 把某个 CPU 侧专家捧热),
  B2 想换成「文本自己漂」(连续分段, 成员靠真语料自己移动)。
  在写 K⁗ 判决树之前必须先知道: 真语料上【上帝成员到底动不动、幅度多大】。
  => 若跨段 top-GC 几乎不动 = 真语料上没有可吃的漂移 => 「动态放置无收益可谈」
     本身就是强负结果(且支撑正文「开局一次性放置」), 工单要么只发一张小表要么不发。

【它不算什么】
  🚨 不建臂、不计时、不碰 K/K′/K″/K‴ 任何一棵树、不产任何判决表 => 打印的一切是
  【模型自跑 · 不进记录】。它只做一件事: 用与装置【同一把尺】量「漂移的刺激强度」。

【与装置共用的尺(逐字同 bench_drifttrack.py)】
  LAYERS / GATE / CAP_LAYER / BS / GC / K=2 / hits_of(每 64 行批 topk(K) + bincount)
  / capture_from(在 MODEL.model.layers[CAP_LAYER].mlp 挂 forward hook 取输入)
  capture 默认 CAPMODE=seg = 每段自己 forward 一次(位置从 0 起), 与装置 phase_ids+capture_from 逐字同;
  CAPMODE=win 只是省时间的近似(窗口内后续段位置起点不同), 两种模式的打印都在段行里带 wid/sid 可核。

【口径】
  · 段 = BS 行连续不重叠, 段序 = 文本序。
  · 开局成员 = 该语料最前段的 top-GC(= 装置里「开局一次性放置」的对应物)。
  · cov = 该段命中落在给定 GC 个成员内的行占比(top-K 命中里有几成落在常驻 GPU 成员上)。
  · 可吃缺口 = ncpu(开局成员) - ncpu(本段自己的 top-GC) = 换成员能省下的【每行 CPU 专家个数】,
    相对% = 缺口 / ncpu(开局成员)。🚨 它是【延迟上界的代理】, 不许拿去换算 GAIN/T,
    它只回答「刺激量级」; cov(命中份额)同打印, 与装置 hist/fav_share 同口径便于对账。
  · 末尾附 CRAFTED 对照 = 与装置同法挑行(捧 order_A[GC] / order_A[GC+1]) 但只量 cov,
    用来回答「真语料的自然漂移是 r1~r5 人造漂移的几成」。

【用法】
  cd <GV> && NSEG=8 STRIDE=0 MAXWIN=1 CORPORA=readme,input.txt \
    /home/adomlans/miniconda3/envs/dl_lab/bin/python scan_drift.py
"""
import os, sys, time, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import real_smollm as R

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, K = 42, 2
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS = int(os.environ.get("BS", 512))
GC = int(os.environ.get("GC", 3))
E = len(LAYERS)
CORPORA = [c.strip() for c in os.environ.get("CORPORA", "readme,input.txt").split(",") if c.strip()]
NSEG = int(os.environ.get("NSEG", 8))          # 每窗口段数(窗口 = NSEG*BS 行)
STRIDE = int(os.environ.get("STRIDE", 0))      # 0 = 只跑首窗; >0 = 窗口起点等间距(token)
MAXWIN = int(os.environ.get("MAXWIN", 1))      # 每语料最多几个窗口
CAPMODE = os.environ.get("CAPMODE", "seg")     # seg=每段单独 forward(逐字同装置) / win=一次 forward 切段
POOL = int(os.environ.get("POOL", 4096))       # CRAFTED 对照的池子行数(与装置同)


def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 这台机器 DNS->127.0.0.1, 强制离线
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = R.snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        mdl = AutoModelForCausalLM.from_pretrained(snap)
        mdl.eval().to(torch.float32)
        for p in mdl.parameters():
            p.requires_grad_(False)
        return mdl, tok
    finally:
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


def capture_from(ids):
    box = {}

    def _h(_m, inp, _o):
        box["x"] = inp[0].detach()

    hk = MODEL.model.layers[CAP_LAYER].mlp.register_forward_hook(_h)
    try:
        with torch.no_grad():
            MODEL(ids)
    finally:
        hk.remove()
    return box["x"][0].float().contiguous()


def hits_of(gate_w, rows):
    W = gate_w.float()
    cnt = torch.zeros(E)
    for i in range(0, rows.shape[0], 64):
        sc = rows[i:i + 64] @ W.t()
        cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
    return cnt


def prof_of(gate_w, rows, members):
    """给定常驻成员 members(GC 个), 返回两个量:
       cov  = 命中次数里落在 members 的份额%(与装置 hist/fav_share 同一口径)
       ncpu = 每行平均要走【CPU 侧】的专家个数 = K - (落在 members 的命中数 / 行数)
              🚨 这才是延迟代理: CPU 腿是 GEMV 次数的线性函数, 少一个 = 少一份 CPU 时间。"""
    W = gate_w.float()
    hit = 0
    for i in range(0, rows.shape[0], 64):
        ti = (rows[i:i + 64] @ W.t()).topk(K, dim=1).indices
        hit += int((ti.unsqueeze(-1) == torch.tensor(list(members)).unsqueeze(0).unsqueeze(1)).sum())
    cov = hit / (rows.shape[0] * K) * 100.0
    return cov, K - hit / rows.shape[0]


def top_of(cnt, n=GC):
    return torch.argsort(cnt, descending=True).tolist()[:n]


def pct(c):
    return [round(float(v) / float(sum(c)) * 100, 1) for v in c.tolist()]


def text_of(name):
    if name == "readme":
        return R.real_text(N_CHARS)
    return open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()[:N_CHARS]


torch.manual_seed(SEED)
_SD = R.state_dict()
MODEL, TOK = _load()
BASE = R.real_src(LAYERS, gate_mode=GATE)
GW = BASE["gate_w"]
print(f"[探针头] LAYERS={list(LAYERS)} GATE={GATE} CAP_LAYER={CAP_LAYER} BS={BS} GC={GC} "
      f"K={K} E={E} N_CHARS={N_CHARS} NSEG={NSEG} STRIDE={STRIDE} MAXWIN={MAXWIN}")
print("[探针头] 🚨 本脚本不建臂/不计时/不参与任何 K 树 => 全部输出【模型自跑 · 不进记录】")

t0 = time.perf_counter()
ALL = {}
for name in CORPORA:
    txt = text_of(name)
    ids = TOK(txt, return_tensors="pt").input_ids
    T = ids.shape[1]
    W = NSEG * BS
    starts = [0] if STRIDE <= 0 else [s for s in range(0, T - W + 1, STRIDE)][:MAXWIN]
    print(f"\n=== 语料 {name}: chars={len(txt)} tokens={T} 窗口={W}tok({NSEG}段x{BS}行) "
          f"窗口起点={starts} 理论可切段数={max(0, (T - BS) // BS + 1)} ===")
    segs = []                                        # 每段: wid/sid/tok/rows/cnt/top/cov_*
    for wid, s in enumerate(starts):
        if CAPMODE == "win":                          # 省时间: 一次 forward 整窗再切段(位置起点与装置不同)
            x = capture_from(ids[:, s:s + W])
            assert x.shape[0] >= W, f"capture 只拿到 {x.shape[0]} 行 < {W}"
            parts = [(sid, x[sid * BS:(sid + 1) * BS].contiguous()) for sid in range(NSEG)]
            del x
        else:                                         # 逐字同装置: 每段自己 forward, 位置从 0 起
            parts = [(sid, capture_from(ids[:, s + sid * BS:s + (sid + 1) * BS])) for sid in range(NSEG)]
        for sid, rows in parts:
            segs.append(dict(wid=wid, sid=sid, tok=s + sid * BS, rows=rows,
                             cnt=hits_of(GW, rows), top=None))
            segs[-1]["top"] = top_of(segs[-1]["cnt"])
        print(f"  [窗口{wid} CAPMODE={CAPMODE}] capture+直方图 累计 {time.perf_counter()-t0:6.1f}s")
    # 开局成员 = 全局最前段的 top-GC(= 装置里「开局一次性放置」的对应物)
    home = segs[0]["top"]
    for g in segs:
        g["cov_own"], g["ncpu_own"] = prof_of(GW, g["rows"], g["top"])
        g["cov_home"], g["ncpu_home"] = prof_of(GW, g["rows"], home)
        g["rel"] = ((g["ncpu_home"] - g["ncpu_own"]) / g["ncpu_home"] * 100.0
                    if g["ncpu_home"] else 0.0)
    print(f"  [cov] 算完 累计 {time.perf_counter()-t0:6.1f}s")
    print(f"  开局成员(最前段 top{GC}) = {home}")
    print(f"  {'段':>6} {'tok':>7} | 命中pct(E=8)                 | top{GC}          "
          f"| cov%own cov%home | ncpu_own ncpu_home 省ncpu  相对% | 重合home")
    gaps = []
    chg = 0
    prev_top = None
    for g in segs:
        ov = len(set(g["top"]) & set(home))
        gp = g["ncpu_home"] - g["ncpu_own"]
        gaps.append(gp)
        if prev_top is not None and g["top"] != prev_top:
            chg += 1
        prev_top = g["top"]
        print(f"  {('w%ds%d' % (g['wid'], g['sid'])):>6} {g['tok']:>7} | "
              f"{str(pct(g['cnt'])):<28} | {str(g['top']):<13} | "
              f"{g['cov_own']:6.2f} {g['cov_home']:7.2f} | "
              f"{g['ncpu_own']:8.3f} {g['ncpu_home']:9.3f} {gp:+7.3f} "
              f"{(gp/g['ncpu_home']*100 if g['ncpu_home'] else 0):+6.1f} | {ov}/{GC}"
              + ("  <=top 集合变" if ov < GC else ""))
    n = len(segs)
    print(f"  【小结 {name}】段数={n} top{GC}集合变化次数={chg}/{n-1} "
          f"平均重合={sum(len(set(g['top']) & set(home)) for g in segs)/n:.2f}/{GC} "
          f"可吃缺口(ncpu): 均值 {sum(gaps)/n:+.3f} 最大 {max(gaps):+.3f} 最小 {min(gaps):+.3f} "
          f"相对% 均值 {sum(g['rel'] for g in segs)/n:+.1f} 最大 {max(g['rel'] for g in segs):+.1f}")
    # 【漂移的时间尺度】同一 top-GC【集合】能连续保持几段 = flip 成本能摊到几行
    runs, cur = [], 1
    for i in range(1, n):
        if frozenset(segs[i]["top"]) == frozenset(segs[i - 1]["top"]):
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    print(f"  【漂移尺度】同一 top{GC}【集合】连续游程: 均值 {sum(runs)/len(runs):.2f} 段 = "
          f"{sum(runs)/len(runs)*BS:.0f} 行、最大 {max(runs)*BS} 行、游程数 {len(runs)}/{n} "
          f"=> 追踪器每 ~{sum(runs)/len(runs)*BS:.0f} 行就想 flip 一次(flip = 19.9ms = 4.89 帧)")
    # 【可预测性 + 错置代价尺度】用来判断上面那些缺口是【可利用的漂移】还是【抽样噪声】
    if n > 1:
        cr, pv = [], []
        for i, g in enumerate(segs):
            _c, nn = prof_of(GW, g["rows"], segs[(i + 1) % n]["top"])   # 非因果: 拿下一段的 top-GC 打本段
            cr.append(nn - g["ncpu_own"])
            _c, np_ = prof_of(GW, g["rows"], segs[(i - 1) % n]["top"])  # 因果可得: 拿上一段的 top-GC
            pv.append(np_ - g["ncpu_own"])
        print(f"  【因果追踪】用『上一段的 top{GC}』打本段 => 比自家最优多付 ncpu 均值 {sum(pv)/n:+.3f} "
              f"=> 与 home 缺口均值 {sum(gaps)/n:+.3f} 同号比较: 前者大 = 按历史切换是净亏")
        alt = [i for i in range(E) if i not in home][:GC]                # 固定的非最优集合
        rc = []
        for g in segs:
            _c, nn = prof_of(GW, g["rows"], alt)
            rc.append(nn - g["ncpu_own"])
        print(f"  【可预测性】用『下一段的 top{GC}』打本段 => 比自家最优多付 ncpu 均值 {sum(cr)/n:+.3f} "
              f"(0 = 漂移完全可预测; 可与『home 缺口均值 {sum(gaps)/n:+.3f}』比大小)")
        print(f"  【错置尺度】固定用非最优集合 {alt} 打本段 => 比自家最优多付 ncpu 均值 {sum(rc)/n:+.3f}")
    ALL[name] = dict(segs=segs, home=home, gaps=gaps, chg=chg, n=n)

# ------------------------------------------------ CRAFTED 对照(与装置同法挑行, 只量 cov 不算延迟)
print("\n=== CRAFTED 对照(= r1~r5 装置的造漂移法, 去掉计时) ===")
name0 = "readme" if "readme" in ALL else CORPORA[0]
txt = text_of(name0)
ids = TOK(txt, return_tensors="pt").input_ids[:, :POOL]
XP = capture_from(ids)
cnt_e = hits_of(GW, XP[:BS])
order_A = torch.argsort(cnt_e, descending=True).tolist()
home = ALL[name0]["home"]
FAV = [None, order_A[GC], order_A[GC + 1]]
print(f"  语料={name0} 池={POOL}行 order_A前6={order_A[:6]} FAV={FAV} 开局成员={home}")
for b, fav in enumerate(FAV):
    if fav is None:
        sel = torch.arange(BS)
    else:
        # 🚨 空间对账: 装置里 route 的 ti 在【槽位】空间(引擎 experts 已按 order 置换), 所以它比
        #    eid == favor; 这里 GW = BASE["gate_w"] 未置换 => ti 已在【expert id】空间 => 直接比 favor。
        ti = torch.cat([(XP[i:i + 64] @ GW.t()).topk(K, dim=1).indices for i in range(0, XP.shape[0], 64)])
        pos = int(fav)
        score = (ti == pos).sum(1).float()
        sel = torch.argsort(-score, stable=True)[:BS].sort().values
    rows = XP[sel]
    c = hits_of(GW, rows)
    tp = top_of(c)
    co, no = prof_of(GW, rows, tp)
    ch, nh = prof_of(GW, rows, home)
    print(f"  blk{b} favor={str(fav):>4} 行数={rows.shape[0]} 命中pct={str(pct(c)):<28} "
          f"top{GC}={str(tp):<12} cov%own={co:6.2f} cov%home={ch:6.2f} "
          f"ncpu_own={no:6.3f} ncpu_home={nh:6.3f} 省ncpu={nh-no:+6.3f} "
          f"相对%={(nh-no)/nh*100 if nh else 0:+6.1f} "
          f"fav_share={float(c[int(fav)])/float(c.sum())*100 if fav is not None else float('nan'):5.1f}%")

print(f"\n[探针尾] 总用时 {time.perf_counter()-t0:.1f}s")

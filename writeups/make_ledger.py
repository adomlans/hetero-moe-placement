#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P0-④b 交付件之一：总账图（两张 PNG）
  fig1_ladder.png   前 14 刀的累计账（OPTLOG §1 `LADDER` 的可视化）
  fig2_exchange.png 三方交换图（显存 ↔ 延迟 ↔ token级质量；唯一数据源 = P0-④a 两张归档表）

🚨 这个脚本【不测任何东西】。它画的全部数字都是从归档判决表里【抄】进来的，
   而且抄对没有不由我决定：draw 之前先跑 verify()，把每个数回 `results/` 逐格对账，
   对不上就 SystemExit 拒绝出图（归档件全是用户自跑的 ⇒ 图不可能编数）。
   数据源：
     results/2026-11-21_P0-④a_swap_3way.txt        （主表，date 10:41:44，md5 a1dabef9…）
     results/2026-11-21_P0-④a附_bs128_crossproc.txt（bs=128 单独进程复现轮，date 11:35:42，md5 a6b759d7…）
     OPTLOG.md §1 LADDER 表
   ④b 引用纪律（P0-④a附）：A bs=128 gc=3 bf16 的 p50 是【跨进程双峰】(1.349/1.735ms)，
   图上按纪律把两峰都画出来并标注，不选一个当"真值"。
"""
import io, os, re, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
GV = os.path.dirname(HERE)
RES = os.path.join(GV, "results")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK = 2  # ttc 里 Noto Sans CJK SC 的面号
S = 2    # 2x 超采样后 Lanczos 缩小（PIL 的线不抗锯齿）

MAIN = "2026-11-21_P0-④a_swap_3way.txt"
FU = "2026-11-21_P0-④a附_bs128_crossproc.txt"

# ───────────────────────────── 归档解析 ─────────────────────────────
ROW = re.compile(
    r"^\s*(\d)\s+(fp32|bf16)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+")

def parse_swap(fname):
    """把 bench_swap 的两张归档表读成 {(cfg,bs,gc,dtype): {...}}"""
    out, cfg, bs = {}, None, None
    for ln in io.open(os.path.join(RES, fname), encoding="utf-8"):
        m = re.search(r"配置([AC]).*?bs=\s*(\d+)", ln)
        if m and "swiglu |" in ln:
            cfg, bs = m.group(1), int(m.group(2)); continue
        m = ROW.match(ln)
        if not m or cfg is None:
            continue
        gc, dt = int(m.group(1)), m.group(2)
        rest = ln[m.end():].split()
        # rest: [vs纯CPU, GPU腿slot, CPU腿slot, CPU驻MB, relRMSE, top1, KL, TV, ...]
        out[(cfg, bs, gc, dt)] = dict(
            peak=float(m.group(3)), steady=float(m.group(4)),
            p50=float(m.group(5)), sigma=int(m.group(6)),
            cpu_mb=float(rest[3]), kl=float(rest[6]))
    return out

# ─────────────────── 要画进图的数（从归档抄，先验证后作图） ───────────────────
# bs=1 / bs=32 取主表；bs=128 取单独进程复现轮（更干净的窗口），双峰那格另存两峰
SWAP = {
    1: {"fp32": [(0, 0.1, 0.246, 4.928e-03), (1, 33.8, 0.267, 7.214e-03),
                 (3, 101.3, 0.257, 6.105e-11), (5, 169.3, 0.322, 6.105e-11),
                 (7, 238.9, 0.395, 6.105e-11), (8, 273.1, 0.434, 6.105e-11)],
        "bf16": [(0, 0.1, 0.273, 4.928e-03), (1, 22.0, 0.240, 7.554e-03),
                 (3, 65.9, 0.215, 1.353e-05), (5, 109.8, 0.263, 1.353e-05),
                 (7, 154.4, 0.303, 1.353e-05), (8, 177.6, 0.317, 1.353e-05)],
        "front": {"fp32": {0, 3}, "bf16": {1, 3}}, "src": MAIN},
    32: {"fp32": [(0, 0.3, 1.710, 2.841e-04), (1, 34.0, 1.126, 3.498e-04),
                  (3, 101.5, 0.756, 2.220e-05), (5, 169.5, 0.528, 1.204e-05),
                  (7, 239.2, 0.744, 3.682e-06), (8, 273.3, 0.742, 1.126e-11)],
         "bf16": [(0, 0.3, 1.378, 2.841e-04), (1, 22.2, 1.401, 3.635e-04),
                  (3, 66.1, 0.743, 2.166e-05), (5, 110.0, 0.390, 1.180e-05),
                  (7, 154.6, 0.350, 4.045e-06), (8, 177.8, 0.352, 9.376e-07)],
         "front": {"fp32": {1, 8}, "bf16": {0, 3, 5, 7, 8}}, "src": MAIN},
    128: {"fp32": [(0, 1.0, 3.413, 1.016e-04), (1, 34.7, 2.212, 1.049e-04),
                   (3, 102.2, 1.287, 9.041e-06), (5, 170.2, 0.783, 4.086e-06),
                   (7, 239.8, 1.080, 1.247e-06), (8, 274.0, 1.108, 4.979e-12)],
          "bf16": [(0, 1.0, 3.427, 1.016e-04), (1, 22.9, 2.305, 1.083e-04),
                   (3, 66.8, 1.735, 9.635e-06), (5, 110.7, 0.639, 4.546e-06),
                   (7, 155.3, 0.493, 1.532e-06), (8, 178.5, 0.578, 3.705e-07)],
          "front": {"fp32": {0, 1, 3, 8}, "bf16": {0, 1, 3, 5, 7, 8}}, "src": FU},
}
BIMODAL = dict(cfg="A", bs=128, gc=3, dtype="bf16", other_peak=(66.8, 1.349, 1.735))

# LADDER（OPTLOG §1）：档位 -> (A/d0, C/d0)，取"这一刀之后"的值
LAD = [("T4 基线", 1.758, 6.109), ("+T5", 1.262, 1.463), ("+T6", 0.750, 1.311),
       ("+T7", 0.387, 0.917), ("+T9 v1", 0.401, 0.847), ("+T11", 0.365, 0.817),
       ("+T12", 0.322, 0.766), ("+T13 v2", 0.295, 0.755), ("+T14", 0.300, 0.756)]
LAD_OFF = [("+T8 否证", 0.401, 0.847, "T8 等D2H挪worker：否证，留档"),
           ("+T12b", 0.297, 0.706, "T12b 尾图：默认【关】，只当定价用")]
LAD_KEYS = {"T4 基线": "T4 基线", "+T5": "+T5 tok_block", "+T6": "+T6a dense",
            "+T7": "+T7 full graph", "+T9 v1": "+T9 放置规则 v1", "+T11": "+T11 前段融合",
            "+T12": "+T12 单流", "+T13 v2": "+T13 放置规则 v2", "+T14": "+T14 OMP guided",
            "+T8 否证": "T8 等D2H挪worker", "+T12b": "+T12b 尾图"}

def parse_front(fname):
    """归档里每个块尾部那行"三方帕累托前沿(...)" -> {(cfg,bs): {(gc,dtype),...}}"""
    out, cfg, bs = {}, None, None
    for ln in io.open(os.path.join(RES, fname), encoding="utf-8"):
        m = re.search(r"配置([AC]).*?bs=\s*(\d+)", ln)
        if m and "swiglu |" in ln:
            cfg, bs = m.group(1), int(m.group(2)); continue
        if "三方帕累托前沿" in ln and cfg:
            out[(cfg, bs)] = {(int(g), dt) for g, dt in re.findall(r"gc=(\d+)(fp32|bf16)", ln)}
    return out


def verify():
    bad = []
    for bs, blk in SWAP.items():
        art = parse_swap(blk["src"])
        for dt in ("fp32", "bf16"):
            for (gc, steady, p50, kl) in blk[dt]:
                k = ("A", bs, gc, dt)
                if k not in art:
                    bad.append("缺格 %s" % (k,)); continue
                g = art[k]
                if abs(g["steady"] - steady) > 0.06: bad.append("稳态MB %s %s!=%s" % (k, steady, g["steady"]))
                if abs(g["p50"] - p50) > 0.006:      bad.append("p50 %s %s!=%s" % (k, p50, g["p50"]))
                if abs(g["kl"] - kl) > max(1e-16, abs(kl) * 3e-3): bad.append("KL %s %s!=%s" % (k, kl, g["kl"]))
        # 帕累托前沿也要跟归档那行"三方帕累托前沿"一致
        want = parse_front(blk["src"]).get(("A", bs))
        mine = {(gc, dt) for dt in ("fp32", "bf16") for gc in blk["front"][dt]}
        if want is None: bad.append("归档里找不到 bs=%s 的前沿行" % bs)
        elif want != mine: bad.append("前沿 bs=%s 归档%s != 图上%s" % (bs, sorted(want), sorted(mine)))
    b = BIMODAL
    a1 = parse_swap(MAIN); a2 = parse_swap(FU)
    k = (b["cfg"], b["bs"], b["gc"], b["dtype"])
    if abs(a1[k]["p50"] - 1.349) > 0.006 or abs(a2[k]["p50"] - 1.735) > 0.006:
        bad.append("双峰两峰对不上: 主表%s / 附%s" % (a1[k]["p50"], a2[k]["p50"]))
    opt = io.open(os.path.join(GV, "OPTLOG.md"), encoding="utf-8").read().split("\n")
    for name, a, c in LAD + [(x[0], x[1], x[2]) for x in LAD_OFF]:
        key = LAD_KEYS[name]
        row = next((l for l in opt if l.strip().startswith("|") and key in l), None)
        if row is None: bad.append("LADDER 找不到行 %s" % key); continue
        cells = [c2.strip() for c2 in row.strip().strip("|").split("|")]
        got = []
        for cell in cells[1:3]:
            nums = re.findall(r"[\d.]+", cell.replace("~", ""))
            got.append(float(nums[-1]) if nums else None)
        if got[0] is None or abs(got[0] - a) > 0.0051: bad.append("LADDER A %s %s!=%s" % (name, a, got[0]))
        if got[1] is None or abs(got[1] - c) > 0.0051: bad.append("LADDER C %s %s!=%s" % (name, c, got[1]))
    return bad

# ───────────────────────────── 画图小工具 ─────────────────────────────
import math

class C:
    bg=(252,252,250); fg=(30,34,40); grid=(214,218,224); sub=(110,118,128)
    A=(31,86,150); C_=(216,118,26); red=(196,42,42); green=(30,124,74); purple=(110,60,150); gray=(150,156,164)

def canvas(w, h):
    im = Image.new("RGB", (w*S, h*S), C.bg); return im, ImageDraw.Draw(im)

def F(size, bold=False): return ImageFont.truetype(FONT_B if bold else FONT, int(size*S), index=CJK)

def T(d, x, y, s, size=13, fill=None, bold=False, anchor="la"):
    d.text((x*S, y*S), s, font=F(size, bold), fill=fill or C.fg, anchor=anchor)

def line(d, pts, fill, w=2, dash=0):
    P = [(x*S, y*S) for x, y in pts]
    if not dash:
        d.line(P, fill=fill, width=w*S); return
    for i in range(len(P)-1):
        (x0,y0),(x1,y1) = P[i], P[i+1]
        seg = ((x1-x0)**2+(y1-y0)**2) ** .5
        n = max(1, int(seg) // (dash*S))
        for j in range(n):
            t0, t1 = j/n, (j+0.55)/n
            d.line([(x0+(x1-x0)*t0, y0+(y1-y0)*t0), (x0+(x1-x0)*t1, y0+(y1-y0)*t1)], fill=fill, width=w*S)

def rect(d, x0, y0, x1, y1, outline, w=1):
    d.rectangle([x0*S, y0*S, x1*S, y1*S], outline=outline, width=w*S)

def dot(d, x, y, r, fill=None, edge=None, w=2):
    d.ellipse([(x-r)*S,(y-r)*S,(x+r)*S,(y+r)*S], fill=fill, outline=edge, width=w*S)

def diamond(d, x, y, r, fill):
    d.polygon([(x*S,(y-r)*S),((x+r)*S,y*S),(x*S,(y+r)*S),((x-r)*S,y*S)], fill=fill)

def log_ticks(lo, hi):
    out, e = [], -3
    while 10.0**e < hi*10:
        for m in (1, 2, 5):
            v = m*10.0**e
            if lo <= v <= hi: out.append(v)
        e += 1
    return out

def mkY(lo, hi, y0, y1):
    l10, h10 = math.log10(lo), math.log10(hi)
    return lambda v: y1 - (math.log10(v)-l10)/(h10-l10)*(y1-y0)

KL_BUCKETS = [(1e-3, "KL ≥ 1e-3  int8 字节墙", (196,42,42)),
              (1e-4, "1e-4 ~ 1e-3", (222,120,40)),
              (1e-5, "1e-5 ~ 1e-4", (214,170,50)),
              (1e-6, "1e-6 ~ 1e-5", (120,166,90)),
              (1e-9, "1e-9 ~ 1e-6", (60,140,170)),
              (0,    "< 1e-9 ≈ 逐位相同", (40,80,150))]

def kl_color(kl):
    for thr, _, col in KL_BUCKETS:
        if kl >= thr: return col
    return KL_BUCKETS[-1][2]

# ───────────────────────────── fig1 · LADDER ─────────────────────────────
def fig1(path):
    W, H = 1500, 880
    im, d = canvas(W, H)
    T(d, 40, 30, "总账图 ① 前 14 刀累计账（LADDER）", 26, bold=True)
    T(d, 40, 70, "口径：同进程轮转 5 轮取中位、绝对 ms、y 取 log。A/d0 = 热专家常驻 GPU（打字节墙）；C/d0 = 冷专家进 GPU（打算力墙，跨次噪声底 8~18%）。", 14, C.sub)
    x0, x1, y0, y1 = 120, 1440, 150, 660
    Y = mkY(0.25, 8.0, y0, y1)
    for v in log_ticks(0.25, 8.0):
        y = Y(v); line(d, [(x0,y),(x1,y)], C.grid, 1)
        T(d, x0-12, y-6, ("%.2f" % v if v < 1 else "%g" % v), 12, C.sub, anchor="ra")
    T(d, x0-12, y0-26, "ms", 12, C.sub, anchor="ra")
    n = len(LAD); xs = [x0+60 + i*((x1-x0-90)/(n-1)) for i in range(n)]
    for i, (nm, _, _) in enumerate(LAD):
        line(d, [(xs[i], y0),(xs[i], y1)], (234,236,240), 1)
        T(d, xs[i], y1+12, nm, 13, C.fg, anchor="ma")
    yA = [Y(r[1]) for r in LAD]; yC = [Y(r[2]) for r in LAD]
    line(d, list(zip(xs, yA)), C.A, 3)
    line(d, list(zip(xs, yC)), C.C_, 3)
    for i, x in enumerate(xs):
        gap = abs(yA[i]-yC[i])
        # 两线贴近(如 +T5 的 1.262/1.463 只差 22px)时, 内侧标签必须外推: A 往【下】、C 往【上】,
        # 否则 A 标签正好压在 C 点上、C 标签正好压在 A 点上(+T5 第一版就是这么糊的)。
        if gap > 60:
            tA, tC = yA[i]-20, yC[i]+12
        else:
            tA, tC = yA[i]+22, yC[i]-18
        dot(d, x, yA[i], 6, fill=C.A); T(d, x, tA, "%.3f" % LAD[i][1], 12, C.A, bold=True, anchor="ma")
        dot(d, x, yC[i], 6, fill=C.C_); T(d, x, tC, "%.3f" % LAD[i][2], 12, C.C_, bold=True, anchor="ma")
    for nm, a, c, note in LAD_OFF:
        i = {"+T8 否证": 4, "+T12b": 7}[nm]        # 画在它所属两档之间
        x = (xs[i-1]+xs[i])/2
        diamond(d, x, Y(a), 7, C.gray); diamond(d, x, Y(c), 7, C.gray)
        T(d, x, Y(a)-30, "%s A %g" % (nm, a), 11, C.gray, anchor="ma")
        T(d, x, Y(c)+16, "C %g" % c, 11, C.gray, anchor="ma")
    T(d, x1-8, 172, "A/d0 1.758 → 0.300ms（−83.0%，谷底 T13 0.295）", 16, C.A, bold=True, anchor="ra")
    T(d, x1-8, 200, "C/d0 6.109 → 0.756ms（−87.6%）—— C 档大头是 T5/T6 把算力墙搬走的", 16, C.C_, bold=True, anchor="ra")
    T(d, x1-8, 232, "灰色菱形 = 主线之外的两档：+T8 否证留档 / +T12b 默认【关】只当定价用", 13, C.sub, anchor="ra")
    rect(d, x0, y1+66, x1, H-24, (200,204,210))
    notes = [
     "读这张表的三条纪律（照抄 OPTLOG §1，别在图上做新的解读）：",
     "① C 列跨次运行噪声底 8~18% ⇒ C 曲线上任何一段小平滑都别读成收益；",
     "② T14 在 ladder 口径下是平的（A 0.295→0.300），它的真收益 A −6.8%/C −12.4% 在专用 11 轮配对表 bench_t14_ab.py —— 口径不同，两个数都真；",
     "③ 这 14 刀全部是 bs=1 的演示形状（D=1024/N=2048 随机权重）。bs 变成变量、换上真权重真形状之后还剩多少，见总账图 ② 与 P0-①/乙-1/乙-2。",
    ]
    for i, t in enumerate(notes):
        T(d, x0+14, y1+78+i*22, t, 13, C.fg if i == 0 else C.sub, bold=(i == 0))
    im.resize((W, H), Image.LANCZOS).save(path)
    return path

# ───────────────────────────── fig2 · 三方交换 ─────────────────────────────
CLAIM = {1:  ("bf16", 3, "花 66MB ⇒ 快 21.2%、KL 好 364x", "🚨 bs=1 每帧只有 2 个 slot ⇒ 放置规则在此选不出档（退化）"),
         32: ("bf16", 5, "花 110MB ⇒ 快 71.7%、KL 好 24x", "规则选 gc=5 = fp32 实测最优；bf16 规则真值 6、实测最优 7"),
         128:("bf16", 7, "花 155MB ⇒ 快 85.6%、KL 好 66x（两轮同值）", "规则选 gc=5 = fp32 实测最优；放错档欠给 +101~106%")}

def fig2(path):
    W, H = 1500, 950
    im, d = canvas(W, H)
    T(d, 40, 26, "总账图 ② 三方交换：显存 ↔ 延迟 ↔ token级质量（同一格、同权重、同裁判）", 24, bold=True)
    T(d, 40, 62, "x = 稳态显存 MB（暖机后仍驻留，不是峰值）| y = p50 延迟 ms（逐点独占 7 轮 ×20 次取中位，log 轴）| 圆色 = KL(P‖Q) | 空心 = fp32、实心 = bf16 | 粗环 = 帕累托前沿", 13, C.sub)
    T(d, 40, 84, "数据源：results/2026-11-21_P0-④a_swap_3way.txt（bs=1/32）+ …④a附_bs128_crossproc.txt（bs=128 单独进程复现轮）。P = 同架构 fp32 稠密裁判，KL 是【相对量】（参照行见底部③）。", 13, C.sub)
    panels = [(1, 0.18, 0.50, "bs=1（decode：每帧 1 个 token，2 个 slot）"),
              (32, 0.30, 2.0, "bs=32"),
              (128, 0.40, 4.2, "bs=128")]
    pw, gap, x_left, py0, py1 = 430, 28, 96, 156, 648
    for pi, (bs, LO, HI, title) in enumerate(panels):
        px0 = x_left + pi*(pw+gap); px1 = px0+pw
        Y = mkY(LO, HI, py0, py1)
        def X(mb): return px0 + mb/300.0*pw
        rect(d, px0, py0, px1, py1, (198,202,208))
        for v in log_ticks(LO, HI):
            y = Y(v); line(d, [(px0,y),(px1,y)], (236,238,241), 1)
            T(d, px0-8, y-6, ("%g" % v if v >= 1 else "%.2f" % v), 11, C.sub, anchor="ra")
        for mb in (0, 100, 200, 300):
            x = X(mb); line(d, [(x,py0),(x,py1)], (236,238,241), 1)
            T(d, x, py1+8, str(mb), 11, C.sub, anchor="ma")
        T(d, (px0+px1)/2, py1+26, "稳态显存 MB", 12, C.sub, anchor="ma")
        T(d, px0, py0-26, title, 16, C.fg, bold=True)
        blk = SWAP[bs]
        for dt in ("fp32", "bf16"):
            pts = [(X(p[1]), Y(p[2])) for p in blk[dt]]
            line(d, pts, C.gray, 2, dash=7)
            for (gc, steady, p50, kl), (x, y) in zip(blk[dt], pts):
                front = gc in blk["front"][dt]; col = kl_color(kl)
                dot(d, x, y, 8 if front else 5.5, fill=(col if dt == "bf16" else None),
                    edge=col, w=3 if front else 2)
                if dt == "bf16":
                    T(d, x+11, y-6, "gc%d" % gc, 10, (140,146,154))
        dt, gc, claim, note2 = CLAIM[bs]
        base = [p for p in blk["bf16"] if p[0] == 0][0]
        hit = [p for p in blk[dt] if p[0] == gc][0]
        line(d, [(X(base[1]), Y(base[2])), (X(hit[1]), Y(hit[2]))], C.green, 2, dash=9)
        for p in (base, hit): dot(d, X(p[1]), Y(p[2]), 12, edge=C.green, w=3)
        # 主张写在面板【下方】而不是面板内左下角: bs=128 的 gc=7 bf16(最优那点)就贴在面板底边,
        # 写在里面必然压字(P0-④a附 的引用点就是它, 不能为了排版把它挪位或藏掉)。
        T(d, px0, py1+50, claim, 13, C.green, bold=True)
        T(d, px0, py1+70, note2, 11, C.sub)
        if bs == 128:   # 跨进程双峰（④b 引用纪律：两峰都画，不选一个当真值）
            b = BIMODAL; x = X(b["other_peak"][0]); ya, yb = Y(1.349), Y(1.735)
            d.rectangle([ (x-9)*S, (yb-9)*S, (x+9)*S, (yb+9)*S ], outline=C.purple, width=2*S)
            d.rectangle([ (x-9)*S, (ya-9)*S, (x+9)*S, (ya+9)*S ], outline=C.purple, width=2*S)
            line(d, [(x, ya), (x, yb)], C.purple, 2)
            T(d, x-14, (ya+yb)/2-16, "跨进程双峰 28.6%", 12, C.purple, bold=True, anchor="ra")
            T(d, x-14, (ya+yb)/2+4, "1.349(主表σ1%) / 1.735(附轮)", 11, C.purple, anchor="ra")
    lx, ly = 96, 758
    T(d, lx, ly-4, "KL(P‖Q) 色阶 →", 13, C.fg, bold=True)
    for i, (thr, lab, col) in enumerate(KL_BUCKETS):
        dot(d, lx+136+i*204, ly+2, 7, fill=col)
        T(d, lx+148+i*204, ly-6, lab, 12, C.fg)
    rect(d, 92, 784, 1440, 940, (200,204,210))
    notes = [
     "① 三根轴同格同权重同裁判 ⇒ 这张图不跨表借数：闸门⓪ 已把它与【另一份代码】bench_metrics 归档的 60 个 KL 对过账（不符 0 格，最大相对偏差 2.9e-04）。",
     "② 显存报的是【稳态 MB】。峰值 MB 那一列 bf16【不降】是应该的（建图瞬时走 fp32 staging）：gc=5 两档峰值同为 186.1MB，稳态才是 169.5 vs 110.0。",
     "③ KL 只报相对量。参照行「把真第 8 层 MLP 换成 8 选 2 的 fp32 稠密 MoE 本身」= bs=128 KL 8.854e-02 / top1 86.72% ⇒ 比引擎最狠那档还贵 871x，别拿它当这一刀的代价。",
     "④ 紫框那格按 P0-④a附 纪律处理：A bs=128 gc=3 bf16 是跨进程双峰（轮内 σ=1% 却跨进程差 28.6%；形态 = CPU 驻 12.7MB ≈ L3 12MB 悬崖）⇒ 两峰都画，不选一个当真值。",
     "⑤ 曲线不是配对：逐点独占 ⇒ 相邻两点差 <噪声底（放对 2~3%）不读。表里唯一的配对是 fp32↔bf16 那一刀，且只认【配对可用】✓ 的行（gc=7/8 的 −48~−53%）。",
     "⑥ 形状读法：gc↑ ⇒ 显存单调↑、KL 单调↓；延迟是 U 形 —— bs 小是 GPU 半边纯付权重的钱，bs 大才是 CPU 腿 int8 字节墙被搬走（3.4ms → 0.5ms）。",
    ]
    for i, t in enumerate(notes):
        T(d, 104, 798+i*22, t, 12.5, C.sub)  # 最后一行底边 ~922, 必须留在读图框(940)里面
    im.resize((W, H), Image.LANCZOS).save(path)
    return path

if __name__ == "__main__":
    bad = verify()
    if bad:
        print("🚨 对账失败 %d 条 —— 拒绝出图（图里的数必须能在 results/ 归档里逐格找到）：" % len(bad))
        for b in bad[:20]: print("   ", b)
        sys.exit(1)
    print("✅ 对账通过：36 个交换格 + 前沿集合 + 双峰两峰 + 11 行 LADDER 全部与归档逐格一致")
    print("wrote", fig1(os.path.join(HERE, "fig1_ladder.png")))
    print("wrote", fig2(os.path.join(HERE, "fig2_exchange.png")))

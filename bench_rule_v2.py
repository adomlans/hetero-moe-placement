"""放置规则 v2 的【离线重放验证器】(P1 第一优先的验证侧)。

自跑工具 => 输出【不进记录】(HANDOFF §2: 判决表由用户自跑, 模型自跑只作归因)。
它一次都不测延迟, 只做三件事:
  1) 归档复现: 把 bench_swap 印过的"规则选 gc ... 比最优贵 +x%"用解析器+规则重算一遍,
     对得上 => 证明解析器与规则输入没漂(这是本脚本的命, 同 闸门⓪ 的逻辑);
  2) 重放对账: 把 v1(plan_gpu_capacity) 与 v2(plan_gpu_capacity_v2) 放回归档实测网格选档,
     量各自比【全域实测最优】贵多少 —— 让"八条偏贵 +14.5%~+37.6%"变成两版可比的数;
  3) 分辨力测量: 同一决策单元内 max(两腿) 预测与实测 p50 的排序一致率 + 残差 +
     "要把 band 开多大才装得下实测最优" —— tie_band 该取多大由它说话, 不由"让答案变对"说话。

数据纪律:
  · 实测 p50/稳态MB/KL/GPU腿slot 全部解析 results/ 归档文本, 本脚本不重测;
  · 规则输入(每 bs 的 per-专家 slot 数)是确定量 => 现加载模型重算, 且有【自证闸门】:
    重算的 sum(hits[:gc]) 与归档 GPU腿slot 必须逐格相等, 不等就整块作废;
  · 🚨 跨进程不混用: bs=1 同时存在于 ④a(只有 cat) 与 乙-2d(cat+traffic), 两次进程绝对值
    差到 12%(方法论 26)。每个决策单元只用覆盖实现档最全的那一份当主口径, 另一份不参与选档。

用法: /home/adomlans/miniconda3/envs/dl_lab/bin/python bench_rule_v2.py
env: BANDS=0.08,0.2,0.35,0.5  GCMIN=0,2  GPU_BW_S=250,333,390 (GB/s, 敏感性诊断, 不是重标)
"""
import os, re, sys, time, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FILES = [("④a", "2026-11-21_P0-④a_swap_3way.txt"),
         ("乙-2d", "2026-11-21_乙-2d_swap_bs1_traffic.txt")]
BANDS = [float(v) for v in os.environ.get("BANDS", "0.08,0.2,0.35,0.5").split(",")]
GCMINS = [int(v) for v in os.environ.get("GCMIN", "0,2").split(",")]
GPU_BW_S = [float(v) * 1e9 for v in os.environ.get("GPU_BW_S", "250,333,390").split(",")]
ELS = {"fp32": 4, "fp16": 2, "bf16": 2}
IMPL_NAMES = {"cat", "traffic", "bmm", "sparse"}
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
K, E, SEED = 2, len(LAYERS), 42
HDR = re.compile(r"^swiglu \| 配置([AC]) .*?\| bs=\s*(\d+) \|")
ROW_RE = re.compile(r"^ {2}(\d+)\s+(fp32|bf16|fp16)\s+(.*)$")
NAN = float("nan")


def fnum(s):
    try:
        return float(s.rstrip("%"))
    except ValueError:
        return NAN


def parse_swap(path, tag):
    """归档文本 -> {cfg: {bs: {dtype: {impl: {gc: rec}}}}}"""
    out, blk = {}, None
    for ln in open(path, encoding="utf-8", errors="ignore").read().splitlines():
        m = HDR.match(ln)
        if m:
            blk = out.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})
            continue
        m = ROW_RE.match(ln)
        if not m or blk is None:
            continue
        gc, dt, rest = int(m.group(1)), m.group(2), m.group(3).split()
        impl = "cat"
        if rest and rest[0] in IMPL_NAMES:
            impl, rest = rest[0], rest[1:]
        if len(rest) < 13:
            continue
        blk.setdefault(dt, {}).setdefault(impl, {})[gc] = dict(
            pk=fnum(rest[0]), st=fnum(rest[1]), p=fnum(rest[2]), sig=fnum(rest[3]),
            ngpu=int(fnum(rest[5])), ncpu=int(fnum(rest[6])), kl=fnum(rest[10]))
    ncell = sum(len(r) for c in out.values() for b in c.values()
                for v in b.values() for r in v.values())
    print(f"  解析 {tag:<7} 配置{sorted(out)} bs={sorted({b for c in out.values() for b in c})}"
          f" 交换格 {ncell}")
    return out


def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
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
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def nearest(g, gs, bias="small"):
    """规则真值落到实测网格(与 bench_swap 同一规矩: 取最近档, 平手取更小 gc)。
    返回 (档位, 是否平手歧义)。"""
    if g in gs:
        return g, False
    d = min(abs(g - x) for x in gs)
    c = [x for x in gs if abs(g - x) == d]
    return (min(c) if bias == "small" else max(c)), len(c) > 1


def main():
    print("=" * 120)
    print("放置规则 v2 离线重放 —— 模型自跑【不进记录】: 实测值全来自 results/, 规则输入现算 + 自证闸门")
    print("=" * 120)
    GRID, PROV = {}, {}
    for tag, name in FILES:
        p = os.path.join(RES, name)
        if not os.path.exists(p):
            print(f"  跳过 {tag}: 缺 {name}")
            continue
        for cfg, bys in parse_swap(p, tag).items():
            for bs, bydt in bys.items():
                for dt, byim in bydt.items():
                    for im, rows in byim.items():
                        GRID.setdefault((tag, cfg, bs, dt, im), {}).update(rows)
                        PROV.setdefault((cfg, bs, dt), {}).setdefault(tag, []).append(im)
    PRIM = {}          # 主进程 = 同一决策单元里实现档覆盖最全的那一份(跨进程不混用)
    for (cfg, bs, dt), bytag in PROV.items():
        PRIM[(cfg, bs, dt)] = max(bytag.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]

    # ---------------------------------------------------------- 规则输入(确定量, 现算)
    t0 = time.perf_counter()
    _SD = R.state_dict(); _R_sd = R.state_dict
    R.state_dict = lambda: _SD
    MODEL, TOK = _load(); R.state_dict = _R_sd
    IDS = TOK(R.real_text(24000), return_tensors="pt").input_ids

    def capture(ids):
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
        W = gate_w.float(); cnt = torch.zeros(E)
        for i in range(0, rows.shape[0], 64):
            sc = rows[i:i + 64] @ W.t()
            cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
        return cnt

    gate = R.real_src(LAYERS, gate_mode=GATE)["gate_w"]
    cnt_old = hits_of(gate, capture(IDS[:, :512]))
    ORD = {"A": torch.argsort(cnt_old, descending=True).tolist(),
           "C": torch.argsort(cnt_old).tolist()}
    dims = R.dims(_SD); eb = dims["N"] * dims["D"] * 3
    print(f"  [规则输入] 512 行命中%={[round(float(v)/float(cnt_old.sum())*100, 1) for v in cnt_old]}"
          f" A序={ORD['A']} C序={ORD['C']} | eb={eb} 加载 {time.perf_counter()-t0:.1f}s")
    HITS = {}
    for bs in sorted({b for (t, c, b, d, i) in GRID}):
        cnt_bs = hits_of(gate, capture(IDS[:, :bs]))          # 每 bs 只现抓一次 x
        for cfg, order in ORD.items():
            HITS[(cfg, bs)] = [int(v) for v in cnt_bs[order]]

    # ---------------------------------------------------------- 自证闸门
    print("\n" + "-" * 120)
    print("自证闸门 | 重算 sum(hits[:gc])/sum(hits[gc:]) == 归档 GPU腿slot/CPU腿slot | 不等 => 该块作废")
    bad, ok = [], 0
    for (tagk, cfg, bs, dt, im), rows in sorted(GRID.items()):
        h = HITS.get((cfg, bs))
        if h is None:
            continue
        for g, rec in sorted(rows.items()):
            gpu = sum(h[:g])
            if gpu == rec["ngpu"] and sum(h) - gpu == rec["ncpu"]:
                ok += 1
            else:
                bad.append(f"{tagk} {cfg}/bs{bs}/{dt}/{im}/gc{g}: 重算 {gpu}/{sum(h)-gpu}"
                           f" vs 归档 {rec['ngpu']}/{rec['ncpu']}")
    print(f"  覆盖 {ok + len(bad)} 格 => 一致 {ok} / 不一致 {len(bad)}")
    for b in bad[:10]:
        print("   ", b)

    units = sorted({(c, b, d) for (t, c, b, d, i) in GRID},
                 key=lambda k: (k[0], k[1], k[2]))

    # ---------------------------------------------------------- ① 归档复现(解析器自证)
    print("\n" + "-" * 120)
    print("① 归档复现 | v1 逐 (dtype/实现) 子网格选档 => 应与归档里印过的'规则选 gc ... 比最优贵'逐字对上")
    print(f"  {'决策单元':<20}{'进程/实现':>12}{'子网格最优':>16}{'v1 真值->落格':>18}{'偏贵%':>9}")
    for (cfg, bs, dt) in units:
        for tg in sorted({t for (t, c, b, d, i) in GRID if (c, b, d) == (cfg, bs, dt)}):
          for im in ("cat", "traffic"):
            if (tg, cfg, bs, dt, im) not in GRID:
                continue
            rows = GRID[(tg, cfg, bs, dt, im)]
            gs = sorted(rows)
            best = min(gs, key=lambda g: rows[g]["p"])
            g0 = H.plan_gpu_capacity(HITS[(cfg, bs)], eb, eb * ELS[dt], gpu_expert_count=2)[0]
            gm, amb = nearest(g0, gs)
            lab = cfg + " bs=" + str(bs) + " " + dt
            blab = "gc=%d %.3f" % (best, rows[best]["p"])
            glab = "gc=%d->gc=%d" % (g0, gm) + ("(歧义)" if amb else "")
            print(f"  {lab:<20}{tg + '/' + im:>12}{blab:>16}{glab:>18}"
                  f"{100*(rows[gm]['p']/rows[best]['p']-1):>+8.1f}%")

    # ---------------------------------------------------------- ② 重放对账
    print("\n" + "-" * 120)
    print("② 重放对账 | 全域最优 = 同一决策单元内所有 (gc,实现) 格的最快实测 | 偏贵% 一律对着它")
    print("  五个变体(🚨 全部共用 42GB/s / 250GB/s / tie_band=0.08, 一个常数都不许逐单元挑 => 挑常数=过拟合):")
    print("    v1         = plan_gpu_capacity(归档口径: GPU 腿只有 gc x 权重字节)")
    print("    v2 gmin2/0 = plan_gpu_capacity_v2 + 调用方补的【网格取最近档、平手取更小】疤痕")
    print("    v2 grid    = 网格内直选(grid=实测网格) => 把那道疤痕搬进规则, 规则真值==落格")
    print("    v2 grid+墙 = 再开 CPU 腿第二堵墙 cpu_mac_rate=CPU_MAC_RATE(CPU腿=max(字节墙,算力墙))")
    print("    † = 该进程只归档了 cat, 变体选了 traffic => 按其 gc 落 cat 读数(实现档那半刀不判)")
    UNITS = []
    for (cfg, bs, dt) in units:
        ptag = PRIM[(cfg, bs, dt)]
        imps = [i for i in ("cat", "traffic") if (ptag, cfg, bs, dt, i) in GRID]
        cells = {(g, i): r for i in imps for g, r in GRID[(ptag, cfg, bs, dt, i)].items()}
        gs = sorted({g for g, i in cells})
        if 0 not in gs:
            continue
        best = min(cells, key=lambda k: cells[k]["p"])
        pb = cells[best]["p"]
        h = HITS[(cfg, bs)]
        gwb = eb * ELS[dt]

        def pct(k, sel):
            if k not in cells:
                return "        缺档  " + sel
            return "%+8.1f%%  " % (100 * (cells[k]["p"] / pb - 1)) + sel

        g1 = H.plan_gpu_capacity(h, eb, gwb, gpu_expert_count=2)[0]
        k1 = (nearest(g1, gs)[0], "cat")
        g2, im2, us2, cur = H.plan_gpu_capacity_v2(h, eb, gwb, gc_min=2)
        g2m, amb = nearest(g2, gs)
        k2 = (g2m, im2) if (g2m, im2) in cells else (g2m, "cat")
        g20, im20, _, _ = H.plan_gpu_capacity_v2(h, eb, gwb, gc_min=0)
        g20m, amb20 = nearest(g20, gs)
        k20 = (g20m, im20) if (g20m, im20) in cells else (g20m, "cat")
        g3, im3, us3, _ = H.plan_gpu_capacity_v2(h, eb, gwb, grid=gs)
        k3 = (g3, im3) if (g3, im3) in cells else (g3, "cat")
        g4, im4, us4, _ = H.plan_gpu_capacity_v2(h, eb, gwb, grid=gs,
                                                 cpu_mac_rate=H.CPU_MAC_RATE)
        k4 = (g4, im4) if (g4, im4) in cells else (g4, "cat")
        UNITS.append(dict(unit=(cfg, bs, dt), cells=cells, gs=gs, best=best, pb=pb,
                          h=h, k1=k1, g1=g1, k2=k2, g2=g2, im2=im2, cur=cur,
                          k3=k3, k4=k4))

        def cell_lab(g, im, fell, forced):
            """选档 -> 实际落格。fell!=选档 说明被网格收口/实现档不在了; † 见上面口径。"""
            sel = ("gc=%d/%s -> gc=%d/%s" % (g, im, fell[0], fell[1])) if g != fell[0] \
                else ("gc=%d/%s" % (g, im))
            return sel + (" †" if forced else "")

        lab = cfg + " bs=" + str(bs) + " " + dt
        print("  决策单元 %-14s 进程=%-6s 全域最优 gc=%d/%s %.3fms  网格=%s"
              % (lab, ptag, best[0], best[1], pb, gs))
        print("      %-11s %s" % ("v1", pct(k1, "gc=%d -> gc=%d/cat" % (g1, k1[0]))))
        print("      %-11s %s" % ("v2 gmin2", pct(
            k2, cell_lab(g2, im2, k2, k2[1] != im2)
            + ("(歧义)" if amb else ""))))
        print("      %-11s %s" % ("v2 gmin0", pct(
            k20, cell_lab(g20, im20, k20, k20[1] != im20)
            + ("(歧义)" if amb20 else ""))))
        print("      %-11s %s" % ("v2 grid", pct(k3, cell_lab(g3, im3, k3, k3[1] != im3))))
        print("      %-11s %s" % ("v2 grid+墙", pct(k4, cell_lab(g4, im4, k4, k4[1] != im4))))

    # ---------------------------------------------------------- ③ 分辨力
    print("\n" + "-" * 120)
    print("③ 模型分辨力 | 决策单元内 max(两腿) 预测 vs 实测 p50: 排序一致率 / 残差 / 要多大的 band 才装得下实测最优")
    print(f"  {'决策单元':<18}{'格数':>5}{'一致率':>8}{'预测差中位':>12}{'实测差中位':>12}"
          f"{'去中位残差±':>13}{'最优需band':>12}{'v1/v2 需band':>16}")
    concs = []
    for U in UNITS:
        cfg, bs, dt = U["unit"]
        cur, cum, cbof = H.placement_curve_v2(U["h"], eb, eb * ELS[dt], gc_min=0)
        cells, gs = U["cells"], U["gs"]
        pred = {}
        for (g, im) in cells:
            pred[(g, im)] = cur[im][g][2] if g in cur[im] else NAN
        ks = sorted(cells)
        agree = tot = 0
        for i, a in enumerate(ks):
            for b2 in ks[i + 1:]:
                tot += 1
                pa = pred[a] - pred[b2]
                pm = cells[a]["p"] - cells[b2]["p"]
                if pa == 0 or (pa > 0) == (pm > 0):
                    agree += 1
        conc = agree / tot if tot else NAN
        concs.append(conc)
        res = [cells[k]["p"] * 1000 - pred[k] for k in ks if pred[k] == pred[k]]
        med = statistics.median(res)
        spread = max(abs(r - med) for r in res) / (U["pb"] * 1000) * 100
        dp = statistics.median(sorted(abs(pred[a] - pred[b2]) for i, a in enumerate(ks)
                                     for b2 in ks[i + 1:]
                                     if pred[a] == pred[a] and pred[b2] == pred[b2]))
        dm = statistics.median(sorted((cells[a]["p"] - cells[b2]["p"]) * 1000
                                      for i, a in enumerate(ks) for b2 in ks[i + 1:]))
        pmin = min(v for v in pred.values() if v == v)
        need_best = 100 * (pred[U["best"]] / pmin - 1)
        need1 = 100 * (pred[U["k1"]] / pmin - 1) if U["k1"] in pred else NAN
        need2 = 100 * (pred[U["k2"]] / pmin - 1) if U["k2"] in pred else NAN
        lab = cfg + " bs=" + str(bs) + " " + dt
        nlab = "%+.0f%%/%+.0f%%" % (need1, need2)
        print(f"  {lab:<18}{len(ks):>5}{conc*100:>7.0f}%"
              f"{dp:>11.0f}us{dm:>11.0f}us{spread:>12.1f}%{need_best:>+11.0f}%{nlab:>16}")
    print(f"  平均排序一致率 {statistics.mean(concs)*100:.0f}% 最低 {min(concs)*100:.0f}%"
          f"  => 一致率不高的单元里, '偏贵 x%' 读成【模型没分辨力】而不是效应量")

    # ------------------------------------------------- ③b 第二堵墙值不值(开/关同格对比)
    print("\n" + "-" * 120)
    print("③b 第二堵墙值不值 | 同一批归档格, CPU 腿 = max(字节墙, 算力墙) 的【开/关】各自算分辨力")
    print("   这一栏回答的是: 算力墙是【提高了模型分辨力】还是【只是把答案蒙对了】(方法论 22)")
    print(f"  {'决策单元':<18}{'格数':>5}{'一致率 关→开':>16}{'去中位残差± 关→开':>22}"
          f"{'预测跨度us 关→开':>20}")

    def stats2(cur, cells, pb):
        ks = sorted(cells)
        pred = {k: (cur[k[1]][k[0]][2] if k[0] in cur[k[1]] else NAN) for k in ks}
        agree = tot = 0
        for i, a in enumerate(ks):
            for b2 in ks[i + 1:]:
                tot += 1
                pa = pred[a] - pred[b2]
                pm = cells[a]["p"] - cells[b2]["p"]
                if pa == 0 or (pa > 0) == (pm > 0):
                    agree += 1
        res = [cells[k]["p"] * 1000 - pred[k] for k in ks if pred[k] == pred[k]]
        med = statistics.median(res)
        spread = max(abs(r - med) for r in res) / (pb * 1000) * 100
        vals = [v for v in pred.values() if v == v]
        return (agree / tot if tot else NAN), spread, max(vals) - min(vals)

    concs_w = {"off": [], "on": []}
    for U in UNITS:
        cfg, bs, dt = U["unit"]
        out = {}
        for key, rate in (("off", None), ("on", H.CPU_MAC_RATE)):
            cur, _cum, _cb = H.placement_curve_v2(U["h"], eb, eb * ELS[dt], gc_min=0,
                                                  cpu_mac_rate=rate)
            out[key] = stats2(cur, U["cells"], U["pb"])
            concs_w[key].append(out[key][0])
        lab = cfg + " bs=" + str(bs) + " " + dt
        print(f"  {lab:<18}{len(U['cells']):>5}"
              f"{out['off'][0]*100:>7.0f}% -> {out['on'][0]*100:>3.0f}%"
              f"{out['off'][1]:>10.1f}% -> {out['on'][1]:<7.1f}%"
              f"{out['off'][2]:>9.0f} -> {out['on'][2]:<8.0f}")
    for key in ("off", "on"):
        print(f"  平均一致率(cpu_mac_rate={key:<3}) {statistics.mean(concs_w[key])*100:.0f}%"
              f" 最低 {min(concs_w[key])*100:.0f}%")

    # ---------------------------------------------------------- ④ 敏感性
    print("\n" + "-" * 120)
    print("④ 敏感性 | v2 选档对 (GPU 单价 / band / gc_min) 的响应 —— 诊断: 看规则被哪个常数牵着走")
    print(f"  {'决策单元':<18}{'全域最优':>14}{'实测最优需band':>15}"
          + "".join(f"{'bw' + str(int(b // 1e9)) + 'GB/s':>13}" for b in GPU_BW_S))
    for U in UNITS:
        cfg, bs, dt = U["unit"]
        cells, gs = U["cells"], U["gs"]
        cur0 = H.placement_curve_v2(U["h"], eb, eb * ELS[dt], gc_min=0)[0]
        pmin = min(cur0[i][g][2] for i in cur0 for g in cur0[i])
        need = 100 * (cur0[U["best"][1]][U["best"][0]][2] / pmin - 1)
        cols = []
        for gbw in GPU_BW_S:
            picked = []
            for band in BANDS:
                for gmin in GCMINS:
                    g2, im2, _, _ = H.plan_gpu_capacity_v2(U["h"], eb, eb * ELS[dt], gc_min=gmin,
                                                           tie_band=band, gpu_bw=gbw)
                    g2m, _ = nearest(g2, gs)
                    k2 = (g2m, im2) if (g2m, im2) in cells else (g2m, "cat")
                    if k2 in cells:
                        picked.append(f"{g2m}/{im2[:2]}")
            uniq = sorted(set(picked))
            cols.append((",".join(uniq[:2]) + ("…" if len(uniq) > 2 else "")) or "—")
        lab = cfg + " bs=" + str(bs) + " " + dt
        blab = "gc=%d/%s" % U["best"]
        print(f"  {lab:<18}{blab:>14}{need:>+14.0f}%"
              + "".join(f"{c:>13}" for c in cols))

    print("\n口径提醒: ① 本脚本一次都不测延迟 => 表里每个实测数都能指回 results/ 归档;"
          "\n  ② '最优需band' 是把 tie_band 当唯一解药时要付的价: 它远大于同一行'去中位残差' =>"
          " 说明 bs 小的那几格不是 band 太窄, 是【两腿定价本身在 bs=1 上没有分辨力】;"
          "\n  ③ GPU 单价那一栏只是把'要多少钱规则才选对'摆出来看, 拿它当结论 = 拟合答案(方法论 22)。")


if __name__ == "__main__":
    main()

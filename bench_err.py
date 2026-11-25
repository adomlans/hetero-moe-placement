#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-②c 判决表: 【per-expert int8 误差单价表】+ 「成员集 KL 能不能被加性模型解释」。

【这一刀为什么必须打(P1-②b 把自己逼到这里的)】
  两轮 dyn(results/2026-11-22_P1-②_dyn_gc3_bs512.txt md5 bd8eae3b / ..._startshuffle.txt md5 8f33ba36)
  留下一个没法解释的硬伤:
    · drift 段三个 GPU 成员集, 覆盖流量 65.3 > 58.5 > 55.9, KL 却是 2.079e-05 > 1.564e-05 > 1.301e-05
      —— 覆盖与 KL **完全反序**(Spearman = -1), 不是"不单调"而是**符号反了**;
    · 按直方图追"最优成员"的 hyst_1x 三段覆盖全部追平上帝视角(vs地板 +0.0), 花 4 次迁移 63.08ms,
      KL 却比"根本不追"的四档差 60%。
  => 纯命中判据(覆盖流量%)在"被省略的二阶量随成员身份变化"时给出**反向决策**(OPTLOG 方法论 30)。
     那个二阶量 = 每个专家自己的 int8 误差大小。它现在只是个假设(还带方向性预言: eid5/eid6 应该比
     eid2/eid3 便宜), 没有数就没资格改判据 => P1-②d 被这张表卡住。

【怎么把"单价"量出来: 单专家扰动】
  对每个专家 e 建一个引擎: gc = E-1, 置换序把 e 放到最后一个槽 => **只有 e 在 CPU 腿(int8),
  其余 E-1 个都在 GPU 腿(fp32)**。这一档的 KL(p_e) 就是"只让 e 走 int8"的代价 = 单价 u_e。
  它旁边还有一个可除的量: 该段 e 的命中占比 hit_e => 每-slot 单价 kappa_e = u_e / hit_e。
  kappa_e 才是与流量无关的【专家本体属性】(u_e 里混了流量), 判据要用的是它。

【这一刀真正的赌注: 加性预测(不是"再看一张 KL 表")】
  拿 u_e / kappa_e 去【预测】那些早就归档在 results/ 里的成员集 KL —— 两个都无自由参数的模型:
    (M0 计数模型)  KL(S) ~ KL(全CPU) x [e in CPU(S)] hit_e          = "覆盖流量%" 那一列的数学形式
    (M1 单价模型)  KL(S) ~ [e in CPU(S)] u_e = [e in CPU(S)] hit_e x kappa_e
  被测点 = 归档里已有的 (语料段, GPU 成员集) 组合: P0-④c(readme) / P0-③补(input.txt) 的 gc 扫描
  (A 放对 + C 放错 = 同 bs 同语料的两族成员集) + 两轮 dyn 的 gc=3 成员集 + 上帝视角地板。
  🚨锚点是【解析归档文本】拿的, 不是抄在代码里的常数 —— 读不到文件就明确印"无锚点"而不是假过。
  再加一列旧口径(M2): 同一批点的 L2rel%(向量层误差, 台账用了 45 行的那根轴) —— 它能不能排出同一
  个序, 直接决定"relRMSE 不是最终指标"这句话能不能写成定量版。

【闸门(全绿才有一行可读)】
  ① splice 位置: hook 覆盖前录到的 MLP 输入 == 本段现抓 x, 逐位(与 bench_metrics 同一条判据)。
  ② 路由不变: 每个引擎的 top-k(换回 eid 空间)与基准逐位相同 => KL 里没混进路由漂移。
  ③ 🚨只有选中的那一个专家在 CPU 腿: solo_e 档的 CPU 腿 eid 集合必须恰好是 {e}, 且 CPU 腿 slot 数
     == 该段 route 到 e 的 slot 数; 成员集档的 CPU 腿集合 == 全集\S。这一条不绿, 单价表整个是假的。
  ④ 跨轮 KL 互差: 锚点格整格重建重算, KL 必须逐位相同(这些数是确定量, 不是计时量)。
     🚨前提是每格先【连打到相邻两发自洽】(REDO 默认 4)才记账 —— 本机偶发脏格红线抓不到(见 ⑥),
     一发制的闸门④ 会把脏格误读成"重算不符"。闸门④ 因此是第 3 次独立重算, 不是第 2 发。
     🚨"自洽"是【两级判据】(见 leg_run): ①相邻两发逐位相同(绝大多数格走这条, 第 2 发就返回)
     ②打完 REDO 发仍没有相邻对、但同一个值出现过 >=2 次(A,B,A 这种"跳回来"的序列) => 收并打 jit=1,
     同时印出这一格的【轮间相对极差】= 质量列自己的噪声底。两级都不过 => 该格 KL 记 nan(代码强制)。
  ⑤ 端点自洽: gc=E(CPU 腿 0 slot) 的 KL 必须塌到 ~1e-12 量级; gc=0(全 CPU) 的 KL 必须 ~= sum(u_e)
     —— 后者是【加性假设的全 CPU 端点检验】, 它自己就是这一刀的一条判决。
  ⑥ 幅度红线: |logit| <= LOGIT_RED(bs=512 出现过 4.094e+19 的脏格), 引擎出口 y <= Y_RED;
     P(裁判)脏 => 整段作废; 某格 Q 脏 => 该行作废(印 nan, 拟合自动跳过)。
     🚨红线只判【幅度】, 判不了【稳定性】: 冒烟实测有一格 KL=1.10e+152 而 absmax 只有 33.5 ——
     所以每格先连打到相邻两发 KL/TV/top-1 逐位相同才记账(REDO), 打完仍不自洽 => 该格印 nan。
     🚨P1-②c 冒烟抓到自己违反这条: 口径写"记 nan"、代码却把【最后一发原样】填进了 u_e 表和 M1 配平
     (6/66 格)。=> 现在 unst 在写 CELL 那一刻就把 kl/tv/top1 洗成 nan, 且 min/sum/sorted/Spearman
     全部先过 fvals() 再算 —— 带 nan 的 min/sum 在 Python 里是【顺序相关】的静默错误。
  ⓪ 归档对账: 本表实测 KL vs 归档 KL, 常规档相对偏差 <= ANCHOR_TOL(2e-3 = 归档只有 4 位有效)。
     🚨这一条同时证明两件事: 我复现了归档那一格的【同一批 token】(行区间/语料/x 全对), 以及
     "同一成员集 => 同一 KL"(引擎用 sorted(members) 建, 归档用 order_A/order_C)。
     🚨但 KL <= KL_FLOOR(1e-9) 的档(全 GPU 那种 1e-12)本身就在浮点噪声底, 相对偏差会虚高到百分之几
     => 那一类只判比值 <= FLOOR_RATIO(同数量级), 汇总行把常规档/噪声底档分开印, 别混成一个红。
  换序自检: 同一成员集换槽序【不要求】逐位 —— 换序改了 per-token slot 的 index_add_ 归约序,
     只要求相对互差 <= SLOT_TOL(1e-3; 实测 7.2e-05)。=> "KL 是成员集的函数"这句话要带噪声底读。
  ⑦ 段复现: 各段命中% 与 dyn 归档印的 命中%(eid) 逐一相符(±0.05pp) => 我的 drift 段与那两轮
     是同一批行; 顺带对账 A 序。
env: LAYERS=0,4,8,12,16,20,24,28  GATE=embed  CAP_LAYER=8  N_CHARS=24000
     BS=512  PHASES=readme,input.txt,drift  DRIFT_POOL=2048  DTYPES=fp32  ROUNDS=1  REDO=4
     诊断用: PROBE=solo2:input.txt,set023:readme PROBE_SHOTS=12 => 只建那几格、连打 N 发印每发 KL,
             用来分"轮间不自洽"是【末位抖动】还是【灾难脏格】(探针数不进记录, 跑完即退)
     ARCH_M=results/2026-11-21_P0-④c_metrics_bs512.txt,results/2026-11-21_P0-③补_metrics_corpus-input_txt_bs512.txt
     ARCH_D=results/2026-11-22_P1-②_dyn_gc3_bs512.txt,results/2026-11-22_P1-②b_dyn_gc3_bs512_startshuffle.txt
"""
import os, sys, time, math, re, torch
from gc import collect as _pycollect
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS = int(os.environ.get("BS", 512))
GC_SET = int(os.environ.get("GC_SET", 3))                  # 建 drift 段 + 归档成员集用的 GC 口径
DTYPES = [d for d in os.environ.get("DTYPES", "fp32").split(",") if d]
PHASES = [p.strip() for p in os.environ.get("PHASES", "readme,input.txt,drift").split(",") if p.strip()]
DRIFT_POOL = int(os.environ.get("DRIFT_POOL", 2048))
ROUNDS = int(os.environ.get("ROUNDS", 1))                  # 闸门④ 的重算轮数(每轮把锚点格整格重建重算)
LOGIT_RED = float(os.environ.get("LOGIT_RED", 1e4))
Y_RED = float(os.environ.get("Y_RED", 1e3))
ANCHOR_TOL = float(os.environ.get("ANCHOR_TOL", 2e-3))     # 归档 4 位有效 => 2e-3
KL_FLOOR = float(os.environ.get("KL_FLOOR", 1e-9))         # 低于它 KL 已在浮点噪声底, 相对指标失真
FLOOR_RATIO = float(os.environ.get("FLOOR_RATIO", 1e2))    # 噪声底档只判同数量级(1e-2~1e2 倍)
SLOT_TOL = float(os.environ.get("SLOT_TOL", 1e-3))         # 换槽序 => index_add_ 归约序噪声底
REDO = int(os.environ.get("REDO", 4))                   # 每格最多连打几发: 相邻两发 KL/TV/top-1 逐位相同即记账
                                                       #   (自洽格永远在第 2 发就返回 => 抬高这个数【不多花时间】, 只救打歪的格)
HIT_TOL = float(os.environ.get("HIT_TOL", 0.05))           # 闸门⑦ 命中% 允许的四舍五入差(百分点)
ARCH_M = [p.strip() for p in os.environ.get(
    "ARCH_M", "results/2026-11-21_P0-④c_metrics_bs512.txt,results/2026-11-21_P0-③补_metrics_corpus-input_txt_bs512.txt"
).split(",") if p.strip()]
ARCH_D = [p.strip() for p in os.environ.get(
    "ARCH_D", "results/2026-11-22_P1-②_dyn_gc3_bs512.txt,results/2026-11-22_P1-②b_dyn_gc3_bs512_startshuffle.txt"
).split(",") if p.strip()]
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
E = len(LAYERS)
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
assert set(DTYPES) == {"fp32"}, f"单价表只按 fp32 裁判打(归档口径): {DTYPES}"
assert 0 < GC_SET < E, GC_SET
assert DRIFT_POOL >= BS, f"drift 池({DRIFT_POOL}) < BS({BS})"
ALL = frozenset(range(E))

t_load = time.perf_counter()
print(f"[机器状态] {H.machine_state()}")
dims = R.dims()
D, N, NL = dims["D"], dims["N"], dims["layers"]


# ---------------------------------------------------------------- 真模型 + 抓取(与 bench_metrics 逐字同)
def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 这台机器 DNS->127.0.0.1, 强制离线
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = R.snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        mdl = AutoModelForCausalLM.from_pretrained(snap)
        mdl.eval().to(torch.float32)            # 🚨 config 里是 bf16, 裁判要 fp32
        for p in mdl.parameters():
            p.requires_grad_(False)
        return mdl, tok
    finally:
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


MODEL, TOK = _load()
torch.manual_seed(SEED)


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


def splice(ids_row, y_cpu, box=None):
    def _h(_m, inp, _o):
        if box is not None:
            box["x"] = inp[0].detach().clone()
        return y_cpu.unsqueeze(0)

    hk = MODEL.model.layers[CAP_LAYER].mlp.register_forward_hook(_h)
    try:
        with torch.no_grad():
            return MODEL(ids_row).logits[0].double()
    finally:
        hk.remove()


def plain(ids_row):
    with torch.no_grad():
        return MODEL(ids_row).logits[0].double()


def perm_src(order):
    base = R.real_src(LAYERS, gate_mode=GATE)
    return dict(experts=[base["experts"][i] for i in order],
                gate_w=base["gate_w"][list(order)].contiguous(),
                gate_b=base["gate_b"].clone(),
                note=base["note"] + f" perm={list(order)}")


def hits_of(gate_w, rows, probe_bs=64):
    W = gate_w.float()
    cnt = torch.zeros(E)
    for i in range(0, rows.shape[0], probe_bs):
        sc = rows[i:i + probe_bs] @ W.t()
        cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
    return cnt


def mk(gc, order, dt="fp32"):
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind="swiglu",
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def red_ok(L, thr=None):
    thr = LOGIT_RED if thr is None else thr
    a = max(abs(float(L.max())), abs(float(L.min())))
    return a <= thr, a


def metrics(lp, lq, ids_row):
    T = lp.shape[0]
    P, Q = torch.log_softmax(lp, -1), torch.log_softmax(lq, -1)
    p, q = P.exp(), Q.exp()
    kl = (p * (P - Q)).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    a1p, a1q = lp.argmax(-1), lq.argmax(-1)
    k5p, k5q = lp.topk(5, -1).indices, lq.topk(5, -1).indices
    rec = (k5p.unsqueeze(2) == k5q.unsqueeze(1)).any(2).float().mean().item() * 100
    kp = max(1, min(T, -(-T * 95 // 100)))
    return dict(top1=(a1p == a1q).float().mean().item() * 100.0, rec=rec,
                kl=float(kl.mean()), kl95=float(kl.kthvalue(kp).values), tv=float(tv.mean()))


pct = lambda c: [round(float(v) / float(sum(c)) * 100, 1) for v in (c.tolist() if hasattr(c, "tolist") else c)]
def spearman(a, b):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else float("nan")


# ---------------------------------------------------------------- 归档解析(锚点 = 从文本拿, 不抄常数)
def _pfloat(s):
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_metrics(path):
    """bench_metrics 归档 -> corpus / A序 / C序 / 命中% / cells[(label,gc,dtype,kl,rel,slot)]"""
    cells, corpus, label, orders, hits = [], None, None, {}, None
    for ln in open(path, encoding="utf-8"):
        if "[语料]" in ln:
            corpus = ln.split("CORPUS=")[1].split()[0] if "CORPUS=" in ln else "readme"
        if "配置A" in ln:
            label = "A"
        elif "配置C" in ln:
            label = "C"
        m = re.search(r"A序=\[([0-9,\s]+)\]\s+C序=\[([0-9,\s]+)\]", ln)
        if m:
            orders["A"] = [int(v) for v in m.group(1).split(",")]
            orders["C"] = [int(v) for v in m.group(2).split(",")]
        m = re.search(r"命中%=\[([0-9,\.\s]+)\]", ln)
        if m and hits is None:
            hits = [float(v) for v in m.group(1).split(",")]
        p = ln.split()
        if len(p) >= 8 and p[0].isdigit() and p[1] in ("fp32", "bf16", "fp16") and p[4].endswith("%"):
            cells.append((label, int(p[0]), p[1], _pfloat(p[7]), _pfloat(p[4][:-1]), int(p[3])))
    return dict(corpus=corpus, orders=orders, hits=hits, cells=cells)


POL_RE = r"static_A|static_p1|static_C|shuffle|neutral|naive|hyst_1x|hyst_2x|hysteresis"


def parse_dyn(path):
    """bench_dyn 归档 -> (phases{(name,kth): {hits, cells{memberset:kl}, god}}, 阶段列互校不符行数)

    🚨 dyn 归档是【两段式】: 三条 `阶段N = …` 头行连着印在一起, 策略表整张印在三段头之后。
       所以"当前头行"不能当策略行的阶段 —— 那样全部策略行都会落进【最后一段】。
       P1-②c 首次 3 段冒烟实测踩过: 5 个 drift 锚点里 4 个串成了 readme/input.txt 的值,
       看起来像"本表比归档差 20%~130%", 其实是解析器串行(闸门⓪ 抓到 => 它不是装饰)。
       => 策略行的阶段只认【行内那一列 阶段】, 再拿行内语料名互校; 头行只用来收 命中%/地板。"""
    tag = os.path.basename(path)[:24]
    phases, seq, cur, nbad = {}, [], None, 0
    for ln in open(path, encoding="utf-8"):
        m = re.search(r"阶段(\d+)\s*=\s*(\S+)\s*第(\d)段\s*\|\s*命中%\(eid\)=\[([0-9,\.\s]+)\]", ln)
        if m:
            cur = (m.group(2), int(m.group(3)) - 1)
            seq.append(cur)
            phases[cur] = dict(hits=[float(v) for v in m.group(4).split(",")], cells={}, god=None, tag=tag)
            continue
        if cur is None:
            continue
        m = re.search(r"最优成员\(上帝视角\)=\[([0-9,\s]+)\].*?地板.*?KL=([0-9\.eE+-]+)", ln)
        if m:
            g = frozenset(int(v) for v in m.group(1).split(","))
            phases[cur]["god"] = (g, float(m.group(2)))
            phases[cur]["cells"].setdefault(g, (float(m.group(2)), tag + " 地板"))
            continue
        m = re.match(r"\s*(" + POL_RE + r")\s+(\d)\s+(\S+)\s+.*?\[\s*([0-9,\s]+)\]\s+([0-9.]+)\s+([+-][0-9.]+)\s+([0-9.]+)\s+([0-9\.eE+-]+)\s+([0-9\.eE+-]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", ln)
        if m:
            pol, pno, corpus = m.group(1), int(m.group(2)) - 1, m.group(3)
            k2 = seq[pno] if 0 <= pno < len(seq) else None
            if k2 is not None and k2[0] == corpus:
                pk = k2
            else:
                pk, nbad = (corpus, 0), nbad + 1        # 行内语料名兜底, 同时把不符计数交出去(不静默)
            mem = frozenset(int(v) for v in m.group(4).split(","))
            d = phases.setdefault(pk, dict(hits=None, cells={}, god=None, tag=tag))
            d["cells"].setdefault(mem, (float(m.group(8)), f"{pol} {tag}"))
    return phases, nbad


def load_anchors(ph_keys):
    """-> {(phase_key, memberset): (kl, src)} + 报告(读了哪些文件/跳过什么)"""
    A, notes = {}, []
    for rel in ARCH_M:
        p = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        if not os.path.exists(p):
            notes.append(f"🚨归档缺失 {rel} => 这一族的锚点没了")
            continue
        a = parse_metrics(p)
        pk = (a["corpus"], 0)
        if pk not in ph_keys:
            notes.append(f"归档 {rel} 的语料 {a['corpus']} 不在 PHASES 里 => 跳过它的 {len(a['cells'])} 格")
            continue
        n = 0
        for (label, gc, dt, kl, relv, slot) in a["cells"]:
            if dt != "fp32" or label not in a["orders"]:
                continue
            mem = frozenset(a["orders"][label][:gc])
            A.setdefault((pk, mem), (kl, f"{label} gc={gc} {os.path.basename(rel)[:20]}"))
            n += 1
        notes.append(f"读了 {rel}: 语料={a['corpus']} 格={n} A序={a['orders'].get('A')}")
    for rel in ARCH_D:
        p = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        if not os.path.exists(p):
            notes.append(f"🚨归档缺失 {rel} => dyn 成员集锚点没了")
            continue
        dph, nbad = parse_dyn(p)
        for pk, d in dph.items():
            if pk not in ph_keys:
                continue
            for mem, (kl, src) in d["cells"].items():
                A.setdefault((pk, mem), (kl, src + " dyn"))
        notes.append(f"读了 {rel}: 每段成员集 "
                     + " ".join(f"{k[0]}={len(v['cells'])}" for k, v in dph.items())
                     + (f" | 🚨阶段列互校不符 {nbad} 行" if nbad else " | 阶段列互校全符"))
    return A, notes


# ---------------------------------------------------------------- 阶段构造(与 bench_dyn 同写法)
def phase_ids(name, kth, rows=None):
    rows = BS if rows is None else int(rows)
    txt = (R.real_text(N_CHARS) if name == "readme"
           else open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()[:N_CHARS])
    ids = TOK(txt, return_tensors="pt").input_ids
    lo = kth * BS
    assert ids.shape[1] >= lo + rows, f"语料 {name} 取不出第 {kth} 段 {rows} 行"
    return ids[:, lo:lo + rows], len(txt), ids.shape[1]


def drift_ids(order_A):
    ids_pool = phase_ids("readme", 0, DRIFT_POOL)[0]
    xp = capture_from(ids_pool).cuda()
    # 🚨 BASE_W 是 CPU 上的路由权重(hits_of 就走 CPU 路径), 这里 xp 已经 .cuda() => 必须把 W 搬过去
    W = BASE_W.to(xp.device)
    eid = []
    for i in range(0, xp.shape[0], 128):
        sc = xp[i:i + 128] @ W.t()
        eid.append(sc.topk(K, dim=1).indices)
    eid = torch.cat(eid).cpu()
    cold = torch.zeros(E, dtype=torch.bool)
    cold[order_A[GC_SET:]] = True
    score = cold[eid].sum(1).float()
    try:
        sel = torch.argsort(-score, stable=True)[:BS]
    except TypeError:
        sel = torch.argsort(-score)[:BS]
    sel = sel.sort().values
    share = float(score[sel].sum()) / float(BS * K)
    del xp
    torch.cuda.empty_cache()
    return ids_pool[:, sel], share


BASE_W = R.real_src(LAYERS, gate_mode=GATE)["gate_w"].float()
XREAL, _d = R.capture(layer=CAP_LAYER, bs=BS, n_chars=N_CHARS)
cnt_readme = hits_of(BASE_W, XREAL)
order_A = torch.argsort(cnt_readme, descending=True).tolist()
order_C = torch.argsort(cnt_readme).tolist()
hits_desc = torch.argsort(cnt_readme, descending=True).tolist()
print(f"[真源] SmolLM-135M: D={D} N={N} 层数={NL} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家 top_k={K}"
      f" | 路由={GATE} | 模型加载 {time.perf_counter()-t_load:.1f}s")
print(f"[被测] per-expert int8 误差单价 + 成员集 KL 的加性预测 | BS={BS} gc(归档口径)={GC_SET}"
      f" | 阶段={PHASES} | dtype={DTYPES} | 闸门④重算 {ROUNDS} 轮")
print(f"[口径] ① 单价 u_e = 【单专家扰动】档(gc={E-1}, 只有 e 在 CPU 腿走 int8)的 KL; kappa_e = u_e/hit_e"
      f" 才是与流量无关的专家属性。")
print(f"[口径] ② 被测点是【归档里已有的成员集】(P0-④c/P0-③补 的 gc 扫描 + 两轮 dyn 的 gc=3 成员集),"
      f" 锚点由本脚本【解析 results/ 文本】得到 —— 抄在代码里的常数一律不算对账。")
print(f"[口径] ③ 两个无自由参数模型对打: M0 计数(KL(全CPU) x CPU腿流量占比) vs M1 单价(sum_{{e in CPU}} u_e);"
      f" 另加 M2 = 同格 L2rel%(台账用了 45 行的那根向量轴)看它排不排出同一个序。")
print(f"[口径] ④ 🚨u_e 是【KL 域的经验可加性】, 不是误差上界: int8 误差进 silu 再进残差流, 二阶交叉项"
      f" 不被任何一格包含 => 拟合残差本身是结论(它说「单价 + 一个修正项」够不够)。")
print(f"[口径] ⑤ 质量列是确定量(闸门④), 但【跨语料段】比 KL 绝对值没有意义(裁判换了 P 的分布就换了)"
      f" => 只在段内比, 跨段只比 kappa_e 这种归一化量。")
print(f"[口径] ⑥ 显存/延迟都不报: 这一刀只动【放置身份】, 单价表建在引擎外, 不动运行时热路径。")

# ---- 阶段 -> ids/x/P
PH = []
for k, nm in enumerate(PHASES):
    kth = PHASES[:k].count(nm)
    if nm == "drift":
        ids, share = drift_ids(order_A)
        src = f"人造漂移(池 {DRIFT_POOL} 行, COLD={order_A[GC_SET:]}, 挑完 COLD 占 {share*100:.1f}% 命中)"
    else:
        ids, nch, nid = phase_ids(nm, kth)
        src = f"{nm} 第{kth+1}段 ({nch} 字符 -> {nid} token)"
    xcpu = capture_from(ids)
    PH.append(dict(key=(nm if nm != "drift" else "drift", kth), name=nm, kth=kth, ids=ids,
                   xcpu=xcpu, x=xcpu.cuda(), cnt=hits_of(BASE_W, xcpu), src=src))
    print(f"   · 阶段{k+1:<3d} {src:60s} 命中%={pct(PH[-1]['cnt'])}")
PH_KEYS = [p["key"] for p in PH]
ANCH, ANOTES = load_anchors(set(PH_KEYS))
for ln in ANOTES:
    print(f"   [锚点] {ln}")
print(f"[锚点] 拿到 {len(ANCH)} 个 (段, 成员集) 格; 段 = {PH_KEYS}")
if not ANCH:
    print("🚨没有任何锚点 => 加性预测没法做(只剩单价表)。检查 ARCH_M/ARCH_D 路径与 PHASES 是否对得上")

# ---- 引擎网格
SETS = sorted({mem for (_, mem) in ANCH if 0 < len(mem) < E}, key=lambda s: (len(s), sorted(s)))
DUP = SETS[0] if SETS else None
specs = [dict(tag="cpu0", members=frozenset(), solo=None, note="端点: 全 CPU(gc=0)"),
         dict(tag=f"gpu{E}", members=ALL, solo=None, note=f"端点: 全 GPU(gc={E}, CPU 腿 0 slot)")]
for e in range(E):
    specs.append(dict(tag=f"solo{e}", members=ALL - {e}, solo=e, note=f"单专家扰动: 只有 eid{e} 在 CPU 腿"))
for mem in SETS:
    specs.append(dict(tag="set" + "".join(str(v) for v in sorted(mem)), members=mem, solo=None,
                      note="归档成员集复现(槽序用 sorted, 归档用 order_A/C => 同成员集应同 KL, 到 SLOT_TOL)"))
if DUP is not None:
    specs.append(dict(tag="dup" + "".join(str(v) for v in sorted(DUP)), members=DUP, solo=None, dup=True,
                      note="同一成员集换槽序 => KL 与 set 档的差 = 归约序噪声底(判据 SLOT_TOL)"))
print(f"[网格] {len(specs)} 个引擎(建一个测完 {len(PH)} 段再释放): 端点 2 + 单专家扰动 {E} + 成员集 {len(SETS)}"
      + (" + 换序自检 1" if DUP is not None else ""))


def order_for(sp):
    """members 进前 gc 槽, 其余按 readme 命中降序进 CPU 腿; solo 档把该专家钉在最后一个槽。"""
    rest = [e for e in hits_desc if e not in sp["members"]]
    if sp.get("dup"):
        return sorted(sp["members"], reverse=True) + list(reversed(rest))
    if sp["solo"] is not None:
        return [e for e in sorted(sp["members"])] + [sp["solo"]]
    return sorted(sp["members"]) + rest


def fvals(vs):
    """只留有限值(nan/inf 一律丢掉) => max/min/sorted/sum 之前都先过这一道。"""
    return [v for v in vs if v == v and math.isfinite(v)]


def spair(a, b):
    """成对剔除 nan 之后的 Spearman(两轴缺同一个专家就没法配对)。
       🚨直接把带 nan 的数组喂给 spearman() 是【静默错误】: rank() 里 sorted 遇 nan 顺序相关,
          一个缺失的 u_e 就能把"kappa 排序段间相关 +1.00"变成任意数。返回 (rho, 实到点数)。"""
    idx = [i for i in range(min(len(a), len(b)))
           if a[i] == a[i] and b[i] == b[i] and math.isfinite(a[i]) and math.isfinite(b[i])]
    if len(idx) < 3:
        return float("nan"), len(idx)
    return spearman([a[i] for i in idx], [b[i] for i in idx]), len(idx)


def spreq(seq):
    """一串 KL 的相对极差 (max-min)/min(只数有限值; 不足 2 个有限值 => nan)。
       它是"这一格的轮间噪声底", 用来把【末位抖动】和【灾难脏格】分开: 前者 ~1e-7, 后者 ~1e+152。"""
    fin = [v["kl"] for v in seq if v["kl"] == v["kl"] and math.isfinite(v["kl"])]
    if len(fin) < 2 or min(fin) <= 0:
        return float("nan")
    return (max(fin) - min(fin)) / min(fin)


def _mm_of(d):
    return d["mm"]


def _accept(seq, j, jit=0, unst=0):
    """把"被接受的那一发"打包成旧返回结构(主循环/闸门④ 只认这套键)。
       🚨 y 必须跟着【被接受那一发】走: rel(=L2rel%) 是从 y 算的, 若 mm 取第 j 发而 y 取最后一发,
          同一行就会是两个不同帧的混合读数 —— 这正是 P1-②c 冒烟里"发了 3 次仍不收"那批格处置时
          最容易顺手犯的错(它们最后被拿去填了 u_e)。"""
    d = seq[j]
    r = dict(mm=d["mm"], worst=d["worst"], yok=d["yok"], yamax=d["yamax"], okQ=d["okQ"], aQ=d["aQ"],
             nq=j, natt=len(seq), y=d["y"], L=None, unst=int(unst), jit=int(jit),
             seq=seq, sp=spreq(seq))
    r["bad"] = int(unst or not (d["okQ"] and d["yok"]))
    return r


def leg_run(m, ph):
    """同一格连打, 直到【相邻两发】的 KL/TV/top-1 逐位相同才记账(bench_metrics 给裁判写重试是同一件事)。
       🚨 冒烟实测: 首轮会蹦出【红线抓不到的脏格】—— KL=1.10e+152 而 |logit|<=1e4 照过、重跑计数 0/38,
       同一格整格重建重算是干净的 3.116e-05 => "打一发就记账"在这台机器上不成立, 红线也不是稳定性判据。
       主循环与闸门④ 必须共用这一条 —— 否则闸门④ 测到的是【两条路的差】而不是【确定量的差】。

       【两级判据】(P1-②c 冒烟: 3 发上限下 6/66 格收不了, 而 PROBE 探针 4 格 x 10 发【全部逐位相同】
       => 那些格不是"值不可复现", 而是 REDO 太短撞上了 A,B,A 这种【跳回来】的序列):
         判据① 相邻两发逐位相同 => 收(jit=0) —— 与老口径一致, 也是绝大多数格走的路;
         判据② 打完 REDO 发没有相邻对, 但同一个值【出现过 >=2 次】=> 这个值是可复现的, 收, 但打 jit=1,
                并把这一格的轮间相对极差 sp 交出去 => 表能自证"这一格的噪声底有多大"。
         两级都不过 => unst=1 => 上层把 KL 记 nan 剔除。
       返回: mm/worst/yok/yamax/okQ/aQ=被接受那一发的量, y=同一发的引擎输出(rel 必须由它算),
             nq=被接受发的下标, natt=实际打了几发, jit/unst=见上, seq=每发留档(kl/tv/top1/ybit/yamax/aQ)。"""
    seq = []
    for att in range(max(3, REDO)):
        outs = {mo: m(ph["x"], async_mode=True, mode=mo) for mo in H.MODES}
        worst = max((outs[u] - outs[v]).abs().max().item()
                    for i, u in enumerate(outs) for v in list(outs)[i + 1:])
        y = outs[MODE]
        yok, yamax = red_ok(y, Y_RED)
        L = splice(ph["ids"], y.cpu())
        okQ, aQ = red_ok(L)
        mm = metrics(ph["P"], L, ph["ids"]) if (okQ and yok) else dict(
            top1=float("nan"), rec=float("nan"), kl=float("nan"), kl95=float("nan"), tv=float("nan"))
        # 🚨 y 是引擎的输出缓冲, 下一发很可能就地覆写 => 比"上一发是否逐位"必须先 clone
        ybit = 1 if (seq and bool(torch.equal(y, seq[-1]["y"]))) else 0
        seq.append(dict(att=att, mm=mm, y=y.detach().clone(), worst=worst, yok=yok, yamax=yamax,
                        okQ=okQ, aQ=aQ, ybit=ybit,
                        kl=mm["kl"], tv=mm["tv"], top1=mm["top1"]))
        del outs, y, L
        # 🚨 是 [k] 不是 ["k"]: 上一版写成 ["k"] => 判据① 每次求值都 KeyError(第 2 发必崩, 全表 0 格能跑)
        if att and all(_mm_of(seq[att])[k] == _mm_of(seq[att - 1])[k] for k in ("kl", "tv", "top1")):
            return _accept(seq, att)
        _ = seq[att]["kl"]                       # 只为了给 spreq/判据② 留一份扁平行, 别删(seq 里要用)
    cnt = {}
    for d in seq:
        k = (d["kl"], d["tv"], d["top1"])
        cnt[k] = cnt.get(k, 0) + 1
    if cnt and max(cnt.values()) >= 2:
        kb = max(cnt, key=lambda k: (cnt[k], -abs(math.log(max(k[0], 1e-30)))))
        j = next(i for i, d in enumerate(seq) if (d["kl"], d["tv"], d["top1"]) == kb)
        return _accept(seq, j, jit=1)
    return _accept(seq, len(seq) - 1, unst=1)


# ---------------------------------------------------------------- 主循环: 逐引擎(建->测各段->释放)
CELL = {}        # (spec_tag, phase_key) -> dict
GATE3, GATE2, DJUD, NQ = [], [], [], []
JUDGE = mk(0, order_A)                      # P = 全专家 fp32 稠密(gc=0), 与 P0-③/④ 同口径
for i, ph in enumerate(PH):
    box = {}
    lg = splice(ph["ids"], torch.zeros(BS, D), box)
    d_sp = (box["x"][0] - ph["xcpu"]).abs().max().item()
    bit_sp = int(torch.equal(box["x"][0], ph["xcpu"]))
    ts0, ti0 = JUDGE.route(ph["x"])
    ref0 = JUDGE.fp32_reference(ph["x"], ts0, ti0)
    PL = splice(ph["ids"], ref0.cpu())
    okP, aP = red_ok(PL)
    ORIG = plain(ph["ids"])
    M0 = metrics(ORIG, PL, ph["ids"])
    ph.update(P=PL, ts0=ts0, ti0=ti0, d_sp=d_sp, bit_sp=bit_sp, absmaxP=aP, okP=okP,
              ref=M0, absmaxO=red_ok(ORIG)[1], ref0t=ref0)
    print(f"  阶段{i+1} 闸门① splice互差={d_sp:.9f} 逐位={bit_sp}(必须1) | 裁判 absmax={aP:.3g}"
          f"(红线{LOGIT_RED:g} {'过' if okP else '🚨脏 => 整段作废'}) | 参照(换成 MoE 本身) "
          f"top1={M0['top1']:.2f}% KL={M0['kl']:.3e}")
    del box, lg, ORIG
del JUDGE
_pycollect(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- PROBE(可选, 默认关): 脏格探针
# PROBE=solo2:input.txt,set34567:drift [PROBE_SHOTS=12] => 只建这几个引擎、每格连打 N 发,
# 印每发的 KL / y 是否逐位 => 回答"轮间不自洽"到底是【末位抖动】还是【灾难脏格】。
# 为什么不直接改判据先跑全表: 全表 ~4min, 一个探针 ~20s; 这两类的处置相反(该收 vs 该丢), 猜不得。
PROBE = [p.strip() for p in os.environ.get("PROBE", "").split(",") if p.strip()]
PROBE_SHOTS = int(os.environ.get("PROBE_SHOTS", 12))
if PROBE:
    for spec in PROBE:
        tag, _, pn = spec.partition(":")
        sp = next((s for s in specs if s["tag"] == tag), None)
        ph = next((p for p in PH if p["name"] == pn), None)
        if sp is None or ph is None:
            print(f"🚨PROBE 认不出 {spec!r}  可用 tag={[s['tag'] for s in specs]}"
                  f" 可用段={[p['name'] for p in PH]}")
            continue
        m = mk(len(sp["members"]), order_for(sp))
        y0, kls = None, []
        for sh in range(PROBE_SHOTS):
            outs = {mo: m(ph["x"], async_mode=True, mode=mo) for mo in H.MODES}
            y = outs[MODE]
            if y0 is None:
                y0 = y.detach().cpu().clone()
            same = int(bool(torch.equal(y.cpu(), y0)))
            L = splice(ph["ids"], y.cpu())
            mm = metrics(ph["P"], L, ph["ids"])
            kls.append(mm["kl"])
            print(f"    {tag}/{ph['name']} 发{sh:2d} y逐位=首发{same}"
                  f" |y|absmax={float(y.abs().max()):.4g} KL={mm['kl']:.9e}"
                  f" TV={mm['tv']:.6e} top1={mm['top1']:.4f} 三模式互差={max((outs[u] - outs[v]).abs().max().item() for i, u in enumerate(outs) for v in list(outs)[i + 1:]):.3e}")
            del outs, y, L, mm
        fin = [v for v in kls if math.isfinite(v)]
        print(f"  [{tag}/{ph['name']}] {PROBE_SHOTS} 发: y 逐位不同的发数见上 | KL 不同值 {len(set(kls))} 个"
              f" 极差={max(kls) - min(kls):.3e} 相对极差={spreq([dict(kl=v) for v in kls]):.3e}"
              f" 非有限 {len(kls) - len(fin)} 个")
        del m
        _pycollect(); torch.cuda.empty_cache()
    print("[PROBE] 探针跑完即退出(PROBE 置空 = 回到正常判决表) —— 探针数不进记录")
    sys.exit(0)

t0 = time.perf_counter()
for si, sp in enumerate(specs):
    order = order_for(sp)
    gc = len(sp["members"])
    m = mk(gc, order)
    for ph in PH:
        eid_row = torch.tensor(order_A)[ph["ti0"].detach().cpu().long()]   # 裁判 ti0 属 order_A 槽序
        ts, ti = m.route(ph["x"])
        eid = torch.tensor(order)[ti.detach().cpu().long()]
        idx_d = int((torch.sort(eid, dim=1).values != torch.sort(eid_row, dim=1).values).sum())
        r = leg_run(m, ph)
        mm, y = r["mm"], r["y"]
        NQ.append(dict(tag=sp["tag"], nm=ph["name"], nq=r["nq"], unst=r["unst"], yamax=r["yamax"],
                       aQ=r["aQ"], leg="首轮", seq=r["seq"], jit=r["jit"], natt=r["natt"], sp=r["sp"],
                       mem=sp["members"]))
        if r["bad"]:
            # 🚨口径早写了"打完仍不自洽 => KL 记 nan 不进台账", 代码却没做: 冒烟里 6/66 格照原样
            #    填进了 u_e 表和 M1 配平(等于拿【没自洽的最后一发】当确定量)。这一句才是那句口径。
            mm = dict(top1=float("nan"), rec=float("nan"), kl=float("nan"), kl95=float("nan"),
                      tv=float("nan"))
        refi = m.fp32_reference(ph["x"], ts, ti)
        _, _, rel = H.err_metrics(y, refi)
        djud = (refi - ph["ref0t"]).abs().max().item()
        DJUD.append((sp["tag"], ph["name"], djud))
        cpu_eid = eid[ti.detach().cpu().long() >= gc]
        cpu_set = frozenset(int(v) for v in cpu_eid.unique().tolist())
        rcnt = torch.bincount(eid_row.reshape(-1).long(), minlength=E)
        exp_ncpu = int(sum(int(rcnt[e]) for e in range(E) if e not in sp["members"]))
        exp_set = frozenset(e for e in range(E) if e not in sp["members"] and int(rcnt[e]) > 0)
        CELL[(sp["tag"], ph["key"])] = dict(kl=mm["kl"], tv=mm["tv"], top1=mm["top1"], rec=mm["rec"],
                                            rel=rel, mode=r["worst"], idx=idx_d, ncpu=int(cpu_eid.numel()),
                                            cpu_set=cpu_set, yamax=r["yamax"], aQ=r["aQ"],
                                            unst=r["unst"], bad=r["bad"])
        GATE2.append(idx_d)
        GATE3.append((sp["tag"], ph["name"], gc, cpu_set, sp["members"], int(cpu_eid.numel()),
                      exp_set, exp_ncpu))
        del r, mm, y, refi, eid, eid_row, cpu_eid
    del m
    _pycollect(); torch.cuda.empty_cache()
    print(f"  [{sp['tag']:<8s}] gc={gc} {len(PH)} 段测完 {time.perf_counter()-t0:6.1f}s"
          f" | loadavg={open('/proc/loadavg').read().split()[0]}")

# ---- 闸门④ 锚点格跨轮逐位重算
RECHECK = [(s["tag"], ph["key"]) for s in specs for ph in PH
           if (s["tag"], ph["key"]) in CELL and (ph["key"], s["members"]) in ANCH]
d4, n4, nan4 = 0.0, 0, 0
bad4 = []
NQ2 = []
for rd in range(max(1, ROUNDS)):
    for tag, pk in RECHECK:
        sp = next(s for s in specs if s["tag"] == tag)
        ph = next(p for p in PH if p["key"] == pk)
        m = mk(len(sp["members"]), order_for(sp))
        r = leg_run(m, ph)                       # 🚨 与首轮共用同一条路(含"两发一致才收"判据)
        NQ2.append(dict(tag=tag, nm=ph["name"], nq=r["nq"], unst=r["unst"], yamax=r["yamax"],
                        aQ=r["aQ"], leg="重算", seq=r["seq"], jit=r["jit"], natt=r["natt"], sp=r["sp"],
                        mem=sp["members"]))
        kl2 = r["mm"]["kl"]
        ok2 = bool(r["okQ"]) and not r["unst"]        # 重算这发可用 = 红线内 且 相邻两发自洽
        kl1 = CELL[(tag, pk)]["kl"]
        if not (math.isfinite(kl1) and math.isfinite(kl2)):
            nan4 += 1
        if abs(kl2 - kl1) > d4:
            d4 = abs(kl2 - kl1)
            bad4 = [(tag, pk, ph["name"], sp["members"], kl1, kl2, ok2)]
        elif abs(kl2 - kl1) == d4 and d4 > 0:
            bad4.append((tag, pk, ph["name"], sp["members"], kl1, kl2, ok2))
        n4 += 1
        del m, r
        _pycollect(); torch.cuda.empty_cache()
print(f"  闸门④ 重建重算 KL 互差 max={d4:.3e} (n={n4}, 必须 0 = 这些数是确定量; ROUNDS={ROUNDS})"
      f" | 非有限值 {nan4} 格(nan 不参与 max => 不单独印就会把静默的 0 读成全绿)")
for tag, pk, nm, mem, k1, k2, ok2 in bad4[:4]:
    if d4 > 0:
        lo = min(abs(k1), abs(k2))
        rel4 = abs(k2 - k1) / lo if lo > 0 else float("inf")
        print(f"    🚨最大不符格: {tag} 段={nm} 成员={sorted(mem)} 首轮={k1:.6e} 重算={k2:.6e}"
              f" 差={abs(k2-k1):.3e} 相对={rel4:.3e} 重算是否干净={ok2}")
        ck = ANCH.get((pk, mem))
        if ck is not None and ck[0] > 0:
            av, src = ck
            d1, d2 = abs(k1 - av) / av, abs(k2 - av) / av
            pick = ("首轮" if d1 < d2 else "重算") if d1 != d2 else "平票"
            print(f"      归档投票: 归档={av:.6e}({src}) 首轮偏={d1:.3g} 重算偏={d2:.3g}"
                  f" => 归档支持【{pick}】那一模 => 这一格是【两模格】(两个值都能自洽复现),"
                  f" 不是测量误差。🚨逐位判据不许放宽: 它判的就是逐位 0,"
                  f" 归档容差(2e-3)只回答'哪个值有独立证据', 不回答'这格可以进台账'。")
NAME = {p["key"]: p["name"] for p in PH}
NQD = NQ + NQ2
UNST = [v for v in NQD if v["unst"]]
JIT = [v for v in NQD if v.get("jit")]
shots = sum(v['natt'] for v in NQD) / max(1, len(NQD))
_spd = [v["sp"] for v in NQD if v["sp"] == v["sp"] and math.isfinite(v["sp"])]
print(f"  重打统计(判据: 相邻两发 KL/TV/top-1 逐位相同即记账, 兜底=同一个值出现过 >=2 次, 上限 REDO={REDO}):"
      f" 平均 {shots:.2f} 发/格"
      f" | 首轮 {len(NQ)} 格 平均 {sum(v['natt'] for v in NQ)/max(1, len(NQ)):.2f} 发"
      f" | 闸门④ 重算 {n4} 格 平均 {sum(v['natt'] for v in NQ2)/max(1, n4):.2f} 发"
      f"  => 闸门④ 是在『已自洽』的值上做第 3 次独立重建重算")
_spmax = max(_spd) if _spd else 0.0
print(f"    判据②(非相邻同值)兜底收下的格 {len(JIT)} 个"
      f" | 全表最大【轮间相对极差】={_spmax:.3e}"
      f"(这一列本身不是判据, 记账判据是【相邻两发逐位】/【同值出现>=2 次】)")
if _spmax <= 1e-3:
    print(f"      => 判决: 极差 <=1e-3, 远小于要读的段内 KL 差(10% 量级) => 不影响读表")
else:
    print(f"      🚨判决: 极差 {_spmax:.3e} 不是噪声底量级 => 至少一格是【两模/脏发】,"
          f" 见下面点名行; 那一格的 KL 只有在【逐位判据或归档投票】二选一给了方向之后才可引用,"
          f" 否则按不确定处理(不放宽判据, 只是不许把它当确定量直接进台账)")
_spu = sorted([v for v in NQD if v["sp"] == v["sp"] and math.isfinite(v["sp"])],
              key=lambda v: -v["sp"])[:3]
if _spmax > 1e-3 and _spu:
    print("      轮间相对极差最大 3 格(点名):")
    for v in _spu:
        print(f"        · {v['leg']} {v['tag']} 段={v['nm']} 成员={sorted(v.get('mem', ()))}"
              f" 发了 {v['natt']} 发 y逐位={[x['ybit'] for x in v['seq']]}"
              f" 不同 KL 值 {len({x['kl'] for x in v['seq']})} 个 相对极差={v['sp']:.3e}"
              f" KL=[" + " ".join(f"{x['kl']:.6e}" for x in v["seq"]) + "]")
if JIT:
    for v in JIT[:6]:
        print(f"      · {v['leg']} {v['tag']} 段={v['nm']} 打了 {v['natt']} 发 y逐位={[x['ybit'] for x in v['seq']]}"
              f" KL=[" + " ".join(f"{x['kl']:.6e}" for x in v["seq"]) + f"] 相对极差={v['sp']:.3e}")
if UNST:
    print(f"  🚨打完 {REDO} 发仍不逐位相同的格 {len(UNST)} 个 => KL 记 nan 不进台账"
          f"(本机偶发脏格: 红线 |logit|<=1e4 抓不到它, 冒烟实测首轮 1.10e+152 vs 重算 3.12e-05):")
    for v in UNST[:8]:
        sq = v["sp"]
        kind = ("①灾难脏格(量级都变 => 必须丢)" if not (sq == sq and sq <= 1e-3)
                else ("②引擎末位抖动(y 就逐位不上)" if any(not x["ybit"] for x in v["seq"][1:])
                      else "③下游前向末位抖动(y 逐位, KL 末位不同)"))
        print(f"    · {v['leg']} {v['tag']} 段={v['nm']} 发了 {v['natt']} 次"
              f" y逐位={[x['ybit'] for x in v['seq']]} KL=["
              + " ".join(f"{x['kl']:.6e}" for x in v["seq"]) + f"] 相对极差={sq:.3e}")
        print(f"        y_absmax={v['yamax']:.3g} L_absmax={v['aQ']:.3g}"
              f" (红线 y {Y_RED:g} / L {LOGIT_RED:g} 都没抓到) => {kind}")
SUS = [(t, NAME.get(pk, pk), c["kl"]) for (t, pk), c in CELL.items() if c["kl"] == c["kl"] and c["kl"] > 1e-2]
if SUS:
    print(f"  🚨可疑大 KL 格 {len(SUS)} 个(整表量级在 1e-5~1e-4, 超过 1e-2 一律当脏读, 别拿去拟合):")
    print("      " + " | ".join(f"{t} 段={nm} KL={k:.3e}" for t, nm, k in SUS[:6]))

# ---------------------------------------------------------------- 表A 单价表
print("\n" + "=" * 158)
print(f"swiglu | P1-②c per-expert int8 误差单价表 + 成员集 KL 的加性预测 | BS={BS} E={E} top_k={K}"
      f" dtype=fp32 | 引擎 {len(specs)} 个 | 锚点 {len(ANCH)} 格 | 耗时 {time.perf_counter()-t0:.0f}s")
print("=" * 158)
KAP = {}
for i, ph in enumerate(PH):
    hit = [float(v) / float(ph["cnt"].sum()) for v in ph["cnt"]]
    print(f"\n  阶段{i+1} = {ph['src']}")
    print(f"  参照【未挂钩真模型 -> P】top1={ph['ref']['top1']:.2f}% KL={ph['ref']['kl']:.3e}"
          f"  🚨这条是「换成 MoE 本身」的代价, 不是这一刀的代价")
    k0 = CELL[("cpu0", ph["key"])]["kl"]
    ke = CELL[(f"gpu{E}", ph["key"])]["kl"]
    print(f"{'eid':>4}{'命中%':>8}{'u_e=KL(e only CPU)':>20}{'TV':>12}{'top1一致%':>10}{'L2rel%':>10}"
          f"{'kappa=u/hit':>14}{'相对最便宜':>11}{'CPU腿slot':>10}{'CPU腿eid集':>12}")
    us = [CELL[(f"solo{e}", ph["key"])]["kl"] for e in range(E)]
    # 🚨这一格可能因为"打完 REDO 发仍不自洽"被记成 nan(闸门④/⑥那两行会点名)。
    #    带 nan 的 min()/sum() 在 Python 里是【顺序相关】的静默错误 => 名次与求和都只对有限值算。
    usf = fvals(us)
    nmiss = len(us) - len(usf)
    umin = max(min(usf), 1e-30) if usf else float("nan")
    KAP[ph["key"]] = [(us[e] / max(hit[e], 1e-12) if us[e] == us[e] and math.isfinite(us[e])
                       else float("nan")) for e in range(E)]
    for e in range(E):
        c = CELL[(f"solo{e}", ph["key"])]
        print(f"{e:>4}{hit[e]*100:>8.1f}{us[e]:>20.4e}{c['tv']:>12.3e}{c['top1']:>10.2f}{c['rel']:>10.4f}"
              f"{KAP[ph['key']][e]:>14.3e}{us[e]/umin:>11.2f}{c['ncpu']:>10}{str(sorted(c['cpu_set'])):>12}")
    ssum = sum(usf)
    print(f"  端点: gc=0(全CPU) KL={k0:.4e} TV={CELL[('cpu0', ph['key'])]['tv']:.3e}"
          f" | gc={E}(全GPU) KL={ke:.3e} (CPU 腿 0 slot => 应塌到 ~1e-12)"
          f" | Σu_e={ssum:.4e} => 🚨加性检验 全CPU/Σu = "
          f"{(k0/ssum if ssum > 0 else float('nan')):.3f} (1.0 = int8 误差在 KL 域可加)"
          + (f"  🚨本段缺 {nmiss} 个 u_e(不自洽已记 nan) => Σu_e 只是【部分和】,"
             f" 上式这个比值天然【偏低】, 不能当判决读" if nmiss else ""))

# ---- kappa_e 跨段稳定性: P1-②d 的判据能不能【查表】而不是在线估
if len(PH) >= 2:
    from itertools import combinations
    print(f"\n  kappa_e 跨段稳定性(P1-②d 要用它决定【查表】还是【在线估】) | 段={[p['name'] for p in PH]}")
    print(f"{'eid':>4}{'hit%P1':>8}" + "".join(f"{('kappa@' + p['name'])[:16]:>16}" for p in PH)
          + f"{'极差比':>10}{'kappa 名次(P1)':>15}")
    k0v = KAP[PH_KEYS[0]]
    ord1 = sorted([e for e in range(E) if math.isfinite(k0v[e])], key=lambda e: -k0v[e])
    for e in range(E):
        vs = [KAP[p["key"]][e] for p in PH]
        vsf = fvals(vs)
        rng = (max(vsf) / max(min(vsf), 1e-30)) if len(vsf) == len(vs) else float("nan")
        print(f"{e:>4}{[float(v) / float(PH[0]['cnt'].sum()) * 100 for v in PH[0]['cnt']][e]:>8.1f}"
              + "".join(f"{v:>16.3e}" for v in vs)
              + f"{rng:>10.2f}{(ord1.index(e) + 1) if e in ord1 else '-':>15}"
              + ("   🚨有段缺这一格 => 极差比不算" if len(vsf) != len(vs) else ""))
    rs = []
    for a, b in combinations(PH, 2):
        r, nv = spair(KAP[a["key"]], KAP[b["key"]])
        rs.append((r, nv, f"{a['name']}~{b['name']}"))
    rsf = [r for r, _, _ in rs if r == r]
    print(f"    专家之间 kappa 排序的段间 Spearman: "
          + " ".join(f"{nm}:{'{:+.2f}(n={})'.format(r, nv) if r == r else 'nan(点数不足)'}" for r, nv, nm in rs)
          + (f"  均值={sum(rsf) / len(rsf):+.2f}" if rsf else "")
          + " (≈+1 => 单价表可查表; 低 => 只能在线增量估)")
    us_all = {p["key"]: [CELL[(f"solo{e}", p["key"])]["kl"] for e in range(E)] for p in PH}
    print("    同一格的 u_e 排序段间 Spearman: " + " ".join(
        "{}:{:+.2f}(n={})".format(f"{a['name']}~{b['name']}", *spair(us_all[a["key"]], us_all[b["key"]]))
        for a, b in combinations(PH, 2)))

# ---------------------------------------------------------------- 表B 成员集解释
def pred_m0(pk, cpu):
    k0 = CELL[("cpu0", pk)]["kl"]
    hit = [float(v) / float(PHnext(pk)["cnt"].sum()) for v in PHnext(pk)["cnt"]]
    return k0 * sum(hit[e] for e in cpu)


def pred_m1(pk, cpu, cap_key=None):
    hit = [float(v) / float(PHnext(pk)["cnt"].sum()) for v in PHnext(pk)["cnt"]]
    if cap_key is None:
        return sum(CELL[(f"solo{e}", pk)]["kl"] for e in cpu)
    return sum(hit[e] * KAP[cap_key][e] for e in cpu)


def PHnext(pk):
    return next(p for p in PH if p["key"] == pk)




CELLBYSET = {}
for s in specs:
    if s["solo"] is None and not s.get("dup"):
        for ph in PH:
            c = CELL.get((s["tag"], ph["key"]))
            if c:
                CELLBYSET[(ph["key"], s["members"])] = c
print("\n" + "-" * 158)
print("  表B: 归档成员集 KL 能否被【无自由参数】的两个模型解释(M0 计数/覆盖流量, M1 单价 sum u_e, M2 L2rel%)"
      "   比值 = 预测/实测(1.0 = 模型中靶)")
print(f"{'阶段':>6}{'GPU成员集':>14}{'CPU集':>16}{'CPU流量%':>9}{'实测归档KL':>13}{'实测本表KL':>13}"
      f"{'本表/归档':>10}{'L2rel%':>9}{'M0比值':>9}{'M1同段比值':>11}{'M1跨段比值':>11}{'M2 rel排名':>11}{'来源':>26}")
fit = []
for i, ph in enumerate(PH):
    rows = []
    for mem in [frozenset()] + SETS:
        ck = ANCH.get((ph["key"], mem))
        c = CELLBYSET.get((ph["key"], mem))
        if (ck is None or c is None or not math.isfinite(c["kl"])
                or not math.isfinite(c["rel"]) or c["kl"] <= 1e-10):
            # 脏格(nan)/全 GPU 端点(kl<=1e-10)都不进配平, 也不参与 rel 名次 ——
            # Python 的 sorted/max 遇到 nan 是顺序相关的, 一个脏格能把名次排成假的
            continue
        cpu = ALL - mem
        hit = [float(v) / float(ph["cnt"].sum()) for v in ph["cnt"]]
        ka, kb = ck[0], c["kl"]
        p0, p1 = pred_m0(ph["key"], cpu), pred_m1(ph["key"], cpu)
        p1x = pred_m1(ph["key"], cpu, cap_key=PH_KEYS[0])
        rows.append(dict(mem=mem, cpu=cpu, cpu_hit=100.0 * sum(hit[e] for e in cpu), ka=ka, kb=kb,
                         rel=c["rel"], r0=p0 / kb, r1=p1 / kb, r1x=p1x / kb, src=ck[1],
                         j=len(rows), ep=(len(mem) == 0)))
    if not rows:
        print(f"  阶段{i+1}: 没有任何可配的锚点格")
        continue
    ord_rel = sorted(range(len(rows)), key=lambda j: -rows[j]["rel"])
    rk = {j: pp + 1 for pp, j in enumerate(ord_rel)}
    for r in sorted(rows, key=lambda r: -r["kb"]):
        print(f"{('P' + str(i+1)):>6}{str(sorted(r['mem'])):>14}{str(sorted(r['cpu'])):>16}"
              f"{r['cpu_hit']:>9.1f}{r['ka']:>13.4e}{r['kb']:>13.4e}{r['kb']/r['ka']:>10.3f}"
              f"{r['rel']:>9.4f}{r['r0']:>9.2f}{r['r1']:>11.2f}{r['r1x']:>11.2f}"
              f"{('第' + str(rk[r['j']]) + '名'):>11}{r['src']:>26}")
    fit.append(dict(pk=ph["key"], rows=rows))
    # 🚨非有限值(kl/r0/r1/r1x/rel 任一 nan)的格子从【段内统计】里剔掉: Python 的 max/sorted 遇到 nan
    #   是【顺序相关】的, 一个脏格能把整段的极差比变成 nan 或假的 1.00。剔除数照印, 不静默。
    okrows = [r for r in rows
              if all(isinstance(r[k], float) and math.isfinite(r[k]) for k in ("kb", "r0", "r1", "r1x", "rel"))]
    ks = [r["kb"] for r in okrows]
    if len(okrows) >= 3:
        m0v = [r["r0"] * r["kb"] for r in okrows]        # == pred_m0, 但已保证有限(nan 格在上面剔除)
        m1v = [r["r1"] * r["kb"] for r in okrows]
        relv = [r["rel"] for r in okrows]
        covv = [r["cpu_hit"] for r in okrows]
        r1s = [r["r1"] for r in okrows]
        print(f"       段内 Spearman(实测KL vs): 覆盖流量={spearman(covv, ks):+.2f}"
              f"  M0计数={spearman(m0v, ks):+.2f}  M1单价={spearman(m1v, ks):+.2f}"
              f"  M2 L2rel%={spearman(relv, ks):+.2f}   n={len(okrows)}"
              + (f"  🚨另 {len(rows) - len(okrows)} 格非有限值已剔除(见闸门④/⑥那两行)"
                 if len(okrows) != len(rows) else "")
              + f"   M1 比值极差={min(r1s):.2f}~{max(r1s):.2f}")
        fit[-1]["rho"] = (spearman(covv, ks), spearman(m0v, ks), spearman(m1v, ks), spearman(relv, ks))
    else:
        print(f"       段内可配格 {len(okrows)}/{len(rows)} 个(非 3 => 不算段内 Spearman)")
        fit[-1]["rho"] = None

# ---- 全表(跨段)合并: KL 绝对值跨段不可比 => 只拼【无量纲比值(预测/实测)】
allrows = [r for f in fit for r in f["rows"]]
pool = [r for r in allrows if not r.get("ep")]
if len(pool) >= 3:
    print(f"\n  跨段合并 n={len(pool)} 格(剔掉 gc=0 端点行, 它是 M0 的标定点 r0 恒=1):")
    for nm, key in (("M0 计数(覆盖流量)", "r0"), ("M1 单价(同段 u_e)", "r1"),
                    ("M1 单价(跨段 kappa@P1)", "r1x")):
        vs = sorted(r[key] for r in pool if r[key] == r[key])
        gm = math.exp(sum(math.log(max(v, 1e-30)) for v in vs) / len(vs))
        print(f"    {nm:<26s} 比值(预测/实测) 几何均值={gm:6.2f} 范围={vs[0]:6.2f}~{vs[-1]:6.2f}"
              f" 落在[0.5,2]内 {sum(1 for v in vs if 0.5 <= v <= 2)}/{len(vs)}")
    rs = [f["rho"] for f in fit if f.get("rho")]
    if rs:
        mid = lambda vals: sorted(vals)[len(vals) // 2]
        print(f"    段内 Spearman(实测 KL vs) 各段中位: 覆盖流量={mid([r[0] for r in rs]):+.2f}"
              f"  M0={mid([r[1] for r in rs]):+.2f}  M1={mid([r[2] for r in rs]):+.2f}"
              f"  M2_L2rel={mid([r[3] for r in rs]):+.2f}  (n={len(rs)} 段)")

# ---------------------------------------------------------------- 闸门汇总
bad3 = [g for g in GATE3 if g[3] != g[6] or g[5] != g[7]]
print("\n" + "-" * 158)
print(f"  闸门① splice 逐位: {sum(p['bit_sp'] for p in PH)}/{len(PH)} 过")
print(f"  闸门② 路由 eid 集与基准逐位: 不符 {sum(1 for v in GATE2 if v)} 处 / {len(GATE2)} 格(必须 0)")
print(f"  闸门③ CPU 腿成员集 == 预期: 🚨不符 {len(bad3)} 格"
      + (" | " + "; ".join(str(b) for b in bad3[:4]) if bad3 else " => 单价表的每一格都只动了该动的那个人"))
print(f"  闸门④ 跨轮 KL 互差 max={d4:.3e} (n={n4}) | 非有限值 {nan4} 格"
      + (" 全绿" if d4 == 0 and nan4 == 0 else " 🚨有格不符或不可用(不可用=红线越界或打完 REDO 发仍不自洽)"))
print(f"  闸门④b 裁判重算互差(换槽序后 fp32_reference vs 建段时那份; 归约序噪声底, 非判决):"
      f" max={max(d for _, _, d in DJUD):.3e} (n={len(DJUD)})")
_ep = fvals([CELL[(f'gpu{E}', p['key'])]['kl'] for p in PH])
print(f"  闸门⑤ 端点: gc={E} 全 GPU KL max={max(_ep) if _ep else float('nan'):.3e}"
      f"(红线 1e-9) | gc=0 全 CPU 与 Σu_e 的比值见每段末行")
print(f"  闸门⑥ 红线 |logit|<={LOGIT_RED:g}: 越界格 {sum(1 for c in CELL.values() if c['bad'])}"
      f"/{len(CELL)} | P absmax max={max(p['absmaxP'] for p in PH):.3g}"
      f" | 引擎出口 y absmax max={max(c['yamax'] for c in CELL.values()):.3g}(红线 {Y_RED:g})")
worst = 0.0
ncell = nbad = nfloored = nfloorbad = 0
for (pk, mem), (ka, src) in ANCH.items():
    c = CELLBYSET.get((pk, mem))
    if c is None or not (c["kl"] == c["kl"]):
        nbad += 1
        continue
    if max(c["kl"], ka) <= KL_FLOOR:
        nfloored += 1
        r = max(c["kl"], ka) / max(min(c["kl"], ka), 1e-30)
        if r > FLOOR_RATIO:
            nfloorbad += 1
            print(f"  🚨闸门⓪ 噪声底档连数量级都不符: 段={pk} 成员集={sorted(mem)} 本表={c['kl']:.4e}"
                  f" 归档={ka:.4e} 比值={r:.3g}")
        continue
    ncell += 1
    d = abs(c["kl"] - ka) / max(ka, 1e-30)
    if d > ANCHOR_TOL:
        print(f"  🚨闸门⓪ 归档不符: 段={pk} 成员集={sorted(mem)} 本表={c['kl']:.4e} 归档={ka:.4e}"
              f" 相对偏差={d:.3g} (阈值 {ANCHOR_TOL:g})")
    worst = max(worst, d)
print(f"  闸门⓪ 归档对账: 常规 {ncell} 格(阈值 {ANCHOR_TOL:g} = 归档 4 位有效) 最大相对偏差 {worst:.3g}"
      + (" <= 阈值" if worst <= ANCHOR_TOL else " 🚨超阈值")
      + f" | 噪声底档 {nfloored} 格(两值都 <= KL_FLOOR {KL_FLOOR:g}, KL 已在浮点噪声底 => 只判比值"
      + f" <= {FLOOR_RATIO:g} 即同数量级)"
      + (" 全过" if nfloorbad == 0 else f" 🚨{nfloorbad} 格不符"))
# 闸门⑦ 段复现 + 换序自检
for i, ph in enumerate(PH):
    for rel in ARCH_D:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        dph, _nb = parse_dyn(p)
        if ph["key"] in dph and dph[ph["key"]]["hits"]:
            dd = max(abs(a - b) for a, b in zip(pct(ph["cnt"]), dph[ph["key"]]["hits"]))
            print(f"  闸门⑦ 段{i+1}({ph['name']}) 命中% vs {os.path.basename(rel)[:22]}: 最大差"
                  f" {dd:.2f}pp (阈值 {HIT_TOL}pp) {'过' if dd <= HIT_TOL else '🚨不是同一批行'}")
    for rel in ARCH_M:
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        am = parse_metrics(p)
        if am.get("corpus") == ph["name"] and am.get("hits"):
            dd = max(abs(a - b) for a, b in zip(pct(ph["cnt"]), am["hits"]))
            print(f"  闸门⑦ 段{i+1}({ph['name']}) 命中% vs {os.path.basename(rel)[:26]}: 最大差"
                  f" {dd:.2f}pp (阈值 {HIT_TOL}pp) {'过' if dd <= HIT_TOL else '🚨不是同一批行'}")
for s in specs:
    if s.get("dup"):
        base = "set" + "".join(str(v) for v in sorted(s["members"]))
        dd = [(ph["name"], CELL[(base, ph["key"])]["kl"], CELL[(s["tag"], ph["key"])]["kl"]) for ph in PH]
        rd = [abs(a - b) / max(max(a, b), 1e-30) for _, a, b in dd]
        wd = max(rd)
        bit = all(a == b for _, a, b in dd)
        verdict = ("逐位相同" if bit else
                   (f"逐位不同但最大相对互差 {wd:.3g} <= SLOT_TOL {SLOT_TOL:g}" if wd <= SLOT_TOL
                    else f"🚨最大相对互差 {wd:.3g} > SLOT_TOL {SLOT_TOL:g} => KL 真的不只依赖成员集"))
        print(f"  换序自检 成员集{sorted(s['members'])} 换槽序后 KL: "
              + " | ".join(f"{n} {a:.4e} vs {b:.4e}" for n, a, b in dd))
        print(f"       => {verdict}: 换槽序改了 per-token slot 的 index_add_ 归约序 => 【逐位】只在同序时成立,"
              f" KL 只能读成『成员集的函数』并带上这把噪声底 {SLOT_TOL:g}(P1-②b 那句『纯函数』按此收窄)")

print("""
判读提示: ① 先读闸门③: 它说"这一格真的只把 eid{e} 换成了 int8"。它红 = 单价表整个是假的, 别读后面的数。
       ② u_e 那一列是这一刀的产物: 它是【单专家扰动】的 KL, 已经是 hit_e 加权之后的量, 所以
          跨专家比 u_e 会混进流量; 要"这个专家的 int8 有多贵"就读 kappa = u/hit 那一列。
       ③ 段末那行 全CPU/Σu 是加性假设的端点检验: ~1 = int8 误差在 KL 域可加 => M1 有资格当判据;
          远小于 1 = 有饱和(误差之间互相抵消/被 softmax 吸收) => M1 只能读序不能读绝对值。
       ④ 表B 三个比值列的读法: M0(计数模型)在 P3 段若系统性 >1 或 <1, 就是"覆盖流量% 会给出反向
          决策"的定量版; M1 同段比值应当在一段之内围着 1 打转(它没有自由参数, 偏了就是加性坏了)。
          M1 跨段(拿阶段1 的 kappa 去预测后面两段)= 判据真能上线的形态 —— 运行时只有阶段1 的价。
       ⑤ M2 = 台账旧口径(L2rel%)的 Spearman: 它若明显低于 M1, "relRMSE 不是最终指标"就从口号变成
          有数的判决; 它若和 M1 一样高, 那说明这批点的 KL 序其实是向量误差序的影子, 结论要降级。
       ⑥ 🚨 kappa 的跨段稳定性决定一切: 三段 kappa 极差 > 30% 就说明"专家本体误差"这个概念本身
          没那么干净(它依赖输入分布), 那时 P1-②d 的判据只能用【在线增量估计】而不是查表。
       ⑦ 归档锚点是【别的代码】(bench_metrics/bench_dyn)在别的进程里印的数: 闸门⓪ 过 = 本表的
          x/语料/行区间/裁判与那两轮完全同一件东西; 不过 => 行区间或语料漂了, 整表作废。""")

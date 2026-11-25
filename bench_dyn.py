#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-② 判决表: 【运行时动态专家放置】到底值不值 —— 三方(延迟/质量/迁移成本)同格。

【这一刀测的是什么(不是"加个功能")】
  P0-③补(results/2026-11-21_P0-③补_metrics_corpus-input_txt_bs512.txt, md5 0e0694aa) 量到:
  同一模型、同一路由、同样 512 行真激活, 只换语料, 命中直方图就变 =>
      readme: 命中%=[42.3,13.2,17.7,6.8,1.4,12.6,6.0,0.1] 最热/最冷 433.0x  A序[0,2,1,5,...]
      input : 命中%=[39.3,11.7,16.9,9.6,1.6,18.8,2.1,0.2] 最热/最冷 201.0x  A序[0,5,2,1,...]
  gc=3 的 GPU 成员集合从 {0,2,1} 变成 {0,5,2}(要换 1<->5), gc>=5 才重合; 而"放错 vs 放对"
  的同档税是 KL 5.9x(gc=3) / 20.9x(gc=5) / 309x(gc=7)。
  => 静态放置的成员是【离线在一份语料上】选的, 到目标流量上可能已经放错。动态放置就是在
     运行时把这笔重定价付掉。本表量它三个数: 它省多少 KL、它自己花多少 ms、阈值开不开火。

【表结构】外层 = 策略(每个策略一个引擎, 建一次跑完所有阶段), 内层 = 阶段(=流量段)。
  阶段 = 语料名: readme/input.txt 取该语料第 k 次出现的第 k 段 BS 行(同名重复 = 同语料换行区间,
  不是复读); 特殊名 drift = 【人造漂移】=> 不换语料, 从 readme 前 DRIFT_POOL 行真激活里按命中
  画像挑 BS 行(专挑"静态成员留在 CPU 腿的那几个专家"命中的行)。它换的只是【混合比例】,
  每行仍是真 token 的真激活; 作用是把"漂移值多少钱"这个效应量抬到噪声底之上。
  每段还建一个【该段上帝视角成员】的静态引擎打一发 = 该段【地板/可达下界】, 表里 KL/地板 读它。
  每个阶段内: 暖机 WARM 帧 -> FRAMES 帧(每帧计时一次 + 在【计时窗口外】跑 dyn.step) ->
  再打一发取 y 做质量(top-1/KL/TV)。策略之间共用【同一批 x、同一份裁判 P】。

【策略(全部读同一份计数器, 差别只在决策)】
  static_A   上帝视角放对(整本 readme 的直方图) —— 上界锚点, 不开决策
  neutral    中立序(eid 序)不动 —— 【没做动态放置】的对照格
  static_p1  中立序起步, 只在阶段 1 自适应, 之后冻结 —— "启动期自适应"够不够
  static_C   上帝视角放错(最冷的进 GPU) —— 下界锚点
  naive      外部建议的原型: CPU>5% 提升 / GPU<1% 降级, 两条独立阈值, 无冷却
  hysteresis 补容量判据(gain=3x)+ 冷却 16 帧
  hyst_1x/hyst_2x  同上但 gain=1.0/2.0 —— gain(成对判据)那条腿的三档刻度
  —— 以上都只比【命中占比】。P1-②c 量到 int8 误差单价 kappa_e 的谱(150x)远陡于流量谱
     (<0.5x) => "谁更热"不是"谁让质量变差"的充分代理。下面三档同一副骨架, 只换比较的量:
  price_cal  判据 = 预测 KL 贡献 u_e(P1-②c 单价表, 离线标定段 PRICE_CAL) —— 上线真拿得到
  price_hit  判据 = kappa_e x 窗内实测占比 —— 离线只标"专家本体多脆", 流量交给在线计数器
  price_orc  价 = 【本段自己的真价】= 上帝价 => "在线估若免费"的【上界锚点】, 不可上线
  🚨 static_* 从【自己的 order】起步; 动态策略默认从【static_A 的成员】起步(START_DYN=A),
     因为"离线在一份语料上选好成员、上线后流量漂了"才是动态放置真正要接的那一杆活。
     START_DYN=neutral|shuffle 可切到中立序/打乱序起跑(冷启动那一档; 本机 eid 序与
     static_A 同成员 => 冷启动那一档要用 shuffle 才有信息量, 见 POLICY_PRESETS 注释)。

【闸门(这一刀的命门: 迁移不能改数值)】
  ⓪ 归档对账: phase0/readme + fp32 + static_A/static_C 的 KL 必须等于 P0-④c 归档(md5 a1dabef9)
  ① 计数器 == route(x): counts_from_engine(m) 与 torch.bincount(route(x)[1]) 逐位相同
  ② 迁移不改数值: 末阶段结束后按【运行时 slot2eid】重建一个新引擎, 输出与迁移后的引擎逐位相同
  ③ 图不重录: id(_g_pre)/id(_g_hot)/_gkey/_wver 从暖机后到末帧全程不变(变了 = 每帧重录图)
  ④ 裁判自洽: 裁判 P 重算互差必须 0(它自己抖 => 整表 KL 全废)
  ⑤ 幅度红线: |logit| <= LOGIT_RED(默认 1e4; 本仓 bs=512 出现过 4.094e+19 的脏格)
  ⑥ 记账 == 内容: slot2eid 必须与 gpu_experts/gpu_up/gpu_down/gate.weight 的实际内容逐位一致
     (回写方向反了 = 拿假直方图做决策, 这一条把它钉死)
  ⑧ 价表(P1-②d 新增): 逐行解析出 E 行 + Σ(解析 u) 与该段归档 Σu_e 自洽 + 解析命中% 与本运行
     重算命中% 同源(<=0.05pp) + 标定段不能落在后续阶段(否则 price_cal 用了未来信息)。
     🚨这一刀不重跑单价表 —— 价是【别的进程】印的, 闸门⑧ 就是它的跨进程对账(同 ⓪ 的规矩)。
  ⑨ 质量列有效性(P1-②d 判决轮补): 阶段末那一发要同时过【幅度】和【定义域】两组检查、并且在
     REDO 发内自洽, 才有资格进表 —— (a) Q 侧 |logit|<=LOGIT_RED; (b) 参照侧 PL 在 metrics 时刻
     的 absmax 仍等于建段时那份(容差 1e-9 相对)【且它自己还要过一道绝对红线: 只比"变没变"对
     【生下来就脏】的裁判全盲 —— r3 复核轮 |P|absmax=5.85e+19, 闸门⑤ 红而 ⑨ 全绿 = 自相矛盾】;
     (c) KL 有限且 0<=KL<=KL_CAP x 本段参照、TV∈[0,1];
     (d) 相邻两发 (KL,TV,top-1) 逐位同, 兜底 = 同一组值出现>=2 次。判脏 => 那四列印『脏』不印数。
     动因: 判决轮 static_C 阶段1 印出 KL=5.77e+151(TV 4.35e-35)而 Q 侧红线【通过】、闸门①②③⑥⑦
     全绿, 只有闸门⓪ 抓到 => 旧设计只核了 Q 侧幅度 + 只在建段时核了一次 P 侧 = 只核一侧等于没核。
env: LAYERS=0,4,8,12,16,20,24,28  GATE=embed  CAP_LAYER=8  N_CHARS=24000
     BS=512  GC=3  DTYPE=fp32  FRAMES=64  CHECK=8  WINDOW=16  WARM=16  REDO=3  KL_CAP=10
     POLICIES=static_A,shuffle,static_p1,static_C,naive,hysteresis,hyst_1x
     PHASES=readme,input.txt,readme   (readme = R.real_text(); 其余按脚本目录下相对路径读)
     PRICE_TABLE=results/2026-11-22_P1-②c_err单价_bs512.txt  PRICE_CAL=readme  PRICE_THR=-1(<0=preset)
"""
import os, sys, time, statistics, re, torch
from gc import collect as _pycollect
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R
from dyn_placement import (DynPlacer, counts_from_engine, slot_counts, policy_from_name,
                           POLICY_PRESETS, NAIVE, HYSTERESIS, PRICE)

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS = int(os.environ.get("BS", 512))
GC = int(os.environ.get("GC", 3))
DTYPE = os.environ.get("DTYPE", "fp32").strip()
FRAMES = int(os.environ.get("FRAMES", 64))
CHECK = int(os.environ.get("CHECK", 8))
WINDOW = int(os.environ.get("WINDOW", 16))
WARM = int(os.environ.get("WARM", 16))
WARM_ENG = 12                       # 引擎级暖机(建图/缓存), 与 bench_swap.warm(m,x,12) 同
TAIL = float(os.environ.get("TAIL", "0.667"))   # p50尾 = 只用后 2/3 帧(把首帧尖峰剔掉再读)
POLICIES = [p.strip() for p in os.environ.get(
    "POLICIES", "static_A,shuffle,static_p1,static_C,naive,hyst_1x,hyst_2x,hysteresis"
).split(",") if p.strip()]
PHASES = [p.strip() for p in os.environ.get("PHASES", "readme,input.txt,drift").split(",") if p.strip()]
DRIFT_POOL = int(os.environ.get("DRIFT_POOL", 2048))   # drift 段从 readme 前这么多行里挑 BS 行
START_DYN = os.environ.get("START_DYN", "A").strip()   # 动态策略起跑序: A=离线选好的成员
LOGIT_RED = float(os.environ.get("LOGIT_RED", 1e4))
# 质量列是【确定量】=> 记账资格 = 同一发打出 REDO 次后自洽(闸门⑨), 不是降噪。REDO∈[2,4]
REDO = max(2, min(4, int(os.environ.get("REDO", 3))))
KL_CAP = float(os.environ.get("KL_CAP", 10.0))   # KL 上限 = 本段参照(换成 MoE 本身)的 KL_CAP 倍
# ---- P1-②d 价判据: 价来自 P1-②c 判决表的【文本】(🚨不抄常数, 台账改了这里跟着变)
PRICE_TABLE = os.environ.get("PRICE_TABLE", os.path.join(HERE, "results",
                                                         "2026-11-22_P1-②c_err单价_bs512.txt"))
PRICE_CAL = os.environ.get("PRICE_CAL", "readme")       # 离线标定段 = 上线前真能拿到价的那一段
PRICE_THR = float(os.environ.get("PRICE_THR", "-1"))    # <0 => 用 preset 的 thr_rel
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
E = len(LAYERS)
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
JUDGE_GC = 0                        # 裁判固定 gc=0 = 全专家 fp32 稠密(与 P0-③/④ 同口径)
assert set(POLICIES) <= set(POLICY_PRESETS), f"未知策略名, 可选 {sorted(POLICY_PRESETS)}"
assert 0 < GC < E, f"这一刀要的是【有放置自由度】的档: 0 < GC={GC} < E={E}"
assert DTYPE in ("fp32", "bf16", "fp16"), DTYPE
assert WARM <= FRAMES and WINDOW <= FRAMES, "暖机/窗口比 FRAMES 还长 => 一帧决策都开不出来"
assert DRIFT_POOL >= BS, f"drift 段的池子({DRIFT_POOL}行)比 BS({BS}) 还短 => 挑不出东西"
assert START_DYN in ("A", "neutral", "shuffle"), START_DYN
TAILN = max(1, int(round(FRAMES * TAIL)))

# ---------------------------------------------------------------- 闸门⓪ 锚点(P0-④c 归档值)
# 来源: results/2026-11-21_P0-④c_metrics_bs512.txt (bench_metrics 用户跑, md5 a1dabef9…)
# 键 = "配置 bs gc dtype" -> KL(P‖Q)。🚨只判 fp32: 那份归档的 A bs=512 gc=3 bf16 有一格脏
# logits(absmax 4.094e+19, 台账已按"作废精确到格 x 列"处理) => bf16 行不进锚点。
ANCHORS = {}
for _l in """
A 512 0 fp32 5.240e-05
A 512 3 fp32 9.711e-06
A 512 5 fp32 2.972e-06
A 512 7 fp32 3.468e-07
A 512 8 fp32 3.686e-12
C 512 0 fp32 5.240e-05
C 512 3 fp32 4.962e-05
C 512 5 fp32 4.485e-05
C 512 7 fp32 1.805e-05
C 512 8 fp32 3.686e-12
""".strip().splitlines():
    _lb, _bs, _gc, _dt, _kl = _l.split()
    ANCHORS[(_lb, int(_bs), int(_gc), _dt)] = float(_kl)


# ---------------------------------------------------------------- P1-②d 价表(读 P1-②c 归档文本)
def parse_price_table(path):
    """从 P1-②c 判决表文本里解析 per-expert int8 误差单价表 -> {段名: dict(rows/u/kappa/hit/sumu)}。
    只吃【表头含 u_e=KL(e only CPU) 的那个块】里的 E 行: 那份归档在 kappa 稳定性块里也印
    8 个以数字开头的行, 不锁块就会串台。段名 = readme / input.txt / drift(归档里那一段头印作
    "人造漂移(...)")。u_e 按 eid 索引 => 它与放置序无关, 换阶段不用换索引方式。"""
    out, seg, rows = {}, None, None
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):   # 🚨用内置 open: 本模块没有 import io
        t = ln.strip()
        if t.startswith("阶段") and " = " in t:
            head = t.split("=", 1)[1].strip()
            m = re.match(r"(\S+)\s+第\d+段", head)
            seg = "drift" if "人造漂移" in head else (m.group(1) if m else None)
            rows = None
            continue
        if "u_e=KL(e only CPU)" in t:
            rows = {}
            continue
        if t.startswith("端点") and seg in out:      # 🚨这条必须在 rows is None 之前: 收满 E 行
            m = re.search(r"u_e=([0-9.eE+-]+)", t)   #    后 rows 已被置 None, 放后面永远走不到
            if m:
                out[seg]["sumu"] = float(m.group(1))
        if rows is None:
            continue
        p = t.split()
        if len(p) == 10 and p[0].isdigit():
            rows[int(p[0])] = dict(eid=int(p[0]), hit=float(p[1]), u=float(p[2]), tv=float(p[3]),
                                   top1=float(p[4]), l2rel=float(p[5]), kappa=float(p[6]))
            if len(rows) == E:
                out[seg] = dict(rows=[rows[e] for e in sorted(rows)], seg=seg, sumu=float("nan"))
                rows = None
            continue
    for d in out.values():
        d["u"] = [r["u"] for r in d["rows"]]
        d["kappa"] = [r["kappa"] for r in d["rows"]]
        d["hit"] = [r["hit"] for r in d["rows"]]
    return out


PRICE_BY_NAME = parse_price_table(PRICE_TABLE)

# ---------------------------------------------------------------- 真模型(只 load 一次)
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


t_load = time.perf_counter()
print(f"[机器状态] {H.machine_state()}")
_SD = R.state_dict()
_R_sd = R.state_dict
R.state_dict = lambda: _SD            # 要建十几个引擎, 别每次重读 safetensors(与 bench_swap 同)
dims = R.dims(_SD)
D, N, NL = dims["D"], dims["N"], dims["layers"]
MODEL, TOK = _load()
V = MODEL.get_input_embeddings().weight.shape[0]
torch.manual_seed(SEED)


def capture_from(ids):
    """在真模型第 CAP_LAYER 层 MLP 入口抓 x(与 bench_metrics/bench_swap 逐字同一挂法)。"""
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


def perm_src(order):
    """按 order 重排专家: order[槽位] = 专家 eid; 门控行跟着走(与 bench_swap 逐字同)。"""
    base = R.real_src(LAYERS, gate_mode=GATE)
    return dict(experts=[base["experts"][i] for i in order],
                gate_w=base["gate_w"][list(order)].contiguous(),
                gate_b=base["gate_b"].clone(),
                note=base["note"] + f" perm={list(order)}")


def hits_of(gate_w, rows):
    W = gate_w.float()
    cnt = torch.zeros(E)
    for i in range(0, rows.shape[0], 64):
        sc = rows[i:i + 64] @ W.t()
        cnt += torch.bincount(sc.topk(K, dim=1).indices.flatten(), minlength=E).cpu()
    return cnt


BASE = R.real_src(LAYERS, gate_mode=GATE)      # 🚨 未置换的基表: 迁移换入换出的内容按 eid 从这里取


def mk(gc, order, dt=DTYPE):
    """建模型 + 上 GPU + 暖一次缓存。与 bench_swap.mk 逐字同(闸门⓪ 靠这个才成立)。"""
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind="swiglu",
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def timed(m, x, n):
    """逐帧计时(前后各 sync 一次)。返回 ms 列表。"""
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fwd(m, x)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out, y


def splice(ids_row, y_cpu, box=None):
    """把第 CAP_LAYER 层 MLP 出口换成 y_cpu(T,D), 跑完剩余层 => 真 logits (T,V) float64。"""
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


def red_ok(L):
    mx, mn = float(L.max()), float(L.min())
    return max(abs(mx), abs(mn)) <= LOGIT_RED, max(abs(mx), abs(mn))


def metrics(lp, lq, ids_row):
    """P/Q 的 logits -> token 级指标(float64, log_softmax 域)。= bench_swap.metrics。"""
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


# ------------------------------------------------- 质量列有效性(闸门⑨, 见判读提示⑫)
def fmt_ratio(x, w, nd=2):
    """比值印刷护栏: 非有限 -> 'nan', |x|>1e6 -> 科学计数, 其余定点。
    动因: 判决轮那发 KL=5.77e+151 被 `{:>8.2f}` 印成 158 位巨数, 把整张表顶歪还看不出是脏。"""
    if x != x or abs(x) == float("inf"):
        return f"{'nan':>{w}}"
    return f"{x:>{w}.{nd}e}" if abs(x) > 1e6 else f"{x:>{w}.{nd}f}"


def quality_domain(P, Mq, aq, aPn):
    """阶段末那一发的幅度+定义域校验 -> 违规项名列表(空 = 这一发可引用)。
    🚨(b) 是这一刀的命: 参照侧 PL 的 absmax 必须在【metrics 时刻】仍等于建段时那份 ——
    旧闸门⑤ 只在建段时核一次 P 侧, 判决轮那发 KL=5.77e+151 偏生 Q 侧红线是【过】的。
    🚨(b2) r3 复核轮补: (b) 只查"变没变", 裁判【生下来就脏】时它与自己比 => 必绿, 于是出现
    "闸门⑤ 红(|P|=5.85e+19) + 闸门⑨ 全绿"的矛盾读数 => 参照侧还要过一道【绝对】红线。
    一致性检查不能代替绝对红线(方法论 36)。"""
    bad = []
    if aq > LOGIT_RED:
        bad.append(f"Q|absmax|={aq:.3g}>红线{LOGIT_RED:.0f}")
    if abs(aPn - P["aP"]) > 1e-9 * max(P["aP"], 1.0):
        bad.append(f"参照侧absmax {P['aP']:.4g}->{aPn:.4g}(建段后被改写)")
    if aPn > LOGIT_RED:      # (b2) born-dirty: 裁判自己越红线 => 本段所有 KL/TV 都是垃圾
        bad.append(f"参照侧|absmax|={aPn:.3g}>红线{LOGIT_RED:.0f}(裁判 P 生下来就脏)")
    kl, tv, t1 = Mq["kl"], Mq["tv"], Mq["top1"]
    for nm, v in (("KL", kl), ("TV", tv), ("top1", t1)):
        if v != v or abs(v) == float("inf"):
            bad.append(f"{nm}非有限={v}")
    if kl == kl and abs(kl) != float("inf"):
        if kl < 0.0:
            bad.append(f"KL={kl:.3g}<0(定义域坏)")
        if P["M0"]["kl"] > 0 and kl > KL_CAP * P["M0"]["kl"]:
            bad.append(f"KL={kl:.3g}>本段参照的{KL_CAP:g}倍")
    if tv == tv and abs(tv) != float("inf") and not 0.0 <= tv <= 1.0:
        bad.append(f"TV={tv:.3g}不在[0,1]")
    return bad


def quality_shot(P, m):
    """阶段末那一发 => 质量 + 记账资格(打 REDO 次):
      资格① 相邻两发 (kl,tv,top1) 逐位同(主判据, 同 P1-②c/bench_err 的规矩);
      资格② 兜底 = 同一组值出现过 >=2 次(= 有一发脏发被剔掉了, 表下点名);
      两条都不满足, 或【被收下那一发】没过 quality_domain => dirty(表里印『脏』)。
    🚨这是【记账资格】不是降噪: 质量列是确定量(闸门④ 已证互差=0), 两发不同 = 有非确定性脏发。"""
    shots, flags, rule, pick = [], [], None, None
    for i in range(REDO):
        yq = fwd(m, P["x"]).float().cpu().clone()
        Lq = splice(P["ids"], yq)
        _, aq = red_ok(Lq)
        _, aPn = red_ok(P["PL"])            # 🚨参照侧【此刻】的幅度, 不是建段时存的那份
        Mq = metrics(P["PL"], Lq, P["ids"])
        shots.append(((Mq["kl"], Mq["tv"], Mq["top1"]), Mq, aq, aPn))
        d = quality_domain(P, Mq, aq, aPn)
        if d:
            flags.append((i, d))
        del Lq, yq
        if i and shots[i][0] == shots[i - 1][0]:
            rule, pick = "相邻逐位同", i
            break
        grp = {}
        for j, sh in enumerate(shots):
            grp.setdefault(sh[0], []).append(j)
        rep2 = [v for v in grp.values() if len(v) >= 2]
        if rep2:
            rule, pick = "同值出现>=2次", rep2[-1][-1]
            break
    if pick is None:
        pick = len(shots) - 1
    key, Mq, aq, aPn = shots[pick]
    dom = quality_domain(P, Mq, aq, aPn)
    why = ([] if rule else [f"{REDO}发内既没相邻逐位同也没有同值>=2次"])
    why += [f"收下的第{pick + 1}发仍违规: {x}" for x in dom]
    return dict(kl=key[0], tv=key[1], top1=key[2], Mq=Mq, aq=aq, aPn=aPn, nshot=len(shots),
                qrule=rule, shots=[f"{sh[0][0]:.4g}" for sh in shots], flag=flags,
                dirty=(rule is None or bool(dom)), why=why)


pct = lambda c: [round(float(v) / float(sum(c)) * 100, 1) for v in (c.tolist() if hasattr(c, "tolist") else c)]

# ---------------------------------------------------------------- 阶段语料 -> ids/x
def phase_ids(name, kth, rows=None):
    """第 k 次出现该语料 => 取第 k 段; rows=None 就是 BS 行(drift 段要一个更长的池子)。"""
    rows = BS if rows is None else int(rows)
    txt = (R.real_text(N_CHARS) if name == "readme"
           else open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()[:N_CHARS])
    ids = TOK(txt, return_tensors="pt").input_ids
    lo = kth * BS
    assert ids.shape[1] >= lo + rows, f"阶段语料 {name} 只有 {ids.shape[1]} token, 取不出第 {kth} 段 {rows} 行"
    return ids[:, lo:lo + rows], len(txt), ids.shape[1]


def drift_ids():
    """人造漂移段(见 docstring): 真激活按【命中画像】重排, 不换语料也不造假数据。
    挑分 = 该行 top-K 里落在 COLD 集(= 静态成员留在 CPU 腿的那 E-GC 个 eid)的个数, 从多到少。
    🚨 它是【放大器的角色】: 没有它, 两段真语料把 top-GC 成员只挪动一个 eid / 覆盖差 7pp,
       动态放置的效应量落在噪声底里 => 表会退化成"什么都没发生"。放大之后的数值只读
       【方向和量级】, 不当绝对结论(方法论 22)。"""
    ids_pool = phase_ids("readme", 0, DRIFT_POOL)[0]
    xp = capture_from(ids_pool).cuda()
    _ts, ti = jm.route(xp)
    eid = torch.tensor(order_A)[ti.detach().cpu().long()].reshape(-1, K)
    cold = torch.zeros(E, dtype=torch.bool)
    cold[order_A[GC:]] = True
    score = cold[eid].sum(1).float()
    try:
        sel = torch.argsort(-score, stable=True)[:BS]
    except TypeError:                                  # 老 torch 没有 stable
        sel = torch.argsort(-score)[:BS]
    sel = sel.sort().values
    share = float(score[sel].sum()) / float(BS * K)    # 挑完之后 COLD 集占总命中的比例
    del xp
    torch.cuda.empty_cache()
    return ids_pool[:, sel], share


# x 的归档路径那份(readme 前 BS 行) —— 只作互差对照, 与 bench_swap 的 XREAL 同物
XREAL, _d = R.capture(layer=CAP_LAYER, bs=BS, n_chars=N_CHARS)
cnt_e = hits_of(BASE["gate_w"], XREAL)
order_A = torch.argsort(cnt_e, descending=True).tolist()
order_C = torch.argsort(cnt_e).tolist()
NEUTRAL = list(range(E))
# 🚨 真对照需要【GPU 成员集】与 static_A 不同的起跑序; 但 eid 序在本真源上热专家恰好
#    就是 {0,1,2} => set(NEUTRAL[:GC]) == set(order_A[:GC]) => neutral 行与 static_A 同格
#    (槽内换序不改成员集 => KL 逐位相同) => 它只能当占位/自检。shuffle: 固定种子打乱,
#    并保证 GPU 成员集既 != A 也 != C(否则换下一个种子), 它才是真【不做动态放置】的对照。
SHUF, _sd = None, 7
while SHUF is None:
    cand = torch.randperm(E, generator=torch.Generator().manual_seed(_sd)).tolist()
    if set(cand[:GC]) != set(order_A[:GC]) and set(cand[:GC]) != set(order_C[:GC]):
        SHUF = cand
    else:
        _sd += 1
print(f"[真源] SmolLM-135M: D={D} N={N} 层数={NL} 词表={V} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家"
      f" top_k={K} | 路由={GATE} | 模型加载 {time.perf_counter()-t_load:.1f}s")
print(f"[阶段] {len(PHASES)} 段 x {FRAMES} 帧(计时) | 每段再打一发取质量 | BS={BS} GC={GC} dtype={DTYPE}")
for nm in PHASES:
    if nm == "drift":
        print(f"   · drift        人造漂移: 从 readme 前 {DRIFT_POOL} 行真激活里按命中画像挑 {BS} 行"
              f"(COLD 集={order_A[GC:]} = 静态成员留在 CPU 腿的那几个) | 起跑序={START_DYN}")
        continue
    ids, nch, ntok = phase_ids(nm, 0)
    print(f"   · {nm:<12} {nch} 字符 -> {ntok} token"
          + ("(本仓库 readme 拼贴, 域外)" if nm == "readme" else "(本机文件, 大概率被见过 => 只读稳健性)"))
print(f"[命中] 阶段1(readme 前 {BS} 行) 累计 命中%={pct(cnt_e)} 最热/最冷="
      f"{float(cnt_e.max())/max(float(cnt_e.min()),1.0):.1f}x | A序={order_A} C序={order_C} 中立序={NEUTRAL}")
print(f"[被测] 运行时动态放置: 计数器(m._pin[idx], 零额外 D2H) -> 滑窗{WINDOW} -> 决策 -> 就地迁移"
      f" | 检查点每 {CHECK} 帧, 暖机 {WARM} 帧, 每阶段最多 {(FRAMES-WARM)//CHECK+1} 次决策")
print(f"[口径] ① 延迟 = 每帧一发前后 sync 取中位; p50全=全 {FRAMES} 帧中位, p50尾=后 {TAILN} 帧中位"
      f"(首帧含冷启动/迁移当帧的尖峰 => 读趋势用尾段)")
print(f"[口径] ② 迁移成本【单列】且实测(ms/MB), 不并进 p50(step 在计时窗口外); 折合帧数 = 迁移ms/p50尾,"
      f" 由读者自己给流量定价 —— 这就是这一刀的价签")
print(f"[口径] ③ 质量是【阶段末那一发】: 同一份裁判 P(全专家 fp32 稠密), 只报相对量(KL/TV/top-1),"
      f" KL/A = 同阶段 static_A 的多少倍(参照行见每段开头)")
print(f"[口径] ④ 同名语料重复 = 【同一份语料换行区间】不是复读; 换语料 => 直方图会漂,"
      f" 漂多少见每段开头印的 命中%(eid 空间, 与策略无关); drift 段是【人造漂移】= 真激活按"
      f" 命中画像重排 => 它的绝对值只读方向, 漂移幅度是放大器不是流量画像")
print(f"[口径] ⑤ 曲线不是配对: 策略之间不轮转(一个引擎跑完三阶段才换), 相邻策略 <噪声底(A 2~3%)的差别读;"
      f" 唯一的配对是【同策略同阶段的 p50全 vs p50尾】和 static_* 三档对同一根杆的位置")
print(f"[口径] ⑥ 显存不报: 动态放置【不新增常驻字节】(只改内容不改地址, 闸门③), 稳态列就是 P0-④a 那一列")
print(f"[口径] ⑦ 抓到热/热总 = 本段【真值直方图】(eid 空间, 与策略无关)里占比>1% 的专家中, "
      f"末态仍在 GPU 的有几个 —— 它是这一刀唯一真正的【效果量】, KL/A 只是它的下游")
print(f"[口径] ⑧ 迁移次/ms/MB/轨迹 都是【本阶段】的量(进阶段存快照出阶段减差); 轨迹行末那个"
      f" 累计(跨阶段) 才是引擎全生命周期的总数, 别把两者混读")
print(f"[口径] ⑨ 地板 = 拿【这一段自己的】上帝视角成员另建一个静态引擎打一发(同 bs 同裁判):"
      f" KL/地板 = 这一刀追回了多少【可达收益】; 阶段1 的 static_A 本来就在地板上 => 该列必为 1.00")
print(f"[口径] ⑩ 价判据三档 = 同一副骨架(容量约束 + 冷却 + 一进一出)只换【比较的量】和【价的来源】:"
      f" price_cal = u_e@标定段'{PRICE_CAL}'(离线标定 => 上线真可得) / price_hit = kappa_e x 窗内占比"
      f"@标定段(离线只标专家本体, 流量形状交给在线计数器) / price_orc = u_e@本段真价 =【上帝价】"
      f"=> '在线估若免费'的【上界锚点】, 它用了本段未来流量, 不可上线")
print(f"[口径] ⑪ 价判据的阈值: promote_min/demote_max 全置 0(P1-②/②b 实测 demote_max=1% 在真漂移下"
      f" 把降级那一半永久锁死), 换成 成对价杆 gain={PRICE['gain']} + 预测KL降幅门槛 "
      f"thr_rel={100*PRICE['thr_rel']:.0f}%(迁移一次实测 ~15ms = 3~4 帧 => 降幅低于价签就不该动手;"
      f" PRICE_THR 可当旋钮消融); 价 = 解析 {os.path.basename(PRICE_TABLE)}(闸门⑧ 对账)")
print(f"[口径] ⑫ 质量列的记账资格 = 闸门⑨: 阶段末那一发打 REDO={REDO} 次, 【相邻两发 (KL,TV,top-1)"
      f" 逐位同】才记账(兜底 = 同一组值出现>=2 次); 每发同时核 Q 侧幅度、参照侧 PL【此刻】的 absmax"
      f"(建段后被改写 = 参照没了)、KL∈[0,{KL_CAP:g}x本段参照] 与 TV∈[0,1] —— 判脏的行那四列印『脏』"
      f"而不是印垃圾, 兜底收下的行表下点名(它有一发脏发被剔除)")


def price_gap(u, mem):
    """(M1预测, 价gap%) —— 纯算术, 不碰引擎; 由闸门⑧c 用【已知答案的格】做恒等式自检。

    M1预测 = Σ_{e∈CPU腿} u_e (P1-②c: 预测/实测 段内 0.92~1.11)
    价gap% = max(0, max_{e∈CPU腿} u_e − min_{e∈GPU腿} u_e) / M1预测 x 100
      = 0  => 末态已是该段【价最优成员】(CPU 腿最贵的也比 GPU 腿最便宜的便宜 => 没有一对可赚)
      > 0  => 还有一对可赚, 这个数就是它值多少比例的 KL
    🚨方向必须与 `dyn_placement._decide_price` 一致(v_in = CPU 腿最贵, v_out = GPU 腿最便宜,
       赚 = v_in − v_out)。判决表 `3ce6836d` 那一列把两个操作数写反了 => 价最优行印出
       9.4/23.7、还有得赚的行反而印 0.0(它不参与任何决策, 所以那轮闸门全绿没人报警)。
    """
    if u is None:
        return float("nan"), float("nan")
    mem = [int(e) for e in mem]
    cpu = [e for e in range(len(u)) if e not in mem]
    if not mem or not cpu:
        return float("nan"), float("nan")
    pred = sum(u[e] for e in cpu)
    if not pred > 0.0:
        return pred, float("nan")
    return pred, max(0.0, (max(u[e] for e in cpu) - min(u[e] for e in mem)) / pred * 100.0)


# ---------------------------------------------------------------- 闸门⑧ 价表
def gate_price():
    """价表可用性 + 跨进程同源。它红 = 三档 price_* 全表作废: 价是别人印的, 没人核就没人负责。"""
    bad = []
    print(f"\n  闸门⑧ 价表: 源={os.path.basename(PRICE_TABLE)} 解析 {len(PRICE_BY_NAME)} 段"
          f" {[d['seg'] for d in PRICE_BY_NAME.values()]} | 标定段 PRICE_CAL={PRICE_CAL}")
    if PRICE_CAL not in PRICE_BY_NAME:
        bad.append("标定段缺价")
        print(f"    🚨PRICE_CAL={PRICE_CAL} 不在价表里 => price_cal/price_hit 全程无价(沉默)")
    if PRICE_CAL in [p["seg"] for p in PH[1:]]:
        bad.append("标定段落在后续阶段")
        print(f"    🚨PRICE_CAL={PRICE_CAL} 出现在后续阶段 => price_cal 用了未来信息,"
              f" 它不再是'上线可得'那一档(只有 price_orc 允许这样, 它本来就标了上帝价)")
    for k, P in enumerate(PH):
        d, tag = PRICE_BY_NAME.get(P["seg"]), f"段{k+1} {P['seg']}#{P['kth']}"
        if d is None or len(d["rows"]) != E:
            bad.append(tag)
            print(f"    🚨{tag} 价表缺段/行数不齐({0 if d is None else len(d['rows'])}/{E})"
                  f" => 该段 M1预测/价gap=nan, price_orc 不开火")
            continue
        ds = abs(d["sumu"] - sum(d["u"])) / max(abs(d["sumu"]), 1e-300)
        dh = max(abs(a - b) for a, b in zip(d["hit"], pct(P["cnt"])))
        note = "本段真价可得" if P["u"] is not None else \
            f"同语料第{P['kth']+1}段: 价表只标了第1段 => 本段无真价(那两列 nan)"
        if P["kth"] == 0 and dh > 0.05:
            bad.append(tag)
            note += " 🚨命中%与那份归档不同源(不是同一批行 => 价不适用于这批行)"
        if not ds < 1e-3:
            bad.append(tag)
            note += (" 🚨归档端点行的 Σu_e 没解析到" if d["sumu"] != d["sumu"]
                     else " 🚨Σu 与逐行和不自洽(解析串台)")
        print(f"    {tag:<20} 行{len(d['rows'])}/{E} Σu 归档={d['sumu']:.4e}"
              f" 解析和={sum(d['u']):.4e} 相对差={ds:.1e}"
              f" | 命中%最大差={dh:.2f}pp | {note}")
        # 闸门⑧c 价杆自证: 派生列只有一行代码, 但它决定整根价杆的读法方向 —— 拿一个
        # 【答案已知的格】(按 u_e 降序取 GC 个 = 价最优成员)喂给 price_gap, 它必须给 0。
        if P["u"] is not None:
            _opt = sorted(range(E), key=lambda e: -d["u"][e])[:GC]
            _p0, _g0 = price_gap(d["u"], _opt)
            _p1, _g1 = price_gap(d["u"], P["top"])
            _id_ok = (_g0 == 0.0)
            if not _id_ok:
                bad.append(f"{tag} 价杆自证")
            print(f"      闸门⑧c 价最优成员={_opt} M1={_p0:.3e} gap={_g0:.1f}"
                  f"(恒等式: 必须 0){'' if _id_ok else ' 🚨方向反了/实现坏了'}"
                  f" | 覆盖最优成员(上帝视角)={sorted(P['top'])} M1={_p1:.3e} gap={_g1:.1f}")
    print("    => 价与本运行直方图同源, 三档价判据可引用" if not bad else f"    🚨闸门⑧ 不符: {bad}")
    return bad

# ---------------------------------------------------------------- 裁判 P(每阶段一份, 策略共用)
PH = []
jm = mk(JUDGE_GC, order_A, "fp32")
for k, nm in enumerate(PHASES):
    kth = PHASES[:k].count(nm)
    if nm == "drift":
        ids, _share = drift_ids()
    else:
        ids, nch, ntok = phase_ids(nm, kth)
    xcpu = capture_from(ids)
    box = {}
    splice(ids, torch.zeros(BS, D), box)
    d_sp = (box["x"][0] - xcpu).abs().max().item()
    bit_sp = int(torch.equal(box["x"][0], xcpu))
    d_cap = (xcpu - XREAL[:BS]).abs().max().item() if kth == 0 and nm == "readme" else float("nan")
    x = xcpu.cuda()
    ts0, ti0 = jm.route(x)
    # 本段真值直方图, 记在【eid 空间】: 路由是置换不变的 => 它与策略/放置无关,
    # 所以表头那列"上帝视角最优成员"和表里"CPU有流量"两列都用它, 不掺策略的账。
    _c0 = slot_counts(ti0.detach().cpu(), E).float()
    _cnt = torch.zeros(E)
    for _s, _e in enumerate(order_A):
        _cnt[_e] = _c0[_s]
    ref0 = jm.fp32_reference(x, ts0, ti0)
    ref0b = jm.fp32_reference(x, ts0, ti0)
    d_jud = (ref0 - ref0b).abs().max().item()
    PL = splice(ids, ref0.cpu())
    ORIG = plain(ids)
    M0 = metrics(ORIG, PL, ids)
    okP, aP = red_ok(PL)
    okO, aO = red_ok(ORIG)
    # 地板 = 用【这一段自己的】上帝视角成员另建一个静态引擎: 它是这一段的可达下界。
    # 没有它, KL/A 只能说"比静态放对差几倍", 而静态放对在漂移段本身就是错的成员集。
    top_o = torch.argsort(_cnt, descending=True).tolist()[:GC]
    mo = mk(GC, top_o + [e for e in range(E) if e not in top_o], DTYPE)
    for _ in range(WARM_ENG):
        fwd(mo, x)
    Lo = splice(ids, fwd(mo, x).float().cpu())
    Mor = metrics(PL, Lo, ids)
    ook, aOr = red_ok(Lo)
    del mo, Lo
    torch.cuda.empty_cache()
    # 本段【真价】只在"该语料的第 1 段 且 在价表里"时可得: 同语料换行区间(kth>0)那一段的
    # u_e 没量过 => M1预测/价gap 两列必须 nan, 不能拿第 1 段的价冒充(那是自欺)。
    _pd = PRICE_BY_NAME.get(nm) if kth == 0 else None
    PH.append(dict(name=nm, kth=kth, seg=nm,
                   u=list(_pd["u"]) if _pd and len(_pd["rows"]) == E else None,
                   ids=ids, x=x, PL=PL, M0=M0, d_jud=d_jud,
                   floor=Mor, top=top_o, ook=ook, aOr=aOr,
                   d_sp=d_sp, bit_sp=bit_sp, d_cap=d_cap, aP=aP, aO=aO,
                   cnt=_cnt))
    del ref0b, PL, ORIG
print("\n  闸门① splice位置(每段): " + " | ".join(
    f"阶段{k+1}{p['name']}#{p['kth']} 互差={p['d_sp']:.9f} 逐位={p['bit_sp']}" for k, p in enumerate(PH)))
_fouled = [f"阶段{k + 1}{p['name']}#{p['kth']}|P|={p['aP']:.3g}|未挂钩|={p['aO']:.3g}"
           for k, p in enumerate(PH) if p['aP'] > LOGIT_RED or p['aO'] > LOGIT_RED]
print(f"  闸门④ 裁判重算互差 max={max(p['d_jud'] for p in PH):.3g}(必须0) "
      f"| 闸门⑤ 红线 逐段|P|absmax[" +
      " ".join(f"{k + 1}:{p['name']}={p['aP']:.3g}" for k, p in enumerate(PH)) +
      f"] max={max(p['aP'] for p in PH):.3g} |未挂钩 max={max(p['aO'] for p in PH):.3g}"
      f" |地板档 max={max(p['aOr'] for p in PH):.3g}(红线{LOGIT_RED:.0f})"
      + ("  🚨裁判脏 => 元凶=" + " + ".join(_fouled) +
         " 的那几段【质量列】作废(闸门⑨ 会把它们印成『脏』; 决策/迁移/覆盖列不依赖裁判 P, 仍可引用)"
         if _fouled else "  过")
      + ("" if all(p['ook'] for p in PH) else "  🚨地板档越红线"))
print(f"  阶段1 现抓 x 与归档路径(R.capture {BS} 行长)互差={PH[0]['d_cap']:.3g}"
      f"(同长同语料 => 应为 0)")
GATE8 = gate_price()

# ---------------------------------------------------------------- 逐策略
GATE1 = []       # 闸门① 计数器
GATE2 = []       # 闸门② 迁移后 vs 按新序重建
GATE3 = []       # 闸门③ 图不重录
GATE6 = []       # 闸门⑥ 记账 == 内容
GATE7 = []       # 闸门⑦ 静态档(enabled=False)零迁移 = 对照格没被自己污染
ANCH = []        # 闸门⓪ 归档对账
LOG = {}         # pol -> 决策日志逐条(dyn.pol.reasons)
SWAPH = {}       # pol -> 按阶段汇总的迁移账(dyn.summary())
ROWS = []
PRICE_HIT = []   # 闸门⑧b: 每个 策略x阶段 是否真拿到了价(缺价时必须能从表里看出来)


def price_vec_for(pol_name, k):
    """按策略给【第 k 阶段】的价向量(eid 空间) -> (vec 或 None, 印刷标签)。
    calib = 标定段 PRICE_CAL 的价(与目标流量无关的一份离线标定) / oracle = 本段真价(上帝价)。
    frozen 给 u_e(它本身已按 e 自己的流量加权过, 再乘窗内占比 = 把流量算两遍);
    hit 给 kappa_e(=u_e/hit_e, 与流量无关的专家属性), 乘占比在策略内部完成。"""
    cfg = POLICY_PRESETS.get(pol_name, {})
    pm = cfg.get("price_mode", "off")
    if pm == "off":
        return None, "-"
    src = cfg.get("price_src", "calib")
    seg = PRICE_CAL if src == "calib" else PH[k]["seg"]
    d = PRICE_BY_NAME.get(seg)
    if d is None or len(d["rows"]) != E or (src == "oracle" and PH[k]["u"] is None):
        return None, "缺价"
    return (list(d["kappa"] if pm == "hit" else d["u"]),
            f"{'k' if pm == 'hit' else 'u'}@{seg}")


t_all = time.perf_counter()
for pol_name in POLICIES:
    _dyn_start = dict(A=order_A, neutral=NEUTRAL, shuffle=SHUF)[START_DYN]
    start = dict(static_A=order_A, static_C=order_C, neutral=NEUTRAL,
                 shuffle=SHUF).get(pol_name, _dyn_start)
    enabled = bool(POLICY_PRESETS.get(pol_name, {}).get("enabled", True))
    phase0_only = pol_name == "static_p1"
    t0 = time.perf_counter()
    m = mk(GC, start, DTYPE)
    # 🚨 DynPlacer.__init__ 不转发 src_gate(那是 ExpertSwapper 的参数) => 建完立刻挂上,
    #    闸门⑥ 才会连 gate.weight 行 + _gate_wt 列一起核。
    dyn = DynPlacer(m, BASE["experts"], GC, policy=policy_from_name(pol_name, GC),
                    window=WINDOW, check_every=CHECK, warm=WARM, order=start)
    dyn.sw.src_gate = BASE["gate_w"]
    if PRICE_THR >= 0 and dyn.pol.price_mode != "off":
        dyn.pol.thr_rel = PRICE_THR          # 消融旋钮: PRICE_THR=0 => 只看成对价杆, 不看降幅门槛
    for k, P in enumerate(PH):
        # 每阶段快照: 出表时减差 => 迁移次/ms/MB 是【本阶段】的量, 不是跨阶段累计
        s_n, s_th = dyn.pol.n_swap, dyn.pol.n_thrash
        s_us, s_mb = dyn.sw.total_us, dyn.sw.bytes_total
        s_sw = len(dyn.swaps)
        # 价在【每阶段开头】挂/摘: 价按 eid 索引, eid 空间跨阶段不变(变的是槽位) => 换阶段
        # 只换"用哪份价", 不动 slot2eid。缺价 => price=None => 策略沉默(不偷偷退回命中判据)
        _pv, _ptag = price_vec_for(pol_name, k)
        dyn.pol.price = None if _pv is None else list(_pv)
        PRICE_HIT.append((pol_name, k, dyn.pol.price_mode, dyn.pol.price is not None, _ptag))
        for _ in range(WARM_ENG):
            fwd(m, P["x"])
        torch.cuda.synchronize()
        gk = (id(m._g_pre), id(m._g_hot), m._gkey, tuple(m._wver or ()))
        if k == 0:
            gk0 = gk
        GATE3.append((pol_name, k, gk == gk0))
        # 🚨 必须 AND 上 preset 的 enabled: 早先这里漏了 => 静态档(shuffle/neutral/static_*)
        #    在阶段循环里被重新点亮, 对照格自己偷偷迁移 = 对照组污染(冒烟实测 shuffle 阶段1
        #    真的迁移了 1 次, 表里那一行是脏的)。static_p1 的"冻结"才由 phase0_only 负责。
        dyn.pol.enabled = enabled and ((not phase0_only) or k == 0)
        ss, _y = timed(m, P["x"], 1)               # 第一帧: 建 pinned 缓冲, 兼计数器闸门
        c = counts_from_engine(m, E)
        ts, ti = m.route(P["x"])
        GATE1.append((pol_name, k, int(torch.equal(c, slot_counts(ti.detach().cpu(), E))),
                      int(c.sum())))
        dyn.step(k * FRAMES + 0, k, counts=c)   # 第 0 帧也进窗口(否则 warm/检查点整格错位)
        for f in range(1, FRAMES):
            s, y = timed(m, P["x"], 1)
            ss += s
            dyn.step(k * FRAMES + f, k)         # 计数在计时窗口【外】读(那份 pinned 只活一帧)
        GATE6.append((pol_name, k, dyn.sw.check_content()))
        QD = quality_shot(P, m)              # 阶段末那一发(REDO 次) => 质量 + 记账资格
        Mq = QD["Mq"]
        # 两根新杆一律用【本段真价】当公共杆(与该行策略决策时用了哪份价无关, 才能横着比):
        #   M1预测 = Σ_{e∈CPU腿} u_e = 这个成员集的无自由参数 KL 预测(P1-②c: 预测/实测 0.92~1.11)
        #   价gap% = max(0, max_CPU u - min_GPU u)/M1预测 => 0 = 末态已是该段【价最优成员】,
        #            >0 = 还有一对可赚且这里写着它值多少 KL —— 机制量, 不随 KL 抖
        #            (与 _decide_price 同方向; 走 price_gap() 而不是就地写, 因为闸门⑧c 要用
        #             同一个函数对同一个表做恒等式自检 —— 两份实现会漂移)
        _mem = list(dyn.sw.hot_members())
        _u = P["u"]
        _pred, _pgap = price_gap(_u, _mem)
        ROWS.append(dict(pol=pol_name, k=k, name=P["name"], kth=P["kth"], ptag=_ptag,
                         pred=_pred, pgap=_pgap,
                         p50=statistics.median(ss), p50t=statistics.median(ss[-TAILN:]),
                         pmin=min(ss), nswap=dyn.pol.n_swap - s_n,
                         thrash=dyn.pol.n_thrash - s_th,
                         swap_ms=(dyn.sw.total_us - s_us) / 1e3,
                         swap_mb=(dyn.sw.bytes_total - s_mb) / 2 ** 20,
                         nsw_total=dyn.pol.n_swap, swap_ms_total=dyn.swap_ms,
                         moves=[(f, r["eid_out"], r["eid_in"]) for (f, pp, r) in dyn.swaps[s_sw:]],
                         members=list(dyn.sw.hot_members()),
                         cov=100.0 * sum(float(P["cnt"][e]) for e in dyn.sw.hot_members())
                                / float(P["cnt"].sum()),
                         cov_top=100.0 * sum(float(P["cnt"][e]) for e in P["top"])
                                   / float(P["cnt"].sum()),
                         ncpu_hot=sum(1 for e in range(E) if e not in dyn.sw.hot_members()
                                      and float(P["cnt"][e]) / float(P["cnt"].sum()) > 0.01),
                         top1=QD["top1"], kl=QD["kl"], tv=QD["tv"], rec=Mq["rec"],
                         aq=QD["aq"], aPn=QD["aPn"], nshot=QD["nshot"], qrule=QD["qrule"],
                         shots=QD["shots"], flag=QD["flag"], dirty=QD["dirty"], why=QD["why"],
                         ss=list(ss), nobs=dyn.n_obs))
    GATE7.append((pol_name, enabled, dyn.pol.n_swap))
    LOG[pol_name] = list(dyn.pol.reasons)
    SWAPH[pol_name] = dyn.summary()
    last_order = list(dyn.sw.slot2eid)
    if dyn.pol.n_swap:
        m2 = mk(GC, last_order, DTYPE)
        for _ in range(WARM_ENG):
            fwd(m2, PH[-1]["x"])
        yb = fwd(m2, PH[-1]["x"]).float().cpu().clone()
        ya = fwd(m, PH[-1]["x"]).float().cpu().clone()
        GATE2.append((pol_name, last_order, int(torch.equal(ya, yb)),
                      (ya - yb).abs().max().item()))
        del m2, yb, ya
    del m, dyn
    _pycollect(); torch.cuda.empty_cache()
    print(f"  [{pol_name:<12}] 建+跑完 {len(PH)} 阶段 {(time.perf_counter()-t0):.1f}s "
          f"| loadavg={open('/proc/loadavg').read().split()[0]}")

# ---------------------------------------------------------------- 打印
print("\n" + "=" * 168)
print(f"swiglu | P1-② 动态放置 | GC={GC} BS={BS} dtype={DTYPE} | 阶段={PHASES} | 每段 {FRAMES} 帧"
      f" 每 {CHECK} 帧一次决策(暖机 {WARM} 帧) 窗口 {WINDOW} | 总耗时 {time.perf_counter()-t_all:.0f}s")
print("=" * 168)
for k, P in enumerate(PH):
    # 参照行: 裁判 P 自己越红线 => 那个 KL 是垃圾, 印『脏』而不是印数(同闸门⑨ 印『脏』的规矩)
    _ref = (f"KL=脏(裁判|P|absmax={P['aP']:.3g}>红线{LOGIT_RED:.0f} => 本段参照与质量列全体作废)"
            if P['aP'] > LOGIT_RED else f"KL={P['M0']['kl']:.3e}")
    print(f"  阶段{k+1} = {P['name']} 第{P['kth']+1}段{'' if P['kth']==0 else '(同语料不同行区间)'}"
          f" | 命中%(eid)={pct(P['cnt'])} 最热/最冷="
          f"{float(P['cnt'].max())/max(float(P['cnt'].min()),1.0):.1f}x"
          f" | 参照(换成 MoE 本身) top1={P['M0']['top1']:.2f}% {_ref}")
    # 🚨起跑成员必须按 START_DYN 真值取(早先写死"A 否则 NEUTRAL" => START_DYN=shuffle 时
    #    表头会印成中立序成员, 与真实起跑成员(shuffle)不一致 = 印刷性撒谎, 判决轮会读错)
    _sm = dict(A=order_A, neutral=NEUTRAL, shuffle=SHUF)[START_DYN][:GC]
    _ct, _s = P["cnt"], float(P["cnt"].sum())
    cov_of = lambda mem: 100.0 * sum(float(_ct[e]) for e in mem) / _s
    print(("    🚨中立序(eid)的 GPU 成员集与 static_A 同一个集合 => neutral 行 KL 必与 "
           "static_A 逐位同, 它是占位不是对照; 真对照看 shuffle\n")
          if set(NEUTRAL[:GC]) == set(P['top']) else "")
    print(f"    该段按整段直方图的最优成员(上帝视角)={P['top']} 中立序成员={NEUTRAL[:GC]} shuffle对照成员={SHUF[:GC]}"
          f" 动态策略起跑成员={_sm}(覆盖 {cov_of(_sm):.1f}%) 上帝视角覆盖 {cov_of(P['top']):.1f}%"
          f" | 地板(另建静态引擎打这一段的上帝视角成员) KL={P['floor']['kl']:.3e}"
          f" TV={P['floor']['tv']:.3e} top1={P['floor']['top1']:.2f}%")
hdr = (f"{'策略':<12}{'阶段':>4}{'语料':>12}{'p50全':>8}{'p50尾':>8}{'最快帧':>8}"
       f"{'迁移次':>7}{'来回抖':>7}{'迁移ms':>9}{'迁移MB':>8}{'折合帧':>7}"
       f"{'GPU成员eid':>14}{'覆盖流量%':>12}{'vs地板覆盖':>12}{'top1一致%':>10}{'KL(P‖Q)':>12}{'TV距离':>11}"
       f"{'KL/A':>8}{'KL/地板':>9}{'价源':>12}{'M1预测':>12}{'价gap%':>8}")
print("\n" + hdr)
by_phase = {}
for r in ROWS:
    by_phase.setdefault(r["k"], {})[r["pol"]] = r
for r in ROWS:
    ref = by_phase[r["k"]].get("static_A")
    # 分母脏 => 这一行的两个比值同样不可引用(参照行自己没记账资格)
    dq = r["dirty"] or (ref is not None and ref["dirty"])
    ratio = r["kl"] / ref["kl"] if ref and ref["kl"] > 0 else float("nan")
    conv = r["swap_ms"] / r["p50t"] if r["swap_ms"] > 0 else 0.0
    _fl = PH[r["k"]]["floor"]["kl"]
    rf = r["kl"] / _fl if _fl > 0 else float("nan")              # 追回到可达下界的几分之一
    _kl = f"{'脏':>12}" if dq else f"{r['kl']:>12.3e}"
    _tv = f"{'脏':>11}" if dq else f"{r['tv']:>11.3e}"
    _ra = f"{'脏':>8}" if dq else fmt_ratio(ratio, 8)
    _rf = f"{'脏':>9}" if dq else fmt_ratio(rf, 9)
    print(f"{r['pol']:<12}{r['k']+1:>4d}{r['name'][:11]:>12}{r['p50']:>8.3f}{r['p50t']:>8.3f}"
          f"{r['pmin']:>8.3f}{r['nswap']:>7d}{r['thrash']:>7d}{r['swap_ms']:>9.2f}{r['swap_mb']:>8.1f}"
          f"{conv:>7.1f}{str(r['members']):>14}"
          f"{r['cov']:>12.1f}{r['cov_top'] - r['cov']:>+12.1f}{r['top1']:>10.2f}"
          f"{_kl}{_tv}{_ra}{_rf}"
          f"{r['ptag']:>12}{r['pred']:>12.3e}{r['pgap']:>8.1f}")
    if r["moves"]:
        print(f"    └ 迁移轨迹: " + " ".join(f"f{f}(出{a}←入{b})" for f, a, b in r["moves"])
              + f" | 累计(跨阶段)迁移 {r['nsw_total']} 次 {r['swap_ms_total']:.2f}ms")
    if r["dirty"] or r["flag"]:
        _tag = "判脏: 那四列印『脏』不可引用" if r["dirty"] else "收下: 但有一发脏发被剔掉了"
        _fl = " ".join(f"第{i + 1}发[{'/'.join(v)}]" for i, v in r["flag"])
        _why = "; ".join(r["why"]) if r["why"] else (_fl or "脏发已剔除")
        print(f"    🚨闸门⑨[{r['pol']} 阶段{r['k']+1}] {_tag} 发数={r['nshot']}"
              f" 判据={r['qrule'] or '未自洽'} KL各发={r['shots']}"
              f" |Q|absmax={r['aq']:.4g} |P|absmax@metrics={r['aPn']:.4g}"
              f"(建段时={PH[r['k']]['aP']:.4g}) :: {_why}")

# 决策日志(逐条, 来自 pol.reasons + dyn.swaps)
print("\n  决策日志(命中判据 rule_a/rule_b: hot=窗内最热CPU占比 cold=窗内最冷GPU占比;"
      " 价判据 rule_p(frozen|hit): v_in/v_out = 进/出专家的预测KL贡献, 预测降 = 换这一对能在"
      "【当前 CPU 腿】里省下的比例, 它必须 > thr_rel 才动手):")
for pol_name in POLICIES:
    lg = LOG.get(pol_name, [])
    ph = SWAPH.get(pol_name, {})
    if not lg:
        print(f"    {pol_name:<12} 一次都没开火")
        continue
    for s in lg:
        print(f"    {pol_name:<12} {s}")
    for k, d in sorted(ph.items()):
        print(f"    {'':<12} 阶段{k+1}: 迁移 {d['n']} 次 {d['us']/1e3:.2f}ms "
              f"对={d['pairs']} 首帧={d['first']} 末帧={d['last']}")

# 闸门汇总
bad1 = [g for g in GATE1 if not g[2]]
print(f"\n  闸门① 计数器==route(x) 逐位: {len(GATE1)-len(bad1)}/{len(GATE1)} 过"
      + (f"  🚨不符 {bad1[:3]}" if bad1 else ""))
if GATE2:
    print("  闸门② 迁移后 vs 按运行时 slot2eid 重建引擎(逐位必须=1): "
          + " | ".join(f"{g[0]} 序={g[1]} 逐位={g[2]} 互差={g[3]:.3g}" for g in GATE2))
else:
    print("  闸门② 本轮没有策略发生迁移 => 无对象(不是跳过, 是没资格)")
bad3 = [g for g in GATE3 if not g[2]]
print(f"  闸门③ CUDA Graph 不重录(id/键/_wver 全程不变): {len(GATE3)-len(bad3)}/{len(GATE3)} 过"
      + (f"  🚨重录 {[(g[0],g[1]) for g in bad3][:5]}" if bad3 else "  => 迁移只改内容, 没付重录图的价"))
bad6 = [g for g in GATE6 if g[2]]        # 🚨 存的是【不符项列表】: 空 = 绿(不是 bool)
print(f"  闸门⑥ slot2eid 与权重/门控实际内容逐位一致: {len(GATE6)-len(bad6)}/{len(GATE6)} 过"
      + ("" if not bad6 else f"  🚨{[(g[0], g[1], g[2][:3]) for g in bad6][:5]}"))
stat_rows = [g for g in GATE7 if not g[1]]        # 这一轮里的静态/对照档
bad7 = [g for g in stat_rows if g[2]]
print(f"  闸门⑦ 静态档(enabled=False)零迁移: {len(stat_rows)-len(bad7)}/{len(stat_rows)} 过"
      + (f"  🚨对照格自己迁移了 {bad7} => 那些行的 GPU成员/KL 全脏"
         if bad7 else "  => 对照格(static_*/neutral/shuffle)确实一步没动"
         if stat_rows else "  (本轮没有静态档)"))
_dyn_p = [g for g in PRICE_HIT if g[2] != "off"]
if _dyn_p:
    _miss = [f"{g[0]}@阶段{g[1]+1}({g[4]})" for g in _dyn_p if not g[3]]
    print(f"  闸门⑧b 价判据每格都真拿到价: {len(_dyn_p)-len(_miss)}/{len(_dyn_p)} 格"
          + (f"  🚨缺价 {str(_miss)[:160]} => 那几行的'一次都没开火'是缺价, 不是判据沉默"
             if _miss else "  => 表里 price_* 的'一次都没开火'可以读成判据自己住手"))
else:
    print("  闸门⑧b 本轮没选价判据策略(price_*) => 价表只当参照列")
dq9 = [r for r in ROWS if r["dirty"]]
bk9 = [r for r in ROWS if not r["dirty"] and r["qrule"] != "相邻逐位同"]
print(f"  闸门⑨ 质量列有效性(Q侧红线 + 参照侧 absmax@metrics + KL/TV 定义域 + REDO={REDO} 发内自洽):"
      f" {len(ROWS) - len(dq9)}/{len(ROWS)} 格可引用 | 平均 {sum(r['nshot'] for r in ROWS) / max(1, len(ROWS)):.2f} 发/格"
      + ("  => 表里每个 KL/TV 都有记账资格" if not dq9 else ""))
if bk9:
    print("    判据②(同值非相邻)兜底收下的格 "
          + ", ".join(f"{r['pol']}@阶段{r['k'] + 1}(KL各发={r['shots']})" for r in bk9)
          + "  => 这些格有一发脏发被剔除, 引用时把这句一起带上")
if dq9:
    print("    🚨判脏格 " + ", ".join(f"{r['pol']}@阶段{r['k'] + 1}" for r in dq9)
          + " => 那几格 KL/TV/KL-A/KL地板 印的是『脏』不是数; 同 P1-②c 的规矩: 判脏的 KL 只有在"
            "【逐位判据或归档投票】二选一给了方向之后才可引用, 否则按不确定处理(不放宽判据)")
    for r in dq9:
        print(f"      · {r['pol']}@阶段{r['k']+1}: {'; '.join(r['why']) or '脏发'}"
              f" KL各发={r['shots']} 发数={r['nshot']}")
if DTYPE == "fp32":
    for r in ROWS:
        if r["k"] == 0 and r["pol"] in ("static_A", "static_C") and BS in (512,):
            key = (r["pol"][-1], BS, GC, DTYPE)
            if key in ANCHORS:
                a = ANCHORS[key]
                dev = abs(r["kl"] - a) / max(a, 1e-300)
                ANCH.append((r["pol"], r["kl"], a, dev, r["dirty"]))
    if ANCH:
        nb = [a for a in ANCH if a[3] > 2e-3]
        print(f"  闸门⓪ 与 P0-④c 归档(md5 a1dabef9)对账: 覆盖 {len(ANCH)} 格 不符 {len(nb)} 格 "
              f"最大相对偏差 {max(a[3] for a in ANCH):.3g} (阈值 2e-3 = 归档只 4 位有效)"
              + ("" if not nb else "  🚨" + str(nb)))
        for pol, got, exp, dev, dirty9 in ANCH:
            print(f"    · {pol} gc={GC} {DTYPE} 本表 KL={got:.4e} 归档={exp:.4e} 相对偏差={dev:.3g}"
                  + ("  〔闸门⑨ 判脏 => 这一行只是【归档第三方票】, 本表那一发不可引用〕" if dirty9 else ""))
    else:
        print(f"  闸门⓪ 本轮没有可锚的格子(要 phase0=readme + static_A/static_C + BS/GC 在归档里)")
else:
    print(f"  闸门⓪ 只覆盖 fp32: 归档那份 A bs=512 gc=3 bf16 有一格脏 logits(已作废) => {DTYPE} 行不锚")

print("""
判读提示: ① 这一刀的问法是【自适应值不值】, 不是"能不能自适应":
          每行右边四列(GPU成员/覆盖流量%/vs地板覆盖/KL/地板)回答"它有没有追到该追的成员",
          中间四列(迁移次/迁移ms/折合帧/来回抖)回答"它为此付了多少"。
          值不值 = KL/A 降了几倍 vs 折合帧数花了几个帧, 两个数都在表里, 不要凭感觉。
       ② 决策日志里出现 "一次都没开火" 是【结论】不是 bug: 外部原型的两条独立阈值
          (CPU>5% 提升 / GPU<1% 降级)在真实漂移下常常【永远等不到降级那一半】
          —— 最冷的 GPU 专家还有百分之几, demote_max=1% 就把它锁死了。
          gain(成对判据)那条腿就是为这个加的, 它开不开火看 rule_a/rule_b 标的是哪个。
       ③ p50全 vs p50尾 是【同策略同阶段】的唯一配对: 差得多 = 有帧内事件(迁移当帧、
          掉核、首帧冷启动)。迁移当帧的尖峰会进 p50全 但【不该】读成引擎变慢 => 读趋势看尾段。
       ④ 迁移成本的大头(本机实测归因, 见 OPTLOG P1-② 小节): CPU 腿重量化 ~6.7ms +
          设备热权重缓冲重写 ~3.6ms + H2D ~2.3ms + 门控行 ~0.2ms。前两项属于 P2
          (启动期把所有专家的 int8 冷表一次建好, 迁移退化成指针级), 这里【不优化只定价】。
       ⑤ 闸门② 是这一刀的命: 逐位=1 才说明"换内容不换地址"这条路走得通(图不用重录、
          质量不会漂)。它红 = 记账方向或某处布局漏改, 整表作废, 别去读 KL。
       ⑥ 中立序(neutral)那一行才是【没做动态放置】的真对照: static_A 是上帝视角
          (离线拿整本语料的直方图选成员, 运行时不可能有), 只能当上界锚点写进台账。
       ⑦ KL/top-1 是相对量: 每段开头那条参照(换成 MoE 本身)比引擎最狠那档贵几个数量级
          (P0-③ 判决⑤), 所以只报 KL/A 和 KL/地板 这两列的比, 别报绝对 ppl。
       ⑧ KL/地板 = 相对【该段上帝视角成员】的那一档: 阶段1 的 static_A 本来就在地板上 =>
          该列必为 1.00(它不是 1.00 = 成员选错或表建错) = 追平; 越大越没追上。
          🚨但"地板"是【按命中直方图】选的成员, 不是 KL 最小化器 —— 每个专家的 int8 误差
          不一样大, 所以 KL/地板 < 1 是【合法读数】(覆盖率更高的成员集反而更差时就会 <1),
          它只说明"直方图最优 != KL 最优", 别读成"引擎突破了可达下界"。
          真正单调的是 覆盖流量% 这一列(= GPU 成员吃掉多少比例的流量), 它是机制量;
          KL 是它的下游, 两者方向不一致时以机制量讲结论、以 KL 报代价。
       ⑨ drift 段是【放大器】不是流量画像: 它把 CPU 腿专家的命中占比人为拉高, 好让效应量
          高出噪声底。它证明的是"漂移一旦发生, 动态放置能不能追上 + 追一次多少钱", 不证明
          "真实流量会这么漂" —— 真漂多少见 P0-③补(换语料 A序 1<->5 换一个人)。
       ⑩ 价判据(price_*)与命中判据(hyst_*)【不是同一根杆】: 后者问"谁更热", 前者问"谁让 int8
          多付 KL"(P1-②c: kappa 谱 150x 远陡于流量谱 <0.5x => "热"只是它的代理, 而且在漂移段
          代理失效: 覆盖流量那根杆在 drift 只有 Spearman +0.60, 单价那根还是 +1.00)。
          🚨别拿 price_* 的 gain=1 和 hyst_1x 的 gain=1 当同一个刻度读 —— 一个比价一个比热,
          没有可比性; 但两族共用冷却/容量/迁移机制, 所以"迁移次/折合帧/来回抖"可以横着比。
       ⑪ 最后三列是【价杆】: 价源(u@标定段 / k@标定段 / u@本段真价=上帝价 / 缺价)= 这一行决策时
          看到的是哪份价; M1预测 与 价gap% 一律用【本段真价】算, 不随策略换了价源而变, 是机制量。
          · 价gap% = 0 => 末态已是该段价最优成员; >0 => 还有一对可赚, 这个数就是它值多少 KL 比例;
          · price_orc 减 price_cal 的差 = "在线估价 / 换标定语料"这门生意的【最大彩金】: 小 =>
            一次离线标定就够, 不要建影子探针(省掉 P1-②e); 大且方向稳定 => 才值得为它写那 40 行;
          · price_cal 与 price_hit 末态成员集相同 => 用在线命中给单价重定价【不改决策】=> 上线
            直接查 u_e 表就够, 计数器留给别的判据用; 不同 => 才需要决定信哪一份(kappa 跨段
            Spearman 只有 +0.16, 见 P1-②c 判决 —— 那时读"命中重定价"比读"查 kappa"更要紧)。
       ⑫ 闸门⑨ = 质量列的【记账资格】: KL/TV 是确定量(闸门④ 互差=0), 所以"两发逐位同"是资格
          不是降噪。判脏的行那四列印『脏』而不是印垃圾 —— 判决轮曾印出 KL=5.77e+151 / TV=4.35e-35
          把表顶歪, 而那时 Q 侧红线是【通过】的: 🚨只核一侧幅度的红线等于没核, 所以现在同时核
          参照侧 absmax 在建段后有没有变(一致性)+ 它自己有没有过【绝对】红线(born-dirty)
          + KL 有没有超本段参照的 KL_CAP 倍 + TV 在不在 [0,1]。🚨r3 复核轮的教训: 裁判
          |P|absmax=5.85e+19 时 ⑤ 红而 ⑨ 绿 —— 因为 (b) 只比"变没变", 生下来就脏的裁判与
          自己比必绿。**两把闸门冲突时以红的那把为准 = 该段质量列作废**, 一致性检查永不代替绝对红线。
          兜底(同值非相邻 >=2 次)收下的行仍可用, 但要连带那句"有一发脏发被剔除";
          判脏的行: 引用走归档对账或重跑投票, 不直接进台账(方法论 32)。
""")

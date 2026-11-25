#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-④a 判决表: 把【显存 <-> 延迟 <-> 质量】三根轴装进同一次运行(同一套真权重、同一个 gc 网格)。

【为什么必须做它】
  P0-④ 的主图是"三方交换曲线": 显存轴 + 延迟轴 + 质量轴。可这两根轴原来分属两套口径:
    · P0-② 的 gc 扫描(bench_baselines.py)有【峰值显存 + 延迟】, 但权重是 D=1024 随机
      std=1 的玩具, 精度只到 relRMSE;
    · P0-③(bench_metrics.py)有【真权重 + token 级质量】, 但整份代码【一次计时都没有】,
      也没有峰值显存(它那一列 CUDA驻MB 是"权重驻留"口径, 不是窗口高水位)。
  收工 P0-③ 时我把"两边的显存列禁止交叉引用"写进了台账 => 那张图【现在画不出来】。
  本表把三根轴一次跑齐: 同一批真激活 x、同一个裁判 P、同一个 gc 网格, 每格同时出
  峰值显存 / p50独占 / top-1 + KL + TV。

【口径(四条, 别看错)】
  1) 逐点【独占】: 每个 gc/dtype 点都是"建一个 -> 量一个 -> 删一个", 窗口开之前查空载。
     => 这一张是【曲线】不是【配对】: 两点之差 <噪声底(A 2~3% / C 8~18%)不要下结论。
     (同一 gc 的 fp32/bf16 是例外, 见第 3 条。)
  2) 显存出【两列】(2026-11-21 探针 /tmp/probe_pk.py 定下来的, 一列会骗人):
     · 峰值MB = 独占窗口高水位 - 空载基线, 与 P0-② gc 扫描同口径。🚨它【不含】降精的收益:
       建图/暖机的瞬时是 fp32 staging(gc=5 实测 fp32 与 bf16 的高水位都是 186.1MB)。
     · 稳态MB = 暖机完成后仍驻留的字节, 与 P0-③ 那列 CUDA驻MB 同口径(169.3/109.8 逐位对上)。
       降精/去死权重的显存收益只在稳态列上看得见 => 三方交换图的显存轴用【稳态MB】,
       峰值列只用来跟 P0-② 对齐 + 暴露"建图瞬时也要这么多"这件事。
     两列都比"权重算式"大: 差额 = wcat 副本 + 死权重 bmm 副本 + CUDA 图静态缓冲。
  3) dtype 之间(=降精这一刀)另起【轮转配对段】: 同一 gc 的 fp32/bf16 两个引擎同时常驻,
     逐轮配对 + 单双号换序(位置偏置自查)。同 gc => CPU 侧 int8 专家集相同 => 轮转公平。
     质量数【只从独占段取】(确定量不需要配对), 配对段只借它量时间。
  4) 延迟吃的是【真激活】x(与质量段同一批), 命中率由真路由决定 => 绝对值不必等于 P0-②
     那张玩具权重表, 本表只在自己的曲线内部比。

【闸门】
  ⓪ 跨bench对账: 同格 KL 必须与 P0-③ 归档表在 4 位有效内一致(锚点直接抄在下面的
     ANCHORS 里, 60 格)。它证明本表新加的计时/显存窗口【没改动数值路径】, 顺带证明
     语料与 HF 快照没漂。这是这张表最硬的一道闸, 因为它对的是另一份代码的归档值。
  ① splice 位置: hook 覆盖前录到的 MLP 输入与现抓 x 逐位相同(必须 1)。
  ② 路由: 每格各 gc/dtype 的 top-k 与裁判档(gc=0 fp32)逐位相同(idx差 = 0)。
  ③ 三模式互差: 引擎 sync/L0/L1/L2 逐位 0。
  ④ 裁判自检: 各档自己的 fp32_reference vs 裁判 P 的 ref(index_add 原子序, 不保证 0)。
  ⑤ 空载闸门: 每个窗口开之前空载必须 <2MB, 否则显存列当场作废报警。
  ⑥ 共驻不互污: 配对段两个引擎同时活着时, 各自输出与独占段那份【逐位 0】。
  ⑦ 幅度红线(= bench_metrics 闸门⑥(a), 纯观测不重算): |logit| <= LOGIT_RED(默认 1e4; 真 logits
     absmax ~O(30), 而 bs=512 那轮出现过 6.59e+19 的脏值)。本表已有闸门⓪ 逐格对 P0-③ 归档的 60 个
     KL, 红线只是它的【廉价双保险】: 闸门口径不一样 —— 闸门⓪ 对的是平均量(KL), 一个脏元素会被 T
     行摊薄; 红线看的是单元素幅度, 摊不掉。这里不重算: 本表每格只有一次独占前向的额度, 重算会把
     时间/显存列的窗口口径改掉。

【乙-2d 加的轴】IMPLS=cat,traffic —— GPU 半边【怎么读权重】。
  cat(默认, 入库行为逐字不变) = 稠密: 每帧读满 gc 个专家的三张矩阵, 门控为 0 也读。
  traffic = 网格开在 (bs*top_k) 的【流量】上、在 device 上按 idx 间接寻址专家,
            零流量专家一个字节都不读(实现与单价算式见 gpu_traffic.py 文件头)。
  🚨 口径三条: (a) 闸门⓪ 的锚点全是 cat 档 => 只判 cat 行, traffic 行不覆盖;
  (b) traffic 换了归约序 => 逐位不可能等于 cat, 闸门⑧ 判的是【归约序噪声底】
      (fp32 1e-4 / bf16 5e-2 相对量级), 不是"接近"; (c) 稳态显存两档【应该相同】
      (读同一组 _wcat_gu/_wstack_d, 只差几 KB scratch) => 两档在同一条显存竖线上,
      谁在时间轴左边谁赢, 这是三方交换图上新增的"同价位换时间"那一刀。
env: LAYERS=0,4,8,12,16,20,24,28  GATE=embed  CAP_LAYER=8  HIT_BS=512  N_CHARS=24000
     BS=1,32  GC=0,1,3,5,7,8  DTYPES=fp32,bf16  ROUNDS=7  ITERS=20  CORPUS=(空)
     IMPLS=cat(默认) | cat,traffic(乙-2d 判决跑法) —— IMPLS[0] 必须是 cat(基准档)
     形状钉死 swiglu: gemm 档专家输出宽 1536 拼不回 576 维残差流, "下一个词的分布"语义不成立。
"""
import os, sys, time, math, statistics, torch
from gc import collect as _pycollect          # ✅ 不叫 gc: 下面循环变量就叫 gc
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R

LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
HIT_BS = int(os.environ.get("HIT_BS", 512))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS_LIST = [int(v) for v in os.environ.get("BS", "1,32").split(",")]
GC_LIST = [int(v) for v in os.environ.get("GC", "0,1,3,5,7,8").split(",")]
DTYPES = [d for d in os.environ.get("DTYPES", "fp32,bf16").split(",") if d]
# ✅ 乙-2d: GPU 腿实现轴。默认只有 cat => 本表退回 P0-④a 入库形态(逐字不变)。
IMPLS = [i.strip() for i in os.environ.get("IMPLS", "cat").split(",") if i.strip()]
# ✅ 放置规则轴: ✅ 默认已切 v2 —— 依据 = 用户判决轮 bs=32 RULE=v2(md5 f17fc0d0, OPTLOG 第 2 节):
#    四个决策单元对着【全域实测最优】偏贵 0.0/8.4/3.0/0.0%(平均 2.8%), 同口径 v1 是 0.0/23.4/14.8/206.9%(平均 61.3%),
#    且 v2 四格全选 cat = 实现档那条腿在边界正确一侧。老那一行"规则选 gc"(v1 口径)照印不动。
#    🚨复现 P0-④a / 乙-2d 那两份旧表要显式 RULE=v1(它们印的是 v1 默认形态, v2 会【多印一行】)。
#    bs=128 的 v2 实测已到手(用户判决轮 2026-11-21 19:38, md5 25746271): 四个决策单元【全部】正中全域
#    实测最优、平均偏贵 0.0%(同口径 v1 是 114.6%, 最差 +303.7%) => 只剩 bs=1 还欠实测(离线重放已有)。
RULE = os.environ.get("RULE", "v2").strip()
MAC_WALL = os.environ.get("MAC_WALL", "1").strip() in ("1", "true", "on")
assert RULE in ("v1", "v2"), f"RULE 要的是 v1/v2, 收到 {RULE!r}"
# 独占段/打印段共用的格子序: 同一 gc 内先 fp32-cat 再 fp32-traffic ...
# 🚨 顺序有讲究: cat 必须排在同 dtype 的 traffic 前面, 因为 traffic 行的闸门⑧
#    要拿【同一 gc/dtype 的 cat 那份 y】当基准, 它得先算出来。
CELL = [(dt, im) for dt in DTYPES for im in IMPLS]
ROUNDS = int(os.environ.get("ROUNDS", 7))
ITERS = int(os.environ.get("ITERS", 20))
CORPUS = os.environ.get("CORPUS", "").strip()
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
# DIAG=1 => 每行尾多印【已有但没进表】的两个纯观测量 dmax(max|P-Q| logits) / kl95(P95 token 的 KL)。
#   用途 = 归因: 闸门⓪ 抓到偶发脏格时, dmax 能一眼分清【单个 token 的 logit 尾被搬走】还是【整片量级变了】。
#   默认关 => 入库表格式逐字不变。这两个数本来就算好了(metrics() 里), 加它零成本、不碰数值路径。
DIAG = os.environ.get("DIAG", "").strip() in ("1", "true", "on")
E = len(LAYERS)
LOGIT_RED = float(os.environ.get("LOGIT_RED", 1e4))   # 闸门⑦ 幅度红线(同 bench_metrics 闸门⑥(a))
JUDGE_GC = 0            # 🚨 裁判 P 固定用 gc=0 的 fp32 档 —— P0-③ 归档跑法就是 GC[0]=0
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
ELS = {"fp32": 4, "fp16": 2, "bf16": 2}
NOISE_T = 0.02          # 噪声底: 配对前沿时时间/显存轴上 2% 以内当平手(A 档口径, 保守)

assert set([k for k in os.environ.get("EKIND", "swiglu").split(",") if k]) == {"swiglu"}, \
    "P0-④a 只能在 swiglu 档做(见 docstring 末行)"
assert set(DTYPES) <= {"fp32", "fp16", "bf16"}, DTYPES
assert set(IMPLS) <= {"sparse", "cat", "bmm", "traffic"}, IMPLS
assert IMPLS[0] == "cat", "IMPLS[0] 必须是 cat: 闸门⓪/闸门⑧/配对段全拿它当基准档"
assert 0 <= CAP_LAYER < 30 and E >= max(GC_LIST) and min(GC_LIST) >= 0, (E, GC_LIST)
#    🚨 上界是 >= 不是 >: gc=E(纯 GPU)是这条曲线的上界锚点(P0-②/P0-③ 都拿它当端点)。
if JUDGE_GC not in GC_LIST:
    print(f"🚨 GC={GC_LIST} 里没有 {JUDGE_GC}: 锚点闸门⓪ 只能覆盖 gc≠裁判档的格子")

# ---------------------------------------------------------------- 闸门⓪ 锚点(P0-③ 归档值)
# 来源: results/2026-11-21_P0-③_metrics_top1-kl.txt (bench_metrics.py 用户跑, md5 fba72657…)
# 键 = "配置 bs gc dtype", 值 = KL(P‖Q)。4 位有效是归档格式, 所以判定阈值取 2e-3 相对偏差。
ANCHORS = {}
for _l in """
A 1 0 fp32 4.928e-03
A 1 0 bf16 4.928e-03
A 1 3 fp32 6.105e-11
A 1 3 bf16 1.353e-05
A 1 5 fp32 6.105e-11
A 1 5 bf16 1.353e-05
A 1 7 fp32 6.105e-11
A 1 7 bf16 1.353e-05
A 1 8 fp32 6.105e-11
A 1 8 bf16 1.353e-05
C 1 0 fp32 4.928e-03
C 1 0 bf16 4.928e-03
C 1 3 fp32 4.928e-03
C 1 3 bf16 4.928e-03
C 1 5 fp32 4.928e-03
C 1 5 bf16 4.928e-03
C 1 7 fp32 1.882e-04
C 1 7 bf16 1.355e-04
C 1 8 fp32 6.105e-11
C 1 8 bf16 1.353e-05
A 32 0 fp32 2.841e-04
A 32 0 bf16 2.841e-04
A 32 3 fp32 2.220e-05
A 32 3 bf16 2.166e-05
A 32 5 fp32 1.204e-05
A 32 5 bf16 1.180e-05
A 32 7 fp32 3.682e-06
A 32 7 bf16 4.045e-06
A 32 8 fp32 1.126e-11
A 32 8 bf16 9.376e-07
C 32 0 fp32 2.841e-04
C 32 0 bf16 2.841e-04
C 32 3 fp32 2.574e-04
C 32 3 bf16 2.579e-04
C 32 5 fp32 2.446e-04
C 32 5 bf16 2.434e-04
C 32 7 fp32 2.091e-05
C 32 7 bf16 1.969e-05
C 32 8 fp32 1.126e-11
C 32 8 bf16 9.376e-07
A 128 0 fp32 1.016e-04
A 128 0 bf16 1.016e-04
A 128 3 fp32 9.041e-06
A 128 3 bf16 9.635e-06
A 128 5 fp32 4.086e-06
A 128 5 bf16 4.546e-06
A 128 7 fp32 1.247e-06
A 128 7 bf16 1.532e-06
A 128 8 fp32 4.979e-12
A 128 8 bf16 3.705e-07
C 128 0 fp32 1.016e-04
C 128 0 bf16 1.016e-04
C 128 3 fp32 9.341e-05
C 128 3 bf16 9.396e-05
C 128 5 fp32 8.901e-05
C 128 5 bf16 8.929e-05
C 128 7 fp32 1.723e-05
C 128 7 bf16 1.710e-05
C 128 8 fp32 4.979e-12
C 128 8 bf16 3.705e-07
""".strip().splitlines():
    _lb, _bs, _gc, _dt, _kl = _l.split()
    ANCHORS[(_lb, int(_bs), int(_gc), _dt)] = float(_kl)

# ---------------------------------------------------------------- 真模型(只 load 一次)
def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 这台机器 DNS->127.0.0.1, 强制离线
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = R.snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        mdl = AutoModelForCausalLM.from_pretrained(snap)
        # 🚨 必须显式 .to(float32): config 里 torch_dtype=bfloat16, 不转则 hook 塞回的 fp32 y
        #    在之后那层 rmsnorm 上报 dtype 不同, 而且裁判 P 就不是 fp32 了。
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
_SD = R.state_dict()                  # 只读一次 safetensors(~500MB)
_R_sd = R.state_dict
R.state_dict = lambda: _SD            # ✅ 本表要建上百个引擎, 每次 build 都重读 safetensors 太亏。
#    安全性: real_src 里 expert_mats 一律 .float().t().contiguous() => 每次拿到的是【新张量】,
#    引擎拿到的是副本(探针 hetero_moe_int8.py:252/259 全是 .float()/.contiguous()), 不会被共用污染。
dims = R.dims(_SD)
D, N, NL = dims["D"], dims["N"], dims["layers"]
CORPUS_TEXT = (open(CORPUS, encoding="utf-8", errors="ignore").read()[:N_CHARS]
               if CORPUS else R.real_text(N_CHARS))
MODEL, TOK = _load()
IDS_FULL = TOK(CORPUS_TEXT, return_tensors="pt").input_ids
V = MODEL.get_input_embeddings().weight.shape[0]
torch.manual_seed(SEED)


def capture_from(ids):
    """在真模型第 CAP_LAYER 层 MLP 入口抓 x(与 bench_metrics 逐字同一挂法)。"""
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


if CORPUS:
    XREAL = capture_from(IDS_FULL[:, :HIT_BS])
else:
    XREAL, _d = R.capture(layer=CAP_LAYER, bs=HIT_BS, n_chars=N_CHARS)
assert XREAL.shape[1] == D, (XREAL.shape, D)
assert IDS_FULL.shape[1] >= HIT_BS, IDS_FULL.shape


# ---------------------------------------------------------------- 引擎侧(与 bench_metrics 同写法)
def perm_src(order):
    """按 order 重排专家: order[新 id] = 旧 id; gate_w 的行必须跟着走。"""
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


def mk(gc, order, kind, dt, impl=None):
    """建模型 + 上 GPU + 暖一次缓存。🚨 与 bench_metrics.mk 逐字同(闸门⓪ 靠这个才成立)。
    impl=None 或 "cat" => 用 FULL(= 入库形态逐字不变); 乙-2d 才传 "traffic"。"""
    torch.manual_seed(SEED)
    kw = FULL if impl in (None, FULL["gpu_impl"]) else dict(FULL, gpu_impl=impl)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind=kind,
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **kw).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def win():
    """干净窗口: 回收 -> sync -> empty_cache -> 读空载基线 -> 清零高水位(闸门⑤)。"""
    _pycollect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    if base > 2.0:
        print(f"  🚨窗口不干净: 空载仍有 {base:.1f}MB 常驻 => 这一段显存数字作废")
    return base


def warm(m, x, n=12):
    for _ in range(n):
        fwd(m, x)
    torch.cuda.synchronize()


def one_round(m, x):
    """一轮 = ITERS 次取中位(方法论第 1 条: 禁 min-of-N)。"""
    s = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd(m, x)
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


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


# ------------------------------------------------- 闸门⑦ 幅度红线探针(纯观测, 不重算)
def red_ok(L):
    """只看 max/min —— 不造 abs 临时(bs=512 时 V=49152, 一个 fp64 临时就是 200MB)。"""
    mx, mn = float(L.max()), float(L.min())
    a = max(abs(mx), abs(mn))
    return a <= LOGIT_RED, a


def red_locate(L, tag):
    """越红线才付的代价: 脏元素在哪、位置固不固定、是不是同一个值。"""
    bad = L.abs() > LOGIT_RED
    idx = bad.nonzero()
    rows, cols = idx[:, 0].unique(), idx[:, 1].unique()
    v = L[bad].flatten()
    print(f"    🚨[{tag}] 脏 n={int(v.numel())}/{L.numel()} "
          f"行={int(rows.numel())}{tuple(rows[:5].tolist())} 列={int(cols.numel())}{tuple(cols[:5].tolist())} "
          f"唯一值={int(torch.unique(v).numel())} absmax={float(v.abs().max()):.4g} 首值={float(v[0]):.6g}")


def metrics(lp, lq, ids_row):
    """P/Q 的 logits -> token 级指标。float64, 全程 log_softmax 域。(= bench_metrics.metrics)"""
    T = lp.shape[0]
    P, Q = torch.log_softmax(lp, -1), torch.log_softmax(lq, -1)
    p, q = P.exp(), Q.exp()
    kl = (p * (P - Q)).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    a1p, a1q = lp.argmax(-1), lq.argmax(-1)
    k5p, k5q = lp.topk(5, -1).indices, lq.topk(5, -1).indices
    rec = (k5p.unsqueeze(2) == k5q.unsqueeze(1)).any(2).float().mean().item() * 100
    tgt = int(a1p[-1])
    rank_last = int((lq[-1] > lq[-1, tgt]).sum().item()) + 1
    nllp = nllq = float("nan")
    if T >= 2:
        lab = ids_row[0, 1:].long().unsqueeze(1)
        nllp = float(-P[:-1].gather(1, lab).mean())
        nllq = float(-Q[:-1].gather(1, lab).mean())
    kp = max(1, min(T, -(-T * 95 // 100)))
    return dict(top1=(a1p == a1q).float().mean().item() * 100.0, rec=rec,
                kl=float(kl.mean()), kl95=float(kl.kthvalue(kp).values), tv=float(tv.mean()),
                dmax=float((lp - lq).abs().max()), nllp=nllp, nllq=nllq,
                rank=rank_last, last=int(a1p[-1] == a1q[-1]),
                argq=int(a1q[-1]), argp=int(a1p[-1]))


pct = lambda c: [round(float(v) / float(c.sum()) * 100, 1) for v in c]
cnt_e = hits_of(R.real_src(LAYERS, gate_mode=GATE)["gate_w"], XREAL)
order_A = torch.argsort(cnt_e, descending=True).tolist()
order_C = torch.argsort(cnt_e).tolist()
CFGS = [("A", order_A, "放对: 真最热的 gc 个进 GPU"),
        ("C", order_C, "放错: 真最冷的 gc 个进 GPU")]
eb = N * D * 3                                    # 一个专家的元素数(swiglu = 3 个矩阵)
fmt = lambda v, spec, w: (f"{v:{spec}}" if v == v else f"{'—':>{w}}")

print(f"[真源] SmolLM-135M: D={D} N={N} 层数={NL} 词表={V} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家"
      f" top_k={K} | 路由={GATE} | 模型加载 {time.perf_counter()-t_load:.1f}s")
print(f"[语料] {'CORPUS=' + CORPUS if CORPUS else '本仓库 readme 拼贴(域外, 只有 Δ 可比)'}"
      f" {len(CORPUS_TEXT)} 字符 -> {IDS_FULL.shape[1]} token | x=第 {CAP_LAYER} 层 MLP 入口 {tuple(XREAL.shape)}")
print(f"[被测] 三方交换: 峰值显存(独占窗口) + p50独占(逐点独占 {ROUNDS}轮x{ITERS}次取中位) + "
      f"token级质量(top-1/KL/TV) | 同一批真激活同一份裁判 | bs={BS_LIST} gc={GC_LIST} dtype={DTYPES}")
print(f"[口径] ① 逐点独占 => 这是【曲线】不是【配对】, 两点之差 <噪声底(A 2~3%/C 8~18%)不要下结论;"
      f" 只有 dtype 那一栏是配对(见 ③)")
print(f"[口径] ② 显存两列: 峰值MB=窗口高水位-空载(同 P0-② gc 扫描, 🚨建图瞬时是 fp32 staging"
      f" => 降精不动它) | 稳态MB=暖机后仍驻留(同 P0-③ 的 CUDA驻MB, 降精只在这列看得见)。"
      f" 权重算式 fp32={eb*4/2**20:.1f}MB/专家 bf16={eb*2/2**20:.1f}MB/专家"
      f" int8={eb/2**20:.2f}MB/专家; 两列都比它大 = wcat+死bmm副本+图静态缓冲(不是 bug)")
print(f"[口径] ③ dtype 那一刀另起【轮转配对段】: 同 gc 两档同时常驻 + 单双号换序;"
      f" 质量数一律来自独占段(确定量), 配对段只借它量时间")
print(f"[口径] ④ 延迟吃真激活 => 绝对值不必等于 P0-② 玩具表; ΔNLL/ppl 全表不报"
      f"(P0-③ 判决⑥: 那些数在噪声里), 质量只报 top-1/KL/TV")
if RULE == "v2":      # 🚨 RULE=v1(默认)时这一行【不印】=> 归档表头逐字不变
    print(f"[口径] ⑤ 放置规则轴 RULE=v2 => 每个 dtype【多印一行】'规则v2选 …'"
          f"(网格直选 + 算力墙{'on 288GMAC/s' if MAC_WALL else 'off'});"
          f" 老的'规则选 gc'(v1 口径)照印 => 两版并排读")
print(f"[命中] {HIT_BS} 行真激活累计 命中%={pct(cnt_e)} 最热/最冷="
      f"{float(cnt_e.max())/max(float(cnt_e.min()),1.0):.1f}x | A序={order_A} C序={order_C}")

# ---------------------------------------------------------------- 逐格
for bs in BS_LIST:
    ids_row = IDS_FULL[:, :bs]
    # 🚨 x 必须在【同一个 T】上现抓: 归档那次是 512 行一起前向, 不同 batch 长度走不同 GEMM
    #    分块/归约序 => 差 ~5e-7, 闸门① 不可能逐位 0。
    xcpu = capture_from(ids_row)
    d_cap = (xcpu - XREAL[:bs]).abs().max().item()
    x = xcpu.cuda()
    for label, order, note in CFGS:
        print("\n" + "=" * 150)
        print(f"swiglu | 配置{label} {note} | bs={bs:>3d} | 三方交换(显存/延迟/质量同格) "
              f"| dtype={DTYPES} gc={GC_LIST} | 新序命中%={pct(cnt_e[order])} "
              f"| loadavg={open('/proc/loadavg').read().split()[0]}")
        print("=" * 150)
        t_blk = time.perf_counter()
        # ---- 裁判段(独占): 建 gc=0 fp32 -> 拿 P -> 删。P 与 Q 无关, 每格只有一个 P。
        wbase = win()
        jm = mk(JUDGE_GC, order, "swiglu", "fp32")
        judge_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) - wbase
        ts0, ti0 = jm.route(x)
        ref0 = jm.fp32_reference(x, ts0, ti0)
        PL = splice(ids_row, ref0.cpu())
        ORIG = plain(ids_row)
        M0 = metrics(ORIG, PL, ids_row)
        okP, aP = red_ok(PL)                       # 闸门⑦: 裁判 P 的幅度红线
        okO, aO = red_ok(ORIG)
        P_DIRTY = not (okP and okO)
        if not okP:
            red_locate(PL, f"P bs={bs} 配置{label}")
        if not okO:
            red_locate(ORIG, f"未挂钩 bs={bs} 配置{label}")
        del jm                      # 🚨 ref0(=裁判 P 的 MoE 输出)留到本块结束: 闸门④ 逐格要对它
        box = {}
        splice(ids_row, torch.zeros(bs, D), box)
        d_splice = (box["x"][0] - xcpu).abs().max().item()
        bit_splice = int(torch.equal(box["x"][0], xcpu))
        win()
        print(f"  样本 = {bs} 个 token 位置"
              + ("  🚨T<16: top1/TV 在这一格是【单次抽样】不是比率, 只能读 KL 的量级" if bs < 16 else "")
              + ("  🚨T=1: ΔNLL/ppl 无定义(P0-③ 判决⑥) => 本表这两列干脆不印" if bs == 1 else ""))
        print(f"  闸门① splice互差={d_splice:.9f} 逐位={bit_splice}(必须1) | 现抓 x 与归档路径"
              f"(R.capture 512 行长)互差={d_cap:.3g}(非0 是应该的: 不同 batch 长度 => 不同 GEMM 归约序)"
              f" | 未挂钩 logits rms={ORIG.pow(2).mean().sqrt():.3g} | 裁判段显存 {judge_mb:.1f}MB")
        print(f"  参照【未挂钩真模型 → P】= 把第{CAP_LAYER}层真 MLP 换成 {E} 选 {K} 的 fp32 稠密 MoE"
              f" 本身的代价: top1={M0['top1']:.2f}% top5={M0['rec']:.2f}% KL={M0['kl']:.3e}"
              f"  🚨这条不是这一刀的代价, 是「把真层换成 MoE」的代价(P0-③ 判决⑤: 只报相对量)")

        # ---- 独占段: 逐点"建一个/量一个/删一个", 三根轴一起出
        icol = len(IMPLS) > 1                      # 只有一个实现时不印 impl 列(P0-④a 原样)
        print(f"{'gc':>3}{'dtype':>7}" + (f"{'impl':>9}" if icol else "")
              + f"{'峰值MB':>10}{'稳态MB':>10}{'p50独占':>9}{'σ%':>5}{'vs纯CPU':>10}"
              f"{'GPU腿slot':>11}{'CPU腿slot':>11}{'CPU驻MB':>9}{'relRMSE':>10}{'top1一致%':>10}"
              f"{'KL(P‖Q)':>12}{'TV距离':>11}{'末位rank':>9}{'idx差':>7}{'三模式互差':>12}")
        ROW, YBIT = {}, {}
        pk_q_delta = 0.0
        for gc in GC_LIST:
            for (dt, impl) in CELL:
                wbase = win()
                m = mk(gc, order, "swiglu", dt, impl)
                warm(m, x, 20)
                torch.cuda.synchronize()
                pk = (torch.cuda.max_memory_allocated() / 2 ** 20) - wbase
                st = (torch.cuda.memory_allocated() / 2 ** 20) - wbase   # 稳态=降精看得见的那列
                rs = [one_round(m, x) for _ in range(ROUNDS)]
                p = statistics.median(rs)
                ts, ti = m.route(x)
                idx_d = int((ti != ti0).sum())
                # 闸门③(三模式互差)成立前提: 入库形态 FULL 不带 tail_graph => 默认 False => _finish 走
                # _h2d_result 的 torch.add, 每发都返回新张量 => 三个 mode 是三个不同对象/不同 data_ptr,
                # 所以这一道是真闸门(探针逐位验过: 互差 0.0, 且再打 3 发后回读不被覆写)。
                # 🚨陷阱: 若日后把 tail_graph 接上 True, _finish 会返回复用的 _tail_out => 三 mode 同一张量
                #   => 闸门③ 静默退化成【自我比较】(恒等 0.000000000, 看着全绿其实什么都没比)。
                #   届时要先 .clone() 再比, 或照 check_redlines.py 的覆写探针另单打一发。
                outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
                worst = max((outs[a] - outs[b]).abs().max().item()
                            for i, a in enumerate(outs) for b in list(outs)[i + 1:])
                refi = m.fp32_reference(x, ts, ti)
                _, _, rel = H.err_metrics(outs[MODE], refi)
                djud = (refi - ref0).abs().max().item()
                ncpu = int((ti >= m.gpu_expert_count).sum())
                y = outs[MODE]
                nbad = int(torch.isinf(y).sum() + torch.isnan(y).sum())
                Lq = splice(ids_row, y.cpu())
                okq, aq = red_ok(Lq)               # 闸门⑦: Q 的幅度红线
                if not okq:
                    red_locate(Lq, f"Q bs={bs} 配置{label} gc={gc} {dt} {impl}")
                Mq = metrics(PL, Lq, ids_row)
                # 闸门⑧(乙-2d): 非基准实现档与【同格 cat 档】的 y 互差。归约序不同
                #   => 不判 0, 判量级: 阈值按 dtype 给(fp32 1e-4 / 半精 5e-2), 超了就是
                #   "少读的字节其实是别人该参与的那一笔", 不是浮点重排。
                ycpu = y.float().cpu().clone()
                base_y = YBIT.get((gc, dt, IMPLS[0]))
                dy_max = dy_rel = float("nan")
                if base_y is not None:
                    dy_max = (ycpu - base_y).abs().max().item()
                    dy_rel = (ycpu - base_y).norm().item() / max(base_y.norm().item(), 1e-30)
                YBIT[(gc, dt, impl)] = ycpu
                pk_q_delta = max(pk_q_delta,
                                 (torch.cuda.max_memory_allocated() / 2 ** 20) - pk - wbase)
                del outs, refi, y, m, ts, ti
                ROW[(gc, dt, impl)] = dict(p=p, sd=100 * statistics.pstdev(rs) / p, pk=pk,
                                     st=st, ncpu=ncpu, ngpu=bs * K - ncpu, rel=rel, idx=idx_d,
                                     mode=worst, djud=djud, nbad=nbad, red=0 if okq else 1,
                                     amax=aq, dym=dy_max, dyr=dy_rel, **Mq)
        # vs纯CPU / KL 比的分母 = 同一实现档的 gc 最小格(基准实现 = IMPLS[0] = cat)
        cpu_p = {(dt, im): ROW[(GC_LIST[0], dt, im)]["p"] for (dt, im) in CELL
                 if (GC_LIST[0], dt, im) in ROW}
        cpu_kl = {(dt, im): ROW[(GC_LIST[0], dt, im)]["kl"] for (dt, im) in CELL
                  if (GC_LIST[0], dt, im) in ROW}
        hits_bs = torch.bincount(ti0.flatten(), minlength=E).tolist()
        for gc in GC_LIST:
            for (dt, impl) in CELL:
                r = ROW[(gc, dt, impl)]
                print(f"{gc:>3}{dt:>7}" + (f"{impl:>9}" if icol else "")
                      + f"{r['pk']:>10.1f}{r['st']:>10.1f}{r['p']:>9.3f}{r['sd']:>5.0f}"
                      f"{100*(r['p']/cpu_p[(dt, impl)]-1):>+9.1f}%"
                      f"{r['ngpu']:>11d}{r['ncpu']:>11d}{(E-gc)*eb/2**20:>9.1f}"
                      f"{r['rel']:>9.4f}%{r['top1']:>10.2f}{r['kl']:>12.3e}{r['tv']:>11.3e}"
                      f"{r['rank']:>9d}{r['idx']:>7d}{r['mode']:>12.9f}"
                      + (f" y互差={r['dym']:.2g}" if icol and r["dym"] == r["dym"] else "")
                      + (f" dmax={r['dmax']:.3g} kl95={r['kl95']:.3g}" if DIAG else "")
                      + (f"  🚨inf/nan={r['nbad']}" if r["nbad"] else ""))
        print(f"  闸门② idx差 max={max(r['idx'] for r in ROW.values())} 闸门③ 三模式互差 max="
              f"{max(r['mode'] for r in ROW.values()):.9f} 闸门④ 裁判自检 max={max(r['djud'] for r in ROW.values()):.3g}"
              f" 闸门⑤ 空载告警见上(没有=全绿) 峰值列之后额外高水位={pk_q_delta:+.1f}MB"
              f"(= fp32 裁判自己的 CPU 腿权重 GPU 副本 _cpu_w_dev_cache ≈ (E-gc)x"
              f"{eb*4/2**20:.1f}MB, 是裁判的成本不是引擎的; 峰值/稳态两列都在它之前读的)")
        # 闸门⑧(乙-2d): 非基准实现档 => 归约序噪声底。🚨它【不是】精度闸门: 精度由闸门⓪
        #   (只覆盖 cat 行)与 P0-③ 归档判, 这一道只判"少读的字节有没有改变结果的字面值"。
        if len(IMPLS) > 1:
            thr8 = {dt: (1e-4 if dt == "fp32" else 5e-2) for dt in DTYPES}
            over = [(g, d, im, r["dyr"]) for (g, d, im), r in ROW.items()
                    if im != IMPLS[0] and r["dyr"] == r["dyr"] and r["dyr"] > thr8[d]]
            nn8 = sum(1 for k in ROW if k[2] != IMPLS[0])
            mx8 = max((r["dyr"] for r in ROW.values() if r["dyr"] == r["dyr"]), default=0.0)
            print(f"  闸门⑧ 归约序互差(非 {IMPLS[0]} 档 vs 同格 {IMPLS[0]} 档, 覆盖 {nn8} 格): "
                  f"最大相对互差 {mx8:.3g} | 阈值 fp32 1e-4 / 半精 5e-2"
                  + (f" 🚨超线: {['%s/%s/%s=%.2g' % o for o in over]}" if over else
                     " => 过(这一档只是换了加法顺序)"))

        # ---- 闸门⑦ 幅度红线汇总(不重算, 只判; 脏档必须写进表下面那行)
        dirtyq = [(f"gc={g}{d}/{im}" if len(IMPLS) > 1 else f"gc={g}{d}")
                  for (g, d, im), r in ROW.items() if r["red"]]
        print(f"  闸门⑦ 幅度红线 |logit|<={LOGIT_RED:g}: 通过 {len(ROW) - len(dirtyq)}/{len(ROW)} 档"
              f" |P|absmax={aP:.3g} |未挂钩|absmax={aO:.3g} 最大|Q|absmax="
              f"{max(r['amax'] for r in ROW.values()):.3g}"
              + (f" 🚨越线档: {' '.join(dirtyq)}" if dirtyq else "")
              + (" | 🚨P 越线 => 本块 KL/top-1 全部作废(本表不重算, 复跑)" if P_DIRTY else ""))

        # ---- 闸门⓪ 跨 bench 对账
        hit_n = mis = 0
        worst_rel = 0.0
        for (gc, dt, im), r in ROW.items():
            if im != IMPLS[0]:      # 🚨 锚点是 P0-④a 那次(cat 档)归档的 KL, 换实现不覆盖
                continue
            a = ANCHORS.get((label, bs, gc, dt))
            if a is None:
                continue
            hit_n += 1
            rel_d = abs(r["kl"] - a) / max(a, 1e-30)
            worst_rel = max(worst_rel, rel_d)
            if rel_d > 2e-3:
                mis += 1
                print(f"  🚨闸门⓪ 与 P0-③ 归档不符: gc={gc} {dt} 本表 KL={r['kl']:.4e} "
                      f"归档={a:.4e} 相对偏差={rel_d:.3g}")
        n_anch = sum(1 for k in ROW if k[2] == IMPLS[0])
        if hit_n:
            print(f"  闸门⓪ 跨bench对账(vs P0-③ 归档 KL"
                  + (f", 只判 {IMPLS[0]} 档" if len(IMPLS) > 1 else "") + "): "
                  f"覆盖 {hit_n}/{n_anch} 格, "
                  f"不符 {mis} 格, 最大相对偏差 {worst_rel:.2g} (阈值 2e-3 = 归档只有 4 位有效)")
        else:
            else_ = len(ROW) - n_anch
            print("  🚨闸门⓪ 这一格没有任何锚点(BS/GC 超出 P0-③ 归档范围) => 数值路径未获对账"
                  + (f"; 另有 {else_} 格非 {IMPLS[0]} 档本来就不在锚点内(闸门⑧ 判它们)"
                     if else_ else ""))

        # ---- 曲线摘要 + 放置规则 + 三方帕累托
        for dt in DTYPES:
          for impl in IMPLS:
            tag = dt if len(IMPLS) == 1 else f"{dt}/{impl}"
            sub = {gc: ROW[(gc, dt, impl)] for gc in GC_LIST}
            best = min(sub, key=lambda g: sub[g]["p"])
            kb = min(sub, key=lambda g: sub[g]["kl"])
            gc_r0 = H.plan_gpu_capacity(hits_bs, eb, eb * ELS[dt], gpu_expert_count=2)[0]
            gc_r = gc_r0 if gc_r0 in sub else min(GC_LIST, key=lambda g: (abs(g - gc_r0), g))
            print(f"  [{tag}] 实测最快 gc={best}({sub[best]['p']:.3f}ms, 峰值{sub[best]['pk']:.0f}"
                  f"/稳态{sub[best]['st']:.0f}MB, KL={sub[best]['kl']:.3e}) | "
                  f"最省 KL gc={kb}(稳态{sub[kb]['st']:.0f}MB, KL={sub[kb]['kl']:.3e}) | "
                  f"规则选 gc={gc_r}{'(网格内取最近档, 规则真值 gc=' + str(gc_r0) + ')' if gc_r != gc_r0 else ''}"
                  f"({sub[gc_r]['p']:.3f}ms, 峰值{sub[gc_r]['pk']:.0f}/稳态{sub[gc_r]['st']:.0f}MB,"
                  f" 比最优贵 {100*(sub[gc_r]['p']/sub[best]['p']-1):+.1f}%)"
                  + ("  🚨规则输入退化: bs=1 只有 2 个 slot" if bs * K <= 2 * E - 2 else ""))
            g_lo, g_hi = GC_LIST[0], GC_LIST[-1]      # 🚨 别写死 sub[0]: GC 不含 0 时会 KeyError
            print(f"        端点: gc={g_lo} {sub[g_lo]['p']:.3f}ms/峰值{sub[g_lo]['pk']:.0f}"
                  f"/稳态{sub[g_lo]['st']:.0f}MB"
                  f"/KL={sub[g_lo]['kl']:.3e} -> gc={g_hi} {sub[g_hi]['p']:.3f}ms"
                  f"/峰值{sub[GC_LIST[-1]]['pk']:.0f}/稳态{sub[GC_LIST[-1]]['st']:.0f}MB"
                  f"/KL={sub[GC_LIST[-1]]['kl']:.3e}")
          if RULE == "v2":
              # 规则v2: 在【本 bs/dtype 的整张 (gc x 实现档) 网格】里直选, 不回手补疤痕。
              # 对照对象 = 全域最优(跨实现档), 与 bench_rule_v2.py 的 ② 同口径。
              cells = {(g, im): ROW[(g, dt, im)] for g in GC_LIST for im in IMPLS}
              bo = min(cells, key=lambda k: cells[k]["p"])
              gv, iv, ugv, _ = H.plan_gpu_capacity_v2(
                  hits_bs, eb, eb * ELS[dt], grid=GC_LIST, impls=tuple(IMPLS),
                  cpu_mac_rate=(H.CPU_MAC_RATE if MAC_WALL else None))
              kv = (gv, iv) if (gv, iv) in cells else None
              if kv is None:                      # 理论上不会(网格直选必在网格内)
                  print(f"        🚨规则v2 落格 gc={gv}/{iv} 不在本表网格 => 不判")
              else:
                  rv, rb = cells[kv], cells[bo]
                  print(f"        规则v2选 gc={gv}/{iv}(网格直选, 算力墙"
                        f"{'=on 288GMAC/s' if MAC_WALL else '=off'},"
                        f" 预测{ugv:.0f}us): {rv['p']:.3f}ms,"
                        f" 峰值{rv['pk']:.0f}/稳态{rv['st']:.0f}MB, KL={rv['kl']:.3e},"
                        f" 比全域最优(gc={bo[0]}/{bo[1]} {rb['p']:.3f}ms)"
                        f"贵 {100*(rv['p']/rb['p']-1):+.1f}%")
        pts = [(gc, dt, im) for gc in GC_LIST for (dt, im) in CELL]
        lab8 = lambda a: (f"gc={a[0]}{a[1]}" if len(IMPLS) == 1
                          else f"gc={a[0]}{a[1]}/{a[2]}")
        le = lambda a, b, e: a <= b * (1 + e) + 1e-12
        front, dom = [], []
        for a in pts:
            ra = ROW[a]
            all_b = False
            for b in pts:
                if b == a:
                    continue
                rb = ROW[b]
                not_worse = (le(rb["p"], ra["p"], NOISE_T) and le(rb["st"], ra["st"], NOISE_T)
                             and le(rb["kl"], ra["kl"], 0.05))
                strictly = (rb["p"] < ra["p"] * (1 - NOISE_T) or rb["st"] < ra["st"] * (1 - NOISE_T)
                            or rb["kl"] < ra["kl"] * 0.95)
                if not_worse and strictly:
                    all_b = True
                    break
            (dom if all_b else front).append(a)
        print(f"  三方帕累托前沿(显存轴=稳态MB↓/延迟↓/KL↓ 三者都不被支配, 平手带={NOISE_T:.0%}): "
              + " ".join(lab8(a) for a in front))
        if dom:
            print(f"  被支配(白付钱): " + " ".join(lab8(a) for a in dom))

        # ---- 配对段: 同一 gc 的两个 dtype 同时常驻, 逐轮配对 + 单双号换序
        # 🚨 这一刀只量【降精】, 所以实现档固定在基准档 cat: 混进 traffic 就把两刀
        #    的效应搅在一根轴上了(P0-④a 那 6 行台账全部是 cat 档, 对账也才对得上)。
        base_dt, BASE_IM = DTYPES[0], IMPLS[0]
        others = [d for d in DTYPES[1:]]
        if others:
            print(f"\n  配对段(降精这一刀): {base_dt} vs 其余 | 同 gc 两档同时常驻(🚨CPU 侧 int8 "
                  f"专家集相同 => 轮转公平; 但两份 int8 => 比独占段更挤 L3, 绝对值会一起抬高)")
            print(f"{'gc':>3}{'配对':>12}{f'p50_{base_dt}':>11}{'p50_other':>11}{'配对Δ%':>10}"
                  f"{'逐轮Δ中位%':>12}{'更快轮':>9}{'踩L3':>16}{'稳态省MB':>10}"
                  f"{'KL比(other/主)':>16}{'闸门⑥共驻互差':>16}{'配对可用':>12}")
            for gc in GC_LIST:
                for dt in others:
                    wbase = win()
                    ma = mk(gc, order, "swiglu", base_dt, BASE_IM)
                    mb = mk(gc, order, "swiglu", dt, BASE_IM)
                    warm(ma, x, 20); warm(mb, x, 20)
                    ra, rb = [], []
                    for rr in range(ROUNDS):
                        if rr % 2 == 0:
                            ra.append(one_round(ma, x)); rb.append(one_round(mb, x))
                        else:
                            rb.append(one_round(mb, x)); ra.append(one_round(ma, x))
                    d_co = max((fwd(ma, x).float().cpu()
                               - YBIT[(gc, base_dt, BASE_IM)]).abs().max().item(),
                               (fwd(mb, x).float().cpu()
                               - YBIT[(gc, dt, BASE_IM)]).abs().max().item())
                    del ma, mb
                    win()
                    pa, pb = statistics.median(ra), statistics.median(rb)
                    pd_ = statistics.median([100 * (rb[i] / ra[i] - 1) for i in range(ROUNDS)])
                    faster = sum(1 for i in range(ROUNDS) if rb[i] < ra[i])
                    dirt = (100 * (pa / ROW[(gc, base_dt, BASE_IM)]["p"] - 1),
                            100 * (pb / ROW[(gc, dt, BASE_IM)]["p"] - 1))
                    kla, klb = ROW[(gc, base_dt, BASE_IM)]["kl"], ROW[(gc, dt, BASE_IM)]["kl"]
                    # 配对段可用判据: 两腿共驻都没把自己踩脏(±10%)且逐轮方向一致
                    usable = ("✓" if abs(dirt[0]) <= 10 and abs(dirt[1]) <= 10
                              and faster in (0, ROUNDS) else "✗共驻污染")
                    print(f"{gc:>3}{base_dt+'→'+dt:>12}{pa:>11.3f}{pb:>11.3f}"
                          f"{100*(pb/pa-1):>+9.1f}%{pd_:>+11.1f}%{faster:>6d}/{ROUNDS}"
                          f"{f'{dirt[0]:+.0f}/{dirt[1]:+.0f}%':>16}"
                          f"{ROW[(gc, base_dt, BASE_IM)]['st']-ROW[(gc, dt, BASE_IM)]['st']:>10.1f}"
                          f"{klb/kla if kla>0 else float('nan'):>16.3g}{d_co:>16.9f}"
                          f"{(usable[0] if usable else '?'):>12}")
                    if ROW[(gc, dt, BASE_IM)]["ngpu"] == 0:
                        # 🚨方法论 25: 数值闸门(闸门⑤/P0-③)判的是【输出逐位相同】= 降精在数值上
                        # 没有入口, 这句话【不能外推到计时列】—— 稠密写法照样把 gc 个专家全算一遍,
                        # 0 流量仍有关键路径成本。所以这一行读作【稠密税被降精打掉多少】(乙-2d 靶子),
                        # 不是噪声。实证: 本轮 bs=1 C gc=5 配对 -16.1%(7/7) 与归档 乙-2c 端到端
                        # -16.3%(配置C bs=1, GPU 半边 0.472->0.288ms) 是同一件事。
                        print(f"       ↑ gc={gc} 这一格 GPU 腿【0 个 slot】= 零流量: 数值无入口"
                              f"(KL 与 dtype 逐位无关), 但稠密写法照付时间 => 这一行是【为 0 流量"
                              f"付的钱】被降精打掉的部分(乙-2d 靶子), 不是噪声(方法论 25)")
                    elif usable is not None and not usable[0].startswith("✓"):
                        print(f"       ↑ gc={gc} 这一行【不可用】: 共驻段把两腿一起踩脏/换序不利 "
                              f"(判据=两腿踩L3 均在±10% 且 更快轮 {ROUNDS}/{ROUNDS} 或 0/{ROUNDS}); "
                              f"Δ 只能当噪声底读, 别进台账")
        # ---- 实现段(乙-2d): 同 gc 同 dtype, cat(稠密) vs traffic(按真流量) 共驻轮转配对
        # 这一刀和降精那一刀的本质区别: 两档读的是【同一组 GPU 权重】(同一份 _wcat_gu +
        # _wstack_d), 所以显存列【应该相同】(只差 _tH/_tPART 几 KB) => 它在三方交换图上是
        # "同一根显存竖线上把时间轴往左推", 不是"用显存换时间"。
        # 判据与 dtype 配对段同一条: 两腿踩L3 均 ±10% 且 更快轮 7/7 或 0/7 => 才允许进台账。
        if len(IMPLS) > 1:
            print(f"\n  实现段(乙-2d 少读字节这一刀): {BASE_IM} vs 其余 | 同 gc 同 dtype 两档同时"
                  f"常驻(🚨两档共用同一组 GPU 权重 => 显存列本该相同; CPU 腿 int8 专家集相同"
                  f" => 轮转公平)")
            print(f"{'gc':>3}{'dtype':>7}{'实现配对':>18}{'预期字节比':>11}{'p50_cat':>10}"
                  f"{'p50_other':>11}{'配对Δ%':>10}{'逐轮Δ中位%':>12}{'更快轮':>9}{'踩L3':>16}"
                  f"{'稳态差MB':>10}{'KL比(other/主)':>16}{'闸门⑥共驻互差':>16}{'配对可用':>12}")
            for gc in GC_LIST:
                for dt in DTYPES:
                    for impl in IMPLS[1:]:
                        ns = int((ti0 < gc).sum())                 # 本帧真落 GPU 的 slot 数
                        wbase = win()
                        ma = mk(gc, order, "swiglu", dt, BASE_IM)
                        mb = mk(gc, order, "swiglu", dt, impl)
                        warm(ma, x, 20); warm(mb, x, 20)
                        ra, rb = [], []
                        for rr in range(ROUNDS):
                            if rr % 2 == 0:
                                ra.append(one_round(ma, x)); rb.append(one_round(mb, x))
                            else:
                                rb.append(one_round(mb, x)); ra.append(one_round(ma, x))
                        d_co = max((fwd(ma, x).float().cpu()
                                   - YBIT[(gc, dt, BASE_IM)]).abs().max().item(),
                                   (fwd(mb, x).float().cpu()
                                   - YBIT[(gc, dt, impl)]).abs().max().item())
                        del ma, mb
                        win()
                        pa, pb = statistics.median(ra), statistics.median(rb)
                        pd_ = statistics.median([100 * (rb[i] / ra[i] - 1)
                                                 for i in range(ROUNDS)])
                        faster = sum(1 for i in range(ROUNDS) if rb[i] < ra[i])
                        dirt = (100 * (pa / ROW[(gc, dt, BASE_IM)]["p"] - 1),
                                100 * (pb / ROW[(gc, dt, impl)]["p"] - 1))
                        kla, klb = (ROW[(gc, dt, BASE_IM)]["kl"], ROW[(gc, dt, impl)]["kl"])
                        usable = ("✓" if abs(dirt[0]) <= 10 and abs(dirt[1]) <= 10
                                  and faster in (0, ROUNDS) else "✗共驻污染")
                        print(f"{gc:>3}{dt:>7}{BASE_IM+'→'+impl:>18}{ns/gc if gc else 0:>11.2f}"
                              f"{pa:>10.3f}{pb:>11.3f}{100*(pb/pa-1):>+9.1f}%{pd_:>+11.1f}%"
                              f"{faster:>6d}/{ROUNDS}{f'{dirt[0]:+.0f}/{dirt[1]:+.0f}%':>16}"
                              f"{ROW[(gc, dt, impl)]['st']-ROW[(gc, dt, BASE_IM)]['st']:>10.1f}"
                              f"{klb/kla if kla > 0 else float('nan'):>16.3g}{d_co:>16.9f}"
                              f"{usable[0]:>12}")
                        if gc == 0:
                            print(f"       ↑ gc=0 两档都是【纯 CPU 基线】(GPU 腿整个不存在) => "
                                  f"这一行的 Δ 只能是噪声底, 它是【配对段的空载对照】")
                        elif ROW[(gc, dt, BASE_IM)]["ngpu"] == 0:
                            print(f"       ↑ gc={gc} 这一格本帧【没有一个 slot 落 GPU】(放错) => "
                                  f"traffic 一个字节都不读、稠密照读 gc 份: 预期字节比 0.00, "
                                  f"这一行是【稠密税】的直接读数")
                        elif ns > gc:
                            print(f"       ↑ gc={gc} 本帧 GPU slot={ns} > gc={gc} => 同一张权重被"
                                  f"多个 slot 各读一遍, traffic 预期【变差】(算式: 省字节 ⇔ "
                                  f"bs*top_k < gc)。这一行是【适用边界】的证据, 不是失败")
                        elif usable[0] != "✓":
                            print(f"       ↑ gc={gc} 这一行【不可用】: 共驻段把两腿一起踩脏/换序"
                                  f"不利 (判据同 dtype 配对段); Δ 只能当噪声底读, 别进台账")
        YBIT.clear(); ROW.clear()
        del PL, ORIG, M0, ref0
        torch.cuda.empty_cache()
        print(f"  | 本格 {time.perf_counter()-t_blk:.0f}s | loadavg="
              f"{open('/proc/loadavg').read().split()[0]}")

print("""
判读提示: ① 这一张就是 P0-④ 的三方交换图数据源: 同一行里"显存 / p50 / KL"三根轴【同格同
          权重同裁判】, 不存在再跨表借数。画法是 x=稳态MB, y=p50, 颜色=KL 等值线, 标帕累托前沿。
       ② 曲线不是配对: 逐点独占 => 相邻 gc 之间 <噪声底(A 2~3%/C 8~18%)的差不要读;
          唯一配对的是 dtype 那一刀(同 gc 两档同时常驻 + 单双号换序 + 更快轮数), 且它还要过
          【配对可用】这一列(两腿踩L3 均±10% 且 更快轮 7/7 或 0/7)才允许进台账 —— 共驻段本身
          会把两腿一起踩脏, ✗ 的行的 Δ 是噪声底不是效应量。
       ③ KL/top-1 是【相对量】: 参照行(未挂钩真模型→P)已经把"换成 MoE 本身"的代价摆在那,
          它比引擎最狠那档还贵数百倍(P0-③ 判决⑤) => 只报 gc 之间/dtype 之间的比, 别报绝对 ppl。
       ④ 期望形状: gc↑ => 显存↑、KL↓; 延迟是 U 形(bs 小的时候 GPU 半边纯付权重的钱,
          bs 大的时候 CPU 腿的 int8 字节才是瓶颈)。若 bs=1 的最优点落在 gc=0/1,
          那不是失败, 是"decode 下必须给 GPU 半边定价"(P0-② 判读③/T13 同源)。
       ⑤ 显存两列的分工: 峰值列跟 P0-② 对齐(它含建图/暖机的 fp32 staging, 所以 bf16 那一档
          的峰值【不降】是应该的, 不是 bug); 稳态列才是"这颗模型常驻要占多少", 降精/去死权重
          的收益只在这列看得见。谁引用显存, 就说清是哪一列。
       ⑥ 规则选 gc 那一列: fp32 与 bf16 各算一次(gpu_weight_bytes 跟 dtype 走)。bf16 把 GPU
          半边打折之后规则【敢不敢】给更大 gc —— 这是放置规则 v2 缺的那半个价, 也是 P1 的入口。
       ⑦ 闸门⓪ 是本表的命: 对的是【另一份代码】(bench_metrics)归档在 results/ 的 60 个 KL。
          它不过 => 计时/显存窗口污染了数值路径, 或者快照/语料漂了, 整张表作废。
       ⑧ 只在 IMPLS 带第二个实现档时才有的口径(乙-2d):
          · 闸门⓪ 只覆盖 cat 档(锚点就是 cat 归档), 非 cat 档由闸门⑧ 判"归约序噪声底"
            (fp32 1e-4 / 半精 5e-2) —— 它【不是】精度闸门, 别拿它当"精度没问题"的证据;
            非 cat 档的 KL/top-1 仍然算, 但要和同格 cat 档【成对引用】。
          · bf16 下 traffic 常常比 cat 更接近 fp32 参考(g/u 在 fp32 里累加、只在存 h 时
            截半精, 而稠密先截半精再算 silu) => 这条必须如实写, 不许读成"少读字节还变准了"。
          · 实现段那一列【预期字节比】= 本帧 GPU slot / gc, 它是这一刀的【理论上限】:
            实测 Δ 比它更接近 0 => 还有固定成本没打掉; 实测 Δ 比它还负 => 命中比假设更集中。
          · 适用边界是算式不是感觉: 省字节 ⇔ bs*top_k < gc。bs 一大 traffic 必输,
            表里那些"预期变差"的行【要照抄进台账】—— 它们是 v2 放置规则的分段依据。""")

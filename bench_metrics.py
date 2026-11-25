#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-③ 判决表: 把"质量"从 relRMSE 换成【模型选出的词变了多少】。

【这一刀为什么要打】
  台账里 45 行全是延迟/显存, 精度那一列从头到尾只有一个数: relRMSE(整条 L2 输出的
  相对均方根)。它回答的是"这一层的向量差了多少", 不回答读者真正会问的那句
  "那我拿它生成的字变不变"。P0-④ 要把这条线写进博客/论文, 审稿人第一个问题必然是
  top-1 一致率 / KL / ppl —— 异构 int8 路线的全部风险都在"省下来的显存有没有换成错词"。
  OPTLOG §5.4 里 P0-③ 的原话就是"relRMSE 不是任何论文的最终指标"。

【指标怎么才算"端到端"】
  ❌ 被否决的写法: 把 MoE 层的输出直接过 lm_head 投到词表。那是 residual delta
     (第 8 层 MLP 加在残差流上的那一小截), 它自己不是任何 logits, 投出来的分布无语义。
  ✅ 本表的写法: 在真 SmolLM-135M 的第 CAP_LAYER 层 MLP 上挂 forward hook, 用
     【引擎算出的那一层输出】覆盖掉原输出, 然后让剩下的 21 层 + final norm + lm_head
     照常跑完 => 拿到真 logits (T,V)。P(裁判)与 Q(引擎)共用同一批 ids、同一段剩余网络,
     所以 KL 里只剩"int8 CPU 腿 + dtype/放置 GPU 腿"这一项扰动, 没有别的混淆项。
  P = 同架构 fp32 稠密(m.fp32_reference: 读引擎里那份 fp32 存储权重, 不量化, 同一批路由);
  Q = 引擎本体(L2_零CUDA+pinned, CPU 腿 int8 + GPU 腿 fp32/bf16)。

【闸门, 一道花了整张表就作废】
  闸门① splice 位置: hook 在覆盖前录到的 mlp 输入, 必须与 capture() 拿到的 x 逐位相同
        —— 它证明"我替换的就是我量过的那一层、那 T 行", 而不是错位后的别的层。
  闸门② 路由: 同一格内各 gc/dtype 的 top-k indices 与基准 fp32 档【逐位】相同(idx差=0)
        —— 不为 0 就是 KL 里混进了路由漂移(P 和 Q 选了不同的专家, 量的就不是量化了)。
        fused 前段 vs aten 前段的逐位等价由红线 T11 段负责(bs=1/7/32/128, idx/x_q 逐位)。
  闸门③ 三模式互差: 引擎 sync/L0/L1/L2 四条路径输出逐位 0。
  闸门④ 跨轮互差: 同一格重算 ROUNDS 次, KL 必须逐位相同 —— 证明这些指标是【确定量】。
  闸门⑤ dtype 无流量自检: GPU 腿 0 个 slot 的档(纯 CPU 的 gc=0, 以及 bs=1 放对时 CPU 腿
        0 slot 的那几档) fp32 与 bf16 必须【逐位相同】。不相同 = dtype 漏到了不该漏的地方。
        🚨这一条只判【数值】: "数值没有入口"不能外推到计时列 —— 稠密写法照样把 0 流量的专家
        全算一遍, 时间照付(方法论 25; 计时侧的读法见 bench_swap 配对段那一行提示)。
  闸门⑥ 数值红线 + 逐位可复现(2026-11-21 bs=512 那轮"裁判被踩"之后加的, 纯观测):
        (a) 幅度红线 |logit| <= LOGIT_RED(默认 1e4; 真 logits absmax ~O(30), 而 bs=512 出现过的
            脏值是 6.59e+19 / 3.29e+154 —— 差 15 个数量级, 红线不会误伤);
        (b) 未挂钩真模型【同输入同代码】重算必须逐位 0;
        (c) y 重算逐位 0 时, 同一份 y 反射到 logits 也必须逐位 0。🚨旧文案拿"P 走 index_add_
            逐位 0 不保证"当借口是错的: y 都逐位相同了, 21 层前向就没有不逐位相同的理由;
        (d) 每格 Q 的 logits 同样过红线。
        任一不过 => 自动重算(P 最多 P_RETRY 次, Q 最多 Q_RETRY 次)并印"第几次才干净";
        重算仍脏 => P 脏则【整块作废】(P 是本块所有格共用的分母), Q 脏则【该行作废】。
  闸门⑦ 跨块互差: gc=E(全 GPU, CPU 腿 0 slot)时放置序【不该】改变数值 => 同一 bs 的 A/C 两块
        那份 Q logits 必须逐位相同(逐元素比, 不是 KL; KL 是平均量, 一个 1e19 的脏元素会被 512
        行摊薄成看不出来)。容差 CROSS_TOL 只留给 index_add_ 归约序的 ~1e-7。同时印 crc32,
        这样"同一格两次跑出来的 logits"可以跨进程对账。
  参照行: 未挂钩的真模型 vs P。那是"把真第8层 MLP 换成 8 选 2 的 fp32 MoE"本身的代价,
        不是这一刀的代价; 没有它, 表里 top1=9x% 会被误读成 int8 的锅。

【口径声明(和计时表不一样, 别看错)】
  · 这些指标是确定量: 没有 p50、没有 σ%轮、没有 min-of-N 的概念。
  · ROUNDS 只为闸门④存在(跨轮逐位重算), 表里印的是【跨轮互差】不是噪声。
  · ppl 在【本仓库 readme 拼贴的英文散文】上算 —— 域外语料, 绝对值没有意义,
    只有 ΔNLL / ppl比 可比。要真 held-out ppl: 丢一个 txt 给 CORPUS= 就能接上
    (OPTLOG §5.1.5: dl_lab 没 pyarrow, 955MB TinyStories parquet 读不出来)。
  · gc 之间 KL 的差若要当结论, 必须显著大于闸门④的跨轮互差(正常情况下大 4~6 个数量级)。

【为什么只有 swiglu 档】
  gemm 档专家输出宽 = expert_size(1536), 拼不回 576 维残差流 => "下一个词的分布"
  这个指标在 gemm 档语义上不成立。所以本表钉死 expert_kind="swiglu"(输出宽 = hidden)。

env: LAYERS=0,4,8,12,16,20,24,28  GATE=embed  CAP_LAYER=8  HIT_BS=512  N_CHARS=24000
     BS=1,32  GC=0,3,5,8  DTYPES=fp32,bf16  EKIND=swiglu  ROUNDS=3  CORPUS=(空)
"""
import os, sys, time, math, zlib, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R

LAYERS = tuple(int(v) for v in os.environ.get("LAYERS", "0,4,8,12,16,20,24,28").split(","))
GATE = os.environ.get("GATE", "embed")
CAP_LAYER = int(os.environ.get("CAP_LAYER", 8))
HIT_BS = int(os.environ.get("HIT_BS", 512))
N_CHARS = int(os.environ.get("N_CHARS", 24000))
BS_LIST = [int(v) for v in os.environ.get("BS", "1,32").split(",")]
GC_LIST = [int(v) for v in os.environ.get("GC", "0,3,5,8").split(",")]
DTYPES = [d for d in os.environ.get("DTYPES", "fp32,bf16").split(",") if d]
KINDS = [k for k in os.environ.get("EKIND", "swiglu").split(",") if k]
ROUNDS = int(os.environ.get("ROUNDS", 3))
CORPUS = os.environ.get("CORPUS", "").strip()
LOGIT_RED = float(os.environ.get("LOGIT_RED", 1e4))    # 闸门⑥(a): 真 logits absmax ~O(30), 脏值 ~1e19
Y_RED = float(os.environ.get("Y_RED", 1e3))              # 闸门⑥b: 引擎出口 y 的红线(真值 absmax ~O(5))
STABLE_RATIO = float(os.environ.get("STABLE_RATIO", 8))   # 闸门⑥: 越红线但重算逐位相同 且 absmax <= 该倍率x|P| => 判红线紧而非脏
P_RETRY = int(os.environ.get("P_RETRY", 3))            # 闸门⑥: 裁判脏 -> 重算几次
Q_RETRY = int(os.environ.get("Q_RETRY", 1))            # 闸门⑥(d): 某格 Q 脏 -> 重算几次
CROSS_TOL = float(os.environ.get("CROSS_TOL", 1e-5))   # 闸门⑦: 只容纳归约序噪声(实测 ~1e-7)
SEED = 42
K = 2
E = len(LAYERS)
MODE = "L2_零CUDA+pinned"
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
assert set(KINDS) == {"swiglu"}, f"P0-③ 只能在 swiglu 档做(见 docstring 最后一段): {KINDS}"
assert set(DTYPES) <= {"fp32", "fp16", "bf16"}, DTYPES
assert 0 <= CAP_LAYER < 30

t_load = time.perf_counter()
print(f"[机器状态] {H.machine_state()}")
dims = R.dims()
D, N, NL = dims["D"], dims["N"], dims["layers"]
CORPUS_TEXT = (open(CORPUS, encoding="utf-8", errors="ignore").read()[:N_CHARS]
               if CORPUS else R.real_text(N_CHARS))

# ---------------------------------------------------------------- 真模型(只 load 一次)
def _load():
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"          # 🚨 这台机器 DNS->127.0.0.1, 强制离线
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snap = R.snapshot()
        tok = AutoTokenizer.from_pretrained(snap)
        mdl = AutoModelForCausalLM.from_pretrained(snap)
        # 🚨 必须显式 .to(float32): SmolLM 的 config 里 torch_dtype=bfloat16,
        #    from_pretrained 会照着建 bf16 权重 —— 裁判(P)要 fp32, 而且 hook 塞回去的
        #    y 是 fp32, 不转就在那一层之后的 rmsnorm 上报 dtype 不同。
        mdl.eval().to(torch.float32)
        for p in mdl.parameters():
            p.requires_grad_(False)
        return mdl, tok
    finally:
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)


MODEL, TOK = _load()
IDS_FULL = TOK(CORPUS_TEXT, return_tensors="pt").input_ids
V = MODEL.get_input_embeddings().weight.shape[0]
torch.manual_seed(SEED)


def capture_from(ids):
    """在真模型第 CAP_LAYER 层 MLP 入口抓 x(与 real_smollm.capture 同一挂法)。"""
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


# 默认走 P0-① 归档过的同一条路径(R.capture); 给了 CORPUS 才换成本地抓法
if CORPUS:
    XREAL = capture_from(IDS_FULL[:, :HIT_BS])
else:
    XREAL, _d = R.capture(layer=CAP_LAYER, bs=HIT_BS, n_chars=N_CHARS)
assert XREAL.shape[1] == D, (XREAL.shape, D)
assert IDS_FULL.shape[1] >= HIT_BS, IDS_FULL.shape
print(f"[真源] SmolLM-135M: D={D} N={N} 层数={NL} 词表={V} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家"
      f" top_k={K} | 路由={GATE} | 模型加载 {time.perf_counter()-t_load:.1f}s")
print(f"[语料] {'CORPUS=' + CORPUS if CORPUS else '本仓库 readme 拼贴(域外, 只有 Δ 可比)'}"
      f" {len(CORPUS_TEXT)} 字符 -> {IDS_FULL.shape[1]} token | x=第 {CAP_LAYER} 层 MLP 入口 {tuple(XREAL.shape)}")
print(f"[被测] 质量指标 = 下一个词的分布: top-1一致率 / top-5重合率 / KL(P‖Q) / TV / ΔNLL / ppl比"
      f" | bs={BS_LIST} | gc={GC_LIST} | dtype={DTYPES} | ROUNDS={ROUNDS}")
print(f"[口径] 🚨 这些是【确定量】不是计时量: 没有 p50/σ%轮/轮转配对的概念, ROUNDS 只证闸门④(跨轮逐位重算)。")
print(f"[口径] P = 同架构 fp32 稠密裁判(m.fp32_reference, 读引擎里那份 fp32 存储权重, 不量化, 同一路由);"
      f" Q = 引擎({MODE}, CPU 腿 int8 + GPU 腿 dtype), 覆盖第 {CAP_LAYER} 层 MLP 出口后跑完剩余 {NL-CAP_LAYER-1} 层。")


# ---------------------------------------------------------------- 引擎与拼接
def perm_src(order):
    """按 order 重排专家: order[新 id] = 旧 id; gate_w 的行必须跟着走(与 P0-① 同写法)。"""
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


def mk(gc, order, kind, dt):
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind=kind,
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)   # 建缓存, 不进测量
    return m


def splice(ids_row, y_cpu, box=None):
    """把第 CAP_LAYER 层 MLP 的出口换成 y_cpu(T,D), 跑完剩余层 => 真 logits (T,V) float64。"""
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
    """未挂钩的真模型(参照行用)。"""
    with torch.no_grad():
        return MODEL(ids_row).logits[0].double()


# ------------------------------------------- 闸门⑥/⑦ 的数值探针: 纯观测, 不碰数值路径
def red_ok(L, thr=None):
    """幅度红线。只看 max/min —— 不造 abs 临时(bs=512 时 V=49152, 一个 fp64 临时就是 200MB)。"""
    thr = LOGIT_RED if thr is None else thr
    mx, mn = float(L.max()), float(L.min())
    a = max(abs(mx), abs(mn))
    return a <= thr, a


def _f32_bits(v):
    t = torch.tensor([float(v)], dtype=torch.float32)
    return f"0x{int(t.view(torch.int32).item()) & 0xFFFFFFFF:08x}"


def red_locate(L, tag, thr=None):
    """越红线才付的代价: 脏元素在哪、位置固不固定、是不是同一个值。
    位置固定 = 同一块脏内存每次都被分配给我们(可复现的脏, 能顺藤摸到写它的人);
    位置漂 = 分配器复用残值(谁踩过谁负责)。"""
    bad = L.abs() > (LOGIT_RED if thr is None else thr)
    idx = bad.nonzero()
    rows, cols = idx[:, 0].unique(), idx[:, 1].unique()
    v = L[bad].flatten()
    v0 = float(v[0])
    line = (f"    🚨[{tag}] 脏 n={int(v.numel())}/{L.numel()} "
            f"行={int(rows.numel())}{tuple(rows[:5].tolist())} 列={int(cols.numel())}{tuple(cols[:5].tolist())} "
            f"唯一值={int(torch.unique(v).numel())} absmax={float(v.abs().max()):.4g} 首值={v0:.6g}")
    if L.dtype == torch.float64:      # double 的脏: 两个 32 位分别当 fp32 读, 看是不是 fp32 残值拼的
        h = torch.tensor([v0], dtype=torch.float64).view(torch.int32)
        lo = torch.tensor([int(h[0])], dtype=torch.int32).view(torch.float32)
        hi = torch.tensor([int(h[1])], dtype=torch.int32).view(torch.float32)
        line += f" | 低32位当fp32={float(lo):.4g} 高32位当fp32={float(hi):.4g}"
    else:
        line += f" 位型={_f32_bits(v0)}"
    print(line)


def metrics(lp, lq, ids_row):
    """P/Q 的 logits -> token 级指标。float64, 全程 log_softmax 域。"""
    T = lp.shape[0]
    P, Q = torch.log_softmax(lp, -1), torch.log_softmax(lq, -1)
    p, q = P.exp(), Q.exp()
    kl = (p * (P - Q)).sum(-1)                       # (T,) nats/token
    tv = 0.5 * (p - q).abs().sum(-1)
    a1p, a1q = lp.argmax(-1), lq.argmax(-1)
    k5p, k5q = lp.topk(5, -1).indices, lq.topk(5, -1).indices
    # (T,5,5): [t,i,j] = P 的第 i 名 == Q 的第 j 名; any(2) => P 的第 i 名有没有落进 Q 的 top5
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
print(f"[命中] {HIT_BS} 行真激活累计 命中%={pct(cnt_e)} 最热/最冷="
      f"{float(cnt_e.max())/max(float(cnt_e.min()),1.0):.1f}x | A序={order_A} C序={order_C}")
fmt = lambda v, spec, w: (f"{v:{spec}}" if v == v else f"{'—':>{w}}")   # nan -> "—"


def ppl_s(dn, w=9):
    """ppl 比 = exp(Delta NLL)。Delta NLL 无定义(bs=1 没有下一个词) -> "—";
    |dn| 大到 exp 会溢出/下溢 -> 直接印 e^dn 记法, 不给一列假的 0.0000 或 inf。"""
    if dn != dn:
        return f"{'—':>{w}}"
    if abs(dn) > 50:
        return f"e^{dn:+.1f}".rjust(w)
    return f"{math.exp(dn):>{w}.4f}"

# ---------------------------------------------------------------- 逐格
for bs in BS_LIST:
    ids_row = IDS_FULL[:, :bs]
    # 🚨 x 必须在【同一个 T】上现抓: R.capture 那次是 512 行一起前向, 现在 splice 只跑 T 行,
    #    同一段网络在不同 batch 长度下走不同 GEMM 分块/归约顺序 => 差 ~5e-7, 闸门① 就不可能
    #    逐位 0。所以测量用同长现抓的 x, 归档路径(512 行)那份只作互差对照印出来。
    xcpu = capture_from(ids_row)
    x_arch = XREAL[:bs]
    d_cap = (xcpu - x_arch).abs().max().item()
    x = xcpu.cuda()
    PREV7 = {}                    # 闸门⑦ 跨块留样: 同一 bs 内 A/C 共享(不同 bs 的 T 不同, 必须重置)
    for label, order, note in CFGS:
        X7 = {}
        keys = [(gc, dt) for gc in GC_LIST for dt in DTYPES]
        print("\n" + "=" * 132)
        print(f"swiglu | 配置{label} {note} | bs={bs:>3d} | 第{CAP_LAYER}层 MLP 出口被覆盖 "
              f"| dtype={DTYPES} gc={GC_LIST} | 新序命中%={pct(cnt_e[order])} "
              f"| loadavg={open('/proc/loadavg').read().split()[0]}")
        print("=" * 132)
        t0 = time.perf_counter()
        ENG, MEM = {}, {}
        for k in keys:
            a0 = torch.cuda.memory_allocated()
            ENG[k] = mk(k[0], order, "swiglu", k[1])
            torch.cuda.synchronize()
            MEM[k] = (torch.cuda.memory_allocated() - a0) / 2 ** 20
        # ---- 闸门①: splice 位置 —— hook 录到的输入 vs 抓下来的 x
        box = {}
        lg_probe = splice(ids_row, torch.zeros(bs, D), box)
        d_splice = (box["x"][0] - xcpu).abs().max().item()
        bit_splice = int(torch.equal(box["x"][0], xcpu))
        # ---- 裁判 P(本格唯一) + 闸门⑥: P 是整块共用的分母, 它脏 => 整块 KL 全废
        # 判据: (a)幅度红线 (b)未挂钩同输入同代码重算逐位0 (c)y逐位0时反射logits必须逐位0
        kref = (GC_LIST[0], "fp32")
        ts0, ti0 = ENG[kref].route(x)
        ORIG = plain(ids_row)
        P_DIRTY, P_STATIC, P_ATT = True, False, 0
        d_ref = d_refl = d_plain = aP = aO = 0.0
        PL_prev = None
        for att in range(1, P_RETRY + 2):
            P_ATT = att
            ref0 = ENG[kref].fp32_reference(x, ts0, ti0)
            PL = splice(ids_row, ref0.cpu())
            ref0b = ENG[kref].fp32_reference(x, ts0, ti0)      # 裁判重算(y)
            ORIG2 = plain(ids_row)                             # 未挂钩重算(同输入同代码)
            PLb = splice(ids_row, ref0b.cpu())                 # 同一份 y 再反射一次
            d_ref = (ref0 - ref0b).abs().max().item()
            d_refl = (PL - PLb).abs().max().item()
            d_plain = (ORIG - ORIG2).abs().max().item()
            del PLb, ORIG2
            okP, aP = red_ok(PL)
            okO, aO = red_ok(ORIG)
            # flaky = 偶发型(重算会变 => 可能是被踩); red = 幅度型(阈值太紧 或 真被踩)
            why = ([] if okP else [("red", "P越红线 absmax=%.3g" % aP)]) \
                  + ([] if okO else [("red", "未挂钩越红线 absmax=%.3g" % aO)]) \
                  + ([("flaky", "未挂钩重算互差=%.3g" % d_plain)] if d_plain else []) \
                  + ([("flaky", "y逐位同但logits互差=%.3g" % d_refl)] if d_ref == 0 and d_refl else [])
            same = PL_prev is not None and bool(torch.equal(PL, PL_prev))
            del PL_prev
            PL_prev = PL
            if not why:
                P_DIRTY = False
                break
            # 只有"幅度型 + 重算逐位相同"才允许判成【不是脏】: 偶发的内存踩踏不可能次次给出同一份位型
            if same and all(c == "red" for c, _ in why):
                P_DIRTY, P_STATIC = False, True
                print("    闸门⑥ 第%d次: 越红线但重算【逐位相同】(absmax=%.4g) => 不是偶发脏"
                      "(同一份位型重复出现), 是 LOGIT_RED=%g 比真值 absmax 还紧。"
                      "数值可复现 => 表仍可用, 但正文要写明红线值" % (att, max(aP, aO), LOGIT_RED))
                break
            print("    闸门⑥ P路脏(第%d次): " % att + " / ".join(t for _, t in why)
                  + " | 与上次重算逐位相同=%d y重算互差=%.3g(归约序可非0)" % (int(same), d_ref))
            if not okP:
                red_locate(PL, "P bs=%d 配置%s" % (bs, label))
            if not okO:
                red_locate(ORIG, "未挂钩 bs=%d 配置%s" % (bs, label))
        if not (P_DIRTY or P_STATIC) and P_ATT > 1:
            print(f"    闸门⑥ P路第 {P_ATT} 次重算才干净 => 本块表里的 P/Q 数取自最后一份(脏那份已弃)")
        # ---- 参照行: 未挂钩真模型 vs P
        ORIG = plain(ids_row)
        M0 = metrics(ORIG, PL, ids_row)
        print(f"  样本 = {bs} 个 token 位置"
              + ("  🚨T<16: top1/TV 在这个格是【单次抽样】不是比率, 只能读 KL 的量级" if bs < 16 else "")
              + ("  🚨T=1: ΔNLL/ppl比【无定义】(那格没有下一个词可压) => 那两列的 — 是构造性空缺, 不是失败"
                 if bs == 1 else ""))
        print(f"  闸门① splice互差={d_splice:.9f} 逐位={bit_splice}(必须1) | 现抓 x 与归档路径"
              f"(R.capture 512 行长)互差={d_cap:.3g}(非0 是应该的: 同一段网络在不同 batch 长度下"
              f" GEMM 分块/归约顺序不同) | 裁判重算互差 y={d_ref:.3g}/logits={d_refl:.3g}"
              f"(y 非0 可以是 index_add_ 归约序; y 为0 时 logits 必须为0 = 闸门⑥c)"
              f" | 未挂钩重算互差={d_plain:.3g}(必须0) | 未挂钩 logits rms={ORIG.pow(2).mean().sqrt():.3g}")
        print(f"  参照【未挂钩真模型 → P】= 把第{CAP_LAYER}层真 MLP 换成 {E} 选 {K} 的 fp32 稠密 MoE"
              f" 本身的代价: top1={M0['top1']:.2f}% top5={M0['rec']:.2f}% KL={M0['kl']:.3e} "
              f"ΔNLL={fmt(M0['nllq']-M0['nllp'], '+11.3e', 11)} "
              f"ppl比={ppl_s(M0['nllq']-M0['nllp'])} "
              f"(末位: P 的词掉到 Q 的第 {M0['rank']} 名)")
        print(f"{'gc':>3}{'dtype':>7}{'CUDA驻MB':>10}{'CPU腿slot':>10}{'L2rel%':>10}{'top1一致%':>10}"
              f"{'top5重合%':>10}{'KL(P‖Q)':>12}{'KL_p95':>12}{'TV距离':>11}{'maxΔlogit':>11}"
              f"{'ΔNLL':>11}{'ppl比':>9}{'末位rank':>9}{'三模式互差':>12}{'跨轮KL互差':>12}")
        RS = {}
        for rnd in range(ROUNDS):
            for k in keys:
                m = ENG[k]
                if rnd == 0:                                     # 闸门②③ + 逐档 L2
                    ts, ti = m.route(x)
                    outs = {mo: m(x, async_mode=True, mode=mo) for mo in H.MODES}
                    worst = max((outs[a] - outs[b]).abs().max().item()
                                for i, a in enumerate(outs) for b in list(outs)[i + 1:])
                    idx_d = int((ti != ti0).sum())
                    refi = m.fp32_reference(x, ts, ti)
                    _, _, rel = H.err_metrics(outs[MODE], refi)
                    d_judge = (refi - ref0).abs().max().item()
                    y = outs[MODE]
                    RS[k] = dict(mode=worst, idx=idx_d, rel=rel, djud=d_judge,
                                 ncpu=int((ti >= m.gpu_expert_count).sum()),
                                 nbad=int(torch.isinf(y).sum() + torch.isnan(y).sum()),
                                 kl=[])
                    del outs
                else:
                    y = m(x, async_mode=True, mode=MODE)
                yok, yamax = red_ok(y, Y_RED)                   # 闸门⑥b: 引擎出口先过红线
                if not yok:
                    print(f"    🚨闸门⑥b 引擎出口 y 越红线(absmax={yamax:.3g} > {Y_RED:g})"
                          f" gc={k[0]} {k[1]} rnd{rnd} => 脏在【引擎侧】(int8 CPU 腿/GPU 腿/图缓冲),"
                          f" 不是 HF 那 {NL - CAP_LAYER - 1} 层")
                    red_locate(y, f"y bs={bs} 配置{label} gc={k[0]} {k[1]}", Y_RED)
                RS[k]["yamax"] = max(RS[k].get("yamax", 0.0), yamax)
                L = splice(ids_row, y.cpu())
                nq = stableq = 0
                while not red_ok(L)[0] and nq < Q_RETRY:      # 闸门⑥(d): Q 也过红线, 脏则重算
                    red_locate(L, f"Q bs={bs} 配置{label} gc={k[0]} {k[1]} rnd{rnd}")
                    nq += 1
                    y = m(x, async_mode=True, mode=MODE)
                    L2 = splice(ids_row, y.cpu())
                    if not red_ok(L2)[0] and bool(torch.equal(L, L2)):
                        stableq = 1       # 同一份位型重复出现 => 偶发踩踏说不通
                    L = L2
                    del L2
                okq, aq = red_ok(L)
                # 越红线的两种解释: 红线比真值紧(重算逐位同 + 与 P 同量级) / 真被踩(位型会变 或 高出十几个数量级)
                if okq or (stableq and aq <= STABLE_RATIO * max(aP, aO, 1.0)):
                    RS[k]["nq"] = RS[k].get("nq", 0) + nq
                    if stableq:
                        RS[k]["stableq"] = 1
                else:
                    RS[k]["dirtyq"] = 1 + nq
                RS[k]["amax"] = max(RS[k].get("amax", 0.0), aq)
                if rnd == 0 and k[0] == E:      # 闸门⑦: gc=E => CPU腿0slot => 放置序不该改数值
                    Lf = L.float().contiguous()
                    crc = zlib.crc32(Lf.numpy().tobytes()) & 0xFFFFFFFF
                    if k[1] in PREV7:
                        pL, pcrc = PREV7.pop(k[1])
                        X7[k[1]] = (int(torch.equal(Lf, pL)), (Lf - pL).abs().max().item(), pcrc, crc)
                        del pL
                    PREV7[k[1]] = (Lf, crc)
                R_ = metrics(PL, L, ids_row)
                RS[k]["kl"].append(R_["kl"])
                if rnd == 0:
                    RS[k]["m"] = R_
        # ---- 闸门⑤: GPU 腿 0 个 slot 的档, dtype 必须逐位不动
        chk = []
        for gc in GC_LIST:
            if "fp32" in DTYPES and len(DTYPES) > 1 and bs * K - RS[(gc, DTYPES[0])]["ncpu"] == 0:
                a = ENG[(gc, "fp32")](x, async_mode=True, mode=MODE)
                b = ENG[(gc, DTYPES[-1])](x, async_mode=True, mode=MODE)
                chk.append(f"gc={gc} 互差={(a-b).abs().max().item():.9f}")
        for gc in GC_LIST:
            for dt in DTYPES:
                k = (gc, dt)
                s, R_ = RS[k], RS[k]["m"]
                dk = max(s["kl"]) - min(s["kl"])
                dn = R_["nllq"] - R_["nllp"]
                ratio_s = ppl_s(dn)
                print(f"{gc:>3}{dt:>7}{MEM[k]:>9.1f}{s['ncpu']:>10}{s['rel']:>9.4f}%"
                      f"{R_['top1']:>10.2f}{R_['rec']:>10.2f}{R_['kl']:>12.3e}{R_['kl95']:>12.3e}"
                      f"{R_['tv']:>11.3e}{R_['dmax']:>11.3g}"
                      f"{fmt(dn, '+11.3e', 11)}{ratio_s}"
                      f"{R_['rank']:>9}{s['mode']:>12.9f}{dk:>12.3e}"
                      + ("  🚨inf/nan" if s["nbad"] else "")
                      + ("  🚨Q重算后仍越红线=该行作废" if s.get("dirtyq") else "")
                      + ("  [越红线但稳定=红线紧]" if s.get("stableq") else "")
                      + ("  🚨P脏=本格作废" if P_DIRTY else ""))
        print(f"  闸门② idx差={[RS[k]['idx'] for k in keys]}(全0) 闸门③ 三模式互差 max="
              f"{max(s['mode'] for s in RS.values()):.9f} 闸门④ 跨轮KL互差 max="
              f"{max(max(s['kl'])-min(s['kl']) for s in RS.values()):.3e} "
              f"裁判自检(各档 fp32_reference vs P)max={max(s['djud'] for s in RS.values()):.3g}")
        print(f"  闸门⑤ dtype 无流量自检(GPU腿0slot): " + ("  ".join(chk) if chk else "本格无此档")
              + f"  | 末位 P词={TOK.decode([RS[keys[0]]['m']['argp']])!r} "
                f"Q词={TOK.decode([RS[keys[0]]['m']['argq']])!r} | 本格 {time.perf_counter()-t0:.0f}s")
        # ---- 闸门⑥/⑦ 汇总
        badq = [f"gc={k[0]}{k[1]}" for k in keys if RS[k].get("dirtyq")]
        stq = [f"gc={k[0]}{k[1]}" for k in keys if RS[k].get("stableq")]
        badq = [f"gc={k[0]}{k[1]}" for k in keys if RS[k].get("dirtyq")]
        stq = [f"gc={k[0]}{k[1]}" for k in keys if RS[k].get("stableq")]
        nq_tot = sum(RS[k].get("nq", 0) for k in keys)
        pstat = ("(仍脏)" if P_DIRTY else
               "(稳定越线=红线比真值紧)" if P_STATIC else "过")
        print(f"  闸门⑥ 幅度红线 |logit|<={LOGIT_RED:g}: 通过 {len(keys) - len(badq)}/{len(keys)} 档"
              f" |P|absmax={aP:.3g} |未挂钩|absmax={aO:.3g} 最大|Q|absmax="
              f"{max(RS[k].get('amax', 0.0) for k in keys):.3g} 最大|y|absmax="
              f"{max(RS[k].get('yamax', 0.0) for k in keys):.3g}(红线{Y_RED:g})"
              f" | P路重算={P_ATT}次{pstat}"
              f" 未挂钩重算互差={d_plain:.3g}(必须0)"
              + (f" Q腿重算={nq_tot}次(越红线才重算: 脏或红线紧)" if nq_tot else "")
              + (f" 越红线但稳定(absmax与P同{STABLE_RATIO:g}倍内且重算逐位同)={len(stq)}档: {' '.join(stq)}" if stq else "")
              + (f" 🚨重算后仍脏: {' '.join(badq)}" if badq else ""))
        if X7:
            for dt, (bit, dd, c0, c1) in sorted(X7.items()):
                ok7 = bool(bit) or dd <= CROSS_TOL
                print(f"  闸门⑦ 跨块互差(gc={E}=全GPU => CPU腿0slot => 放置序不该改数值) {dt}: "
                      f"逐位={bit} 互差={dd:.3g}(容差{CROSS_TOL:g}=归约序噪声底) "
                      f"{'过' if ok7 else '🚨不过 => 后一块的数值路径被踩过'} | "
                      f"crc32 前块={c0:08x} 本块={c1:08x}(同 crc 可跨进程对账)")
        elif PREV7:
            print(f"  闸门⑦ 跨块互差(gc={E} 档已留样 crc={list(PREV7.values())[0][1]:08x}): "
                  f"本块是第一个配置块, 无对比对象 => 等后一块来了再判")
        else:
            print(f"  闸门⑦ 跨块互差: 未判 —— GC={GC_LIST} 里没有 gc=E={E} 这一档(想判就把它放进 GC)")
        if P_STATIC:
            print(f"  闸门⑥ 稳定越线: P 与未挂钩两次重算【逐位相同】但 |logit| 超过 LOGIT_RED="
                  f"{LOGIT_RED:g}(实测 absmax={max(aP, aO):.4g}) => 判为红线比真值紧、不是脏, 本格仍可读;"
                  f" 想让这行安静就把 LOGIT_RED 调到 {max(aP, aO) * 4:.0f} 以上")
        if P_DIRTY:
            print(f"  🚨闸门⑥ P 路脏(重算 {P_ATT} 次仍未过): P 是本块所有格共用的分母 => "
                  f"【配置{label} 整块作废】, 这块的 KL/top-1/TV/ΔNLL 一律不可入库; "
                  f"处置: 复跑, 或把 bs 降一档(脏只在 bs=512 出现过)")
        if chk:
            print("  🚨闸门⑤ 只判数值(逐位相同 = dtype 无数值入口), 不可外推到计时列: "
                  "稠密写法里 0 流量仍有关键路径成本 => 计时表里那些档照样能被降精省钱(方法论 25)")
        bad = [k for k in keys if RS[k]["m"]["top1"] < 95.0]
        if bad:
            print(f"  🚨 top-1 一致率 <95% 的档: {bad} ⇒ P0-④ 精度段必须按 token 级改写(不许只写 relRMSE)")
        for m in ENG.values():
            del m
        ENG.clear(); RS.clear(); MEM.clear()
        torch.cuda.empty_cache()

print("""
判读提示: ① 这张表回答的是【选出的词变不变】, 不是"向量差多少":
          top1一致% = 引擎与 fp32 稠密裁判 argmax 相同的 token 占比; KL = 每 token 词分布
          平均搬动多少 nats; TV = 搬走的概率质量(0~1, 最直观)。relRMSE 0.5% 好不好看,
          只有这三列能回答 —— 这才是 P0-④ 精度段该出现的数。
       ② 先看【参照行】(未挂钩真模型 → P): 它是"把真第8层 MLP 换成 8 选 2 MoE(不量化)"
          本身的代价, 不是这一刀的代价。表里的 P→Q 才是 int8 + 放置 + dtype 的总价。
          若参照行的 top1 就已经很低, 说明 P0-① 那套"真层 MLP 当专家 + lm_head 行当路由"
          离真模型很远, 本表的 KL 只能读成【相对量】(gc 之间/dtype 之间比), 不能报绝对值。
       ③ 闸门①②③④⑤ 全绿才有一行可读: ①③⑤ 必须逐位 0, ② 必须 0 个 idx,
          ④ 跨轮 KL 互差必须比 gc 之间的 KL 差小若干个数量级。
       ④ 期望形状: gc 越大 => 越多专家走 GPU fp32/bf16、CPU 腿 int8 的 slot 越少 => KL 单调降;
          bs=1 且放对时 CPU 腿常常 0 个 slot(表里的 CPU腿slot 列), 那一档若 KL 仍非 0,
          付钱的是 GPU 半边(乙-2c 的 bf16 0.43~0.59%), 不是 int8 —— 这正好把两笔钱分开了。
       ⑤ 样本数 = 那一格的 bs(T 个 token 位置)。top-1 一致率的分母就是它: T=128 时分辨率
          只有 0.78pp, T=1 时它就是【一个】token 的硬币 —— bs=1 那几行只读 KL/TV 的量级,
          别把 0% / 100% 当结论。要更细的分辨率就把 BS 开到 512(每格多花几秒)。
       ⑥ ppl 列在域外语料上算, 绝对值无意义; 要 held-out ppl 就 CORPUS=/path/to/wt2.txt。
          gc=0 那一行是【纯CPU下界】= 台账里 relRMSE 最差的那档; gc=8 是纯GPU上界。
          把本表的 KL 与 P0-② 的 gc 扫描(峰值显存/延迟)按 gc 对齐, 就是 P0-④ 那张三方交换图。""")

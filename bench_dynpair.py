#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-②附2 · dyn 轮转配对探针: 【放置完全相同】的三台引擎彼此差多少 —— dyn 那列跨策略 p50 还能不能读。

【为什么还要这一刀(不是"再跑一遍 dyn")】
  dyn 表自己的口径⑤ 就挂着这笔欠账: "曲线不是配对: 策略之间不轮转(一个引擎跑完三阶段才换)
   => 相邻策略 <噪声底 的差别读; 要读延迟收益必须轮转配对"。
  r3 复核轮(results/2026-11-22_P1-②d附_dyn_gc3_bs512_r3_coolfix.txt, md5 27f8c744)把它变成事故:
  【GPU 成员都是 [0,2,1]、三段都零迁移】的四台引擎, p50全(readme/input/drift)=
      static_A  5.099 / 5.844 / 7.961        hysteresis  3.812 / 4.668 / 5.949
      static_p1 4.586 / 4.862 / 6.015        naive       4.022 / 4.882 / 6.245
  => 同放置的引擎之间极差 29%/20%/34%, 三段秩次几乎一致。而 dyn 表里"动态放置省 20%~27% 延迟"
     的全部证据就长在这条带子上。它现在有两种读法, 后果完全相反:
   (甲) 引擎身份: 每台引擎自带一个持久偏移(建序 -> 显存地址/页对齐/图布局) => dyn 的
        【跨策略 p50 列整列作废】, "省延迟"这一条主张必须撤下;
   (乙) 位次/时间: r3 是"一台跑完才换下一台", 位次 = 进程总时长, 两者在 r3 里【完全混叠】——
        同放置那四台正好是 POLICIES 的第 1/3/5/8 位而 p50 严格单调下降, 同时自家 loadavg
        从 1.48 一路爬到 4.46 => 这是位次不是身份 => 轮转 + 两序能把它消掉, 延迟列按配对表更正。
  🚨 r3 内部永远分不开甲/乙(名次就是位次)。只有【同进程轮转 + 正反两序】能分:
     身份效应跟着【臂】走, 位次效应跟着【轮内位置】走 —— 这就是这一刀唯一要产出的那个数。

【臂(全部常驻、全程不 del; 🚨故意不带 price_*: 它要第二份 parse_price_table 实现,
  两份实现会漂移, 台账已有 price_gap() 双实现的教训)】
  A1/A2/A3  三台【同一副 static_A 放置】的引擎(enabled=False, 零迁移) = 空对照组:
            它们互相差多少 = 跨策略 p50 列的【真实噪声底】(= 本刀的主判据 底_pair)
  C1        static_C(成员 [7,4,6]) = 正对照: 已知 ~2.6x, 它不过星标 = 这把尺子坏了
  H1/H2     hyst_1x / hyst_2x = 真会迁移的那两档(一次迁移实测 3.2 帧 / 64 帧 = 5% = 目标效应量)
            它的【阶段1】成员与 A1 逐位同且零迁移 => 白送一对空对照配对

【口径】一发 = 该臂连跑三段(readme/input.txt/drift, 与 r3 同三段同挑行);
  每段 = WARM_ENG=12 帧暖机 + FRAMES=64 帧逐帧计时(前后各 sync) + 帧间在【计时窗外】跑 dyn.step;
  一轮 = 六个臂各一发; 🚨奇数轮正序、偶数轮反序(位次在两种序里对称 => 两序之差就是位次底);
  配对 = 【同一轮内】臂/A1, 跨轮取几何均值; 正序轮/反序轮各算一次几何均值 => 位置底 = 两者之差;
  发数序列上按 bimodal()(阈值 BIM_THR=8%, 与 乙-1c/丁/戊 同一把尺)查双峰, 🚨它要 >=6 发且两簇各
  >=2 发 => ROUNDS>=7 才有效, 少于 7 发时双峰列一律读"没查";
  frame 编号按【发】重置(每段 frame=k*FRAMES+f) => 检查点/冷却与 r3 逐帧对齐;
  dyn 的 n_obs 跨轮累积(只有整轮第 1 发的前 16 帧被策略暖机挡住) => 第 1 发的决策轨迹可与 r3 对账。
  🚨 本刀不碰裁判、不做质量列(KL/TV 归 bench_dyn 自己管), 只问延迟这把尺子能分辨多少。

【闸门】① 计数器 == route(x) 逐位; ② 全跑完后按运行时 slot2eid 重建引擎 => 输出逐位差 = 0;
  ③ 图不重录(id(_g_pre)/id(_g_hot)/_gkey/_wver 从暖机后全程不变); ⑥ slot2eid == 权重实际内容;
  ⑦ 静态档 enabled=False 全程零迁移(空对照组不许被自己污染);
  🚨新增 route 放置无关性: A1/C1/H1 同一段的 route 计数逐位相同 —— 它既是"上帝视角直方图与
     放置无关"的现场重证, 也是【本刀用 A1 代替 r3 那台 gc=0 裁判机挑 drift 行】的合法性凭据
     (不挂裁判机 = 少一份常驻显存 + 少一处会漂移的口径);
  ④' r3 对账: 第 1 发(计入表的那一发)的【分段末态成员 + 决策帧 + 迁移次数】必须等于 r3 归档,
     p50 绝对值只印差、只作对账不当分母(不同进程不同批次, 见 乙-1e 的同口径跨进程对账)。

【预设判决树】(跑前写死, 跑完不许改)
  底_pair = 空对照组内(A2/A1, A3/A1)在三段上 |配对比值(跨轮几何均值)-1| 的最大值;
  底_序   = 空对照组 + A1 自己, 正序轮几何均值 / 反序轮几何均值 与 1 的最大偏离(= 位次底);
  星标门槛 = max(2 x max(底_pair, 底_序), TAX_MIN=2.0%); 双峰一律读【不可判】。TARGET=5%。
  A) 底_pair <= 2.5% 且 真实档(C1/H1/H2)至少一格过星标
     => 轮转配对可用, dyn 的延迟列按本表更正, "动态放置省延迟"按本表配对值记账;
  B) 底_pair <= 2.5% 且 真实档全在带内
     => 【"动态放置省延迟"这条主张撤下】: dyn 表里 -20%~-27% 全是同放置引擎自己的噪声底;
  C) 底_pair > TARGET 且 底_序 <= TARGET
     => 引擎身份效应(位次已被两序排除) => dyn 的【跨策略 p50 列永久作废】,
        判读规则改成"只在同一台引擎内 迁移前 vs 迁移后 配对";
  D) 2.5% < 底_pair <= TARGET, 或 底_序 > TARGET(位次没被排除 => 甲/乙还是分不开),
     或 任一臂双峰 => 挂账(如实记"这一刀没能分开身份与位次", 不改任何旧表)。
【事前预测(可证伪, 跑前写死)】押 D, 机制假设 = 【位次/时间爬升】(上面那条乙):
  理由(假设不是结论): r3 同放置四台的 p50 严格单调跟随【建序 = 轮内位次】, 四台全对单调是
  1/24 的事件; 而"每台引擎自带身份偏移"没有任何已知的物理来源能解释它恰好按建序排开。
  竞争假设 = 甲(引擎身份: 建序 -> 显存地址/页对齐 -> AVX2 载入效率), 它若成立本刀应落 C。
  🚨 押 D 与押 C 是相反的下场(D = 旧表不动但要写明 dyn 延迟列读不了; C = 整列作废),
     所以判据与预测都必须在跑之前写死。
【交付边界】产出只有四种: ①逐发配对表(含末态成员 + 迁移次/ms); ②空对照组内的 底_pair 与
  两序位次底; ③位次曲线(轮内第几个臂 vs 相对本轮均值的偏离); ④按预设树自动印出的 A/B/C/D 落点。
  不改任何旧表、不改生产代码、不碰裁判。

【§5.2g 追加树(2026-11-22 r1 定稿后写死; r1 那棵 A/B/C/D 已消耗 => 复跑【不落它】, 只读累积底曲线)】
  r1 落 D 但落点理由是【有双峰臂 + 底_pair=8.9% > FLOOR_A】, 而 8.9%/底_序 16.4% 与"7 发 p50 的抽样误差"
  同量级 => r1 分不清"底本来就大(身份)"与"底只是发数不够(抽样误差)", 这两条的下场仍然相反。
  产出 = 收缩率 k = -ln(底_pair(n2)/底_pair(n1)) / ln(n2/n1), 用 SCALE_N 的前缀逐点重算两条底:
  E1) k >= E_K_HI=0.35, 或 末点 底_pair <= E_F_LO=6.0%
      => r1 那条底 = 7 发的抽样误差, 甲(引擎身份)否证 => dyn 跨策略 p50 列不作废,
         但必须加发数到 n≈90 才够 FLOOR_A=2.5%; "省 20%~27% 延迟" 仍不可记账(门槛仍 > 效应量)
  E2) k < E_K_LO=0.15 且 末点 底_pair >= E_F_HI=8.0%
      => 底与发数无关 = 每台引擎自带身份偏移, 甲成立 => dyn 跨策略 p50 列【永久作废】,
         判读规则改成"只在同一台引擎内 迁移前 vs 迁移后 配对"
  E3) 其余 => 身份与噪声混叠 => 挂账, 判读规则写成"dyn 跨策略列只能读 >= 2x 底 的差"
【事前预测(§5.2g, 跑前写死, PREDICT 传入)】押 E1【竞争假设 = E2】。理由(假设不是结论):
  ① 同放置的 A/H 组 σ 8.6%~12.8%, 而字节墙主导、绝对时长 3 倍长的 C1 σ 只 1.2%~1.8%
     => 绝对抖动是几百 us 量级的公共项, 相对噪声 ~ 1/时长, 4~6ms 的快腿本来就读不到 5% 级差;
  ② SE(p50@7) ~ 1.25σ/sqrt(7) ~ 5% => 底_pair ~ sqrt(2)x5% ~ 7%、底_序 ~ 2x5% ~ 10%, 与实测 8.9%/16.4% 同量级;
  ③ 位次曲线正序/反序两条【不镜像】且幅度只在 ±7%(在自身误差内) => 位次不是主犯, 身份也没有更大证据。
  => 预期 k≈0.5, 底_pair(25) ≈ 8.9 x sqrt(7/25) ≈ 4.7%。
"""
import os, sys, time, math, statistics, torch
sys.path.insert(0, '.'); sys.setswitchinterval(1e-5)
import hetero_moe_int8 as H
import real_smollm as R
# 🚨 分簇/统计的尺子复用 乙-1c 那一份 => 本刀切出来的簇与已入库表是同一刀
from bench_pairtax import stat, bimodal, TAX_MIN, BIM_THR
from dyn_placement import (DynPlacer, counts_from_engine, slot_counts, policy_from_name,
                           POLICY_PRESETS)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED, K, MODE = 42, 2, "L2_零CUDA+pinned"
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
WARM_ENG = 12
TAIL = float(os.environ.get("TAIL", "0.667"))
DRIFT_POOL = int(os.environ.get("DRIFT_POOL", 2048))
E = len(LAYERS)
FULL = dict(gpu_impl="cat", graph_scope="full", n_threads=6, tok_block=8,
            pre_impl="fused", l2_stream="single", cpu_sched=1, d2h_wait="main")
PHASES = ["readme", "input.txt", "drift"]      # 🚨 与 r3 同三段, 否则 闸门④' 对不上
ROUNDS = int(os.environ.get("ROUNDS", 7))      # 计入表的发数; <7 时双峰列读"没查"
WARM_ROUNDS = int(os.environ.get("WARM_ROUNDS", 1))   # 先跑一轮不进表(r3 那条爬升看得见)
ARMS = [a.strip() for a in os.environ.get("ARMS", "A1,A2,A3,C1,H1,H2").split(",") if a.strip()]
FLOOR_A = float(os.environ.get("FLOOR_A", 2.5))
TARGET = float(os.environ.get("TARGET", 5.0))
SCALE_N = [int(v) for v in os.environ.get("SCALE_N", "7,13,19,25").split(",") if v.strip()]   # §5.2g 累积底前缀
PREDICT = os.environ.get("PREDICT", "E1")          # §5.2g 事前预测(跑前写死, 见 docstring)
E_K_HI = float(os.environ.get("E_K_HI", "0.35"))   # E1: k >= 此值 => 纯抽样噪声
E_K_LO = float(os.environ.get("E_K_LO", "0.15"))   # E2: k < 此值 => 与发数无关 = 身份
E_F_LO = float(os.environ.get("E_F_LO", "6.0"))    # E1: 末点底_pair <= 此值
E_F_HI = float(os.environ.get("E_F_HI", "8.0"))    # E2: 末点底_pair >= 此值
# 臂 -> (策略名, 起跑序)  🚨 A 组三台必须逐字同配置: 同策略名 + 同起跑序 + 都 enabled=False
ARMSPEC = {"A1": ("static_A", "A"), "A2": ("static_A", "A"), "A3": ("static_A", "A"),
           "C1": ("static_C", "C"), "H1": ("hyst_1x", "A"), "H2": ("hyst_2x", "A")}
REAL = ["C1", "H1", "H2"]                       # 真实档(要读它延迟收益的那几台)
NULL = [a for a in ("A1", "A2", "A3") if a in ARMS]
assert set(ARMS) <= set(ARMSPEC), f"未知臂, 可选 {sorted(ARMSPEC)}"
assert WARM_ROUNDS + ROUNDS >= 2, "至少一发暖轮 + 一发计表"
assert ROUNDS >= 2, "两序 = 正反各至少一轮"
assert "A1" in ARMS, "A1 是配对分母, 不能摘"
TAILN = max(1, int(round(FRAMES * TAIL)))

# ---------------------------------------------------------------- 真模型(只 load 一次; 与 bench_dyn 逐字同)
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


def rss_mb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS:"):
            return int(ln.split()[1]) / 1024.0
    return float("nan")


t_load = time.perf_counter()
print(f"[机器状态] {H.machine_state()}")
_SD = R.state_dict()
R.state_dict = lambda: _SD            # 要建 6 台常驻引擎, 别每次重读 safetensors(与 bench_dyn 同)
dims = R.dims(_SD)
D, N, NL = dims["D"], dims["N"], dims["layers"]
MODEL, TOK = _load()
torch.manual_seed(SEED)


def capture_from(ids):
    """真模型第 CAP_LAYER 层 MLP 入口抓 x(与 bench_dyn/bench_swap 逐字同一挂法)。"""
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


def pct(c):
    return [round(float(v) / float(sum(c)) * 100, 1) for v in (c.tolist() if hasattr(c, "tolist") else c)]


BASE = R.real_src(LAYERS, gate_mode=GATE)      # 未置换基表: 迁移换入换出按 eid 从这里取


def mk(gc, order, dt=DTYPE):
    """建引擎 + 上 GPU + 暖一次缓存(与 bench_dyn.mk 逐字同 => 闸门②/⑥ 才与 r3 同源)。"""
    torch.manual_seed(SEED)
    m = H.HeteroMoE(D, E, N, gpu_expert_count=gc, top_k=K, expert_kind="swiglu",
                    real_src=perm_src(order), cpu_delay_ms=0.0, gpu_dtype=dt, **FULL).cuda()
    m(torch.zeros(1, D, device="cuda"), async_mode=True, mode=MODE)
    return m


def fwd(m, x):
    with torch.no_grad():
        return m(x, async_mode=True, mode=MODE)


def timed(m, x, n):
    """逐帧计时(前后各 sync 一次)。返回 (ms 列表, 最后一发输出)。"""
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fwd(m, x)
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out, y


# ---------------------------------------------------------------- 阶段语料 -> ids/x(与 bench_dyn 同一挑法)
def phase_ids(name, kth, rows=None):
    rows = BS if rows is None else int(rows)
    txt = (R.real_text(N_CHARS) if name == "readme"
           else open(os.path.join(HERE, name), encoding="utf-8", errors="ignore").read()[:N_CHARS])
    ids = TOK(txt, return_tensors="pt").input_ids
    lo = kth * BS
    assert ids.shape[1] >= lo + rows, f"阶段语料 {name} 只有 {ids.shape[1]} token, 取不出第 {kth} 段 {rows} 行"
    return ids[:, lo:lo + rows], len(txt), ids.shape[1]


def route_eid(m, x, order):
    """route(x) 槽位索引 -> eid 索引。闸门【route 放置无关性】与 drift 挑行都吃这一行。
    🚨 回换必须用【这台引擎自己的起跑序】: slot j 装的是 order[j] 号专家,
       拿 A 序去换 C 序的槽位 = 自己造出"放置改变了路由"的假象(冒烟轮就是这么红的)。"""
    ts, ti = m.route(x)
    return ts, ti, torch.tensor(list(order))[ti.detach().cpu().long()]


def drift_ids(router, order):
    """人造漂移段: 与 bench_dyn.drift_ids 逐字同一挑法(挑分/COLD集/stable argsort/还原行序全同),
    只把挑行用的 route 从【裁判机 gc=0】换成【A1】—— 合法性由闸门【route 放置无关性】现场证明;
    好处 = 少一份常驻显存 + 少一处会漂移的口径(本刀不碰裁判)。"""
    ids_pool = phase_ids("readme", 0, DRIFT_POOL)[0]
    xp = capture_from(ids_pool).cuda()
    _ts, _ti, eid = route_eid(router, xp, order)
    eid = eid.reshape(-1, K)
    cold = torch.zeros(E, dtype=torch.bool)
    cold[order_A[GC:]] = True
    score = cold[eid].sum(1).float()
    try:
        sel = torch.argsort(-score, stable=True)[:BS]
    except TypeError:                                  # 老 torch 没有 stable
        sel = torch.argsort(-score)[:BS]
    sel = sel.sort().values                            # 🚨 还原行序 => 与 r3 逐行同(闸门④' 的前提)
    share = float(score[sel].sum()) / float(BS * K)
    del xp
    torch.cuda.empty_cache()
    return ids_pool[:, sel], share


# ---------------------------------------------------------------- 命中画像 -> A序/C序
XREAL, _d = R.capture(layer=CAP_LAYER, bs=BS, n_chars=N_CHARS)
cnt_e = hits_of(BASE["gate_w"], XREAL)
order_A = torch.argsort(cnt_e, descending=True).tolist()
order_C = torch.argsort(cnt_e).tolist()


# ---------------------------------------------------------------- 6 臂: 一次建齐, 全程常驻, 一次都不 del
# 🚨 这是本刀全部杠杆所在: 只有【同一进程里常驻的六台】才能把"跟着臂走的身份"和
#    "跟着轮内位置走的位次"分开。任何一台被 del 重造, 身份就被重新发牌, 表立刻不可读。
ENG = {}
for a in ARMS:
    pol_name, st_tag = ARMSPEC[a]
    start = order_A if st_tag == "A" else order_C
    t0 = time.perf_counter()
    m = mk(GC, start, DTYPE)
    dyn = DynPlacer(m, BASE["experts"], GC, policy=policy_from_name(pol_name, GC),
                    window=WINDOW, check_every=CHECK, warm=WARM, order=start)
    dyn.sw.src_gate = BASE["gate_w"]      # 🚨 DynPlacer 不转发它, 挂不上 = 闸门⑥ 不核门控行
    ENG[a] = dict(m=m, dyn=dyn, pol=pol_name, start=list(start), gk0=None,
                  enabled=bool(POLICY_PRESETS.get(pol_name, {}).get("enabled", True)),
                  build_s=time.perf_counter() - t0)
    print(f"  [建臂 {a:<3}] {pol_name:<9} 起跑={'A序' if st_tag == 'A' else 'C序'} "
          f"enabled={ENG[a]['enabled']} "
          f"{ENG[a]['build_s']:.1f}s | RSS={rss_mb():.0f}MB "
          f"显存={torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB")

# ---------------------------------------------------------------- 三段语料(先放对/放错两段, drift 要等 A1 建好)
PH = []
for k, nm in enumerate(PHASES):
    kth = PHASES[:k].count(nm)
    if nm == "drift":
        PH.append(dict(name=nm, kth=kth, pending=True))
        continue
    ids, nch, ntok = phase_ids(nm, kth)
    PH.append(dict(name=nm, kth=kth, ids=ids, x=capture_from(ids).cuda(), nch=nch, ntok=ntok))

# ------------------------------------------------- 闸门【route 放置无关性】(挑 drift 行【之前】先核)
GR = []                       # (段, 臂对, 逐位是否相同)
for P in PH:
    if P.get("pending"):
        continue
    seen = {}
    for a in [q for q in ("A1", "C1", "H1") if q in ENG]:
        _ts, _ti, eid = route_eid(ENG[a]["m"], P["x"], ENG[a]["start"])
        seen[a] = torch.bincount(eid.reshape(-1), minlength=E)
    ks = list(seen)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            GR.append((P["name"], f"{a}/{b}", int(torch.equal(seen[a], seen[b]))))
if not all(g[2] for g in GR):
    print(f"  🚨闸门[route 放置无关性] 挑行前就红了: {GR} => 用 A1 挑 drift 行不合法, 本表作废")

for k, P in enumerate(PH):
    if not P.get("pending"):
        continue
    ids, share = drift_ids(ENG["A1"]["m"], order_A)   # 🚨 用 A1 的 route 挑行(上一闸门刚核过)
    PH[k] = dict(name=P["name"], kth=P["kth"], ids=ids, x=capture_from(ids).cuda(),
                 share=share, nch=-1, ntok=int(ids.shape[1]))
    seen = {}
    for a in [q for q in ("A1", "C1", "H1") if q in ENG]:
        _ts, _ti, eid = route_eid(ENG[a]["m"], PH[k]["x"], ENG[a]["start"])   # 🚨 P 还是 pending 那份, 取 PH[k]
        seen[a] = torch.bincount(eid.reshape(-1), minlength=E)
    ks = list(seen)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            GR.append((PH[k]["name"], f"{a}/{b}", int(torch.equal(seen[a], seen[b]))))

# 每段真值直方图(eid 空间, 与放置/策略无关) + 该段上帝视角成员
for P in PH:
    _ts, _ti, eid = route_eid(ENG["A1"]["m"], P["x"], order_A)
    c = torch.bincount(eid.reshape(-1), minlength=E).float()
    P["cnt"] = c
    P["top"] = torch.argsort(c, descending=True).tolist()[:GC]
    P["hot"] = [e for e in range(E) if float(c[e]) / float(c.sum()) > 0.01]

print(f"\n[真源] D={D} N={N} 层数={NL} | 取 {list(LAYERS)} 层 MLP 当 {E} 个专家 top_k={K}"
      f" | 路由={GATE} | GC={GC} BS={BS} dtype={DTYPE} | 建模+建臂 {time.perf_counter() - t_load:.0f}s")
print(f"[臂] {' '.join(a + '=' + ENG[a]['pol'] for a in ARMS)} | 空对照组={NULL} 真实档={[a for a in REAL if a in ARMS]}")
print(f"[序] A序={order_A}(GPU 成员={order_A[:GC]}) C序={order_C}(GPU 成员={order_C[:GC]})")
print(f"[每段] {len(PH)} 段 x (暖机 {WARM_ENG} 帧 + 计时 {FRAMES} 帧) | dyn: 窗口{WINDOW} 暖机{WARM}"
      f" 每{CHECK}帧一决策 | 一轮 = {len(ARMS)} 臂各一发, 奇数轮正序/偶数轮反序 | 计表 {ROUNDS} 轮")
for k, P in enumerate(PH):
    cov = 100.0 * sum(float(P["cnt"][e]) for e in P["top"]) / float(P["cnt"].sum())
    print(f"   · 段{k + 1} {P['name']:<10} 命中%={pct(P['cnt'])} 上帝成员={P['top']}(覆盖{cov:.1f}%)"
          f" 热集>1%={P['hot']}" + (f" drift COLD 占比={100 * P['share']:.1f}%" if "share" in P else ""))
print(f"[常驻] RSS={rss_mb():.0f}MB 显存={torch.cuda.memory_allocated() / 1024 ** 2:.0f}MB"
      f"(6 台常驻 = 本刀故意付的代价; 🚨 它不是被测系统的显存, dyn 本身不新增常驻字节)")

# ---------------------------------------------------------------- r3 归档锚(闸门④' 只作对账, 绝不当分母)
R3 = dict(md5="27f8c744",
          file="2026-11-22_P1-②d附_dyn_gc3_bs512_r3_coolfix.txt",
          order_A=[0, 2, 1, 5, 3, 6, 4, 7], order_C=[7, 4, 6, 3, 5, 1, 2, 0],
          static_A=dict(p50=[5.099, 5.844, 7.961], mem=[[0, 2, 1]] * 3, nsw=[0, 0, 0], dec=[]),
          static_C=dict(p50=[13.377, 13.775, 12.233], mem=[[7, 4, 6]] * 3, nsw=[0, 0, 0], dec=[]),
          hyst_1x=dict(p50=[4.210, 4.722, 4.993], mem=[[0, 2, 1], [0, 2, 5], [0, 6, 5]],
                       nsw=[0, 1, 1], dec=["f71 5->1", "f143 6->2"]),
          hyst_2x=dict(p50=[4.855, 4.905, 5.665], mem=[[0, 2, 1], [0, 2, 1], [0, 2, 5]],
                       nsw=[0, 0, 1], dec=["f143 5->1"]))

# ---------------------------------------------------------------- 一发 = 一台引擎连跑三段(与 r3 同三段同挑行)
LAT = {a: [[] for _ in PH] for a in ARMS}       # 计入表的 p50(全帧中位) 逐轮
NSW = {a: [[] for _ in PH] for a in ARMS}       # 本段迁移次
SWM = {a: [[] for _ in PH] for a in ARMS}       # 本段迁移 ms
MEM = {a: [[] for _ in PH] for a in ARMS}       # 本段末态成员
G1, G3, G6 = [], [], []                          # 闸门①/③/⑥ 逐发明细
TRACE0, TAGS, POS = {}, [], {}                   # 冷发轨迹 / 计表轮的序标签 / 位次曲线原料
SHOT_NO = {a: 0 for a in ARMS}
fm = lambda m: "[" + ",".join(str(int(e)) for e in m) + "]"


def shot(a, tag, keep):
    """一台引擎连跑三段。返回每段的 (p50, 成员, 迁移次, 迁移ms, 决策列表)。"""
    st = ENG[a]
    m, dyn = st["m"], st["dyn"]
    cold = SHOT_NO[a] == 0                       # 进程冷发: 暖机状态与 r3 逐字同 => ④' 对账对象
    out = []
    for k, P in enumerate(PH):
        dyn.pol.enabled = st["enabled"]          # 🚨 每段重挂: 静态档一旦被点亮就是对照组污染
        for _ in range(WARM_ENG):
            fwd(m, P["x"])
        torch.cuda.synchronize()
        gk = (id(m._g_pre), id(m._g_hot), m._gkey, tuple(m._wver or ()))
        if st["gk0"] is None:
            st["gk0"] = gk
        G3.append((a, k, tag, gk == st["gk0"]))
        s_n, s_us = dyn.pol.n_swap, dyn.sw.total_us
        s_rs = len(dyn.pol.reasons)
        ss, _y = timed(m, P["x"], 1)             # 第 0 帧: 建 pinned 缓冲, 兼计数器闸门
        c = counts_from_engine(m, E)
        _ts, ti = m.route(P["x"])
        G1.append((a, k, tag, int(torch.equal(c, slot_counts(ti.detach().cpu(), E))), int(c.sum())))
        dyn.step(k * FRAMES + 0, k, counts=c)    # 第 0 帧也进窗口(否则 warm/检查点整格错位)
        for f in range(1, FRAMES):
            s, _y = timed(m, P["x"], 1)
            ss += s
            dyn.step(k * FRAMES + f, k)          # 迁移在计时窗口【外】: 它的成本单列
        G6.append((a, k, tag, dyn.sw.check_content()))
        p50 = statistics.median(ss)
        memb = list(dyn.sw.hot_members())
        nsw = dyn.pol.n_swap - s_n
        sms = (dyn.sw.total_us - s_us) / 1e3
        dec = [" ".join(r.split()[:2]) for r in dyn.pol.reasons[s_rs:]]
        out.append((p50, memb, nsw, sms, dec))
        if keep:
            LAT[a][k].append(p50); NSW[a][k].append(nsw); SWM[a][k].append(sms); MEM[a][k].append(memb)
        if cold:
            TRACE0.setdefault(a, []).append(dict(p50=p50, mem=memb, nsw=nsw, ms=sms, dec=dec))
    SHOT_NO[a] += 1
    return out


def run_round(r, keep):
    seq = list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
    order_tag = "fwd" if r % 2 == 0 else "rev"
    tag = f"#{r}{'正' if r % 2 == 0 else '反'}{'表' if keep else '暖'}"
    if keep:
        TAGS.append(order_tag)
    t0, got = time.perf_counter(), {}
    for pos, a in enumerate(seq):
        got[a] = shot(a, tag, keep)
        if keep:
            for k in range(len(PH)):
                POS.setdefault((pos, k), []).append((a, got[a][k][0], order_tag))
    if "A1" not in got:
        return
    for pos, a in enumerate(seq):
        rat = [got[a][k][0] / got["A1"][k][0] for k in range(len(PH))]
        print(f"  [{tag} {a:<3} pos{pos + 1}] p50 " + "/".join(f"{got[a][k][0]:7.3f}" for k in range(len(PH)))
              + " | 配A1 " + "/".join(f"{100 * (v - 1):+6.1f}%" for v in rat)
              + " | mem " + "/".join(fm(got[a][k][1]) for k in range(len(PH)))
              + " | 迁 " + "/".join(str(got[a][k][2]) for k in range(len(PH)))
              + "次 " + "/".join(f"{got[a][k][3]:.1f}" for k in range(len(PH)))
              + "ms" + (" | " + ";".join(d for k in range(len(PH)) for d in got[a][k][4]) if any(got[a][k][4] for k in range(len(PH))) else ""))
    print(f"  [{tag}] {(time.perf_counter() - t0):.0f}s | RSS={rss_mb():.0f}MB "
          f"loadavg={open('/proc/loadavg').read().split()[0]}")


for r in range(WARM_ROUNDS):
    print(f"\n---- 暖轮 {r}(不进表; 只为了让 CUDA 图/分配器/滑窗进入稳态) ----")
    run_round(r, False)
for r in range(WARM_ROUNDS, WARM_ROUNDS + ROUNDS):
    print(f"\n---- 第 {r} 轮({'正序' if r % 2 == 0 else '反序'}, 进表) ----")
    run_round(r, True)

# ---------------------------------------------------------------- 闸门② 末态 slot2eid 重建引擎(逐位必须=1)
G2 = []
for a in ARMS:
    dyn, m = ENG[a]["dyn"], ENG[a]["m"]
    if not dyn.pol.n_swap:
        continue                                   # 没迁移过 = 没资格(不是跳过)
    last_order = list(dyn.sw.slot2eid)
    m2 = mk(GC, last_order, DTYPE)
    for _ in range(WARM_ENG):
        fwd(m2, PH[-1]["x"])
    yb = fwd(m2, PH[-1]["x"]).float().cpu().clone()
    ya = fwd(m, PH[-1]["x"]).float().cpu().clone()
    G2.append((a, last_order, int(torch.equal(ya, yb)), (ya - yb).abs().max().item()))
    del m2, yb, ya
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 配对数学
def geo(v):
    return math.exp(sum(math.log(float(x)) for x in v) / len(v))


nP = len(PH)
FWD = [i for i, t in enumerate(TAGS) if t == "fwd"]
REV = [i for i, t in enumerate(TAGS) if t == "rev"]


def pairs(idx):
    if not idx:
        return {a: [float("nan")] * nP for a in ARMS}
    return {a: [geo([LAT[a][k][i] / LAT["A1"][k][i] for i in idx]) for k in range(nP)] for a in ARMS}


ALL_IDX = list(range(len(TAGS)))
PAL, PFW, PRV = pairs(ALL_IDX), pairs(FWD), pairs(REV)
GS = {a: [geo(LAT[a][k]) if LAT[a][k] else float("nan") for k in range(nP)] for a in ARMS}
NULL2 = [a for a in NULL if a != "A1"]
底_pair = (max([abs(PAL[a][k] - 1) * 100 for a in NULL2 for k in range(nP)] or [float("nan")]))
if FWD and REV:
    底_序 = max(abs(PFW[a][k] / PRV[a][k] - 1) * 100 for a in NULL for k in range(nP))
else:
    底_序 = float("nan")
BAD = {(a, k): bimodal(LAT[a][k]) for a in ARMS for k in range(nP) if len(LAT[a][k]) >= 6}
BAD = {kk: v for kk, v in BAD.items() if v}
thr = max(2.0 * max(v for v in (底_pair, 底_序) if v == v), TAX_MIN)
STAR = {(a, k): (a in REAL and (a, k) not in BAD and abs(PAL[a][k] - 1) * 100 > thr)
        for a in ARMS for k in range(nP)}

# ------------------------- 累积底曲线(§5.2g): 只看【前 n 个计表轮】重算两条底 => 量底随发数怎么缩
def floors(n):
    """前 n 个计表轮前缀上的 (底_pair, 底_序)。纯抽样噪声应 ~ n^-0.5(k≈0.5); 身份与 n 无关(k≈0)。"""
    pal = pairs([i for i in ALL_IDX if i < n])
    fp = (max([abs(pal[a][k] - 1) * 100 for a in NULL2 for k in range(nP)]) if NULL2 else float("nan"))
    fw, rv = [i for i in FWD if i < n], [i for i in REV if i < n]
    fo = float("nan")
    if fw and rv:
        pf, pr = pairs(fw), pairs(rv)
        fo = max(abs(pf[a][k] / pr[a][k] - 1) * 100 for a in NULL for k in range(nP))
    return fp, fo


SC_N = sorted({x for x in SCALE_N if 2 <= x <= len(TAGS)} | {len(TAGS)})
SC = [(n,) + floors(n) for n in SC_N]
_KP = [(n, f) for n, f, _ in SC if f == f and f > 0]
k_scale = (-math.log(_KP[-1][1] / _KP[0][1]) / math.log(_KP[-1][0] / _KP[0][0])) if len(_KP) >= 2 else float("nan")
n_end = _KP[-1][0] if _KP else 0
底_pair末 = _KP[-1][1] if _KP else float("nan")
n_need = int(round(n_end * (底_pair末 / FLOOR_A) ** 2)) if _KP else 0
PRED_DESC = {"E1": "E1(k≈0.5 纯抽样噪声; 底_pair(25) 预期 ≈ 8.9 x sqrt(7/25) ≈ 4.7%)",
             "E2": "E2(k≈0 底与发数无关 = 每台引擎自带身份偏移)",
             "E3": "E3(身份与噪声混叠)"}


def verdict_scale():
    """§5.2g 追加树的落点。判据写死在 docstring 与 OPTLOG 5.2g, 跑完不许改。"""
    if not (k_scale == k_scale) or not (底_pair末 == 底_pair末):
        return "E?", "SCALE_N 可用前缀不足 2 点(或空对照组只剩 A1) => 收缩率 k 算不出"
    if k_scale >= E_K_HI or 底_pair末 <= E_F_LO:
        return "E1", (f"k={k_scale:.2f} >= {E_K_HI} 或 底_pair({n_end})={底_pair末:.1f}% <= {E_F_LO}%"
                      f" => 底随发数收缩 = r1 那条 底_pair=8.9% 是 7 发的抽样误差, 甲(身份)否证"
                      f" => dyn 跨策略 p50 列不作废但须加发数到 n≈{n_need} 才够 FLOOR_A={FLOOR_A}%")
    if k_scale < E_K_LO and 底_pair末 >= E_F_HI:
        return "E2", (f"k={k_scale:.2f} < {E_K_LO} 且 底_pair({n_end})={底_pair末:.1f}% >= {E_F_HI}%"
                      f" => 底与发数无关 = 每台引擎自带身份偏移, 甲成立 => dyn 跨策略 p50 列永久作废")
    return "E3", (f"k={k_scale:.2f} 落在 [{E_K_LO},{E_K_HI}) 且 底_pair({n_end})={底_pair末:.1f}%"
                  f" 落在 ({E_F_LO},{E_F_HI}) => 身份与噪声混叠 => 只能读 >= 2x 底 的差")

# ---------------------------------------------------------------- 落点(优先序 = docstring 里写死的那条, 不许看完数再挑)
def verdict():
    if BAD:
        return "D", f"有双峰臂 {sorted(BAD)} => 按树 D 末条挂账"
    if not (底_序 == 底_序):
        return "D", "正序/反序轮有一边为空 => 位次底算不出 = 位次没被排除"
    if 底_序 > TARGET:
        return "D", f"底_序={底_序:.1f}% > TARGET={TARGET:.0f}% => 位次没被两序消掉, 甲/乙仍混叠"
    if not (底_pair == 底_pair):
        return "D", "空对照组只有 A1 一台 => 底_pair 算不出"
    if 底_pair > TARGET:
        return "C", f"底_pair={底_pair:.1f}% > TARGET(位次已消掉) => 引擎身份效应, dyn 跨策略 p50 列作废"
    if 底_pair > FLOOR_A:
        return "D", f"{FLOOR_A}% < 底_pair={底_pair:.1f}% <= TARGET => 空对照自己的底比要读的信号还大"
    hit = [kk for kk, s in STAR.items() if s]
    return ("A" if hit else "B"), (f"底_pair={底_pair:.1f}% <= {FLOOR_A}% 且真实档过星标 {sorted(hit)}"
                                   if hit else
                                   f"底_pair={底_pair:.1f}% <= {FLOOR_A}% 且真实档全在带内(门槛{thr:.1f}%)")


# ---------------------------------------------------------------- 打印
print("\n" + "=" * 150)
print(f"汇总: 配对几何均值 = 【同一轮内】 臂/A1 再跨轮取几何 | 正序/反序各算一次 = 位次底 | 门槛={thr:.1f}%"
      f" (=max(2xmax(底_pair,底_序), TAX_MIN={TAX_MIN}%)) | * = 真实档过门槛 | ✗ = 双峰不可判")
print("=" * 150)
print(f"{'臂':<5}{'段':>3}{'发':>4}{'p50中位':>10}{'σ%':>6}{'min':>9}{'配对(全)':>10}{'配对(正)':>10}"
      f"{'配对(反)':>10}{'Δ%':>8}{'迁次/发':>9}{'末态成员':>12}  判读")
for a in ARMS:
    for k in range(nP):
        v = LAT[a][k]
        if not v:
            continue
        md, sd, mn = stat(v)
        b = BAD.get((a, k))
        bk = "没查" if len(v) < 6 else ("✗" if b else "")
        tag = ("✗双峰" + f"(低{b[1]}≈{b[0]:.3f}/高{b[3]}≈{b[2]:.3f} 隔{b[4]:.0f}%)") if b else \
              ("没查(发<6)" if len(v) < 6 else ("★过门槛" if STAR[(a, k)] else "带内"))
        print(f"{a + '/' + PH[k]['name'][:6]:<8}{k + 1:>2}{len(v):>4}{md:>10.3f}{sd:>6.1f}{mn:>9.3f}"
              f"{PAL[a][k]:>10.4f}{PFW[a][k]:>10.4f}{PRV[a][k]:>10.4f}"
              f"{100 * (PAL[a][k] - 1):>+8.1f}{sum(NSW[a][k]) / len(v):>9.2f}"
              f"{str(sorted(set(tuple(x[0]) for x in [[mm] for mm in MEM[a][k]]))):>12}  {tag}")
print(f"\n  底_pair(空对照组 A2/A3 对 A1, 三段取最大) = {底_pair:.2f}%   门槛 FLOOR_A={FLOOR_A}% TARGET={TARGET}%")
print(f"  底_序  (空对照组+A1, 正序轮/反序轮 配对之差取最大) = {底_序:.2f}%   正序轮{len(FWD)}发 反序轮{len(REV)}发")
print("\n累积底曲线(§5.2g: 只看【前 n 个计表轮】重算两条底) —— 纯抽样噪声 k≈0.5(底 ~ n^-0.5), 引擎身份 k≈0(底不动)")
for n, fp, fo in SC:
    预 = "%.2f%%" % (底_pair * math.sqrt(len(TAGS) / n)) if 底_pair == 底_pair else "n/a"
    print(f"  n={n:>3}  底_pair={fp:7.2f}%  底_序={('n/a' if fo != fo else format(fo, '.2f') + '%'):>8}"
          f"   纯噪声预期底_pair(n^-0.5)={预:>7}   正序{len([i for i in FWD if i < n])}发/反序{len([i for i in REV if i < n])}发")
print(f"  标度指数 k = -ln(底_pair({n_end})/底_pair({_KP[0][0] if _KP else 0})) / ln({n_end}/{_KP[0][0] if _KP else 0})"
      f" = {('nan' if k_scale != k_scale else format(k_scale, '.2f'))}   | 要够 FLOOR_A={FLOOR_A}% 预计需 n≈{n_need}"
      f"   (事前预测 PREDICT={PREDICT})")
print("  逐臂迁移总账: 计表轮 " + " | ".join(f"{a} {sum(sum(x) for x in NSW[a])}次/{sum(sum(x) for x in SWM[a]):.0f}ms"
                                        for a in ARMS))
print("            全程(含暖轮与进程冷发, 上表【迁次/发】只数计表轮 => 计表轮读 0 不代表这台没迁移过): "
      + " | ".join(f"{a} {ENG[a]['dyn'].pol.n_swap}次/{ENG[a]['dyn'].sw.total_us / 1e3:.0f}ms" for a in ARMS))
print("\n位次曲线(每台先除掉【自己】的跨轮几何均值 => 身份被扣干净, 剩下的只可能是位次; 单位=偏离%)")
for k in range(nP):
    for ot, lab in (("fwd", "正序"), ("rev", "反序")):
        cells = []
        for pos in range(len(ARMS)):
            vals = [v / GS[a][k] for (a, v, t) in POS.get((pos, k), []) if t == ot]
            who = ARMS[pos] if ot == "fwd" else ARMS[len(ARMS) - 1 - pos]
            cells.append(f"pos{pos + 1}{who}={100 * (geo(vals) - 1):+5.1f}" if vals else f"pos{pos + 1}{who}=  n/a")
        print(f"   段{k + 1} {PH[k]['name']:<9} {lab} " + " ".join(cells))


# ---------------------------------------------------------------- 闸门汇总(有一红, 上面那张表就没有资格被读)
def _gate(name, rows, ok, note=""):
    bad = [r for r in rows if not ok(r)]
    mark = "OK" if not bad and rows else ("🚨" if bad else "没资格")
    print(f"  闸门{name:<26} {len(rows) - len(bad)}/{len(rows)} {mark}"
          + (f" —— 不符 {bad[:5]}" if bad else "") + (f"  [{note}]" if note else ""))
    return not bad and bool(rows)


print("\n" + "=" * 150)
print("闸门(全绿 = 六台常驻引擎真的在算同一件事, 且图/权重/计数器没被迁移改坏)")
print("=" * 150)
_gate("① 计数器==route 逐位", G1, lambda r: r[3] == 1 and r[4] > 0,
      "逐发核 counts_from_engine vs route(x), nsum>0")
_gate("③ 图不重录", G3, lambda r: r[3], f"id(_g_pre)/id(_g_hot)/_gkey/_wver 从首发起全程不变")
_gate("⑥ slot2eid==权重内容", G6, lambda r: not r[3], "不符项列表为空 = 绿")
_gate("route 放置无关性", GR, lambda r: r[2] == 1, "A1/C1/H1 同段 eid 直方图逐位同 => 用 A1 挑 drift 行合法")
ST0 = [(a, ENG[a]["enabled"], ENG[a]["dyn"].pol.n_swap) for a in ARMS if not ENG[a]["enabled"]]
_gate("⑦ 静态档零迁移", ST0, lambda r: r[2] == 0, "空对照组/静态 C 被点亮 = 对照组自己脏了")
if G2:
    _gate("② 末态重建逐位", G2, lambda r: r[2] == 1, "按运行时 slot2eid 重造引擎再对输出")
else:
    print(f"  闸门② 末态重建逐位          0/0 没资格 —— 本轮【没有任何臂迁移过】, 不是跳过")

# ---------------------------------------------------------------- 闸门④' r3 对账(冷发轨迹; p50 只印差不当分母)
print(f"\n  闸门④' r3 对账(md5 {R3['md5']} {R3['file']}): 进程冷发的【末态成员/迁移次/决策序列】必须逐字等于归档")
MINE_ORD = {"order_A": order_A, "order_C": order_C}
od_bad = [(t, MINE_ORD[t], R3[t]) for t in MINE_ORD if MINE_ORD[t] != R3[t]]
if od_bad:
    print(f"    🚨起跑序与 r3 不一致 => ④' 整条作废(挑行/成员都换了): {od_bad}")
else:
    print(f"    起跑序 A序={order_A} C序={order_C} 与 r3 逐字一致 => ④' 有效")
    for a in ARMS:
        pol = ENG[a]["pol"]
        if pol not in R3 or a not in TRACE0:
            print(f"    {a:<3} {pol:<9} r3 无此档 => 只对 p50 不作对账")
            continue
        ref, tr = R3[pol], TRACE0[a]
        mm = [k for k in range(nP) if sorted(tr[k]["mem"]) != sorted(ref["mem"][k])]
        nn = [k for k in range(nP) if tr[k]["nsw"] != ref["nsw"][k]]
        d_mine = [d for k in range(nP) for d in tr[k]["dec"]]
        dd = ["(段%d)%s" % (k + 1, d) for k in range(nP) for d in tr[k]["dec"] if d not in ref["dec"]]
        d_miss = [d for d in ref["dec"] if d not in d_mine]
        dp = " ".join(f"段{k + 1} {100 * (tr[k]['p50'] / ref['p50'][k] - 1):+6.1f}%" for k in range(nP))
        s_mem = ('OK ' + '/'.join(fm(x['mem']) for x in tr)) if not mm else ('🚨' + str(mm) + str([x['mem'] for x in tr]))
        s_nsw = ('OK ' + '/'.join(str(x['nsw']) for x in tr)) if not nn else ('🚨' + str(nn) + str([x['nsw'] for x in tr]))
        s_dec = ('OK ' + (';'.join(d_mine) if d_mine else '无')) if not dd and not d_miss \
            else ('🚨本机多出' + str(dd) + ' 本机少了' + str(d_miss))
        print(f"    {a:<3} {pol:<9} 末态 {s_mem} | 迁次 {s_nsw} | 决策 {s_dec} | p50 vs r3 {dp}(跨进程 只作对账)")

# ---------------------------------------------------------------- 落点
V, why = verdict()
print("\n" + "=" * 150)
print("预设判决树落点(跑前写死, 跑完不许改)")
print("=" * 150)
print(f"  => 落点: {V}    理由: {why}")
print(f"     事前预测 = D(位次/时间爬升, 竞争假设 = 甲 引擎身份) => " + ("兑现" if V == "D" else "打脸(如实记账)"))
print("     🚨 这棵 r1 树(A/B/C/D)在 r1 已消耗 => 复跑只复读它的落点, 本轮真正的落点看下面 §5.2g 追加树")

V2, why2 = verdict_scale()
print("\n" + "=" * 150)
print("§5.2g 发数标度树落点(判据跑前写死在 docstring 与 OPTLOG 5.2g, 跑完不许改) —— 只读【累积底曲线】, 不读双峰/星标")
print("=" * 150)
print(f"  E1 k >= {E_K_HI} 或 末点底_pair <= {E_F_LO}%  => 甲(身份)否证 = 底是抽样误差 => 加发数(n≈{n_need}) 才能读延迟列")
print(f"  E2 k <  {E_K_LO} 且 末点底_pair >= {E_F_HI}% => 甲(身份)成立 => dyn 跨策略 p50 列永久作废")
print(f"  E3 其余 => 身份与噪声混叠 => 挂账, 只能读 >= 2x 底 的差")
print(f"  => 落点: {V2}    理由: {why2}")
print(f"     事前预测(跑前写死) = {PRED_DESC.get(PREDICT, '(PREDICT 未登记)')} => "
      + ("兑现" if V2 == PREDICT else "打脸(如实记账)"))
print("  判读: 🚨 k 只用【空对照组】算(A2/A3 对 A1), 与真实档无关 => 双峰臂多不影响它; 末点 = 全部计表轮;")
print("        E1 也不等于「省 20%~27%」被证实 —— 它只说明尺子磨得尖(靠加发数), 效应量仍需 <= 门槛的一半。")

print("""
判读提示: ① 底_pair 是【dyn 跨策略 p50 列】自己的真实噪声底 —— r3 里那 29 个百分点的极差到底是不是噪声, 看它;
        ② 底_序 大 = 正反两序没把位次消掉 => 甲/乙仍混叠, 不许把带内读成免疫, 也不许读成身份;
        ③ 双峰格一律不可判(同一把 BIM_THR=8%, 要 >=6 发), 双峰格的配对值不参与落点;
        ④ r3 那一列 p50 只作对账(不同进程不同批次), 绝不当分母; 它对不上先查闸门 1~7 再查口径;
        ⑤ 位次曲线读【正序与反序两条形状是否镜像】(镜像 = 位次; 两条都平而臂间有差 = 身份), 不是读它平不平;
        ⑥ C1 是正对照(已知约 2.6x): 它不过星标 = 这把尺子坏了, 本刀一切负读都作废。""")
print("[耗时] 建模+建臂+跑轮共 {:.0f}s | RSS={:.0f}MB 显存={:.0f}MB".format(
    time.perf_counter() - t_load, rss_mb(), torch.cuda.memory_allocated() / 1024 ** 2))

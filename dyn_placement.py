#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-② 动态专家放置: 运行时按【真流量】换 GPU 常驻专家, 不重录 CUDA Graph、不重分配。

【为什么这一刀不是"加功能", 而是补一个实测事实】
  P0-③补(换语料稳健性轮, results/2026-11-21_P0-③补_metrics_corpus-input_txt_bs512.txt,
  md5 0e0694aa) 量到: 同一个模型、同一路由、同一批 512 行真激活, 只把语料从本仓库
  readme 换成本机 input.txt(莎剧对白), 命中直方图就变了 ——
      readme: 命中%=[42.3,13.2,17.7,6.8,1.4,12.6,6.0,0.1] 最热/最冷 433.0x  A序[0,2,1,5,...]
      input : 命中%=[39.3,11.7,16.9,9.6,1.6,18.8,2.1,0.2] 最热/最冷 201.0x  A序[0,5,2,1,...]
  gc=3 那一档的 GPU 成员集合从 {0,2,1} 变成 {0,5,2}(要换 1<->5), gc>=5 才恰好重合。
  而【放错 vs 放对】的代价同档量级是 KL 5.9x(gc=3) / 20.9x(gc=5) / 309x(gc=7)。
  => 放置规则的输入(命中直方图)是【语料/流量的函数】, 离线在一份语料上选出来的成员
     到目标流量上可能就变成"放错"。静态放置把这次重定价省了, 动态放置在运行时付它。

【外部建议的原型】"forward 里统计每个专家命中, 每 N 帧检查: CPU 专家 >5% 提升进 GPU,
  GPU 专家 <1% 降级回 CPU, 只改调度层不重写 kernel。"
  —— 方向对, 三条前提都成立(放置=槽位下标切两半 / 路由跟着槽位走 / CPU 腿每帧现读
  权重指针)。但按字面实现会【抖】和【换不动】, 所以这一版补三件它没有的东西:
  1) 【容量约束】提升必须绑一个降级, 决策单元是【一进一出的一对槽位】。GPU 常驻集合
     的大小 gc 是显存预算, 不能"因为有个 CPU 专家热了就加进来"。
  2) 【成对判据 gain】"CPU>5%" 与 "GPU<1%" 是两条【独立】阈值: 流量漂移时最冷的 GPU
     专家常常还有 2%, 降级那一半永远等不到 => 要么永不换、要么单向往 GPU 里塞。
     容量约束下的正确比较是【相对比较】: 进来的要比出去的热 gain 倍(默认 3x)。
  3) 【滞回 + 冷却】刚被降下去的专家在 cooldown 帧内不许提回来, 两次迁移之间至少隔
     cooldown 帧。缺这条, 直方图在阈值附近摆动时每次检查都会撤掉上一刀(=来回抖动),
     而每次迁移都是真字节(见 swap 成本), 抖起来比不抖慢。

【为什么能不重录 CUDA Graph(这一刀的全部技术含量在这一条)】
  放置不是指针切换, 而是【列表下标切两半】: 槽位 0..gc-1 在 GPU, gc..E-1 在 CPU
  (bench 用 perm_src(order) 造初始放置, 见 bench_swap.py:250)。所以"迁移一个专家"
  = 把一个槽位的【门控行 + 三张权重 + int8 副本】整体换成另一个专家的东西。
  🚨 门控行必须跟着走: route() 出的是【槽位下标】, 下游按槽位取权重。只换权重不换
     门控行 = 分数说的是 A 专家、算的是 B 专家 => 静默错数, 且 idx 闸门当场就红。
  只要【缓冲地址不变、只改内容】, CUDA Graph 继续读同一批指针 => 新权重直接生效:
    · param.data.copy_() 在 torch 2.15 实测【不 bump _version】(而 no_grad copy_ /
      detach().copy_ 都会 bump) => _ensure_hot_weights 早退、_gkey=(bs,scope,_wver,dev)
      不变 => 既不重建 _wcat* 也不重录图。历史上"每帧重建 _wcat_gu 白付 4.9ms/帧"
      (hetero_moe_int8.py:601 的注释)就是这条不变量被破掉的样子 => 这里用 assert 钉死。
    · 🚨 前段图还 baked 了 pre_fused.gate_wt() 那份【(D,EP) 预转置门控】的地址
      (hetero_moe_int8.py:797 _gate_wt_dev)。它要是被重建成新张量, 图就读旧内存 =>
      路由停在换之前的专家。所以门控行交换要在【两份布局】上都就地做:
      gate.weight 的行 + m._gate_wt 的列。
    · CPU 腿相反, 每帧把 cpu_w_g/u/d + 三个 scale 当实参传进 C++
      (hetero_moe_int8.py:1039), data_ptr 每帧现读 => 就地写行天然安全。

【口径三条】
  1) 计数器读 m._pin["idx"](L2 档本帧路由的 D2H 落点): 零额外 D2H、零 CUDA 调用、
     零额外同步。所有策略(含 static_*)都跑同一份计数器 => 策略之间的 p50 差只反映
     【迁移决策】, 不反映记账成本。
  2) 命中占比记在【专家 eid 空间】, 不记在槽位空间: 迁移之后槽位的内容换了, 记在
     槽位上等于把两个专家的流量混进同一条历史(=又造了一个假直方图)。
  3) 迁移发生在计时窗口【之外】, 迁移自身成本单列(实测 swap_us, 不用常数估:
     方法论 22)。折合帧数 = 迁移总耗时 / 该阶段 p50 帧, 由读者自己定价。
"""
import time
from collections import deque

import torch

__all__ = ["slot_counts", "counts_from_engine", "HitWindow", "PlacementPolicy",
           "ExpertSwapper", "DynPlacer", "POLICY_PRESETS", "policy_from_name"]

# 外部建议的原型阈值(逐字照抄, 便于"照做的那一档"可复现)
NAIVE = dict(promote_min=0.05, demote_max=0.01, gain=0.0, cooldown=0, max_swaps=1)
# 补了容量判据 + 滞回的那一档
HYSTERESIS = dict(promote_min=0.05, demote_max=0.01, gain=3.0, cooldown=16, max_swaps=1)
# P1-②d: 判据换成【预测 KL 贡献】= 命中 x int8 误差单价。阈值口径整个换掉:
#   promote_min/demote_max 置 0 = 【不再用绝对占比阈值】(P1-②/②b 实测: demote_max=1% 在
#   真漂移下把降级那一半永久锁死, hysteresis 那一档三段一次都没开火 = 判据病根),
#   换成【成对价杆 gain】+【预测 KL 降幅门槛 thr_rel】: 换一次迁移真花 ~15ms(=3~4 帧),
#   所以"预测只降 0.5%"的决定就不该做 —— thr_rel 是把迁移价签写进判据的那一项。
PRICE = dict(promote_min=0.0, demote_max=0.0, gain=1.0, cooldown=16, max_swaps=1, thr_rel=0.05)
POLICY_PRESETS = {
    "static_A": dict(enabled=False),          # 上帝视角放对(整本语料的直方图) —— 上界
    "static_C": dict(enabled=False),          # 放错 —— 下界
    "neutral": dict(enabled=False),           # 中立序(eid 序)不动 = 【没做动态放置】的对照格
    "shuffle": dict(enabled=False),           # 打乱序不动 = 真【没做动态放置】的对照格:
    #   🚨 本机真源上 eid 序的热专家恰好就是 {0,1,2} => neutral 的 GPU 成员集与 static_A
    #      【同一个集合】(槽内换序不改集合) => neutral 那一行 KL 与 static_A 逐位相同,
    #      只能当占位/自检, 不能当对照。shuffle 是固定种子打乱且保证 GPU 成员集既 != A
    #      也 != C 的那一档, 它才是【起跑时不知道热序、且不做动态放置】的真对照。
    "hyst_1x": dict(enabled=True, **dict(HYSTERESIS, gain=1.0)),  # 只要严格更热就换(gain 消融的下刻度)
    "hyst_2x": dict(enabled=True, **dict(HYSTERESIS, gain=2.0)),  # gain 消融的中刻度(2x 才换)
    "static_p1": dict(enabled=True, phase_limit=1, **HYSTERESIS),   # 启动期自适应, 之后冻结
    "naive": dict(enabled=True, **NAIVE),
    "hysteresis": dict(enabled=True, **HYSTERESIS),
    # ---- P1-②d: 同一副骨架(容量约束 + 冷却 + 一进一出), 只换【比较的量】和【价的来源】
    # price_mode: frozen = 直接用 u_e(单专家扰动的 KL, 已含它自己的流量) —— 不再乘窗内占比
    #             hit    = kappa_e(=u_e/hit_e, 与流量无关的专家属性) x 窗内实测占比
    #                      => 离线只标"专家有多脆", 流量形状由【在线计数器】现给
    # price_src : calib  = 离线标定段(PRICE_CAL, 默认 readme)的价 = 上线【真拿得到】的东西
    #             oracle = 本段自己的真价 = 上帝价 => 它是"在线估若免费"的【上界锚点】,
    #                      本身不可上线(用了未来流量)。在线估(影子探针)若做, 目标是追平它。
    "price_cal": dict(enabled=True, price_mode="frozen", price_src="calib", **PRICE),
    "price_hit": dict(enabled=True, price_mode="hit", price_src="calib", **PRICE),
    "price_orc": dict(enabled=True, price_mode="frozen", price_src="oracle", **PRICE),
}


def policy_from_name(name, gc, **over):
    """按短名造策略。over 用来在判决表里做消融(例: gain=0 单拆一条腿)。"""
    cfg = dict(POLICY_PRESETS.get(name, {}))
    cfg.update(over)
    en = cfg.pop("enabled", True)
    cfg.pop("phase_limit", None)
    cfg.pop("price_src", None)
    return PlacementPolicy(gc, name=name, enabled=en, **cfg)


# ---------------------------------------------------------------- 计数器
def slot_counts(idx, E):
    """topk 索引 (bs,top_k) -> 每个【槽位】的命中次数(CPU int64, 不碰 CUDA)。"""
    flat = idx.reshape(-1)
    if flat.device.type != "cpu":
        raise RuntimeError("计数器只吃 CPU 索引(pinned 缓冲本来就是 CPU)")
    return torch.bincount(flat.long(), minlength=E)[:E]


def counts_from_engine(m, E=None):
    """读【刚跑完那一帧】的路由计数。
    L2 档 forward 末尾有 current_stream().synchronize(), 返回时本帧 D2H 已落地、
    worker 也已用它算完 => 此刻读 pinned 缓冲既正确又免费。
    🚨 窗口: 只能在 forward 之后、下一帧之前读(下一帧的 D2H 会覆写它)。"""
    p = getattr(m, "_pin", None)
    if p is None or p.get("idx") is None:
        raise RuntimeError("m._pin['idx'] 还没建: 先跑一帧 L2 再开计数器")
    return slot_counts(p["idx"], E or m.num_experts)


class HitWindow:
    """滑窗命中占比, 记在【专家 eid 空间】(见模块口径 2)。"""

    def __init__(self, E, window=24):
        self.E, self.window = int(E), max(1, int(window))
        self.buf = deque(maxlen=self.window)
        self.seen = 0

    def push(self, cnt_slot, slot2eid):
        v = [0.0] * self.E
        for s, eid in enumerate(slot2eid):
            v[eid] = float(cnt_slot[s])
        self.buf.append(v)
        self.seen += 1

    def frac(self):
        """窗内占比(分母 = 窗内总命中数) => 与帧数/bs 无关, 跨阶段可比。合计 = 1。"""
        if not self.buf:
            return [0.0] * self.E
        tot = [sum(r) for r in zip(*self.buf)]
        s = sum(tot)
        return [t / s for t in tot] if s > 0.0 else [0.0] * self.E

    @property
    def ready(self):
        return len(self.buf) >= self.window

    def reset(self):
        self.buf.clear()


# ---------------------------------------------------------------- 决策
class PlacementPolicy:
    """容量约束下的成对换入换出策略。纯函数式(不碰引擎), 所以能单测。

    decide() 返回 [(i_gpu, j_cpu), ...] —— 一条都没有就返回 []。
    两条触发规则(满足其一才动手):
      rule_a 外部原型: 进来的 CPU 专家 > promote_min 且 最冷的 GPU 专家 < demote_max
      rule_b 容量判据: 进来的 > promote_min 且 进来的 >= gain x 最冷的 GPU 专家
      rule_p 价判据(price_mode != "off" 时【独占】, 上面两条不再参与): 比较【预测 KL 贡献】
             val_e = u_e(frozen) 或 kappa_e x 窗内占比(hit); 进来的 >= gain x 出去的
             且 (v_in - v_out)/Σ_CPU val > thr_rel。详见 _decide_price 注释。
    cooldown: 距上次迁移不足 cooldown 帧 -> 不动(滞回)。
    """

    def __init__(self, gc, promote_min=0.05, demote_max=0.01, gain=3.0, cooldown=0,
                 max_swaps=1, enabled=True, name="custom", thrash_k=4,
                 price_mode="off", thr_rel=0.0):
        self.gc = int(gc)
        self.promote_min = float(promote_min)
        self.demote_max = float(demote_max)
        self.gain = float(gain)
        self.cooldown = int(cooldown)
        self.max_swaps = max(1, int(max_swaps))
        self.enabled = bool(enabled)
        self.name = name
        self.thrash_k = int(thrash_k)
        self.price_mode = str(price_mode)     # off | frozen(u_e) | hit(kappa_e x 窗内占比)
        self.thr_rel = float(thr_rel)         # 预测 KL 降幅门槛: 低于它不付迁移钱
        self.price = None                     # list[E], 按【eid】索引; None = 无价 => 沉默
        self.last_frame = -10 ** 9
        self.banned = {}                      # eid -> 解禁帧(刚降下去的不许立刻回来)
        self.hist = deque(maxlen=max(1, thrash_k))     # 最近几次的 (进, 出)
        self.n_swap = self.n_thrash = self.n_rule_a = self.n_rule_b = 0
        self.n_rule_p = 0                     # rule_p(价判据)开火次数
        self.reasons = []

    def _locked(self, eid, frame):
        """这个专家还在冷却期里吗? `banned[eid]` 存的是【解禁帧】, 必须拿它跟当前帧比。

        🚨只查 key 的存在性 = 把限时禁令过成了永久禁令: 剪枝原先只在 `note_swap` 里做,
           于是"最后一次迁移之后再没有迁移"的那些策略, 被它降下去的那个人再也提不回来。
           判决表 `3ce6836d` 的 price_orc@drift 就是这么被锁死的(那一格 gap 8.0% 明明
           还有得赚, 而唯一能赚的那个人 eid3 正是它在阶段2 刚踢出去的人) => 它因此不是
           一个干净的"上帝价上界锚点"。
        """
        exp = self.banned.get(eid)
        return exp is not None and exp > frame

    def decide(self, frac, slot2eid, frame=0):
        if not self.enabled:
            return []
        E = len(slot2eid)
        if self.gc <= 0 or self.gc >= E:
            return []                      # 没有"两半"可换 = 没有放置自由度
        if frame - self.last_frame < self.cooldown:
            return []
        # 顺手剪掉已过期的解禁记录(它只被 _locked 读过, 剪它不影响任何决策)
        self.banned = {k: v for k, v in self.banned.items() if v > frame}
        if self.price_mode != "off":
            # 价判据独占一条腿: 命中阈值(promote_min/demote_max)不参与 —— 再加一道
            # 绝对阈值就变回"三条件与门", 那正是 P1-② 实测到的"永远等不到降级"病根。
            return self._decide_price(frac, slot2eid, frame)
        f = [frac[eid] for eid in slot2eid]          # 槽位空间的占比
        used_i, used_j, out = set(), set(), []
        for _ in range(self.max_swaps):
            gpu = [s for s in range(self.gc) if s not in used_i]
            cpu = [s for s in range(self.gc, E) if s not in used_j
                   and not self._locked(slot2eid[s], frame)]
            if not gpu or not cpu:
                break
            vi = min(gpu, key=lambda s: f[s])        # 最冷的 GPU 槽位
            pj = max(cpu, key=lambda s: f[s])        # 最热的 CPU 槽位
            hot, cold = f[pj], f[vi]
            ra = hot > self.promote_min and cold < self.demote_max
            rb = (self.gain > 0.0 and hot > self.promote_min
                  and hot >= self.gain * max(cold, 1e-12))
            if not (ra or rb):
                break
            out.append((vi, pj))
            used_i.add(vi)
            used_j.add(pj)
            self.reasons.append(f"f{frame} {slot2eid[pj]}->{slot2eid[vi]} "
                                f"hot={hot:.3f} cold={cold:.3f} "
                                f"{'rule_a' if ra else 'rule_b'}")
            self.n_rule_a += int(ra)
            self.n_rule_b += int(not ra)
        return out

    def _decide_price(self, frac, slot2eid, frame):
        """rule_p: 容量约束下比较【预测 KL 贡献】而不是【命中占比】(P1-②c 的产物)。

        val_e = price[e] x (窗内占比 if price_mode=="hit" else 1.0)
          frozen -> price = u_e: P1-②c 的"只有 e 在 CPU 腿走 int8"那一档的 KL, 它本身
                   已经按 e 自己的流量加权过了(实测 u_e/hit_e = kappa_e 谱 150x, 比流量谱
                   陡得多) => 不再乘窗内占比, 乘了是把流量算两遍。
          hit    -> price = kappa_e(与流量无关的专家属性) x 窗内实测占比: 离线只标专家本体,
                   流量形状交给在线计数器 => 真漂移时价会跟着流量重定价。
        为什么这条腿有资格当判据(P1-②c 实测, 0 自由参数): M1 = Σ_{e∈CPU} u_e 对 23 个
        归档成员集格的段内 Spearman 三段全 +1.00、预测/实测比 0.92~1.11 —— 而"覆盖流量%"
        那根旧杆在漂移段只有 +0.60 且会把 CPU 腿只剩一个专家时低估 6.7 倍。
        开火要同时满足两条:
          v_in >= gain x v_out                       成对价杆(容量约束下的相对比较)
          (v_in - v_out) / Σ_CPU val > thr_rel       预测 KL 降幅; 迁移一次 ~15ms = 3~4 帧,
                                                     降幅低于价签的决定就不该做
        🚨 价向量缺失(price is None)时【沉默】而不是退回命中判据: 退回会让同一策略名
           在有无价表时是两个东西, 表里读不出差别 => 缺失由 bench 侧闸门⑧ 点名。
        """
        if self.price is None or len(self.price) < len(slot2eid):
            return []
        E = len(slot2eid)
        hit_w = self.price_mode == "hit"
        vslot = [float(self.price[eid]) * (frac[eid] if hit_w else 1.0) for eid in slot2eid]
        used_i, used_j, out = set(), set(), []
        for _ in range(self.max_swaps):
            gpu = [s for s in range(self.gc) if s not in used_i]
            cpu = [s for s in range(self.gc, E) if s not in used_j
                   and not self._locked(slot2eid[s], frame)]
            if not gpu or not cpu:
                break
            vi = min(gpu, key=lambda s: vslot[s])      # GPU 腿里最便宜的槽位
            pj = max(cpu, key=lambda s: vslot[s])      # CPU 腿里最贵的槽位
            v_in, v_out = vslot[pj], vslot[vi]
            tot = sum(vslot[s] for s in range(self.gc, E))     # = 当前成员集的 M1 预测 KL
            drop = (v_in - v_out) / tot if tot > 0.0 else 0.0
            if not (v_in >= self.gain * v_out and drop > self.thr_rel):
                break
            out.append((vi, pj))
            used_i.add(vi)
            used_j.add(pj)
            self.reasons.append(f"f{frame} {slot2eid[pj]}->{slot2eid[vi]} "
                                f"v_in={v_in:.3e} v_out={v_out:.3e} "
                                f"\u9884\u6d4b\u964d={100.0 * drop:.1f}% "
                                f"rule_p({self.price_mode})")
            self.n_rule_p += 1
        return out

    # 迁移落地后由调用方回调(纯统计, 不参与决策, 所以放在这里不会形成反馈环)
    def note_swap(self, eid_in, eid_out, frame, cooldown_for=None):
        self.last_frame = frame
        cd = self.cooldown if cooldown_for is None else int(cooldown_for)
        if cd > 0:
            self.banned[eid_out] = frame + cd
        self.banned = {k: v for k, v in self.banned.items() if v > frame}
        if (eid_out, eid_in) in self.hist:
            self.n_thrash += 1
        self.hist.append((eid_in, eid_out))
        self.n_swap += 1


# ---------------------------------------------------------------- 执行(就地换入换出)
class ExpertSwapper:
    """把槽位 i(GPU 腿)与槽位 j(CPU 腿)的【内容】整体互换。

    只改内容、不改地址: 换完之后 CUDA Graph 照原样 replay 就读到新权重。
    换的内容一共五处(漏一处就是静默错数, 所以逐条列在这里):
      1) gate.weight 的行 i<->j (+ gate.bias 同序换) —— 路由分数必须跟着专家走
         (route 出的是槽位下标)。🚨 bias 也是【每槽位一格】, 换行不换 bias = 半换;
          真源那条路 gate_b 恒 0(real_smollm.real_src) 所以看不出来, 玩具权重
          (nn.Linear 默认随机 bias)一换就显形 —— 红线 J 段就是拿玩具权重钉这一条的。
      2) m._gate_wt 的列 i<->j       —— 前段图 baked 的是这份预转置副本的地址
      3) GPU 侧存储权重 gpu_experts/up/down[i] <- 槽位 j 的专家(fp32 存储, dtype 无关)
      4) 设备热权重缓冲的【槽位块】: _wcat 行块 / _wstack[i] / _wcat_gu 列块 /
         _wstack_gu[i] / _wstack_d[i] —— 图读的就是这些, 就地 copy_
      5) CPU 侧 fp32 转置存储 + int8 副本(行) + 三个 scale, 量化算式与 ctor 逐字同
    另外要清的缓存: _cpu_w_dev_cache/_cpu_wu_dev/_cpu_wd_dev(裁判的 GPU 副本)、_wdt
    (sparse 档按 dtype 的半精缓存)。不清 = 下一帧裁判/稀疏档读到旧专家。
    """

    def __init__(self, m, src_experts, check=True, refresh_dead=True, order=None,
                 src_gate=None):
        self.m = m
        self.src = src_experts
        self.src_gate = src_gate      # 未置换的 gate_w(eid 行序); 给了才能做闸门⑥ 的门控行核对
        self.check = bool(check)
        self.refresh_dead = bool(refresh_dead)   # False = 只刷"图真读的那几张"(成本下界)
        # 🚨 起始 slot2eid 必须 = 引擎【真实的】槽位->专家映射。bench 用 perm_src(order) 建引擎
        #    (order[槽位] = 专家), 若这里硬写 range(E), static_A/static_C 两档的直方图会记到
        #    错的专家身上 = 拿假直方图做决策, 且 hot_members 报假放置(与 swap 回写方向那条同族)。
        #    src 永远是【未置换的基表】 => 迁移换入换出的内容按 eid 取, 与起始序无关。
        self.slot2eid = list(range(m.num_experts)) if order is None else [int(v) for v in order]
        assert sorted(self.slot2eid) == list(range(m.num_experts)), f"起始序不是置换: {self.slot2eid}"
        self.n_swap = 0
        self.total_us = 0.0
        self.bytes_last = 0
        self.bytes_total = 0
        self._nb = 0
        # §5.2h 分相计时: 键集与 AsyncSwapper.phase_us() 同表(缺的键由打印口印 "-")。
        # 🚨 只插 time.perf_counter() 时间戳, 换权重的语句逐字未动 => 加计时不改行为。
        self.phase_tot = {}
        self.phase_n = 0

    # ---- 内部: 记账式的就地写
    def _cp(self, dst, src):
        self._nb += dst.numel() * dst.element_size() + src.numel() * src.element_size()
        dst.copy_(src.to(dst.dtype) if src.dtype != dst.dtype else src)

    def _mats(self, eid):
        e = self.src[eid]
        return e["g"], e["u"], e["d"]

    def _requant(self, w, q_s8, q_u8, scales, row):
        """与 hetero_moe_int8.py ctor 逐字同的量化算式(对【已转置存好】的那张算)。
        🚨 q_s8/q_u8 传进来【已经是那一行】(cpu_experts_q_s8[jc] 是 2-D 行视图),
           所以这里只能整行 copy_ —— 再 [row] 一次会落到那行的第 jc 个子行: 形状对不上
           当场抛(运气好), 运气不好就是把别的专家的行覆写掉(静默错数)。
           scales 传的才是整条 1-D 向量, 所以只有它需要 [row]。"""
        sc = max((w.abs().max() / 127.0).item(), 1e-8)
        q = (w / sc).round().clamp(-127, 127)
        self._nb += w.numel() * 4 + q.numel() * (1 + (1 if q_u8 is not None else 0))
        assert q_s8.shape == q.shape, f"q_s8 应是行矩阵 {tuple(q.shape)}, 实得 {tuple(q_s8.shape)}"
        q_s8.copy_(q.to(torch.int8))
        if q_u8 is not None:
            assert q_u8.shape == q.shape, f"q_u8 应是行矩阵 {tuple(q.shape)}, 实得 {tuple(q_u8.shape)}"
            q_u8.copy_((q + 128).to(torch.uint8))
        scales[row] = sc

    def _refresh_gpu(self, i, g, u, d):
        m, N = self.m, self.m.expert_size
        gu = None
        if m._wcat is not None:
            h = g.shape[0]
            self._cp(m._wcat[i * h:(i + 1) * h], g)
        if m._wstack is not None and self.refresh_dead:
            self._cp(m._wstack[i], g)
        if m._wcat_gu is not None:
            gu = torch.cat([g, u], 1)
            self._cp(m._wcat_gu[:, i * 2 * N:(i + 1) * 2 * N], gu)
        if m._wstack_gu is not None and self.refresh_dead:
            self._cp(m._wstack_gu[i], gu if gu is not None else torch.cat([g, u], 1))
        if m._wstack_d is not None:
            self._cp(m._wstack_d[i], d)

    def swap(self, i, j, sync=True):
        """把 GPU 槽位 i 与 CPU 槽位 j 的专家整体互换。返回一个可打印的记录 dict。"""
        m = self.m
        gc = m.gpu_expert_count
        assert i < gc <= j, f"槽位要一个在 GPU 腿一个在 CPU 腿: i={i} j={j} gc={gc}"
        eid_in, eid_out = self.slot2eid[j], self.slot2eid[i]
        g_in, u_in, d_in = self._mats(eid_in)
        g_out, u_out, d_out = self._mats(eid_out)
        # 🚨 换入换出的内容全部来自【源权重表 self.src】, 不是从引擎里倒手 => 不存在
        #    "先被覆写再被读"的次序问题; 也保证换出去的专家可以无限次换回来且逐位可复现。
        v_before = self._versions()
        ptr_before = self._ptrs()
        t_front = time.perf_counter()
        if sync:
            torch.cuda.synchronize()
        self._nb = 0                      # 🚨 每次迁移单独记字节(不清零 => bytes_last 是累计值)
        t0 = time.perf_counter()
        # 1) 门控行(两份布局都要动)
        gw, gb = m.gate.weight.data, m.gate.bias.data
        self._nb += 2 * gw.shape[1] * 4
        tmp = gw[i].clone()
        gw[i].copy_(gw[j])
        gw[j].copy_(tmp)
        if gb.numel():
            tb = gb[i].clone()
            gb[i].copy_(gb[j])
            gb[j].copy_(tb)
        if getattr(m, "_gate_wt", None) is not None:
            gt = m._gate_wt                      # (D, EP) 预转置副本, 列 = 槽位
            tg = gt[:, i].clone()
            self._cp(gt[:, i], gt[:, j].clone())
            self._cp(gt[:, j], tg)
        t_gate = time.perf_counter()
        # 2) GPU 侧存储权重(永远 fp32) —— .data.copy_ 不 bump _version(模块 docstring)
        self._cp(m.gpu_experts[i].data, g_in)
        self._cp(m.gpu_up[i].data, u_in)
        self._cp(m.gpu_down[i].data, d_in)
        # 3) 设备热权重缓冲(图读的那几张)
        self._refresh_gpu(i, g_in, u_in, d_in)
        t_gpu = time.perf_counter()
        # 4) CPU 腿: fp32 转置存储 + int8 副本 + scale
        jc = j - gc
        self._cp(m.cpu_experts_t[jc], g_out.t())
        self._requant(m.cpu_experts_t[jc], m.cpu_experts_q_s8[jc],
                      m.cpu_experts_q[jc] if m.w_pack == "biased" else None,
                      m.cpu_experts_scale, jc)
        if self.m.kind == "swiglu":
            self._cp(m.cpu_up_t[jc], u_out.t())
            self._requant(m.cpu_up_t[jc], m.cpu_up_q[jc], None, m.cpu_up_scale, jc)
            self._cp(m.cpu_down_t[jc], d_out.t())
            self._requant(m.cpu_down_t[jc], m.cpu_down_q[jc], None, m.cpu_down_scale, jc)
        t_cpu = time.perf_counter()
        # 5) 清派生缓存(裁判的 GPU 副本 / sparse 档半精缓存)
        m._cpu_w_dev_cache = None
        m._cpu_wu_dev = None
        m._cpu_wd_dev = None
        if getattr(m, "_wdt", None):
            m._wdt.clear()
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        us = (t_end - t0) * 1e6
        for _k, _v in (("front_sync", t0 - t_front), ("gate", t_gate - t0),
                       ("gpu_live", t_gpu - t_gate), ("cpu_leg", t_cpu - t_gpu),
                       ("tail_sync", t_end - t_cpu), ("total", t_end - t_front)):
            self.phase_tot[_k] = self.phase_tot.get(_k, 0.0) + _v * 1e6
        self.phase_n += 1
        # 🚨 槽位留下【换入】的那个专家: GPU 槽位 i 刚被写进 eid_in 的权重, CPU 槽位 j
        #    刚被写进 eid_out 的 => 回写方向反了 = 记账与实际内容相反, 下一窗口的
        #    直方图会记到错的专家身上(= 拿假直方图做决策), 且 hot_members 报假的放置。
        #    红线 J 段直接拿 gpu_experts[i] 与 src[slot2eid[i]] 逐位对, 钉死这一条。
        self.slot2eid[i], self.slot2eid[j] = eid_in, eid_out
        self.n_swap += 1
        self.total_us += us
        self.bytes_last = self._nb
        self.bytes_total += self._nb
        if self.check:
            assert self._versions() == v_before, "🚨 迁移 bump 了 _version => 会重录 CUDA Graph"
            assert self._ptrs() == ptr_before, "🚨 迁移换了缓冲地址 => 图在读旧内存"
        return dict(us=us, mb=self._nb / 2 ** 20, slot_in=i, slot_out=j,
                    eid_in=eid_in, eid_out=eid_out)

    def reset_prof(self):
        """清分相累计(价签测在计时段开头调一次, 把建臂/冷发那几发排除掉)。"""
        self.phase_tot = {k: 0.0 for k in self.phase_tot}
        self.phase_n = 0
        if hasattr(self, "flips"):            # AsyncSwapper 才有 flips(sync 版没有设备 Event)
            self.flips = []

    def phase_us(self):
        """每发均值 us。sync 版没有侧 stream, 故 h2d_enq/d2d_enq 键压根不存在 => 打印口印 "-"。"""
        n = max(1, self.phase_n)
        return {k: v / n for k, v in self.phase_tot.items()}

    swap_us = swap

    def _versions(self):
        m = self.m
        v = tuple(p._version for p in m.gpu_experts)
        if m.kind == "swiglu":
            v += tuple(p._version for p in m.gpu_up) + tuple(p._version for p in m.gpu_down)
        return v + (m.gate.weight._version,)

    def _ptrs(self):
        m = self.m
        p = {}
        for nm in ("_wcat", "_wstack", "_wcat_gu", "_wstack_gu", "_wstack_d", "_gate_wt"):
            t = getattr(m, nm, None)
            p[nm] = None if t is None else t.data_ptr()
        for i in range(m.gpu_expert_count):
            p[f"g{i}"] = m.gpu_experts[i].data_ptr()
            p[f"gu{i}"] = m.gpu_up[i].data_ptr()
            p[f"gd{i}"] = m.gpu_down[i].data_ptr()
        for nm, lst in (("cw", m.cpu_experts_q_s8), ("cu", m.cpu_up_q), ("cd", m.cpu_down_q)):
            p[nm] = lst.data_ptr()
        return p

    def hot_members(self):
        gc = self.m.gpu_expert_count
        return [self.slot2eid[i] for i in range(gc)]

    @staticmethod
    def _same(got, want):
        """记账与内容的逐位对比, 但【按缓冲自己的 dtype】比。
        🚨 为什么要这一层: 乙-2c 之后 _wcat/_wcat_gu/_wstack_d 这五张设备缓冲在
           gpu_dtype=bf16 时是【半精副本】(hetero_moe_int8._sync_hot_weights 最后一步
           整体 .to(_dt)), 而 src 表是 fp32 —— 直接 torch.equal(半精, fp32) 在 bf16 引擎上
           【必然假红】(2026-11-21 红线 J 段第一次跑就红在这里, 报的三张全是设备缓冲)。
           真口径是: 引擎建它是【直接转 dtype】, 迁移写它也是【直接转 dtype】(见 _cp),
           两边同一种舍入 => 把 fp32 源表转成同一 dtype 后必须【逐位】相同。
           所以这条仍然是逐位闸门, 不是精度闸门(它不判 bf16 准不准, 那只归 KL/top-1)。"""
        got = got.cpu()
        return torch.equal(got, want.to(got.dtype))

    def check_content(self):
        """闸门⑥: 记账(slot2eid)必须与【引擎实际内容】逐位一致 —— 五处布局全查。
        回写方向反了 / 某处布局漏改, 在这一条就会红(它比输出逐位更便宜也更早: 输出逐位
        要靠"按新序重建一个引擎"才比得了, 那条放在 bench_dyn 闸门②)。
        返回【不符项】列表, 空 = 全绿。"""
        m, gc, bad = self.m, self.m.gpu_expert_count, []
        for s, eid in enumerate(self.slot2eid):
            g, u, d = self._mats(eid)
            if self.src_gate is not None:
                if not self._same(m.gate.weight.data[s], self.src_gate[eid]):
                    bad.append(f"gate.weight行{s}!=eid{eid}")
                if getattr(m, "_gate_wt", None) is not None and not self._same(
                        m._gate_wt[:, s], self.src_gate[eid]):
                    bad.append(f"_gate_wt列{s}!=eid{eid}")
            if s < gc:
                for nm, got, want in (("gpu_experts", m.gpu_experts[s].data, g),
                                      ("gpu_up", m.gpu_up[s].data, u),
                                      ("gpu_down", m.gpu_down[s].data, d)):
                    if not self._same(got, want):
                        bad.append(f"{nm}[{s}]!=eid{eid}")
                if m._wcat is not None:
                    h = g.shape[0]
                    if not self._same(m._wcat[s * h:(s + 1) * h], g):
                        bad.append(f"_wcat块{s}!=eid{eid}")
                if m._wcat_gu is not None and not self._same(
                        m._wcat_gu[:, s * 2 * m.expert_size:(s + 1) * 2 * m.expert_size],
                        torch.cat([g, u], 1)):
                    bad.append(f"_wcat_gu块{s}!=eid{eid}")
                if m._wstack_d is not None and not self._same(m._wstack_d[s], d):
                    bad.append(f"_wstack_d[{s}]!=eid{eid}")
            else:
                jc = s - gc
                for nm, got, want in (("cpu_experts_t", m.cpu_experts_t[jc], g.t()),
                                      ("cpu_up_t", m.cpu_up_t[jc], u.t()),
                                      ("cpu_down_t", m.cpu_down_t[jc], d.t())):
                    if not torch.equal(got.cpu(), want):
                        bad.append(f"{nm}[{jc}]!=eid{eid}")
                for nm, q, w, scl in (("cpu", m.cpu_experts_q_s8[jc], g.t(), m.cpu_experts_scale),
                                      ("up", m.cpu_up_q[jc], u.t(), m.cpu_up_scale),
                                      ("down", m.cpu_down_q[jc], d.t(), m.cpu_down_scale)):
                    sc = max((w.abs().max() / 127.0).item(), 1e-8)
                    if abs(float(scl[jc]) - sc) > 1e-12:
                        bad.append(f"{nm}_scale[{jc}]!=eid{eid}")
                    if not torch.equal(q.cpu(), (w / sc).round().clamp(-127, 127).to(torch.int8)):
                        bad.append(f"{nm}_int8[{jc}]!=eid{eid}")
        return bad


# --------------------------------------------------------- 执行(§5.2h 异步迁移 = 选项 B)
class AsyncSwapper(ExpertSwapper):
    """§5.2h: 把一次迁移拆成【plan = 侧 stream 把字节搬进 staging】+【flip = 主 stream 提交】,
    让前台不为迁移 stall。起因是价签而不是双峰: 本文件顶上那句「换一次实测 ~15ms = 3~4 帧」
    是 sync 版 swap() 的墙钟, 而 thr_rel 的滞回开关点就按它标定 -> 价签错 = 开关点错。

    sync 版那 15ms 由三样混成, 它们【该不该付】是三个不同的问题:
      (1) 前后各一个 torch.cuda.synchronize() = device-wide 等上一帧排空 => 真·可消除的同步;
      (2) 约 6 次【pageable 主机】H2D copy_ => CUDA 内部先 sync 再边拷边发 = 同步源;
      (3) CPU 腿的 .t().contiguous() + 3 次 requant(abs/max/div/round/clamp/int8(+u8) over
          884k 元素) = 纯 CPU 计算(它不是设备同步 —— _requant 里那个 .item() 落在 CPU 张量上,
          这是本轮读代码时更正的一条事实), 可按专家预先算好。
    本类把 (1) 去掉、(2) 换成 pinned + 侧 stream、(3) 换成按专家预存的表 => 只剩纯 memcpy。

    🚨 两条设计约束(踩了就是静默错数, 比慢一百倍严重):
      A) 图永不许看到 torn 权重: 所有【写进 live 缓冲】的 D2D 一律排在主 stream 上, 于是自动
         排在"上一帧图 replay"之后、"下一帧图 replay"之前 => 不需要任何"swap 语义"。侧 stream
         只许碰 staging, 一个 live 字节都不许碰。
      B) CPU 腿的写必须落在【没有 CPU kernel 在跑】的时刻: 本 harness 全程 d2h_wait="main"
         => fwd() 返回时 CPU 腿已跑完 => flip 时写它是静默点(红线 R4 就是钉这一条的)。
         换 d2h_wait="worker" 必须先重开 R4, 不许默认它仍然成立。

    不可消除的下限: C++ 入口 C 每次收【整块 stacked 张量】(cpu_w_g/cpu_up_q/cpu_down_q + 三个
    scale 向量 + gpu_expert_count), 所以 CPU 腿只能【写行】, 不能交换指针 -> 每次迁移至少
    3 份 fp32 转置 + 3 份 int8 的 host->host memcpy。异步消掉的是【同步】与【重复计算】,
    消不掉这些字节。

    参数: stage_ahead = plan 提前一整帧做(把 H2D 藏进那一帧的计算里, 代价 = 一帧 stale =
      那一帧仍按旧成员算) => 它省的是【设备时间线】, 不是 host stall, 两笔账必须分开记;
      prebuild = 一次性把所有专家的 pinned/预量化表建好(价签测必开, 否则建表成本会混进被计时的
      flip 里; 它自己的代价 = 常驻 pinned 字节, 单独印在价签上); prof = 分相计时 + 每发设备 Event。
    """

    def __init__(self, m, src_experts, check=True, refresh_dead=True, order=None,
                 src_gate=None, stage_ahead=False, prebuild=True, prof=True):
        super().__init__(m, src_experts, check=check, refresh_dead=refresh_dead,
                         order=order, src_gate=src_gate)
        self.stage_ahead = bool(stage_ahead)
        self.prof = bool(prof)
        self.stream = torch.cuda.Stream()          # 侧 stream: 只碰 staging, 永不碰 live
        self.ev_stage = torch.cuda.Event()
        self.ev_flip = torch.cuda.Event()
        self.phase_tot = dict(front_sync=0.0, h2d_enq=0.0, d2d_enq=0.0, gate=0.0,
                              cpu_leg=0.0, total=0.0)
        self.phase_n = 0
        self.flips = []                            # 每发: dict(us, dev_ev, ...)
        D, N = int(m.gate.weight.shape[1]), int(m.expert_size)
        self.D, self.N = D, N
        self.st32 = self._mk_staging(D, N, torch.float32)      # gpu_experts/up/down 是 fp32
        self.stl = self.st32 if m._dt is torch.float32 else self._mk_staging(D, N, m._dt)
        self.st_tmp = torch.empty(max(D, N), device="cuda")
        self.pin32, self.pinl, self.cpu_tab, self.pin_gate = {}, {}, {}, {}
        self.staged = None                         # staging 当前装的是哪个 eid
        self.pinned_mb = 0.0
        # 门控行/bias 也改成【按 eid 写绝对值】而不是 i<->j 互换: 同 stream 上互换本来也安全,
        # 但绝对值写不依赖"gw[i] 当前正好等于 eid_out"这条隐式论证, 且能复用同一份 pinned 表。
        self.gate_abs = src_gate is not None
        if prebuild:
            for eid in range(m.num_experts):
                self._tab(eid)

    # ------------------------------------------------ 表(全部按【专家 eid】存, 与槽位无关)
    def _mk_staging(self, D, N, dt):
        return (torch.empty(D, N, device="cuda", dtype=dt),
                torch.empty(D, N, device="cuda", dtype=dt),
                torch.empty(N, D, device="cuda", dtype=dt))

    def _pin(self, ts):
        out = []
        for t in ts:
            p = t.contiguous().pin_memory()
            self.pinned_mb += p.numel() * p.element_size() / 2.0 ** 20
            out.append(p)
        return tuple(out)

    def _tab(self, eid):
        """建(或取)第 eid 个专家的全套预存表。逐位对齐 sync 版靠两条:
        (a) pinned 里放的是【目标缓冲自己 dtype 的字节】=> H2D 是纯 memcpy, 与 sync 的
            src.to(dtype) 落在同一个舍入点(半精引擎才分叉, fp32 引擎下 stl 就是 st32);
        (b) CPU 腿那份按 ctor 的量化算式【逐字同】预先算好, 所以 flip 只剩 memcpy。"""
        if eid in self.cpu_tab:
            return
        g, u, d = self._mats(eid)
        self.pin32[eid] = self._pin((g.float(), u.float(), d.float()))
        if self.stl is not self.st32:
            self.pinl[eid] = self._pin((g.to(self.m._dt), u.to(self.m._dt), d.to(self.m._dt)))
        if self.gate_abs:
            self.pin_gate[eid] = self._pin((self.src_gate[eid].float(),))
        gt, ut, dt_ = g.t().contiguous(), u.t().contiguous(), d.t().contiguous()

        def q1(w):                                  # 与 ctor / _requant 逐字同
            sc = max((w.abs().max() / 127.0).item(), 1e-8)
            q = (w / sc).round().clamp(-127, 127)
            return sc, q.to(torch.int8), (q + 128).to(torch.uint8)

        sg, qg, bg = q1(gt)
        su, qu, _ = q1(ut)
        sd, qd, _ = q1(dt_)
        self.cpu_tab[eid] = dict(gt=gt, ut=ut, dt=dt_, qg=qg, qu=qu, qd=qd,
                                 sg=sg, su=su, sd=sd, bg=bg,
                                 nb=(gt.numel() + ut.numel() + dt_.numel()) * 4
                                    + (qg.numel() + qu.numel() + qd.numel())
                                    + (bg.numel() if self.m.w_pack == "biased" else 0))

    # ---------------------------- 第一半: 搬进 staging(侧 stream; live 缓冲一个字节都不碰)
    def plan(self, eid_in):
        self._tab(eid_in)
        srcs = [self.pin32[eid_in]]
        dsts = [self.st32]
        if self.stl is not self.st32:
            srcs.append(self.pinl[eid_in])
            dsts.append(self.stl)
        with torch.cuda.stream(self.stream):
            # 🚨 先等上一次 flip 把 staging 读走, 否则会把正在被 D2D 读的 staging 覆写掉
            self.stream.wait_event(self.ev_flip)
            t0 = time.perf_counter()
            for dset, sset in zip(dsts, srcs):
                for dd, ss in zip(dset, sset):
                    dd.copy_(ss, non_blocking=True)
            # 🚨 门控行(2 x D 个 float = 2.3KB)不进 staging: 它太小, 走侧 stream 反而多一处
            #    stream 交叉与一次 ev_stage 依赖; flip 里直接从 pinned 读, 同主 stream 顺序天然对。
            self.phase_tot["h2d_enq"] += (time.perf_counter() - t0) * 1e6
            self.ev_stage.record(self.stream)
        self.staged = int(eid_in)
        return self.staged

    # --------------------------- 第二半: 提交(全排主 stream => 图只看得到换完/没换)
    def flip(self, i, j):
        m, gc = self.m, self.m.gpu_expert_count
        assert i < gc <= j, f"槽位要一个在 GPU 腿一个在 CPU 腿: i={i} j={j} gc={gc}"
        eid_in, eid_out = self.slot2eid[j], self.slot2eid[i]
        v_before, ptr_before = self._versions(), self._ptrs()
        if eid_out not in self.cpu_tab:      # prebuild=False 的兜底: 建表代价会落进这一发 flip
            self._tab(eid_out)
        t0 = time.perf_counter()
        if self.staged != int(eid_in):              # 没提前 plan(或 plan 错了人) => 当场补搬
            self.plan(eid_in)
        main = torch.cuda.current_stream()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        main.wait_event(self.ev_stage)              # 主 stream 等 staging 落地(不 stall host)
        e0.record(main)
        s32, sl = self.st32, self.stl
        # 1) live 存储权重(fp32 永远) + 热权重缓冲的【槽位块】。
        #    🚨 _wcat 是 cat(dim=0) 的 (gc*D, N) => 行块宽是 D; _wcat_gu 是 cat(dim=1) 的
        #       (D, gc*2N) => 列块宽是 2N。两者方向不同, 上一版草稿把 h 写成 expert_size 就错了。
        h = m._wcat.shape[0] // gc if m._wcat is not None else self.D
        N = self.N
        t1 = time.perf_counter()
        m.gpu_experts[i].data.copy_(s32[0])
        if m.kind == "swiglu":
            m.gpu_up[i].data.copy_(s32[1])
            m.gpu_down[i].data.copy_(s32[2])
        if m._wcat is not None:
            m._wcat[i * h:(i + 1) * h].copy_(sl[0])
        if m._wstack is not None and self.refresh_dead:
            m._wstack[i].copy_(sl[0])
        if m._wcat_gu is not None:
            m._wcat_gu[:, i * 2 * N:i * 2 * N + N].copy_(sl[0])
            m._wcat_gu[:, i * 2 * N + N:(i + 1) * 2 * N].copy_(sl[1])
        if m._wstack_gu is not None and self.refresh_dead:
            m._wstack_gu[i, :, :N].copy_(sl[0])
            m._wstack_gu[i, :, N:].copy_(sl[1])
        if m._wstack_d is not None:
            m._wstack_d[i].copy_(sl[2])
        t2 = time.perf_counter()
        # 2) 门控行 + bias(两份布局都要动 —— 漏一份就是路由与权重分家)。bias 真源恒 0
        #    (real_src 的 gate_b) 但这里按通用处理, 玩具权重的 nn.Linear 随机 bias 一换就显形。
        gw, gb = m.gate.weight.data, m.gate.bias.data
        if self.gate_abs:
            # 两行都从【pinned】读 + non_blocking: 换成 self.src_gate[eid_out](pageable)会当场
            # 引入一次 device sync = 本类存在的理由被自己推翻, 所以这里只许碰 pin_gate。
            pin_in, pin_out = self.pin_gate[eid_in][0], self.pin_gate[eid_out][0]
            gw[i].copy_(pin_in, non_blocking=True)
            gw[j].copy_(pin_out, non_blocking=True)
            gt = m._gate_wt
            if gt is not None:
                gt[:, i].copy_(pin_in, non_blocking=True)
                gt[:, j].copy_(pin_out, non_blocking=True)
        else:
            # 没给 src_gate => 退回 swap 语义(同主 stream 上先读后写, 本来也安全)
            self.st_tmp[:gw.shape[1]].copy_(gw[i])
            gw[i].copy_(gw[j])
            gw[j].copy_(self.st_tmp[:gw.shape[1]])
            gt = m._gate_wt
            if gt is not None:
                self.st_tmp[:gt.shape[0]].copy_(gt[:, i])
                gt[:, i].copy_(gt[:, j])
                gt[:, j].copy_(self.st_tmp[:gt.shape[0]])
        if gb.numel():
            # 标量暂存走【预分配】的 st_tmp[0]: flip 里 clone() 会 cudaMalloc(首发动几十 us,
            # 撞上 allocator 抖动就是几百 us), 价签测里这是纯粹的口径污染。st_tmp 到这里已读完。
            self.st_tmp[0].copy_(gb[i])
            gb[i].copy_(gb[j])
            gb[j].copy_(self.st_tmp[0])
        e1.record(main)
        t3 = time.perf_counter()
        # 3) CPU 腿: 从预存表 memcpy(不 transpose、不 requant)。C++ 收整块 stacked => 只能写行
        jc = j - gc
        tab = self.cpu_tab[eid_out]
        m.cpu_experts_t[jc].copy_(tab["gt"])
        m.cpu_experts_q_s8[jc].copy_(tab["qg"])
        if m.w_pack == "biased":
            m.cpu_experts_q[jc].copy_(tab["bg"])
        m.cpu_experts_scale[jc] = tab["sg"]
        if m.kind == "swiglu":
            m.cpu_up_t[jc].copy_(tab["ut"])
            m.cpu_up_q[jc].copy_(tab["qu"])
            m.cpu_up_scale[jc] = tab["su"]
            m.cpu_down_t[jc].copy_(tab["dt"])
            m.cpu_down_q[jc].copy_(tab["qd"])
            m.cpu_down_scale[jc] = tab["sd"]
        # 4) 清派生缓存(裁判的 GPU 副本 / sparse 档半精缓存) —— 与 sync 版一字不差
        m._cpu_w_dev_cache = None
        m._cpu_wu_dev = None
        m._cpu_wd_dev = None
        if getattr(m, "_wdt", None):
            m._wdt.clear()
        t4 = time.perf_counter()
        self.ev_flip.record(main)                   # 下一次 plan 要等 staging 被读走
        self.staged = None
        nb = tab["nb"] + self.st32[0].numel() * 4 + self.st32[1].numel() * 4 \
            + self.st32[2].numel() * 4
        self.phase_tot["d2d_enq"] += (t2 - t1) * 1e6
        self.phase_tot["gate"] += (t3 - t2) * 1e6
        self.phase_tot["cpu_leg"] += (t4 - t3) * 1e6
        self.phase_tot["total"] += (t4 - t0) * 1e6
        self.phase_n += 1
        us = (t4 - t0) * 1e6
        self.slot2eid[i], self.slot2eid[j] = eid_in, eid_out   # 回写方向同 sync 版
        self.n_swap += 1
        self.total_us += us
        self.bytes_last = nb
        self.bytes_total += nb
        rec = dict(us=us, mb=nb / 2 ** 20, slot_in=i, slot_out=j, eid_in=eid_in,
                   eid_out=eid_out, dev_ev=(e0, e1), host_us=us)
        if self.check:
            assert self._versions() == v_before, "🚨 迁移 bump 了 _version => 会重录 CUDA Graph"
            assert self._ptrs() == ptr_before, "🚨 迁移换了缓冲地址 => 图在读旧内存"
        self.flips.append(rec)
        return rec

    swap = flip              # 兼容 DynPlacer 的 self.sw.swap(i, j) 调用形状

    # -------------------------------------------------------------- 汇报
    def phase_us(self):
        n = max(1, self.phase_n)
        return {k: v / n for k, v in self.phase_tot.items()}

    def dev_ms(self):
        """设备侧真正花掉的 ms(flip 提交后主 stream 上那一段)。价签读这个, 不读 host us。"""
        out = []
        for r in self.flips:
            e0, e1 = r["dev_ev"]
            e1.synchronize()
            out.append(e0.elapsed_time(e1))
        return out



# ---------------------------------------------------------------- 门面
class DynPlacer:
    """把 计数器 -> 滑窗 -> 决策 -> 迁移 串起来。bench 在计时窗口外调 step()。"""

    def __init__(self, m, src_experts, gc, policy=None, window=24, check_every=8,
                 warm=None, check=True, refresh_dead=True, order=None,
                 swapper="sync", **swap_kw):
        self.m = m
        # §5.2h: swapper="sync" 走老路(逐字不变); "async_direct"/"async_staged" 走 AsyncSwapper。
        # 默认值 = sync ⇒ 现存的 bench_dyn / bench_dynpair / 红线 J 全部不受影响。
        cls = ExpertSwapper if swapper == "sync" else AsyncSwapper
        self.sw = cls(m, src_experts, check=check, refresh_dead=refresh_dead,
                      order=order, **swap_kw)
        self.pol = policy or PlacementPolicy(gc)
        self.win = HitWindow(m.num_experts, window)
        self.check_every = max(1, int(check_every))
        self.warm = int(warm) if warm else int(window)
        self.swaps = []                    # (frame, phase, record)
        self.n_obs = 0

    # 只数, 不决策(所有策略都跑 => 记账成本不进策略对比)
    def observe(self, counts=None):
        """counts=None => 从引擎 pinned 缓冲读; 传了就复用那一份(闸门① 要拿同一个数去对 route())。"""
        c = counts_from_engine(self.m) if counts is None else counts
        self.win.push(c, self.sw.slot2eid)
        self.n_obs += 1
        return c

    def step(self, frame, phase=0, counts=None):
        """在【计时窗口之外】调。返回本次执行的迁移列表(通常为空)。"""
        self.observe(counts)
        if not self.pol.enabled or not self.sw.slot2eid:
            return []
        if self.n_obs < self.warm:
            return []
        if (frame - self.warm + 1) % self.check_every:
            return []
        done = []
        for (i, j) in self.pol.decide(self.win.frac(), self.sw.slot2eid, frame):
            rec = self.sw.swap(i, j)
            self.pol.note_swap(rec["eid_in"], rec["eid_out"], frame)
            rec.update(frame=frame, phase=phase)
            self.swaps.append((frame, phase, rec))
            done.append(rec)
        return done

    # ---- 汇报用的派生量
    def frac_eid(self):
        return self.win.frac()

    @property
    def swap_ms(self):
        return self.sw.total_us / 1e3

    def summary(self):
        ph = {}
        for (f, p, r) in self.swaps:
            d = ph.setdefault(p, dict(n=0, us=0.0, first=None, last=None, pairs=[]))
            d["n"] += 1
            d["us"] += r["us"]
            d["first"] = f if d["first"] is None else min(d["first"], f)
            d["last"] = f if d["last"] is None else max(d["last"], f)
            d["pairs"].append((r["eid_out"], r["eid_in"]))
        return ph


# ---------------------------------------------------------------- 单测(纯决策层, 不起 GPU)
def _selftest():
    ok = True

    def chk(cond, tag):
        nonlocal ok
        print(f"  {'OK ' if cond else '🚨FAIL '} {tag}")
        ok = ok and bool(cond)

    E, gc = 8, 3

    def apply_once(pol, s2e, frac, frame):
        """模拟一次完整的 决策 -> 落地(改 slot2eid + 回调 note_swap)。返回迁移数。"""
        n = 0
        for (i, j) in pol.decide(frac, s2e, frame):
            eid_in, eid_out = s2e[j], s2e[i]
            s2e[i], s2e[j] = eid_in, eid_out      # 与 ExpertSwapper.swap 同一条方向
            pol.note_swap(eid_in, eid_out, frame)
            n += 1
        return n

    # 1) 直方图稳定 => 换一次就沉默(冷却 + 最热已在 GPU, 两条各自都足以住手)
    p = PlacementPolicy(gc, **HYSTERESIS)
    frac = [0.30, 0.05, 0.05, 0.05, 0.02, 0.40, 0.08, 0.05]   # 初始 槽位==专家
    s2e = list(range(E))
    acts = [apply_once(p, s2e, frac, f) for f in range(40)]
    chk(sum(acts) == 1, f"稳定直方图: 40 帧只换 1 次 => 共换 {sum(acts)} 次")
    # 2) 容量约束: 决策永远是一进一出, GPU 成员数恒为 gc
    chk(sorted(s2e[:gc]) == sorted([0, 5, 2]), f"GPU 成员数恒为 gc: {s2e[:gc]}")
    # 3) 外部原型(两条独立阈值)在漂移场景【换不动】: 最冷的 GPU 槽位还有 2%(>1%)
    pn = PlacementPolicy(gc, **NAIVE)
    drift = [0.20, 0.05, 0.02, 0.06, 0.30, 0.10, 0.17, 0.10]
    chk(pn.decide(drift, list(range(E)), 0) == [],
        "naive: GPU 最冷 2% > demote_max 1% => 不换(流量漂移期卡死)")
    ph = PlacementPolicy(gc, **{k: v for k, v in HYSTERESIS.items() if k != "cooldown"})
    a = ph.decide(drift, list(range(E)), 0)
    chk(a == [(2, 4)], f"gain 判据救回来: 换掉最冷槽位2、换上4 => {a}")
    # 4) 抖动可计数: 同一对来回搬 = 撤销上一次迁移
    pt = PlacementPolicy(gc, thrash_k=4)
    for k in range(6):
        pt.note_swap(*(5, 1) if k % 2 == 0 else (1, 5), frame=k)
    # 6 次交替 = 后 5 次都在撤销前一次(hist 里能找到反方向那一对), 第 1 次无前序 => 5
    chk(pt.n_swap == 6 and pt.n_thrash == 5,
        f"来回抖动被计到: n_swap={pt.n_swap} n_thrash={pt.n_thrash}(期望 6/5)")
    # 5) 冷却真的挡人: cooldown=16 时第 5 帧不许再动
    pc = PlacementPolicy(gc, **HYSTERESIS)
    pc.note_swap(5, 1, 0)
    chk(pc.decide(drift, list(range(E)), 5) == [], "冷却期内不许再迁")
    chk(len(pc.decide(drift, list(range(E)), 40)) == 1, "冷却到期恢复决策")
    # 6b) 起始序是置换时(eid2 在槽 0 / eid0 在槽 2), 决策必须打在【同一个专家对】上:
    #     直方图按 eid 给: eid0=0.40(热) eid2=0.02(冷) eid5=0.30(热) => 槽位序 [2,1,0,...]
    #     GPU 成员 = 槽 0(eid2,0.02) / 槽 1(eid1,0.05) / 槽 2(eid0,0.40) => 最冷 GPU 槽 = 槽 0
    #     => 决策必须是 (槽0 出 eid2, 槽5 进 eid5), 成员从 {2,1,0} 变 {5,1,0}
    p6 = PlacementPolicy(gc, **{k: v for k, v in HYSTERESIS.items() if k != "cooldown"})
    s6 = [2, 1, 0, 3, 4, 5, 6, 7]
    f6 = [0.40, 0.05, 0.02, 0.05, 0.02, 0.30, 0.10, 0.06]      # eid 空间
    a6 = p6.decide(f6, s6, 0)
    ok6 = a6 == [(0, 5)] and s6[a6[0][0]] == 2 and s6[a6[0][1]] == 5
    apply_once(p6, s6, f6, 0)
    chk(bool(ok6) and sorted(s6[:gc]) == sorted([5, 1, 0]),
        f"置换起始序: 期望决策=(槽0出eid2, 槽5进eid5) => 决策={a6} 现成员={s6[:gc]}")
    # 7) 边界: gc=0 / gc=E 没有放置自由度
    chk(PlacementPolicy(0).decide(frac, list(range(E)), 0) == [] and
        PlacementPolicy(E).decide(frac, list(range(E)), 0) == [], "gc=0/gc=E 无放置自由度")
    # 8) 价判据(rule_p): 选中的是【贵】而不是【热】; thr_rel 挡边际迁移; off 档行为不变
    #    u_e 故意造反: eid6 最贵但【不热】, eid5 最热但第二贵 => frozen 选 6, hit 选 5
    f8 = [0.10, 0.12, 0.08, 0.20, 0.12, 0.30, 0.05, 0.03]        # eid 空间, 合计 1.0
    u8 = [1e-06, 2e-06, 5e-07, 1e-06, 5e-08, 3e-06, 9e-06, 1e-08]
    # kappa 那一档另给一组数: 目标是 val=kappa x 占比 之后【最热但第二贵】的 eid5 反超 eid6
    k8 = [1e-06, 1e-06, 1e-06, 2e-06, 1e-06, 4e-06, 1e-05, 1e-06]
    p8f = PlacementPolicy(gc, **{k: v for k, v in PRICE.items() if k != "thr_rel"},
                          price_mode="frozen")
    p8f.price = u8
    a8f = p8f.decide(f8, list(range(E)), 0)
    chk(a8f == [(2, 6)] and p8f.n_rule_p == 1,
        f"frozen(u_e): 该提最贵的 eid6 而不是最热的 eid5 => 决策={a8f}")
    p8h = PlacementPolicy(gc, **{k: v for k, v in PRICE.items() if k != "thr_rel"},
                          price_mode="hit")
    p8h.price = k8
    a8h = p8h.decide(f8, list(range(E)), 0)
    chk(a8h == [(2, 5)], f"hit(kappa x 窗内占比): 流量重新定价后该提 eid5 => 决策={a8h}")
    p8t = PlacementPolicy(gc, **dict(PRICE, thr_rel=0.90), price_mode="frozen")
    p8t.price = u8
    chk(p8t.decide(f8, list(range(E)), 0) == [],
        "thr_rel=0.90: 预测降幅(65%)低于门槛 => 不付迁移钱(价签写进判据)")
    p8o = PlacementPolicy(gc, gain=1.0, cooldown=0, price_mode="off")
    p8o.price = u8
    chk(p8o.decide(f8, list(range(E)), 0) == [(2, 5)],
        "price_mode=off: 价被忽略, 逐位回到命中判据(已定稿的 P1-②/②b 两表可复现)")
    p8n = PlacementPolicy(gc, **PRICE, price_mode="frozen")
    chk(p8n.decide(f8, list(range(E)), 0) == [] and p8n.n_rule_p == 0,
        "price=None(价表缺失): 沉默, 不偷偷退回命中判据")
    # 9) 冷却是【限时】禁令: 解禁帧之后必须能把"上次被降下去的人"提回来
    #    (旧实现查的是 key 存在性 + 只在 note_swap 里剪枝 => 这条会失败: 它不提最贵的
    #     eid2, 而是去提第二贵的 eid5 = 决策打在错的那对上; 判决表 3ce6836d 的实测形态)
    p9 = PlacementPolicy(gc, **dict(PRICE, cooldown=8), price_mode="frozen")
    p9.price = u8
    s9 = list(range(E))
    a9 = p9.decide(f8, s9, 0)
    chk(a9 == [(2, 6)], f"冷却自证前置: 第 0 帧该提最贵的 eid6 => {a9}")
    p9.note_swap(6, 2, 0)                       # 落地: GPU 槽2 换出 eid2, 它的解禁帧 = 8
    s9[2], s9[6] = 6, 2                         # 现 GPU 成员 = {eid0, eid1, eid6}
    u9 = list(u8)
    u9[2] = 9.5e-06                             # 让被踢出去的 eid2 变成 CPU 腿里最贵的那个
    p9.price = u9
    chk(p9.decide(f8, s9, 4) == [], "冷却期内(帧4 < 解禁帧8)不许把 eid2 提回来")
    a9b = p9.decide(f8, s9, 20)      # 期望 (GPU 槽0=eid0 出, CPU 槽6=eid2 进)
    chk(a9b == [(0, 6)],
        f"解禁后必须把最贵的 eid2(在槽6)提回来(旧实现永久锁死它 => 去提第二贵的 eid5=槽5)"
        f" => {a9b}")
    print("ALL POLICY TESTS PASS" if ok else "\U0001f6a8 POLICY TEST FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)

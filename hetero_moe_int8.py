import math
import os
import statistics
import sys
import time

# ============================================================
# ✅ T7-①  最便宜的一刀: 告诉 libgomp 机器的【拓扑】, 让它自己绑核
#
# 本机 i5-10400: 6 物理核, 超线程兄弟是 N 和 N+6。不绑核时由 Linux 调度器
# 随机放 —— 6 条 OMP worker + 1 条主线程 = 7 条 runnable 抢 6 个物理核,
# 主线程被反复挤到超线程兄弟上, 而它干的是 aten dispatch(纯标量延迟活),
# 每次迁移都要付 cache/频率爬坡的税。
# 实测(同一份二进制, 只改这两个变量): 配置 C 的 L2 1.554 -> 1.185ms (-24%)
#
# 🚨 OMP_PLACES=cores 是把"一个核的两个 SMT 兄弟"当成一个 place,
#    所以线程仍然能同时用 N 和 N+6 —— 它【没有】缩小进程 CPU mask。
# 🚨 千万别改用 GOMP_CPU_AFFINITY=1-5 或 os.sched_setaffinity 去限 mask:
#    实测墙钟从 1.2ms 炸到 9~72ms(主线程和 worker 挤在同一核上互相踩)。
#    要留核请用 OMP_NUM_THREADS, 不要用 mask。
# 🚨 必须赶在 import torch 之前设: libgomp 只在 so 加载时读一次这些变量。
#    (已实测: 进程内 setdefault 与外部 env 效果逐位相同)
#    用 setdefault 而不是直接赋值 => 外部 env 依然能覆盖做 A/B。
os.environ.setdefault("OMP_PLACES", "cores")
os.environ.setdefault("OMP_PROC_BIND", "close")

import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from torch.utils.cpp_extension import load

# ✅ T3 前置: Python 默认 5ms 的 GIL 切换间隔会让工作线程"抢不到票"
sys.setswitchinterval(1e-5)

# HDBG=1 => 每次进 swiglu CPU kernel 前打印 idx/scr 真值(C++ 只报"expert id N out of
#    range", 拿不到现场; 默认关 = 零开销)
_HDBG = os.environ.get("HDBG", "") == "1"

print("正在编译 AVX2 INT8 扩展 (maddubs 路径 —— 本机 i5-10400 无 AVX-VNNI)")
avx_gemv_ext = load(
    name="avx_gemv_int8",
    sources=["avx_gemv_int8.cpp"],
    extra_cflags=["-O3", "-mavx2", "-mfma", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False
)

MODES = ["L0_现状", "L1_松GIL", "L2_零CUDA+pinned"]

# ✅ T10: 默认走 signed(可证明无饱和)。"biased" 只用来复现历史数字 / 做 A/B。
#   env MOE_W_PACK=biased 可整体切回旧打包, 供 bench_int8_exactness.py 对照。
W_PACK = os.environ.get("MOE_W_PACK", "signed")

# ✅ T11: 前段(门控 GEMM -> softmax -> topk -> 激活量化)的实现方式。
#   "torch" = 原样 12 张 aten kernel(图内墙钟 bs=32 41.8us); "fused" = 2 张 triton(12.2us)。
#   🚨 fused 的 topk_indices / x_q 与 torch 版【逐位相同】(bench_pre_fuse.py 逐位闸门),
#      所以 check_redlines 的 relRMSE 基线一个数都不该变; 变了就是换了路由人选 = bug。
PRE_IMPL = os.environ.get("MOE_PRE_IMPL", "fused")

# ============================================================
# ✅ T8-① 权重工作集: 判断 CPU kernel 到底是【算力瓶颈】还是【带宽瓶颈】
#
# 🚨🚨 T14 修订: 这里其实有【两堵墙】, 站哪一堵由"每个 CPU 专家摊到几个 slot"决定:
#     字节墙 = 每个有流量的专家都要被整趟读过 2MB   => 2MB / 42GB/s = 49us/专家
#     算力墙 = 每个 slot 要算 expert_size x hidden = 2M 次 int8 MAC,
#              本机 AVX2 maddubs 实测 单核 13.5 MAC/cyc、6 线程 288 GMAC/s
#              => 2M / 288G = 7us/slot(并行), 且与"权重被重扫几遍"无关
#     交叉点: 每专家约 5 个 slot。C 配置(bs=32)每专家 29/26/4 个 => 算力墙;
#             A 配置每专家 1/5 个 => 字节墙。这就是 T8 和 T14 看起来互相打脸的原因。
#     => 放置规则的 CPU 腿该是 max(字节墙, 算力墙), 现在只记了字节墙 => 待办 T15。
#
# kernel 按专家串行扫, 每个专家的权重每帧整块过一遍 L3。所以:
#     工作集 = CPU侧专家数 x expert_size x hidden x 1byte(int8)
# 本机 i5-10400 的 L3 = 12MB(6 核共享)。6 个 CPU 专家 x 2MB = 12MB —— 正好
# 卡在容量上, 算上桌面/内核自己的页就必然漏到 DRAM。实测(见 bench_kernel_micro):
#     1专家(2MB)  0.021ms/槽 => ~95 GB/s   (L3 命中)
#     4专家(8MB)  0.068ms/槽 => ~117 GB/s  (还在 L3)
#     6专家(12MB) 0.283ms/槽 => ~ 42 GB/s  (双通道 DDR4 顶了, 且 4x 槽位不加时间)
# 🚨 "6专家x4槽 ≈ 6专家x1槽" 是决定性证据: 算术 x4 时间不变 => 纯带宽,
#    任何减少算术的 kernel 优化在这里都拿不到钱。要拿钱只能减【字节】:
#    权重 int8->int4(工作集 12MB->6MB), 或把 CPU 侧专家压到 4 个(8MB)。
# ============================================================
def l3_bytes():
    """/sys 里读真实 L3; 读不到就按 8MB 保守估。"""
    import glob
    best = 0
    for d in glob.glob("/sys/devices/system/cpu/cpu*/cache/index*"):
        try:
            if open(d + "/level").read().strip() == "3":
                txt = open(d + "/size").read().strip()
                n = int(txt[:-1]) * (1024 if txt[-1] in "KM" else 1)
                if txt[-1] == "M":
                    n *= 1024
                best = max(best, n)
        except (OSError, ValueError, IndexError):
            pass
    return best or (8 << 20)


L3 = l3_bytes()
_ws_warned = set()


# ============================================================
# 尺子 1: 精度
# ============================================================
def err_metrics(out, ref):
    out = out.float(); ref = ref.float()
    diff = out - ref
    max_abs = diff.abs().max().item()
    n_ref, n_err = ref.norm().item(), diff.norm().item()
    if n_err == 0.0:
        return max_abs, float("inf"), 0.0
    return max_abs, 20.0 * math.log10(n_ref / n_err), n_err / n_ref * 100.0


def _g_r(gen, *shape):
    """✅ P0-①: 从【独立 Generator】抽一张权重(乙-2 的约定: 不碰全局 RNG, 否则 gate 权重会变,
    延迟表就在比两个不同路由的模型)。抽张量的【顺序/形状】与乙-2 入库版逐字相同。"""
    return torch.randn(*shape, generator=gen)


class HeteroMoE(nn.Module):
    def __init__(self, hidden_size, num_experts, expert_size,
                 gpu_expert_count=2, top_k=2, hot_experts=None, cpu_delay_ms=0.0,
                 # ✅ T4/T5/T7: 默认值就是扫出来的最优点。
                 #    🚨 n_threads 千万别留 1 —— CPU kernel 0.5ms 起步,
                 #       会把 GPU 半边(T6/T7)省下的 0.2ms 彻底盖住, 主表看不见新刀。
                 #    T4 扫描: 6 线程撞上 L3 带宽墙; 8 线程因 SMT 兄弟回退。
                 n_threads=6, tok_block=8, gpu_impl="cat", graph_scope="off",
                 d2h_wait="main", w_pack=None, pre_impl=None,
                 l2_stream="single", tail_graph=False,
                 cpu_loop=0, cpu_sched=1,
                 gpu_dtype="fp32",
                 expert_kind="gemm",
                 overlap=True, free_dead_w=False,
                 real_src=None,
                 freeze_params=True):   # ✅ T14: cpu_loop 已否证(默认关) | cpu_sched=1 => OMP guided
        # ✅ 乙-2: expert_kind 是【新维度】, 默认 gemm = 入库行为逐字不变。
        #    "gemm"   = 历史的单矩阵专家 out = x @ W          (输出宽 = expert_size)
        #    "swiglu" = 真 MoE 的三矩阵 h = silu(x@Wg)*(x@Wu), out = h @ Wd
        #               (输出宽 = hidden_size)
        #    🚨 这一刀要回答的不是"快不快", 而是"前面 14 刀的结论是不是只在玩具形状上成立":
        #       字节墙 x3、h 必须再量化一次(新的精度代价)、两堵墙交叉点和最优线程数都要重标。
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.kind = expert_kind
        assert expert_kind in ("gemm", "swiglu"), f"未知 expert_kind: {expert_kind}"
        # ✅ 乙-2c: GPU 半边【热专家权重】的存储精度。默认 fp32 = 入库行为逐字不变。
        #    乙-2b 归因量到的那笔与 slot 无关的开销就在 GPU 半边, 而它等于
        #    "每帧要读的 GPU 权重字节 / 250~270GB/s":
        #      gemm  cat 档读 _wcat                        = D*gc*N     张
        #      swiglu cat 档读 _wcat_gu + _wstack_d         = D*gc*2N + gc*N*D  = 2 张
        #    => gc=5/D=1024/N=2048/fp32 时 gemm 41.9MB、swiglu 125.8MB,
        #       实测 GPU 半边 0.177 / 0.470ms => 237 / 268 GB/s, 都贴着 skinny GEMM
        #       的标定带宽 250GB/s => 【这一半边是字节 bound, 不是算力 bound】。
        #    所以唯一的减价办法是【少读字节】: 精度 /2 (fp16/bf16) 或 /4 (int8),
        #    再乘上"流量稀疏"(bs=1 只有 top_k 个热专家有流量却算了 gc 个)。
        #    🚨 存储权重(self.gpu_experts/gpu_up/gpu_down)本身【一律不动】:
        #       fp32_reference 直接读它们当裁判, 门控 gate 也不动(乙-1 已否证动前段)。
        self.gpu_dtype = gpu_dtype
        assert gpu_dtype in ("fp32", "fp16", "bf16"), f"未知 gpu_dtype: {gpu_dtype}"
        # ✅ P0-②: overlap=False => 两条腿【串行】(先等 CPU 腿跑完, 才发 GPU 的活)。
        #    主线整个主张是"每帧 ≈ max(CPU腿, GPU腿) + 固定串行", 它的反面就是
        #    "每帧 = CPU腿 + GPU腿 + 固定串行"。这一开关就是把 max 换成 sum,
        #    用来给"重叠"本身定价 —— 这是全台账里唯一还没被量过的主线级数字。
        #    🚨 它只改【提交顺序】, 不改任何一条腿的算法/线程/图 => 差值里没有混淆项。
        #    口径固定在默认流(l2_stream="single", 也就是入库配置)上。
        self.overlap = bool(overlap)
        # ✅ P0-②附刀(死权重): _ensure_hot_weights 每次都把热专家权重同时做成 _wcat 和
        #    _wstack 两份(实测 gc=5 gemm: 各 40MB, 占峰值显存的 2/3)。但 cat 档只读
        #    _wcat + _wstack_d, _wstack / _wstack_gu 只有 bmm 档会碰 => 在 cat 档下它们
        #    是【纯占显存的死权重】。默认 False = 入库行为逐字不变(红线不受影响)。
        self.free_dead_w = bool(free_dead_w)
        # ✅ 乙-2d: traffic 档和 cat 档读的是同一组缓冲(_wcat_gu + _wstack_d),
        #    所以 _wstack/_wstack_gu 这两份死副本同样可以不解禁。
        assert not self.free_dead_w or gpu_impl in ("cat", "traffic"), \
            "free_dead_w 只在 gpu_impl='cat'/'traffic' 下安全(bmm 档要读 _wstack/_wstack_gu)"
        self._dt = {"fp32": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}[gpu_dtype]
        self._wdt = {}             # 乙-2c: sparse 档的按专家半精缓存(带 _version)
        # 一个专家要读几张矩阵 = 字节墙的斜率; 输出宽度(sw 的输出是 hidden 不是 expert_size)
        self.n_mat = 3 if self.kind == "swiglu" else 1
        self.out_dim = hidden_size if self.kind == "swiglu" else expert_size
        self._cpu_wu_dev = None       # swiglu: fp32 参考用的 device 缓存(懒建)
        self._cpu_wd_dev = None
        self.gpu_expert_count = gpu_expert_count
        self.top_k = top_k
        self.cpu_delay_ms = cpu_delay_ms
        self.n_threads = n_threads   # ✅ T4: CPU kernel 的 OpenMP 线程数
        self.tok_block = tok_block   # ✅ T5: 同一专家的 token 块大小(寄存器里的累加器个数)
        # ✅ T14: token 块循环在【行循环外面(0)】还是【里面(1)】。
        #    0 = 原状: 命中某专家的 token 每满 8 个就把那 2MB 权重【整趟重读一遍】,
        #        还各配一次 OMP 并行区启动。配置C(bs=32) 专家5 摊 29 个 token = 5 趟。
        #    1 = 换序: 一个专家只进一次并行区, 一行权重(1KB)进 L1 后被它的所有 token
        #        复用 => 每帧每张权重只从 L2/L3 过一遍。寄存器需求没变。
        #    🚨 这条和 T5 的 tok_block 是两回事: tok_block 定"寄存器里塞几个累加器",
        #       loop_order 定"同一张权重被整趟重读几次"。T5 扫 tb 扫不出这个, 因为它
        #       两个方向都在重扫(tb 越大重扫越少, 所以才一路单调到被 clamp 在 8)。
        #    ❌ T14 实测否证: 这个 kernel 是 AVX2 maddubs 算力墙(单核 13.5 MAC/cyc),
        #       重扫的权重全在 L2/L3 里不要钱 —— 换序后 C 独占 459->506us, 抢核时
        #       564->591us, 两头都更差 => 默认关, 只留作 A/B 变体。
        self.cpu_loop = cpu_loop
        # ✅ T14: OMP 行调度。1=guided(自取递减块, 抗掉队) 0=static,32(历史)。
        #    帧内主线程和绑在 core0 的 OMP 线程0 同核抢 CUDA 提交, static 下
        #    【整个并行区等半速的线程0】: 配置C 独占 478us / 有抢核 839us(+75%),
        #    guided 453 / 569us(-32%)。逐位等价(只换"谁算哪一行")。
        self.cpu_sched = cpu_sched
        # ✅ T6: GPU 热专家实现 sparse/cat/bmm | ✅ 乙-2d 加 traffic(按真流量读权重)
        #    traffic = grid 开在 (bs*top_k) 上、按 idx 在 device 上间接寻址专家:
        #    零流量专家【不读字节】。只做了 swiglu(真形状档), 详见 gpu_traffic.py 文件头。
        assert gpu_impl in ("sparse", "cat", "bmm", "traffic"), gpu_impl
        assert not (gpu_impl == "traffic" and expert_kind != "swiglu"), \
            "乙-2d 的 traffic 档只做了 swiglu(真形状); gemm 玩具档请用 cat"
        self.gpu_impl = gpu_impl
        self._tH = None               # 乙-2d: traffic 档的 (bs*top_k, N) 中间量缓冲
        self._tPART = None            # 乙-2d: traffic 档的 (bs*top_k, out_dim) 部分和缓冲
        self._tbuf_key = None
        # ✅ patch9: 告警按【bs】记集合。旧版是布尔 => 建图暖机(bs=1)先把它烧掉,
        #    于是 bs=32/128 的真实帧再也印不出自己的那一行(表里只剩一行 bs=1 的告警,
        #    边界算式 `bs*top_k < gc` 被贴错了 bs)。
        #    🚨不在 bench 侧 pre-seed: bench_swap.mk() 与 bench_metrics.mk() 必须逐字同
        #       (闸门⓪ 靠它), 所以只在引擎侧修, 暖机那一行照印但【标自己的 bs】。
        self._traffic_warn_bs = set()
        self.graph_scope = graph_scope  # ✅ T6b/T7: off | hot(只录GEMM) | full(连路由+量化一起录)
        # ✅ T8-② L2 里"等 D2H 落地"这件事放在谁身上。
        #    🚨 终审 = 否证, 默认保持 main。终审口径见 bench_d2hwait_ab.py
        #       (同进程轮转 + 逐轮配对 + 两种卡尺各 15 轮):
        #         delay=0 A: worker 主表口径 -2.7%(13 轮里 9 轮同向) / p50 -1.5%
        #                    worker_block 主表 -6.9% 但 σ=14% 而 p50 +0.6% —— 典型的
        #                    "min 好看 p50 不动"方差假收益, 不能算
        #         delay=0 C: worker 主表 -1.6% / p50 +0.7%  → 打平
        #         delay=3 A/C: 全落在 ±0.3% 以内, 符号 7/15~10/15 = 抛硬币
        #       道理: worker 本来就要等 D2H 落地才能开工, 换谁等 DMA 都是同一时刻完成,
        #       省掉的只有"线程被唤醒"那十几微秒(≈0.017ms, 正好是 A/delay0 那 -2.7%);
        #       budget 表里的 cur_sync=0.038 其实是【GPU 还没把数据造出来】, 调度学不掉。
        #       第一次量到的 -16% 是 min-of-N 被高方差变体骗了 —— 所以才有 p50 卡尺。
        self.d2h_wait = d2h_wait
        # ✅ T10: CPU 权重打包方式。None => 用模块级 W_PACK(env MOE_W_PACK 可覆盖, 便于 A/B)
        self.w_pack = w_pack or W_PACK
        # ✅ T11: 前段融合 kernel。延迟导入(triton import 约 0.4s, 不该压在每个脚本头上),
        #    转置权重按 device 懒建(见 _gate_wt_dev: .cuda() 不会搬普通属性)。
        self.pre_impl = pre_impl or PRE_IMPL
        self._pre = None
        self._gate_wt = None
        if self.pre_impl == "fused":
            import pre_fused
            self._pre = pre_fused
        self._ev_d2h = None

        # ✅ P0-①: 权重来源。real_src=None => 逐字保持历史的随机 std=1 玩具权重(红线/全部老表不变);
        #    real_src 是一个 dict: {"experts": [{"g":(D,N), "u":(D,N), "d":(N,D)} x E],
        #                           "gate_w": (E,D) 或 None, "gate_b": (E,) 或 None, "note": str}
        #    🚨 约定: 专家按【列表下标】切两半 —— 前 gpu_expert_count 个上 GPU, 其余进 CPU int8。
        #    🚨 有 real_src 时【不再加 hot bias】: 热专家必须由真路由量出来(bench 自己算 top-k),
        #       靠 bias 造热度 = 又在造玩具分布, 这一刀的卖点恰恰是"分布不是我们造的"。
        self.real_src = real_src
        _ex = real_src["experts"] if real_src is not None else None
        if real_src is not None:
            assert len(_ex) == num_experts, f"real_src 专家数 {len(_ex)} != {num_experts}"
            for e in _ex:
                assert e["g"].shape == (hidden_size, expert_size), e["g"].shape
                if self.kind == "swiglu":
                    assert e["u"].shape == (hidden_size, expert_size) and \
                           e["d"].shape == (expert_size, hidden_size), "swiglu 要 u(D,N) 和 d(N,D)"
        _g = lambda e: _ex[e]["g"].float() if real_src is not None else torch.randn(hidden_size, expert_size)
        self.gpu_experts = nn.ParameterList([
            nn.Parameter(_g(e).contiguous())
            for e in range(gpu_expert_count)])

        # 普通 python list => .cuda() 不搬它; 形状 (expert_size, hidden), 与 gpu 侧互为转置
        self.cpu_experts_t = [
            (_ex[gpu_expert_count + i]["g"].float().t().contiguous() if real_src is not None
             else torch.randn(expert_size, hidden_size).contiguous())
            for i in range(num_experts - gpu_expert_count)]

        self.gate = nn.Linear(hidden_size, num_experts)
        with torch.no_grad():
            if real_src is not None and real_src.get("gate_w") is not None:
                self.gate.weight.copy_(real_src["gate_w"].float())
                if real_src.get("gate_b") is not None:
                    self.gate.bias.copy_(real_src["gate_b"].float())
                else:
                    self.gate.bias.zero_()
            else:
                for eid in (hot_experts or []):
                    self.gate.bias[eid] = 1.5

        # ✅ 乙-2: SwiGLU 的另外两张矩阵。
        # 🚨 用【独立 Generator】抽权重, 且约定"上面那张 W 就是 SwiGLU 的 Wg":
        #    这样全局 RNG 的消耗顺序与 gemm 模式一字不差 => 门控/命中率/路由可比。
        #    (若顺手用全局 randn, 后面 gate 权重就变了 => 延迟表实际在比两个不同路由的模型)
        if self.kind == "swiglu":
            if real_src is not None:
                _ru = lambda e: _ex[e]["u"].float().contiguous()
                _rd = lambda e: _ex[e]["d"].float().contiguous()
            else:
                # 🚨 别再叫它 _g: 上面 _g 是"取第 e 个专家 Wg"的 lambda, 同名复用迟早出事
                _gen = torch.Generator().manual_seed(20261120)
                _ru = lambda e: _g_r(_gen, hidden_size, expert_size)
                _rd = lambda e: _g_r(_gen, expert_size, hidden_size)
            # GPU 侧: Wg = gpu_experts[e] 本身 (D,N) / Wu (D,N) / Wd (N,D) —— h@Wd 才是 (bs,D)
            self.gpu_up = nn.ParameterList(
                [nn.Parameter(_ru(e)) for e in range(gpu_expert_count)])
            self.gpu_down = nn.ParameterList(
                [nn.Parameter(_rd(e)) for e in range(gpu_expert_count)])
            # CPU 侧沿用"转置存"惯例(见上面 cpu_experts_t 的注释):
            #   cpu_up_t[i]   (N,D) -> .t() 得 (D,N) 与 gpu 侧互为转置
            #   cpu_down_t[i] (D,N) 直接就是 C++ 入口 C 要的 wq_d 布局(行=输出 j, 长=N)
            ne = num_experts - gpu_expert_count
            self.cpu_up_t = [(_ex[gpu_expert_count + i]["u"].float().t().contiguous()
                              if real_src is not None
                              else _g_r(_gen, expert_size, hidden_size).contiguous())
                             for i in range(ne)]
            # 🚨 P0-①: 真权重这侧要 .t() —— cpu_down_t 的布局是 (D,N)(=gpu 侧 d(N,D) 的转置,
            #    C++ 入口 C 的 stage3 按"行=输出 j、点积长=N"读它)。真源给的 d 是 (N,D),
            #    忘了 .t() 就会在 C++ 里报"输出宽度是 hidden"那句(玩具 1024/2048 下形状
            #    对称得不明显, 第一次换真形状 D=576/N=1536 才炸出来)。
            self.cpu_down_t = [(_ex[gpu_expert_count + i]["d"].float().t().contiguous()
                                if real_src is not None
                                else _g_r(_gen, hidden_size, expert_size).contiguous())
                               for i in range(ne)]
        if real_src is not None:
            # 装载完就把 CPU 侧三张矩阵的形状钉死 —— 布局错了要在 python 侧当场报,
            # 别等 C++ 那句"输出宽度是 hidden"去猜(它自己也说不清是哪一侧错)。
            want = {"cpu_experts_t": (expert_size, hidden_size),
                    "cpu_up_t": (expert_size, hidden_size),
                    "cpu_down_t": (hidden_size, expert_size)}
            for name, shp in want.items():
                mats = getattr(self, name, None)
                for mt in (mats or []):
                    assert tuple(mt.shape) == shp, f"{name} 布局应为 {shp}, 实得 {tuple(mt.shape)}"

        self.cpu_pool = ThreadPoolExecutor(max_workers=4)
        # 三种模式共用一条普通 stream。
        # 🚨 关键: 不需要 non_blocking stream —— 因为 L2 的 worker 里【一个 CUDA 调用都没有】,
        #    不存在"工作线程在 legacy default stream 上开活"这种污染源。
        #    (torch.cuda.Stream 也没有 non_blocking= 这个参数, 别乱猜 API)
        self.gpu_stream = torch.cuda.Stream()

        # ============================================================
        # ✅ T10: 同一份量化格点, 存两种打包, 唯一区别就是【有没有饱和】
        #   biased(uint8): (w/s + 128) -> [0,255], kernel 用 maddubs(w_raw, x) 再减 128*sum_x。
        #     🚨 maddubs 把【相邻两个乘积之和】压进 int16, 一对最坏 2*255*127=64770 > 32767
        #        => 静默饱和, 凭空多 0.48% 相对误差(配置C 1.4660% vs 理想 1.3844%)。
        #   signed(int8): (w/s) -> [-127,127], kernel 用 maddubs(|w|, x*sign(w)) = Σ w·x,
        #     |一对之和| <= 2*127*127 = 32258 < 32767 => 可证明不饱和; 操作数角色也正好
        #     符合 AVX-VNNI 的 dpbusd(u8, s8) 规定 => 换机器不会再算错。
        #   w/s 本来就在 [-127,127](w_scale=amax/127), 所以两种打包量化到【同一批整数】,
        #   A/B 出来只差那 0.48% 的算术误差, 没有别的混淆项。
        # ============================================================
        self.cpu_experts_q = []          # biased uint8 —— 历史行为(带饱和)
        self.cpu_experts_q_s8 = []       # signed int8  —— 默认(可证明精确)
        self.cpu_experts_scale = []
        for i in range(num_experts - gpu_expert_count):
            w = self.cpu_experts_t[i]
            w_scale = max((w.abs().max() / 127.0).item(), 1e-8)
            # 🚨 先算出整数格点, 再【从它派生】biased 打包。别写 (w/s+128).round():
            #    float32 在 128~255 附近 ULP≈1.5e-5, 半个格点上的值会被 .round() 判到
            #    隔壁去 —— 实测 12.58M 字节里有 65 个和 signed 差 1 LSB, A/B 就不干净了。
            q = (w / w_scale).round().clamp(-127, 127)
            self.cpu_experts_q_s8.append(q.to(torch.int8))
            self.cpu_experts_q.append((q + 128).to(torch.uint8))
            self.cpu_experts_scale.append(w_scale)
        # ✅ P0-②: gpu_expert_count == num_experts(纯 GPU 基线)时这里是【空列表】,
        #    torch.stack([]) 直接抛异常 —— 也就是说这套 harness 历史上根本指向不了那两个
        #    极端。空表用 0 长张量占位, 形状/元素类型都对, 只是没有第 0 维。
        self.n_cpu = num_experts - gpu_expert_count
        if self.cpu_experts_q:
            self.cpu_experts_q = torch.stack(self.cpu_experts_q)
            self.cpu_experts_q_s8 = torch.stack(self.cpu_experts_q_s8)
            self.cpu_experts_scale = torch.tensor(self.cpu_experts_scale, dtype=torch.float32)
        else:
            z = (0, expert_size, hidden_size)
            self.cpu_experts_q = torch.zeros(z, dtype=torch.uint8)
            self.cpu_experts_q_s8 = torch.zeros(z, dtype=torch.int8)
            self.cpu_experts_scale = torch.zeros(0, dtype=torch.float32)

        # ✅ 乙-2: 另外两张矩阵走【和上面同一套】量化算式(同 scale 定义/同格点/同舍入)。
        #    🚨 只存 signed: C++ 入口 C 明确挡住 biased(maddubs 的 int16 饱和 + h 侧的
        #       去偏项该按 slot 而不是按 token 定义), 所以这里不需要 biased 那份副本。
        def _quant_stack_s8(mats):
            if not mats:                      # ✅ P0-②: gc=E 时 CPU 侧一张矩阵都没有
                return (torch.zeros((0, expert_size, hidden_size), dtype=torch.int8),
                        torch.zeros(0, dtype=torch.float32))
            qs, ss = [], []
            for w in mats:
                sc = max((w.abs().max() / 127.0).item(), 1e-8)
                qs.append((w / sc).round().clamp(-127, 127).to(torch.int8))
                ss.append(sc)
            return torch.stack(qs), torch.tensor(ss, dtype=torch.float32)
        if self.kind == "swiglu":
            self.cpu_up_q, self.cpu_up_scale = _quant_stack_s8(self.cpu_up_t)
            self.cpu_down_q, self.cpu_down_scale = _quant_stack_s8(self.cpu_down_t)

        self._cpu_w_dev_cache = None
        # ✅ T6 缓存槽: 稠密权重 / graph
        self._wcat = None
        self._wstack = None
        self._wver = None
        # ✅ 乙-2: swiglu 的 (D, gc*2N) 大 cat(一趟 GEMM 出 g 和 u)
        #    ✅ 乙-2d: traffic 档也读它(按列块偏移 eid*2N 间接寻址), 所以它【不是】死权重
        self._wcat_gu = None
        self._wstack_gu = None    # ✅ 乙-2: swiglu 的 (gc, D, 2N)
        self._wstack_d = None     # ✅ 乙-2: swiglu 的 (gc, N, D)
        self._g_pre = None        # 前段图: 路由 + 量化
        self._g_hot = None        # 后段图: 热专家 GEMM
        self._gkey = None
        self._sx = self._sts = self._sti = None
        self._sxq = self._sxs = self._shot = None
        self._pin = None
        self._pin_bs = -1
        self._g_d2h = None       # ✅ T7-③: 第三张图 —— 4 次 D2H 折成 1 次 replay
        self._d2h_key = None
        self._hout = None        # ✅ T7-③: CPU 结果回 GPU 的预分配暂存(省一次分配)
        # ✅ T12-①: GPU 半边(热专家 replay + 收尾)走哪条流。
        #   "single" = 全回默认流(默认)。"sep" = T7 原状: 每帧 new Event + record +
        #       stream ctx + wait_event, 把热专家图丢到 gpu_stream 上。
        #   🚨 那条第二流【从来没换来并行】: ev 是在 d2h 之后 record 的, gpu_stream
        #      又 wait_event(ev) —— 两条流被这个事件完全串行化了。它买到的只有
        #      每帧 4 次额外主机调用。探针 /tmp/probe_1stream.py 逐位相同。
        self.l2_stream = l2_stream
        # ✅ T12-②: 把"H2D + add"折成第四张图(1 次 replay)。默认【关】——
        #   它换来的是"返回图内静态缓冲 => 下一帧覆写上一帧输出"这种接口语义变化,
        #   代价由调用方承担, 所以必须显式打开。详见 _ensure_tail_graph 的说明。
        self.tail_graph = tail_graph
        self._g_tail = None
        self._gtail_key = None
        self._tail_out = None

        # ✅ T8-①: 权重工作集 > L3 => kernel 必掉 DRAM, 打印一次部署级警告
        # ✅ T9-②: 这条是【上界】警告 —— 真预算要记在"这一帧会被摸到的专家"上,
        #    零流量专家根本不读权重, 是免费的。按实测流量算真值用 plan_gpu_capacity()(T13 起带 GPU 半边定价)
        # ✅ 乙-2: 每张专家矩阵的字节数 x 矩阵张数 = 一个专家的工作集。
        # ✅ P0-① 修订: 这两项改成【按 per 现算】。原来写死 "49us x n_mat" 和 "per >> 20 MB",
        #    在玩具形状(每张正好 2MB)下没问题, 换成真权重(D=576 N=1536, 每张 0.88MB /
        #    swiglu 3 张 2.53MB)就同时错两处: 字节 MB 被 >>20 截成 2, 字节墙被报成 147us
        #    (真值 63us, 差 2.3 倍) —— 一条会撒谎的警告比没警告更糟。
        #    注: 历史表里那句 "49us/专家" 是 2MB/42GB/s 的四舍五入, 现算得 50us, 同一公式。
        per = expert_size * hidden_size * self.n_mat
        ws = (self.num_experts - self.gpu_expert_count) * per
        key = (self.num_experts, self.gpu_expert_count, expert_size, hidden_size, self.kind)
        # 阈值 0.75xL3: 实测 8MB/12MB=0.67 还在 L3(111GB/s), 12MB/12MB=1.0 掉 DRAM(45GB/s)
        if ws * 4 > L3 * 3 and key not in _ws_warned:
            _ws_warned.add(key)
            cap = int(L3 * 0.75) // per
            print(f"🚨[工作集] CPU侧常驻 {self.num_experts - self.gpu_expert_count} 专家 x "
                  f"{per / 2**20:.2f}MB = {ws / 2**20:.1f}MB > L3 {L3 >> 20}MB"
                  f" => 字节墙约 {per / CPU_BW_EFF * 1e6:.0f}us/专家({per / 2**20:.2f}MB / 42GB/s); 每专家摊到 >=5 个 slot"
                  f" 就转成 AVX2 算力墙(见 T8-① 注释与 T14)。"
                  f" 0.75xL3 预算 = 最多 {cap} 个常驻;"
                  f" 真预算按【有流量的专家】记, 见 plan_gpu_capacity / T13")

        # ✅ P0-②b 附刀: 推理引擎装载完就【冻结参数】(默认开, 传 False 可退回旧行为)。
        #    为什么这是"刀"而不是代码卫生: 专家权重是 nn.Parameter(requires_grad=True),
        #    而 forward 从来没包 no_grad => 每帧前向都会建 autograd 图, 图里的 saved tensor
        #    拽住 gc 张 8MB 的 fp32 原版权重。只要【输出张量还活着】(bench 拿它算精度、
        #    服务里拿它做下一层输入都算), 这 gc x 8MB 就不会还给显存池。
        #    探针 /tmp/probe_leak5.py: del 掉模型仍残留 41.0MB, 再 del 掉那个 .cpu() 输出
        #    才掉回 0.1MB。=> bench_baselines 的"峰值显存"整列被污染 40~65MB,
        #    而且"去死权重省显存"错位一格(①-1 ③40 ④65, 真值应是 40/64/0)。
        #    冻结后输出【逐位相同】(探针 /tmp/probe_grad.txt: grad 开 vs no_grad 互差 0.000000000)。
        if freeze_params:
            self.requires_grad_(False)

    # ---------------- pinned 暂存区 (只有 L2 用) ----------------
    def _ensure_pins(self, bs, device):
        if self._pin is not None and self._pin_bs == bs:
            return
        self._pin = {
            "xq":   torch.empty(bs, self.hidden_size, dtype=torch.int8, pin_memory=True),
            "sc":   torch.empty(bs, dtype=torch.float32, pin_memory=True),
            "idx":  torch.empty(bs, self.top_k, dtype=torch.int64, pin_memory=True),
            "scr":  torch.empty(bs, self.top_k, dtype=torch.float32, pin_memory=True),
            "out":  torch.empty(bs, self.out_dim, dtype=torch.float32, pin_memory=True),
        }
        self._pin_bs = bs

    def _simulate_cpu_delay(self):
        if self.cpu_delay_ms > 0:
            time.sleep(self.cpu_delay_ms / 1000.0)

    def _stack_or_empty(self, mats):
        """✅ P0-②: 纯 GPU 基线下 CPU 专家列表是空的, torch.stack([]) 会抛。
        返回 0 长张量(形状照 cpu_experts_t 的元素形状), 裁判那边 for 循环本来就跳过全部专家。"""
        if mats:
            return torch.stack(mats)
        return torch.empty(0, self.expert_size, self.hidden_size, dtype=torch.float32)

    def route(self, x):
        scores = F.softmax(self.gate(x), dim=-1)
        return torch.topk(scores, self.top_k, dim=-1)

    @torch.no_grad()
    def fp32_reference(self, x, topk_scores, topk_indices):
        if self.kind == "swiglu":
            return self._fp32_reference_swiglu(x, topk_scores, topk_indices)
        if self._cpu_w_dev_cache is None or self._cpu_w_dev_cache.device != x.device:
            self._cpu_w_dev_cache = self._stack_or_empty(self.cpu_experts_t).to(x.device)
        ref = torch.zeros(x.shape[0], self.expert_size, device=x.device, dtype=x.dtype)
        for eid in range(self.num_experts):
            mask = (topk_indices == eid)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if eid < self.gpu_expert_count:
                w = self.gpu_experts[eid]
            else:
                w = self._cpu_w_dev_cache[eid - self.gpu_expert_count].t()
            sub = x[token_idx] @ w
            sub = sub * topk_scores[token_idx, slot_idx].unsqueeze(1)
            ref.index_add_(0, token_idx, sub)
        return ref

    @torch.no_grad()
    def _fp32_reference_swiglu(self, x, topk_scores, topk_indices):
        """真形状 fp32 裁判: h = silu(x@Wg) * (x@Wu), out = h @ Wd, 再乘门控分按 token 求和。
        🚨 裁判故意走最直白的 per-expert 循环, 不复用 cat/bmm 那些技巧 —— 裁判和被审对象
           共用一条代码路径, kernel bug 就会变成"自己等于自己"(triton_moe_kernel 那个
           "128 列之后全是 0" 就是这么被放过去的)。"""
        if self._cpu_w_dev_cache is None or self._cpu_w_dev_cache.device != x.device:
            self._cpu_w_dev_cache = self._stack_or_empty(self.cpu_experts_t).to(x.device)
            self._cpu_wu_dev = self._stack_or_empty(self.cpu_up_t).to(x.device)
            self._cpu_wd_dev = self._stack_or_empty(self.cpu_down_t).to(x.device)
        ref = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)
        for eid in range(self.num_experts):
            mask = (topk_indices == eid)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            if eid < self.gpu_expert_count:
                wg, wu, wd = self.gpu_experts[eid], self.gpu_up[eid], self.gpu_down[eid]
            else:
                e = eid - self.gpu_expert_count
                wg = self._cpu_w_dev_cache[e].t()      # (D,N)
                wu = self._cpu_wu_dev[e].t()           # (D,N)
                wd = self._cpu_wd_dev[e].t()           # (N,D) —— h 在左, 输出是 hidden
            sub = (F.silu(x[token_idx] @ wg) * (x[token_idx] @ wu)) @ wd
            sub = sub * topk_scores[token_idx, slot_idx].unsqueeze(1)
            ref.index_add_(0, token_idx, sub)
        return ref

    def forward(self, x, async_mode=True, mode="L0_现状"):
        # ✅ T7: 路由/量化挪进各条路径里的 _prepare —— 因为 full 档要把它们一起录进 graph
        if not async_mode:
            return self._forward_sync(x)
        if mode == "L2_零CUDA+pinned":
            return self._forward_async_L2(x)
        return self._forward_async_legacy(x, release_gil=(mode == "L1_松GIL"))

    def _quantize_act(self, x):
        x_scales = torch.clamp(x.abs().max(dim=-1).values / 127.0, min=1e-8)
        x_q = (x / x_scales.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        return x_q, x_scales

    # ============================================================
    # ✅ T6: GPU 半边 —— 稠密化 + 零主机往返
    #
    # 老写法(保留成 impl="sparse")对每个热专家做:
    #     mask.any()      -> 强制一次 GPU->CPU 往返(主机必须知道有没有命中)
    #     mask.nonzero()  -> 再强制一次往返(nonzero 的输出长度主机必须知道)
    # gpu_expert_count=2 => 每次 forward 4 次主机往返; profiler 实测整段
    # 22 张 kernel + 9 次主机往返/拷贝, 而真正算的 SGEMM 只有 ~0.073ms。
    # 配置 A(56 个槽位落 GPU)和配置 C(只有 7 个)墙钟几乎一样 ——
    # 因为这一段是 launch-bound, 不是 compute-bound。
    #
    # 稠密写法把"哪个 token 命中哪个热专家"从【控制流】降级成【数据】:
    # 冷专家槽位的门控权重记 0, 自然贡献 0, 于是整段变成形状固定的
    # 一趟 GEMM —— 分支没了, 同步没了, kernel 从 22 张掉到个位数。
    # ============================================================
    def _hot_gate(self, topk_scores, topk_indices):
        """(bs, gpu_cnt) 热专家门控权重, 冷专家槽位记 0。全程不把任何数读回主机。

        🚨 T9 修的真 bug: 累加目标原来是 `zeros_like(contrib)`, 形状是
           (bs, **top_k**) 而不是 (bs, **gpu_cnt**)。历史上 gpu_expert_count 恒等于
           top_k=2, 两者恰好同形, 一直没暴露; 一旦 gpu_expert_count>2,
           scatter_add_ 会拿 clamp(max=gc-1) 出来的索引【越界写】(release 模式
           不做边界检查, 静默改写显存), 且 xin 只有 top_k*hidden 列,
           GEMM 形状直接对不上。结论: 凡是"用 scatter 把稀疏命中摊成稠密门控"的
           写法, 目标形状必须由【值域】决定, 不能由【输入形状】决定。"""
        gc = self.gpu_expert_count
        # ✅ P0-②: gc=0(CPU-only 基线)时下面那句 clamp(max=gc-1) 会变成 clamp(max=-1),
        #    给 scatter_add_ 喂出 -1 索引 —— 直接挡掉, 返回一个 (bs,0) 的门控。
        if gc == 0:
            return topk_scores.new_zeros(topk_scores.shape[0], 0)
        contrib = topk_scores * (topk_indices < gc)
        gate = torch.zeros(topk_scores.shape[0], gc,
                           device=topk_scores.device, dtype=topk_scores.dtype)
        return gate.scatter_add_(1, topk_indices.clamp(max=gc - 1), contrib)

    def _ensure_hot_weights(self, x):
        """把热专家权重拼成一次 GEMM 能吃的形状。带 _version 校验, 权重一改就重建。"""
        if self.gpu_expert_count == 0:        # ✅ P0-②: 没有热专家 => 不拼任何权重
            self._wver = ()
            return
        vers = tuple(self.gpu_experts[e]._version for e in range(self.gpu_expert_count))
        if self.kind == "swiglu":
            # 🚨🚨 校验向量的长度必须和【下面存进去的】那个一致。上一版把 up/down 的 _version
            #    加在 early-return 之后 => 存的是长向量、比的是短向量 => 每帧都判定"权重变了"
            #    => 每帧重建 _wcat_gu(40MB)+_wstack_gu+stack_d, 被 full graph 吃进图里,
            #    swiglu 档白付 4.9ms/帧(实测 swiglu/gemm = 21x 而不是字节 bound 的 3x)。
            vers += tuple(self.gpu_up[e]._version for e in range(self.gpu_expert_count))
            vers += tuple(self.gpu_down[e]._version for e in range(self.gpu_expert_count))
        if (self._wcat is not None and self._wcat.device == x.device
                and self._wver == vers
                and (self.kind != "swiglu" or self._wcat_gu is not None)):
            return
        self._wcat = torch.cat([self.gpu_experts[e] for e in range(self.gpu_expert_count)],
                               0).contiguous()
        #    ✅ P0-②附刀: cat 档下这是死权重, free_dead_w=True 时干脆不建(省 gc*D*N*4 字节)
        self._wstack = None if self.free_dead_w else torch.stack(
            [self.gpu_experts[e] for e in range(self.gpu_expert_count)])
        if self.kind == "swiglu":
            # 🚨 按【列(dim=1)】拼, 不是按行: gpu_experts[e] 是 (D,N), x@W 的输出宽是 N。
            #    cat(dim=1) => (D, gc*2N), 一趟 x@它 就同时得到所有专家的 g 和 u。
            #    (别照抄 gemm 那条 _wcat: 它是 cat(dim=0) 配 "gate 折进 x" 用的,
            #     SwiGLU 有激活、折不进 x, 所以这里换了个折叠方向)
            gu = [t for e in range(self.gpu_expert_count)
                  for t in (self.gpu_experts[e], self.gpu_up[e])]
            self._wcat_gu = torch.cat(gu, 1).contiguous()                     # (D, gc*2N)
            self._wstack_gu = None if self.free_dead_w else torch.stack(
                [torch.cat([self.gpu_experts[e], self.gpu_up[e]], 1)
                 for e in range(self.gpu_expert_count)]).contiguous()          # (gc, D, 2N)
            self._wstack_d = torch.stack([self.gpu_down[e]
                                          for e in range(self.gpu_expert_count)]).contiguous()
        # ✅ 乙-2c: 降精放在【最后一步】, 五个缓存一起转, 上游代码一律看不见 dtype 分支。
        #    🚨 必须连 _wcat/_wstack 一起转: 只转 swiglu 那三个会让 gemm 档变成假 A/B。
        #    🚨 这里刻意【不保留 fp32 副本】—— 留着就等于把那 125.8MB 又请回来了。
        if self._dt is not torch.float32:
            for nm in ("_wcat", "_wstack", "_wcat_gu", "_wstack_gu", "_wstack_d"):
                t = getattr(self, nm, None)
                if t is not None:
                    setattr(self, nm, t.to(self._dt))
        self._wver = vers

    def _w_hot(self, eid, which="w"):
        """✅ 乙-2c: sparse 档(A/B 基线才走)按 gpu_dtype 取热专家权重, 带 _version 缓存。
        fp32 时原样返回 nn.Parameter => 零额外显存、零拷贝、入库行为逐字不变。"""
        src = {"w": self.gpu_experts,
               "u": getattr(self, "gpu_up", None),
               "d": getattr(self, "gpu_down", None)}[which][eid]
        if self._dt is torch.float32:
            return src
        key = (eid, which, src._version)
        v = self._wdt.get(key)
        if v is None:
            if len(self._wdt) > 48:          # 权重每改一次就换一批 key, 别让它长牙
                self._wdt.clear()
            v = self._wdt[key] = src.to(self._dt)
        return v

    def _hot_compute(self, x, topk_scores, topk_indices, impl=None):
        """✅ 乙-2c: GPU 半边的【唯一入口】, 也因此是 dtype 降/升的唯一收口点。
        入: x 降到 self._dt; 出: 抬回 fp32。两个方向各有一个必须守住的坑:
          🚨 出必须抬 —— _h2d_result 用 gpu_out.dtype 建 _hout(和 fp32 pinned 相加)。
             留着半精, CPU 腿那条 fp32 结果会被这次相加一起截成半精,
             凭空多出一笔【不属于任何量化】的误差, 红线立刻花。
          🚨 进必须降 —— cat 档是 (x*gate) 拼成 xin 去 @ 半精权重; 只要 x 或 gate
             还有一个是 fp32, torch.mm 就报 dtype 不匹配(它不做隐式提升)。
        所以 gate 的降精放在 _hot_gate 的调用点上, 而不是塞进 _hot_gate 本身
        (_hot_gate 的 T9 越界回归测的是它的【形状】语义, 别把 dtype 混进去)。"""
        if self.gpu_expert_count == 0:        # ✅ P0-②: 纯 CPU 基线 => GPU 腿就是零块
            return x.new_zeros(x.shape[0], self.out_dim)
        dt = self._dt
        if dt is torch.float32:
            return self._hot_compute_inner(x, topk_scores, topk_indices, impl)
        return self._hot_compute_inner(x.to(dt), topk_scores, topk_indices, impl, dt).float()

    def _hot_compute_inner(self, x, topk_scores, topk_indices, impl=None, dt=None):
        impl = impl or self.gpu_impl
        dt = dt or torch.float32
        bs = x.shape[0]
        if self.kind == "swiglu":
            return self._hot_compute_swiglu(x, topk_scores, topk_indices, impl, dt)
        if impl == "sparse":                       # 🚨 T5 原状, 只留着做 A/B
            out = torch.zeros(bs, self.expert_size, device=x.device, dtype=x.dtype)
            for eid in range(self.gpu_expert_count):
                mask = (topk_indices == eid)
                if not mask.any():
                    continue
                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                sub = x[token_idx] @ self._w_hot(eid, "w")
                sub = sub * topk_scores[token_idx, slot_idx].unsqueeze(1)
                out.index_add_(0, token_idx, sub)
            return out

        self._ensure_hot_weights(x)
        gate = self._hot_gate(topk_scores, topk_indices)          # (bs, gc)
        if dt is not torch.float32:
            gate = gate.to(dt)
        if impl == "cat":
            # xin[b, e*hidden+h] = gate[b,e]*x[b,h]  ->  一趟 (bs, gc*hidden)@(gc*hidden, E)
            xin = (x.unsqueeze(1) * gate.unsqueeze(-1)).reshape(bs, -1)
            return xin @ self._wcat
        if impl == "bmm":
            # 只算 gc 份 GEMM(gc 倍算力, 而不是 gc^2 倍), 再沿专家轴加权求和
            y = torch.bmm(x.unsqueeze(0).expand(self.gpu_expert_count, bs, self.hidden_size),
                          self._wstack)                            # (gc, bs, E)
            return (y * gate.transpose(0, 1).unsqueeze(-1)).sum(0)
        if impl == "traffic":
            raise NotImplementedError(
                "乙-2d 的 traffic 档只做了 swiglu(真形状那档)。gemm 玩具档请用 cat。")
        raise ValueError(f"未知 gpu_impl: {impl}")

    def _hot_compute_swiglu(self, x, topk_scores, topk_indices, impl, dt=None):
        # ✅ 乙-2c: dt 由 _hot_compute 传下来(fp32 = 原状)。中间的 g/u/h 跟着 x 一起
        #    落在半精上 —— 这不是妥协, 这就是真形状跑 bf16 权重时的实际数值路径。
        #    🚨 已知风险(乙-2 量过): swiglu 中间量级 ~7e4, 【fp16 上限 65504 会溢出】,
        #       bf16 有位宽无精度(尾数 8bit => 相对 ~4e-3)。判决表要把这俩分开看。
        dt = dt or torch.float32
        """✅ 乙-2: GPU 热专家的 SwiGLU 半边。三种 impl 全部保留 —— 否则 bench 的
        T4 基线(sparse) / 全开(cat) 在 swiglu 下就没法对比了(方法论第 8 条)。

        🚨 T6 那个"cat 技巧"在 SwiGLU 下【只能保住一半】:
           gemm 时 out = Σ_e gate_e * (x @ W_e) = x @ (Σ_e gate_e * W_e) —— 激活是线性的,
           所以能先把跨专家的加和折叠进一趟 GEMM。SwiGLU 的 silu 在【专家内部】,
           先加后激活 ≠ 先激活后加, 所以折叠不了。
           但 gate/up 那一层还是线性的 => 仍然能把 gc 份 GEMM 折成【一趟】大 GEMM
           (bs,D)@(D,gc*2N); down 那一层退回 per-expert bmm(gc 份, 不是 gc^2 份)。
           形状全静态 => 照样能整段录进 CUDA Graph(T7 的收益不会丢)。"""
        bs = x.shape[0]
        gc = self.gpu_expert_count
        if impl == "sparse":                      # 🚨 T5 原状口径, 只留着做 A/B(2 次主机往返/专家)
            out = torch.zeros(bs, self.out_dim, device=x.device, dtype=x.dtype)
            for eid in range(gc):
                mask = (topk_indices == eid)
                if not mask.any():
                    continue
                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                xi = x[token_idx]
                sub = (F.silu(xi @ self._w_hot(eid, "w")) * (xi @ self._w_hot(eid, "u"))) \
                    @ self._w_hot(eid, "d")
                sub = sub * topk_scores[token_idx, slot_idx].unsqueeze(1)
                out.index_add_(0, token_idx, sub)
            return out

        if impl == "traffic":                     # ✅ 乙-2d: 按真流量读权重
            return self._hot_compute_traffic(x, topk_scores, topk_indices)

        self._ensure_hot_weights(x)
        gate = self._hot_gate(topk_scores, topk_indices)          # (bs, gc)
        if dt is not torch.float32:
            gate = gate.to(dt)
        if impl == "cat":
            gu = (x @ self._wcat_gu).view(bs, gc, -1)             # 1 张 GEMM -> (bs, gc, 2N)
            g, u = gu.chunk(2, dim=-1)
            h = F.silu(g) * u                                     # (bs, gc, N)
            y = torch.bmm(h.transpose(0, 1), self._wstack_d)      # (gc, bs, D)
            return (y * gate.transpose(0, 1).unsqueeze(-1)).sum(0)
        if impl == "bmm":
            gu = torch.bmm(x.unsqueeze(0).expand(gc, bs, self.hidden_size),
                           self._wstack_gu)                       # (gc, bs, 2N)
            g, u = gu.chunk(2, dim=-1)
            y = torch.bmm(F.silu(g) * u, self._wstack_d)           # (gc, bs, D)
            return (y * gate.transpose(0, 1).unsqueeze(-1)).sum(0)
        raise ValueError(f"未知 gpu_impl: {impl}")

    # ============================================================
    # ✅ 乙-2d: traffic-sparse GPU 半边(按【本帧真流量】读权重)
    # 稠密档每帧读 gc 个专家的三张矩阵; 这一档每帧只读"真命中 GPU 的 (token,slot) 数"
    # 个专家。省的是字节, 不动路由/量化/放置/精度 => idx 逐位不变、KL 只换归约序。
    # 🚨 省字节的条件 = bs*top_k < gc(bs 大了稠密才划算, 见 gpu_traffic.py 文件头)。
    # ============================================================
    def _ensure_traffic_bufs(self, bs, device):
        """traffic 档的静态缓冲。键带 dtype, 换降精就重建(和 _wcat 一样是"每引擎一份")。

        🚨🚨 这两块必须在 CUDA Graph【外面】分配(_ensure_graphs 里先调一次)。
           录图期间的分配落在 graph 私有显存池里, 池子会被后面的图复用 ——
           把长期复用的 scratch 留在池里 = 等着被别的图覆写。
        🚨 h 缓冲跟着 gpu_dtype 走(和稠密档"中间量落在半精上"同口径);
           part 缓冲一律 fp32(最后一加在 fp32 里做, 这点和稠密档不同, 写进判决表)。"""
        key = (bs, self._dt, device)
        if self._tbuf_key == key and self._tH is not None:
            return
        slots = bs * self.top_k
        self._tH = torch.empty(slots, self.expert_size, device=device, dtype=self._dt)
        self._tPART = torch.empty(slots, self.out_dim, device=device, dtype=torch.float32)
        self._tbuf_key = key

    def _hot_compute_traffic(self, x, topk_scores, topk_indices):
        from gpu_traffic import traffic_hot        # 延迟导入: triton import ~0.4s
        _bs = int(x.shape[0])
        if _bs not in self._traffic_warn_bs and _bs * self.top_k > self.gpu_expert_count:
            # 不挡, 只提醒: 这一档是按"每 slot 读一遍专家权重"算的, bs 大了比重读还贵。
            print(f"  🚨[traffic] bs={_bs} × top_k={self.top_k} = {_bs * self.top_k} 个 slot 已经多于 "
                  f"gc={self.gpu_expert_count} => 同一张权重要被多个 slot 各读一遍, "
                  f"这一档只在 bs*top_k < gc 时比稠密省字节(见 gpu_traffic.py 文件头)")
            self._traffic_warn_bs.add(_bs)
        self._ensure_hot_weights(x)
        self._ensure_traffic_bufs(x.shape[0], x.device)
        return traffic_hot(self, x, topk_scores, topk_indices, self._tH, self._tPART)

    # ---------------- CUDA Graph: 把主机从热路径上彻底赶走 ----------------
    def _gate_wt_dev(self, device):
        """✅ T11: gate.weight 的预转置 (D, EP) 连续版, 按 device 缓存。
        🚨 为什么要懒建: _gate_wt 是普通 python 属性(不是 buffer), 调 .cuda() 时
           nn.Module 根本不会搬它 —— 在 ctor 里建好就会变成「CPU 权重 + GPU 激活」
           这种 Illegal memory access / 隐式同步。
        🚨 为什么必须转置: tl.dot 的 B 操作数按列取, 直接跨步读 (E,D) 完全不能合并
           —— 实测同一张卡 24.5us(跨步) vs 9.6us(连续)。权重静态, 只转一次。"""
        if self._gate_wt is None or self._gate_wt.device != device:
            self._gate_wt = self._pre.gate_wt(self).to(device)
        return self._gate_wt

    def _route_quant(self, x):
        """图内前段: 门控 -> softmax -> topk -> 激活量化。
        全是固定形状、零主机分支, 所以能整段录进 graph。
        ✅ T11: fused 档换成 2 张 triton kernel(逐位等价, 见 pre_fused.py 文件头)。"""
        if self._pre is not None:
            return self._pre.route_quant_fused(
                self, x, self._pre.alloc_pre_out(self, x.shape[0], x.device),
                self._gate_wt_dev(x.device))
        scores = F.softmax(self.gate(x), dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        x_q, x_scales = self._quantize_act(x)
        return topk_scores, topk_indices, x_q, x_scales

    def _warmup_on_side_stream(self, fn):
        cap = torch.cuda.Stream()
        cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cap):
            for _ in range(3):                     # cublas 启发式/workspace 必须先落地
                fn()
        torch.cuda.current_stream().wait_stream(cap)
        torch.cuda.synchronize()

    def _ensure_graphs(self, x):
        """scope="hot"  : 只录热专家 GEMM(T6b)
        scope="full" : 录两张 —— 前段(路由+量化) / 后段(热专家 GEMM)。
            🚨 分两张而不是一张: CPU worker 只等前段(~0.05ms)就能开工,
               不用等热专家 GEMM, 两边的重叠保住。
        🚨 必须在 with torch.cuda.stream(...) 之外调用(里面有 synchronize)。"""
        if self.graph_scope == "off":
            return
        self._ensure_hot_weights(x)
        if self.gpu_impl == "traffic" and self.kind == "swiglu" and self.gpu_expert_count:
            # ✅ 乙-2d: scratch 必须在池外(见 _ensure_traffic_bufs 的 🚨🚨)
            self._ensure_traffic_bufs(x.shape[0], x.device)
        key = (x.shape[0], self.graph_scope, self._wver, x.device)
        if self._gkey == key:
            return
        bs, dev = x.shape[0], x.device
        self._sx = torch.empty_like(x)
        self._sx.copy_(x)
        self._g_pre = None
        if self.graph_scope == "full":
            g1 = torch.cuda.CUDAGraph()
            self._warmup_on_side_stream(lambda: self._route_quant(self._sx))
            with torch.cuda.graph(g1):
                # 这些输出活在 graph 的显存池里: 只要引用还在, 分配器就不会回收
                self._sts, self._sti, self._sxq, self._sxs = self._route_quant(self._sx)
            self._g_pre = g1
        else:
            self._sts = torch.empty(bs, self.top_k, device=dev, dtype=torch.float32)
            self._sti = torch.empty(bs, self.top_k, device=dev, dtype=torch.int64)
            self._sxq = self._sxs = None
            self._sts.zero_(); self._sti.zero_()
        if self.gpu_expert_count == 0:
            # ✅ P0-②: 纯 CPU 基线 —— 后段【没有活可录】。给一块常驻零缓冲,
            #    让 _finish 那句 add(gpu_out, cpu_result) 结构上不变(它付的 H2D+add
            #    是真成本, 不该被这条基线白躲掉)。
            self._g_hot = None
            if self._shot is None or self._shot.shape[0] != bs or self._shot.device != dev:
                self._shot = torch.zeros(bs, self.out_dim, device=dev)
            self._gkey = key
            return
        g2 = torch.cuda.CUDAGraph()
        self._warmup_on_side_stream(
            lambda: self._hot_compute(self._sx, self._sts, self._sti))
        with torch.cuda.graph(g2):
            self._shot = self._hot_compute(self._sx, self._sts, self._sti)
        self._g_hot, self._gkey = g2, key

    def _d2h_to_pins(self, x_q, x_scales, ti, ts):
        """把路由/量化结果 DMA 到 pinned 内存。full 档走第三张图(1 次 replay)。
        抽成方法是为了让 budget_check.py 的手抄版和真路径【逐行同构】, 否则分账表
        量的是老写法, T7 的收益在明细里看不见。"""
        self._ensure_d2h_graph()
        if self._g_d2h is not None:
            self._g_d2h.replay()
            return
        p = self._pin
        p["xq"].copy_(x_q, non_blocking=True)
        p["sc"].copy_(x_scales, non_blocking=True)
        p["idx"].copy_(ti, non_blocking=True)
        p["scr"].copy_(ts, non_blocking=True)

    def _h2d_result(self, gpu_out):
        """CPU 专家的结果回 GPU 并与热专家结果相加。预分配落点省一次分配。
        🚨 不用 torch.add(..., out=): 那样每轮返回同一块缓冲, 调用方存下来的
           结果下一轮就被覆写 —— 量出来的精度是假的。"""
        bs = self._pin["out"].shape[0]
        if (self._hout is None or self._hout.shape[0] != bs
                or self._hout.device != gpu_out.device):
            self._hout = torch.empty(bs, self.out_dim,
                                     device=gpu_out.device, dtype=gpu_out.dtype)
        self._hout.copy_(self._pin["out"], non_blocking=True)
        return torch.add(gpu_out, self._hout)

    def _ensure_d2h_graph(self):
        """✅ T7-③: 把 L2 里"把路由/量化结果 DMA 到 pinned 内存"的 4 次 copy_ 录成一张图。
        微卡尺实测主机入队: 4 次 copy_ = 0.0218ms  ->  1 次 replay = 0.0064ms。

        🚨 合法前提: 图的源地址必须永远不变。只有 full 档下 x_q/x_scales/topk
           才是 _g_pre 的【静态输出】; off/hot 档它们每次都是新分配的 eager 张量,
           地址会变, 录进去就是静默读旧内存 —— 所以那两档直接不录。
        🚨 目标 pinned 缓冲换 bs 时会重分配 => 缓存键必须带上 _pin_bs。"""
        if self.graph_scope != "full" or self._pin is None:
            self._g_d2h = None
            return
        key = (self._gkey, self._pin_bs)
        if self._g_d2h is not None and self._d2h_key == key:
            return
        p = self._pin
        pairs = [(p["xq"], self._sxq), (p["sc"], self._sxs),
                 (p["idx"], self._sti), (p["scr"], self._sts)]
        self._warmup_on_side_stream(
            lambda: [d.copy_(src, non_blocking=True) for d, src in pairs])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for d, src in pairs:
                d.copy_(src, non_blocking=True)
        self._g_d2h, self._d2h_key = g, key

    def _ensure_tail_graph(self):
        """✅ T12-②: 把"CPU 结果 H2D + 与热专家结果相加"这 2 次主机调用折成 1 次 replay。
        合法前提 = 三个地址都静态: _shot(热专家图的输出) / _pin["out"] / _hout。
        scope="off" 时 _shot 每次都是新分配的 eager 张量 => 录进去就是读旧显存, 直接不录。

        🚨 代价(这是【接口语义】变化, 不是实现细节): 返回的 out 活在图私有显存池里,
           下一帧 replay 会把它覆写。CUDA Graph 固有语义(vLLM 的 cudagraph 输出同理),
           但调用方若要把输出留着必须自己 .clone()。
           check_redlines.py 里有一条专门的"连续两帧覆写"探针把它【打印】出来,
           免得日后有人拿这条路径当普通张量用, 静默拿到上一帧的数。"""
        if (not self.tail_graph or self.graph_scope == "off"
                or self._pin is None or self._shot is None):
            self._g_tail = None
            return
        key = (self._gkey, self._pin_bs)
        if self._g_tail is not None and self._gtail_key == key:
            return
        self._h2d_result(self._shot)          # 先按 bs 把 _hout 分配出来
        def tail():
            self._hout.copy_(self._pin["out"], non_blocking=True)
            return torch.add(self._shot, self._hout)
        self._warmup_on_side_stream(tail)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = tail()
        self._g_tail, self._tail_out, self._gtail_key = g, out, key

    def _finish(self, gpu_out):
        """GPU 半边的收尾: CPU 结果回 GPU 并与热专家结果相加。"""
        if self._g_tail is not None:
            self._g_tail.replay()
            return self._tail_out
        return self._h2d_result(gpu_out)

    def _prepare(self, x):
        """取 (topk_scores, topk_indices, x_q, x_scales)。
        图内: 2 次主机调用(1 copy + 1 replay)搞定 13 个 aten 算子。"""
        if self.graph_scope == "full":
            self._ensure_graphs(x)
            self._sx.copy_(x, non_blocking=True)
            self._g_pre.replay()
            return self._sts, self._sti, self._sxq, self._sxs
        topk_scores, topk_indices = self.route(x)
        x_q, x_scales = self._quantize_act(x)
        if self.graph_scope == "hot":                 # 只有 GEMM 进图, 路由/量化留在 eager
            self._ensure_graphs(x)
            self._sx.copy_(x, non_blocking=True)
            self._sts.copy_(topk_scores, non_blocking=True)
            self._sti.copy_(topk_indices, non_blocking=True)
        return topk_scores, topk_indices, x_q, x_scales

    def _hot_half(self, x, topk_scores, topk_indices):
        if self.graph_scope == "off":
            return self._hot_compute(x, topk_scores, topk_indices)
        if self._g_hot is None:          # ✅ P0-②: gc=0 => 后段图根本没录(_ensure_graphs 里挡的)
            return self._shot
        self._g_hot.replay()
        return self._shot

    def _gpu_hot_experts(self, x, topk_scores, topk_indices, stream):
        """单独量 GPU 半边用(budget_check.py)。full 档连前段一起量。"""
        if self.graph_scope != "off":
            self._ensure_graphs(x)
        if self.graph_scope == "full":
            with torch.cuda.stream(stream):
                self._sx.copy_(x, non_blocking=True)
                self._g_pre.replay()
                self._g_hot.replay()
                out = self._shot
                stream.synchronize()
            return out
        if self.graph_scope == "hot":
            self._sts.copy_(topk_scores, non_blocking=True)
            self._sti.copy_(topk_indices, non_blocking=True)
            with torch.cuda.stream(stream):
                self._sx.copy_(x, non_blocking=True)
                self._g_hot.replay()
                out = self._shot
                stream.synchronize()
            return out
        with torch.cuda.stream(stream):
            out = self._hot_compute(x, topk_scores, topk_indices)
            stream.synchronize()
        return out

    @property
    def cpu_w(self):
        """按 self.w_pack 选权重张量。C++ 侧按 dtype 自动选 kernel(int8=>signed),
        所以这里【只要选对张量就不会配错】。"""
        return self.cpu_experts_q_s8 if self.w_pack == "signed" else self.cpu_experts_q

    # ✅ 乙-2: SwiGLU 三张矩阵的 int8 版。Wg 就是历史那张 W(见 ctor 的约定),
    #    Wd 只有 signed 一份 —— 入口 C 明确挡住 biased, 所以不走 self.w_pack。
    @property
    def cpu_w_g(self):
        return self.cpu_w

    @property
    def cpu_w_u(self):
        return self.cpu_up_q

    @property
    def cpu_w_d(self):
        return self.cpu_down_q

    def expert_bytes(self):
        """一个专家在 CPU 侧要读多少字节。字节墙/工作集/放置规则的【唯一】算法入口,
        别再在各 bench 里手算 N*D —— 那正是乙-2 会把两套口径算岔的地方。"""
        return self.expert_size * self.hidden_size * self.n_mat

    # ---------------- 乙-2: CPU 半边 = SwiGLU 三矩阵 ----------------
    def _cpu_swiglu_into(self, x_cpu, x_scales_cpu, ti_cpu, ts_cpu, out_cpu, release_gil):
        """CPU 那几条腿的 SwiGLU(入口 C)。🚨 与 gemm 版同一条铁律: 传进来的必须全是 CPU
        张量, C++ 函数体内零 CUDA 调用; out 由 C++ 侧 zero_() 后再累加(第三阶段是 +=)。"""
        if _HDBG:                                    # 归因钩子(见文件头 HDBG 说明)
            print(f"[HDBG] 进 C++ 前 idx={ti_cpu.tolist()} scr={ts_cpu.tolist()} "
                  f"xq={tuple(x_cpu.shape)} gc={self.gpu_expert_count} n_cpu={self.n_cpu} "
                  f"top_k={self.top_k} nt={self.n_threads} tb={self.tok_block} "
                  f"sched={self.cpu_sched} scope={self.graph_scope}", flush=True)
        avx_gemv_ext.avx2_moe_cpu_experts_int8_swiglu_into(
            # 🚨 实参顺序照 C++ 签名: 激活 -> 门控 -> 三张权重 -> 三个 scale -> out
            x_cpu, x_scales_cpu, ti_cpu, ts_cpu,
            self.cpu_w_g, self.cpu_w_u, self.cpu_w_d,
            self.cpu_experts_scale, self.cpu_up_scale, self.cpu_down_scale,
            out_cpu, self.gpu_expert_count, release_gil,
            self.n_threads, self.tok_block, -1, self.cpu_sched)
        return out_cpu

    # ------------------------------------------- sync 基线
    def _forward_sync(self, x):
        ts, ti, x_q, x_scales = self._prepare(x)
        out = self._hot_half(x, ts, ti)
        if self.n_cpu == 0:                  # ✅ P0-②: 纯 GPU 基线, sync 口径也要能跑
            self._simulate_cpu_delay()
            return out
        if self.kind == "swiglu":
            # sync 口径没有 pinned 缓冲, 就地给一块 CPU 落点(.cpu()/.to() 在这里是合法的,
            # sync 档本来就不是"零 CUDA 调用"的那条路)
            buf = torch.empty(x.shape[0], self.out_dim, dtype=torch.float32)
            self._cpu_swiglu_into(x_q.cpu(), x_scales.cpu(), ti.cpu(), ts.cpu(), buf, False)
            self._simulate_cpu_delay()
            return out + buf.to(x.device)
        out = out + avx_gemv_ext.avx2_moe_cpu_experts_int8(
            # 🚨 实参顺序照 C++ 签名 (x, x_scales, gate_idx, gate_scores, ...): 索引在前
            x_q, x_scales, ti, ts,
            self.cpu_w, self.cpu_experts_scale,
            self.gpu_expert_count, False, self.n_threads, self.tok_block,
            -1, self.cpu_loop, self.cpu_sched)
        self._simulate_cpu_delay()
        return out

    # ------------------------------------------- L0 / L1
    def _forward_async_legacy(self, x, release_gil):
        ts, ti, x_q, x_scales = self._prepare(x)

        def cpu_worker():
            if self.kind == "swiglu":
                # 入口 C 只吃 CPU 张量 => .cpu() 发生在这里。这正是 L0/L1 的老毛病
                # (阻塞式 D2H + CUDA 调用在工作线程里), L2 才把它消掉 —— 故意保留做对照。
                buf = torch.empty(x.shape[0], self.out_dim, dtype=torch.float32)
                self._cpu_swiglu_into(x_q.cpu(), x_scales.cpu(), ti.cpu(), ts.cpu(),
                                      buf, release_gil)
                self._simulate_cpu_delay()
                return buf.to(x.device)
            # 入口 A 内部 .cpu() 走 legacy default stream, 会自己排到 _prepare 之后
            res = avx_gemv_ext.avx2_moe_cpu_experts_int8(
                x_q, x_scales, ti, ts,          # 🚨 索引在前, 门控在后
                self.cpu_w, self.cpu_experts_scale,
                self.gpu_expert_count, release_gil, self.n_threads, self.tok_block,
                -1, self.cpu_loop, self.cpu_sched)
            self._simulate_cpu_delay()
            return res

        fut = None if self.n_cpu == 0 else self.cpu_pool.submit(cpu_worker)
        gpu_out = self._hot_half(x, ts, ti)
        if fut is None:                      # ✅ P0-②: 纯 GPU 基线(gc=E), 没有 CPU 腿
            return gpu_out
        return gpu_out + fut.result()

    # ------------------------------------------- L2: 真并行
    def _forward_async_L2(self, x):
        bs = x.shape[0]
        ts, ti, x_q, x_scales = self._prepare(x)      # full 档: 图已经算好路由+量化
        self._ensure_pins(bs, x.device)
        p = self._pin

        # ① 主线程把 <40KB 的元数据 async DMA 到 pinned 内存
        self._d2h_to_pins(x_q, x_scales, ti, ts)
        # ✅ T8-②: 两种等法。main = 原状, 主线程 sync 之后再走下面的 replay
        #    (budget 实测这一段 = 0.038ms 纯主机干等);
        #    worker / worker_block = 只 record 一个 event 就往下跑, 让 worker 自己去等, 于是
        #    "D2H 落地" 和 "主线程 replay 热专家图" 重叠, 主线程不再干等。
        #    🚨 worker 里出现 CUDA 调用是有意破 L2 的"零 CUDA API"洁癖:
        #       Event.synchronize 会释放 GIL, 且 event 只 record 不查询, 不与
        #       worker 的 OMP 线程抢 CUDA 锁。要不要保留以 bench_scope_ab 的数为准。
        ev_d2h = None
        if self.d2h_wait.startswith("worker"):
            if self._ev_d2h is None:
                # 🚨 blocking=True => cudaEventBlockingSync: cudaEventSynchronize 会
                #    睡眠让核; 默认(False)是【自旋】—— 自旋会占掉一个核, 而这条路径
                #    上已经有 6 条 OMP worker + 1 条主线程在抢 6 个物理核。
                self._ev_d2h = torch.cuda.Event(
                    enable_timing=False,
                    blocking=(self.d2h_wait == "worker_block"))
            ev_d2h = self._ev_d2h
            ev_d2h.record()
        else:
            torch.cuda.current_stream().synchronize()   # 原状: 全流程唯一一次主机等待

        # ② 提交纯 CPU worker: 函数体内零 CUDA 调用 + 整块松开 GIL
        def cpu_worker():
            if ev_d2h is not None:
                ev_d2h.synchronize()      # ✅ T8-②: 等 DMA 落地(释放 GIL 的阻塞等)
            if self.kind == "swiglu":
                # ✅ 乙-2: 三矩阵版走的仍是"pinned 进 / pinned 出 + 零 CUDA 调用"
                self._cpu_swiglu_into(p["xq"], p["sc"], p["idx"], p["scr"], p["out"], True)
            else:
                avx_gemv_ext.avx2_moe_cpu_experts_int8_into(
                    p["xq"], p["sc"], p["idx"], p["scr"],
                    self.cpu_w, self.cpu_experts_scale, p["out"],
                    self.gpu_expert_count, True, self.n_threads, self.tok_block,
                    -1, self.cpu_loop, self.cpu_sched)
            self._simulate_cpu_delay()
            return p["out"]

        # ✅ T12-②: 尾图(默认关)在这里录, 而不是在 stream ctx 里 / worker 在飞时录
        #    —— 录图流程含 torch.cuda.synchronize(), 放进 stream 上下文或和 worker 的
        #    pinned 写并发都是给自己挖坑。它有缓存, 只有首帧/换 bs 时真录。
        self._ensure_tail_graph()
        if self.n_cpu == 0:
            # ✅ P0-②: 纯 GPU 基线(gc=E)—— CPU 腿一条腿都没有。
            #    不能把 0 长的 int8 权重表派给 worker(C++ 那边按 wq.size(0) 建指针数组),
            #    也不能指望有人清零 pinned 落点: 平时 out.zero_() 是 C++ 干的, 现在得自己来。
            #    注: cpu_delay_ms 在这条基线上不生效 —— 没有 CPU 专家就没有"CPU 专家耗时",
            #    这不是漏算, 是这条基线的定义。
            p["out"].zero_()
            fut = None
        else:
            fut = self.cpu_pool.submit(cpu_worker)

        # ✅ P0-②: overlap=False = 主线主张的反面。CPU 腿跑完之前 GPU 一条指令都不发
        #    => 每帧应当变成 CPU腿 + GPU腿 + 固定串行(sum 而不是 max)。
        #    除"提交顺序"外什么都没动: 同一条 worker、同一张图、同一组线程。
        if not self.overlap:
            if fut is not None:
                fut.result()                 # 主线程在这儿阻塞, GPU 全程空转
            gpu_out = self._hot_half(x, ts, ti)
            out = self._finish(gpu_out)
            torch.cuda.current_stream().synchronize()
            return out

        # ③ 主线程同时把 GPU 热专家丢出去
        if self.l2_stream == "single":
            # ✅ T12-①: 全回默认流。热专家 replay 与它的源数据(pre 图 / d2h 图)本来就
            #    被默认流的顺序天然隔开, 不需要 Event 跨流同步。
            gpu_out = self._hot_half(x, ts, ti)
            if fut is not None:              # ✅ P0-②: gc=E 时没有 CPU 腿
                fut.result()
            out = self._finish(gpu_out)
            # 🚨 必须等这条流: 否则下一帧的 D2H 会提前覆写 pinned 缓冲
            torch.cuda.current_stream().synchronize()
            return out

        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(self.gpu_stream):
            self.gpu_stream.wait_event(ev)
            gpu_out = self._hot_half(x, ts, ti)
            # ④ CPU 结果从 pinned 内存 async H2D, 排在同一条流上, 顺序天然正确
            if fut is not None:              # ✅ P0-②
                fut.result()
            out = self._finish(gpu_out)          # ✅ T7-③ 预分配 H2D 落点
            self.gpu_stream.synchronize()   # 也保护 pinned buf 不被下一轮提前覆写
        return out


# ============ 测试台 ============
def plan_cpu_residency(hit_counts, weight_bytes, gpu_expert_count=0,
                       l3_bytes=None, budget_frac=0.5):
    """✅ T9-② 放置规则: 求最小的 GPU 容量, 使 CPU 侧【每帧真会被摸到】的权重字节
    <= budget_frac x L3。返回 (gpu_needed, cpu_traffic_bytes)。
    🚨 原型约束"GPU 只能放 id < gpu_expert_count", 所以只能沿 id 顺序往上抬。

    三条从 T9 实测里长出来的规则:
      规则1(主菜, 值 2.5x): 热专家应该在 GPU —— 配置A(热在GPU) 0.356ms vs
        配置C(热在CPU) 0.905ms。GPU 容量给得起就用这条。
      规则2(GPU 容量锁死时的兜底, 值 -13.8%): 把"会被摸到的 CPU 权重"压回预算。
        🚨 预算记在【流量>0 的专家】上, 不是【CPU 常驻专家数】上: 一个专家只要摊到
        1 个槽位, 它那 2MB 就得整趟过一遍(一个 token 也要读完整张权重); 反过来 0
        流量的专家连读都不读, 完全免费。配置C 的三档正好把机制钉死:
            gc=2  有流量 4 个 =  8MB -> 1.051ms
            gc=3  有流量 3 个 =  6MB -> 0.905ms  (-13.8%)
            gc=4  有流量 3 个 =  6MB -> 0.941ms  (+3.4% vs gc=3)
        gc=3 与 gc=4 的 CPU 侧完全一样(专家 3 零流量), 唯一区别是 GPU 多算一个专家
        的 dense GEMM => 那 +3.4% 就是"多给 GPU 一个专家"的标价, 所以一达标就停手。
      规则3(budget_frac=0.5 而不是 0.75): 微基准里 8MB(=0.67L3) 还能跑 111GB/s,
        但那是空机器单 kernel; 端到端 8MB 那档实测就比 6MB 慢 13.8% —— 桌面上
        微信/Edge/VSCode 也在吃 L3。所以留一半, 按 6MB 用。
    """
    if l3_bytes is None:
        l3_bytes = L3
    budget = int(l3_bytes * budget_frac)
    gc = max(int(gpu_expert_count), 0)
    n = len(hit_counts)

    def traffic(g):
        return sum(weight_bytes for e in range(g, n) if hit_counts[e] > 0)

    while gc < n and traffic(gc) > budget:
        gc += 1
    return gc, traffic(gc)


# ============================================================
# ✅ T13: 放置规则 v2 —— 给 GPU 半边也定价(两条腿的 roofline)
#   下面两个带宽是【本机实测标定值】, 换机器要重标(bench_halves.py 会重新量):
#     42GB/s  = CPU kernel 读 int8 权重的有效带宽。DDR4-2666 双通道理论顶就是
#               ~42GB/s, 端到端和桌面抢核之后还是 42(T8 的结论)。
#     250GB/s = 热专家 dense GEMM 读 fp32 权重的有效带宽。RTX3060 峰值 ~360GB/s,
#               实测每多一个 GPU 专家 +32us/帧(8MB fp32) => 8MB/32us = 250GB/s。
#               🚨 别填 360: bs=32 这种 skinny GEMM 吃不满峰值, 填了就会高估
#                  GPU 侧成本 => 规则不敢抬 gc => 少赚 12%。
CPU_BW_EFF = 42e9
GPU_BW_EFF = 250e9
#     288GMAC/s = CPU 侧 int8 kernel 的算力上限(T14 卡尺实测: 单核 13.5 MAC/cyc、
#               6 线程 265~288 GMAC/s, 且与"权重被重扫几遍"无关 => 它是【第二堵墙】)。
#               int8 点积里"一个权重元素 = 一次 MAC", 所以同一个 eb 既是字节数也是 MAC 数:
#               每 slot 的算力墙 = eb / 288GMAC/s = 9.2us(swiglu D576/N1536)。
#               交叉点 = 9.2us/63us ≈ 每专家 6.8 个 slot(与 OPTLOG §T8-① 的"约 5 个"同量级)。
CPU_MAC_RATE = 288e9


def placement_curve(hit_counts, cpu_weight_bytes, gpu_weight_bytes,
                    gc_min=0, cpu_bw=CPU_BW_EFF, gpu_bw=GPU_BW_EFF):
    """每个候选 gc 的两半边成本(us), 返回 {gc: (cpu_us, gpu_us, max_us)}。
    CPU 半边 = 这帧真会被摸到的 int8 权重字节 / 42GB/s;
    GPU 半边 = gc 张 fp32 权重 / 250GB/s(dense cat 的 GEMM 每帧把 wcat 整条读一遍)。
    🚨 两条腿取 max 不是相加: CPU worker 和 GPU 是【重叠】的(L2 的整个设计就是让它俩
       并行), 相加会把收益算两遍。固定串行的主机开销(~160us, T7/T11/T12 砍剩下的)
       对每个 gc 都一样, 在 argmin 里自己抵消, 所以不用建模。"""
    n = len(hit_counts)

    def traffic(g):
        return sum(cpu_weight_bytes for e in range(g, n) if hit_counts[e] > 0)

    out = {}
    for g in range(max(int(gc_min), 0), n + 1):
        c = traffic(g) / cpu_bw * 1e6
        v = g * gpu_weight_bytes / gpu_bw * 1e6
        out[g] = (c, v, max(c, v))
    return out


def plan_gpu_capacity(hit_counts, cpu_weight_bytes, gpu_weight_bytes,
                      gpu_expert_count=0, cpu_bw=CPU_BW_EFF, gpu_bw=GPU_BW_EFF,
                      tie_band=0.08):
    """✅ T13 放置规则 v2: gc = argmin_gc max(CPU半边, GPU半边), 平手取更小的 gc。
    返回 (gc, 预测关键路径us)。

    为什么要换掉 T9 那条"CPU 被摸字节 <= 0.5xL3 就停手":
      它只有一条腿 —— 只看 CPU 字节, 完全没给 GPU 半边定价, 于是它的停手条件是
      "CPU 侧不再超预算", 而不是"再买一个 GPU 专家已经不划算了"。T9 当时定 0.5 是
      拿 T7 那台机器标出来的, 恰好蒙对了(那时 GPU 半边很贵, 早停是对的)。
      T11(前段融合)+T12(单流)把 GPU 半边和主机侧砍下去之后, 同一套系数立刻过期:
      配置A gc=3 实测 0.286 vs 规则选的 gc=2 0.329, 11 轮配对 10/11 同向(+12.4%)。
      🚨 缺的不是系数, 是【另一半的成本】—— 单系数修不好: 要让规则给 A 选 gc=3
         就得把预算压到 4MB 以下, 那配置C 会被推到 gc=5(实测比最优慢 2.6%)。

    两条腿的模型一次跑通两个配置(bench_halves.py 里那 8 个格子的实测 argmin):
      A: max(143,64) / max(95,96) / max(95,128) / max(95,160) -> 143/96/128/160 => gc=3 ✓
      C: max(190,64) / max(143,96) / max(143,128)/max(143,160) -> 190/143/143/143 => gc=3 ✓
      (单位 us; 第一个数=CPU半边 第二个=GPU半边。C 的后三档 CPU 半边持平, argmin
       落在最小的那一档, 正好是实测最优 —— "平手取小 gc"这条不是随便定的)
    拟合质量: 配置A 端到端 - max(两半边) = 0.171/0.161/0.163/0.154ms, 四个 gc 档
    方差只有 +-4% => "固定串行开销 + 两条腿取 max"这个结构在本机是站得住的。

    🚨 适用边界(别拿它当万能公式):
      1) 只在被测过的 gc∈[2,5]、bs∈[1,32] 内验证过; 超出范围它只会外推, 不会自知。
      2) GPU 半边没有 bs 项, 因为这块 GEMM 是【权重带宽 bound】(32x1024 @ 1024xN,
         算力占用不到 1%), 不是"bs 无所谓"。真上到大 bs(计算 bound)必须补一项。
      3) tie_band=0.08 是【模型分辨力】而不是物理常数: 见下面破平手那段。
      4) 它默认 CPU 侧权重是 int8(2MB/专家)、GPU 侧是 fp32(8MB/专家)。哪天 CPU 侧
         换了打包或 GPU 侧上了 fp16/int8 权重, 两个字节数都要跟着改, 不然规则会
         拿旧的价格做决定 —— 这类"改了实现没改标定"的账, T13 就是活教材。
    """
    cur = placement_curve(hit_counts, cpu_weight_bytes, gpu_weight_bytes,
                          gc_min=gpu_expert_count, cpu_bw=cpu_bw, gpu_bw=gpu_bw)
    best = min(cur, key=lambda g: (cur[g][2], g))
    band = cur[best][2] * (1.0 + tie_band)
    cands = [g for g in cur if cur[g][2] <= band]

    def traffic(g):
        return sum(1 for e in range(g, len(hit_counts)) if hit_counts[e] > 0)

    # 🚨 平手怎么破, 是这条规则的第二半, 不是收尾细节。实测出来的分辨力:
    #    bs=16 配置A 的 gc=2(95us) 与 gc=3(101us) 模型只差 1us(6%), 实测差 12.8%
    #    —— 说明在这个尺度上【模型没有分辨力】, 而它缺的恰恰是 CPU 那一侧的账
    #    (42GB/s 是"和桌面抢核"之后的带宽, 抖得比 GPU 侧那 32us/专家 厉害得多)。
    #    所以平手时优先【CPU 侧要搬的字节更少】的那一档, 再平手才取更小 gc。
    #    加上这一条之后, bs∈{1,4,16,32} x {A,C} 八个格子全部选中实测最优或
    #    只付 <=1.2%(噪声带内) 的代价; 不加它, bs=16/A 会多付 12.8%。
    gc = min(cands, key=lambda g: (traffic(g), g))
    return gc, cur[gc][2]



def placement_curve_v2(hit_counts, cpu_weight_bytes, gpu_weight_bytes,
                       gc_min=0, cpu_bw=CPU_BW_EFF, gpu_bw=GPU_BW_EFF,
                       impls=("cat", "traffic"), cpu_mac_rate=None):
    """放置规则 v2 的曲线: 每个候选 gc 在【每个 GPU 实现档】下的两条腿成本(us)。
    返回 (cur, cum_slots, cpu_bytes_of) —— cur[impl][gc] = (cpu_us, gpu_us, max_us)。

    cpu_mac_rate=None => CPU 腿只记字节墙(与 v1 逐字同价, 归档表不受影响);
    传 CPU_MAC_RATE => CPU 腿 = max(字节墙, 算力墙), 补的是 OPTLOG §T8-①/§T14 早就记下
    却没进规则的**第二堵墙**: "放置规则的 CPU 腿应当是 max(字节墙,算力墙), 现在只记了
    字节墙(配置C 会被严重低估)"。它的代价在离线重放里量出来了(bench_rule_v2 ② 表):
    只记字节墙时 v1/v2 在 配置C bs=32/128 上比实测最优贵 +10.8%/+101.6%(fp32)
    与 +195%/+297%(bf16) —— 因为放错时 CPU 腿既没少字节也没少 slot, 规则却以为它很轻。

    与 v1(placement_curve)唯一的差别在 GPU 半边: v1 只会按"稠密 cat 每帧读 gc 张权重"
    定价, 而乙-2d 之后 GPU 半边有两个实现档, 读字节的次数【不再是 gc 的函数】:
      cat     : gc 张专家权重 —— 稠密写法把"谁命中谁"降级成门控, 门控为 0 也整条读回来
      traffic : slots_gpu(gc) 张专家权重 —— 网格开在 (bs*top_k) 的流量上, 一个 slot
                读它自己那一张(见 gpu_traffic.py 文件头)
    slots_gpu(gc) = sum(hit_counts[:gc]) = 本帧落在 GPU 腿上的 (token,slot) 数。
    => traffic 的 GPU 半边与 gc 无关, 只与"本帧真有多少 slot 落 GPU"有关。把这一条
       接进去, 乙-2d 实测的两件事自动掉出来:
       ① bs 小时 GPU 半边对 gc 不再涨价(gc=3→8 涨幅 +70.5% → +8.4%)= 延迟腿退休;
       ② 放错时 GPU 腿 0 个 slot => 两档预测成本相同 => 稠密税被 CPU 腿的重叠藏掉
          (C 的 gc<=5 六格字节比 0.00 却只有 -0.6%~-12.9% —— max(两腿) 结构第一次
           做出定性预测并被证实, 见 OPTLOG 乙-2d 判决⑧)。

    🚨 口径: slots_gpu 是【上界】—— 同一个专家被多个 slot 命中时第二次起会被 L2 兜掉
       一部分, 所以 bs 大时这一版会【高估】traffic 的 GPU 半边(方向保守: 宁可退回 cat)。
       高估多少正是乙-2d 边界轮(bs=32, bs*top_k=64 > gc)要量的那把尺。
    """
    n = len(hit_counts)
    total = sum(int(v) for v in hit_counts)
    cum = []                                     # cum[g] = 落在 GPU 腿(专家序号 < g)的 slot 数
    s = 0
    for g in range(n + 1):
        cum.append(s)
        if g < n:
            s += int(hit_counts[g])

    def cpu_bytes(g):
        return sum(cpu_weight_bytes for e in range(g, n) if hit_counts[e] > 0)

    # CPU 腿的第二堵墙: 这帧落在 CPU 腿上的 slot 数 x 每 slot 的 int8 算力代价。
    # 🚨 用 max 不用相加: 一个专家的字节读与它的 MAC 是同一段 kernel 里重叠发生的
    #    (T14: MAC/s 与权重重扫趟数无关 => 重扫的字节不要钱)。
    slots_cpu = [total - c for c in cum]
    mac_us = (lambda g: slots_cpu[g] * cpu_weight_bytes / cpu_mac_rate * 1e6) \
        if cpu_mac_rate else (lambda g: 0.0)

    cur = {}
    for impl in impls:
        if impl == "traffic":
            f = lambda g: cum[g]
        elif impl in ("cat", "bmm", "sparse"):   # 稠密系: 读满 gc 张(sparse/bmm 是旧档, 同价)
            f = lambda g: g
        else:
            raise ValueError(f"未知实现档 {impl}(要的是 cat/bmm/sparse/traffic)")
        rows = {}
        for g in range(max(int(gc_min), 0), n + 1):
            c = max(cpu_bytes(g) / cpu_bw, mac_us(g) / 1e6) * 1e6
            v = f(g) * gpu_weight_bytes / gpu_bw * 1e6
            rows[g] = (c, v, max(c, v))
        cur[impl] = rows
    return cur, cum, cpu_bytes


def plan_gpu_capacity_v2(hit_counts, cpu_weight_bytes, gpu_weight_bytes,
                         gc_min=0, cpu_bw=CPU_BW_EFF, gpu_bw=GPU_BW_EFF,
                         tie_band=0.08, impls=("cat", "traffic"), kl_slot_cap=None,
                         grid=None, cpu_mac_rate=None):
    """✅ 放置规则 v2: 在【(gc, GPU 实现档) 这个二维格子】里取 argmin max(CPU腿, GPU腿)。
    返回 (gc, impl, 预测关键路径us, cur)。

    它补的是 v1(plan_gpu_capacity)缺的那半个价 —— v1 的 GPU 半边只会按 `gc x 权重字节`
    记, 于是乙-2d 之后它两件事都看不见:
      (a) 同一个 gc 有两档算法, 价是 `min(cat, traffic)`;
      (b) traffic 的价 = 本帧 GPU slot 数 x 一张权重, 与 gc 无关。
    实测后果(P0-④a / 乙-2d 两张归档表里的"规则选 gc"八条): v1 全表偏贵
    +14.5%~+37.6%, 而且 bs=1 时它照样把 gc 往大里推 —— 那里 traffic 已经把
    "gc 越大越慢"这条腿拆掉了。

    三条设计说明:
      1) 【零新增常数】: 两条带宽(42GB/s / 250GB/s)、tie_band、平手规则全部沿用 v1。
         自证: bs=1 fp32 gc=8 traffic 预测 = 2 slot x 10.6MB / 250GB/s = 84us + 固定
         串行 ~160us = 0.244ms, 实测 0.232/0.234ms; 同格 cat 预测 8 x 42us + 160us =
         0.496ms, 实测 0.433ms(高估 15%, 方向保守)。
      2) 平手仍按 v1 的老规矩: 优先【CPU 侧要搬的字节更少】, 再平手取更小 gc, 再平手取
         【cat】—— cat 是归档形态、traffic 在大 bs 侧还没被判决过, 平手时不赌新档。
      3) kl_slot_cap(默认 None=关掉)是质量腿的【约束】而不是价格: 0.0~1.0, 要求候选的
         CPU 腿 slot 占比 <= cap(放错档那种"多数流量还在 int8 上算"的格子会被踢掉)。
         为什么不在这版把它做成价格: KL 曲面现在只有四个 bs 的归档截面(④c: 放错多付
         5.11/15.1/52.0x), 拿它拟合系数就是过拟合 => 等 v2 判决轮拿到曲面再说。
      4) grid(默认 None = 不收口, 规则真值可以是任意 0..E): 取消 v1 在 bench_swap 里
         被【调用方】补的那道"网格内取最近档、平手取更小"疤痕。疤痕的实测代价见
         bench_rule_v2.py 的重放: A bs=1 bf16 规则真值 gc=2 被抹到 gc=1, 白付 +48.9%
         (同格网格内直选是 +0.0%)。传 grid 时规则【只在实测过的档里选】, 于是
         "取消疤痕"和"仍然可执行"同时成立 —— 这不是放宽规则, 是把疤痕搬进规则里。
      5) cpu_mac_rate(默认 None): 非 None 时 CPU 腿 = max(字节墙, 算力墙), 见
         placement_curve_v2 的说明。为什么默认关: 加了它就要回答"C 大 bs 那
         +195%/+297% 的偏贵到底能不能被它解释", 这句话必须由重放/判决轮来说,
         不能由我改常数来说(方法论 22)。重放结论决定它最终是默认 on 还是留开关。
    """
    cur, cum, cpu_bytes = placement_curve_v2(
        hit_counts, cpu_weight_bytes, gpu_weight_bytes, gc_min=gc_min,
        cpu_bw=cpu_bw, gpu_bw=gpu_bw, impls=impls, cpu_mac_rate=cpu_mac_rate)
    total_slots = sum(int(v) for v in hit_counts)
    allow = None if grid is None else {int(g) for g in grid}
    flat = [(impl, g) for impl in cur for g in cur[impl]
            if allow is None or g in allow]
    if not flat:
        raise ValueError("没有候选 gc(检查 gc_min/grid 与专家数)")

    best = min(flat, key=lambda k: (cur[k[0]][k[1]][2], k[1], 0 if k[0] == "cat" else 1))
    band = cur[best[0]][best[1]][2] * (1.0 + tie_band)
    cands = [k for k in flat if cur[k[0]][k[1]][2] <= band]
    if kl_slot_cap is not None and total_slots:
        ok = [k for k in cands
              if (total_slots - cum[k[1]]) / total_slots <= kl_slot_cap + 1e-12]
        if ok:                                   # 全被踢光就不约束(规则不能无解)
            cands = ok
    # 平手键: CPU 侧要搬的字节 -> 【CPU 腿 slot 数】-> 偏好 cat -> 更小 gc。
    # 第二条是本版新加的, 背书是 ④c: 同样字节数下 slot 越多 = 越多 token 落在 int8 上,
    # 而 int8 的 KL 单价是 bf16 的 274~364 倍(④c 表) => 时间打平时先买质量, 不花钱。
    impl, gc = min(cands, key=lambda k: (cpu_bytes(k[1]), total_slots - cum[k[1]],
                                         0 if k[0] == "cat" else 1, k[1]))
    return gc, impl, cur[impl][gc][2], cur

def measure_hit_rates(model, x):
    _, ti = model.route(x)
    tot = ti.numel()
    return [round((ti == e).sum().item() / tot * 100, 1) for e in range(model.num_experts)]


def bench(model, x, async_mode, mode, warmup=5, n_iter=30, reps=3):
    ts = []
    for _ in range(reps):
        for _ in range(warmup):
            model(x, async_mode=async_mode, mode=mode)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x, async_mode=async_mode, mode=mode)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / n_iter * 1000)
    return min(ts), ts


def machine_state():
    """把测量条件打在每张表头上 —— 这台机器的绝对值严重依赖下面这三行。
    🚨 实测: 桌面(微信/Edge/VSCode)常年占着 ~2 个核, loadavg≈2 时 6 线程的
       CPU kernel 会被拖慢 30%+, 这个干扰比大多数代码改动本身还大。"""
    def rd(p):
        try:
            return open(p).read().strip()
        except OSError:
            return "?"
    gov = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    # 🚨 MHz 必须【多次采样取峰值】: 单次读的是 import 那一刻的空载值, 空载时 cpu0 常在
    #    最低倍频(本机 800MHz), 于是乙-1d 那份表头印出 MHz=800.00 差点被当成"计时窗口被压频"
    #    而误判重跑。真判据是【同格 p50 绝对值对账已入库表】, 不是这一行。
    freqs = []
    for _ in range(6):
        raw = rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if raw != "?":
            try:
                freqs.append(float(raw) / 1000.0)
            except ValueError:
                pass
        else:
            try:
                freqs.append(float(rd("/proc/cpuinfo").split("cpu MHz")[1].split(":")[1].strip()[:6]))
            except (ValueError, IndexError):
                pass
        time.sleep(0.05)
    mhz = f"{max(freqs):.2f}" if freqs else "?"
    load = rd("/proc/loadavg").split()[:2]
    warn = "  🚨 有背景负载在抢核, 绝对值偏高(相对 A/B 仍可用)" if float(load[0]) > 1.0 else ""
    # 峰值仍然很低 = 采样期整机空载(import 阶段必然如此) => 这一行不代表计时窗口
    if freqs and max(freqs) < 2000:
        warn += ("  🚨采样期空载(6 发峰值仅 %s/%s MHz, 采样跨度 %.1fs 在 import 阶段): "
                 "本行【不代表计时窗口】, 别据此判废重跑 —— 判据是同格 p50 与已入库表对账"
                 % (f"{min(freqs):.0f}", f"{max(freqs):.0f}", 0.3))
    return (f"governor={gov} MHz={mhz} loadavg={load[0]}/{load[1]} "
            f"OMP_PLACES={os.environ.get('OMP_PLACES')} "
            f"OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND')} threads={os.cpu_count()}"
            f"{warn}")


if __name__ == "__main__":
    print(f"[机器状态] {machine_state()}")
    torch.manual_seed(42)
    hidden_size, num_experts, expert_size = 1024, 8, 2048
    top_k, bs = 2, 32
    x = torch.randn(bs, hidden_size).cuda()

    configs = [("配置 A (热专家在GPU)", 2, [0, 1]),
               ("配置 C (冷专家在GPU)", 2, [5, 6])]

    for delay in [0.0, 3.0]:
        print(f"\n{'#' * 66}\n### CPU 延迟: {delay} ms\n{'#' * 66}")
        for label, gpu_cnt, hot in configs:
            print(f"\n{'=' * 66}\n{label} | delay={delay}ms | 命中率(仅delay=0打印)\n{'=' * 66}")
            torch.manual_seed(42)
            model = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                              hot_experts=hot, cpu_delay_ms=delay).cuda()
            if delay == 0.0:
                print(f"命中率: {measure_hit_rates(model, x)}")

            def block(mm, tag):
                """一张表的四行。🚨 每个模型都必须在 torch.manual_seed(42) 之后构造,
                否则两个模型的权重不同, 精度列就对不上号(踩过两次)。"""
                ts_, ti_ = mm.route(x)
                ref = mm.fp32_reference(x, ts_, ti_)
                print(f"{'路径':<22}{'max_abs':>10}{'SNR(dB)':>10}{'relRMSE%':>10}"
                      f"{'t(ms)':>9}{'三遍':>22}{'vs sync':>10}")
                t_sync, s3 = bench(mm, x, False, None)
                outs = {mo: mm(x, async_mode=True, mode=mo) for mo in MODES}
                worst = max((outs[a] - outs[b]).abs().max().item()
                            for i, a in enumerate(outs) for b in list(outs)[i + 1:])
                print(f"{'sync (基线)':<22}{'-':>10}{'-':>10}{'-':>10}"
                      f"{t_sync:>9.3f}{('/'.join(f'{v:.2f}' for v in s3)):>22}{'1.00x':>10}")
                print(f"{'三模式输出一致性':<22}{worst:>10.6f}   "
                      f"relRMSE 三模式同: {err_metrics(outs[MODES[0]], ref)[2]:.4f}%   [{tag}]")
                for mo in MODES:
                    t, r3 = bench(mm, x, True, mo)
                    ma, snr, rel = err_metrics(outs[mo], ref)
                    print(f"{mo:<22}{ma:>10.5f}{snr:>10.2f}{rel:>10.4f}"
                          f"{t:>9.3f}{('/'.join(f'{v:.2f}' for v in r3)):>22}"
                          f"{t_sync / t:>9.2f}x")

            block(model, "graph_scope=off")
            # ✅ T7: 同一张表再补一组 full-graph。🚨 同样先 seed(42) => 权重与上面逐位相同,
            #    所以两组的 ms 列可以直接相减, 差值就是 T7 在这一格上的净收益。
            torch.manual_seed(42)
            mg = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                           hot_experts=hot, cpu_delay_ms=delay, graph_scope="full").cuda()
            block(mg, f"graph_scope=full + 前段={mg.pre_impl} + 流={mg.l2_stream} ✅T7/T11/T12")


    # ============================================================
    # ✅ T4: OpenMP 线程数扫描 —— 这一轮攻的是 T_cpu 本身
    #    注意看 sync 列: 它也随线程数变快, 因为 kernel 是两条路径共用的
    # ============================================================
    print(f"\n{'#' * 66}\n### T4 OpenMP 线程扫描 (delay=0, 绝对延迟 ms)\n{'#' * 66}")
    for label, gpu_cnt, hot in configs:
        print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
        print(f"{'threads':>8}{'sync':>10}{'L1_松GIL':>11}{'L2_pinned':>11}{'L2 vs sync':>13}")
        for nt in [1, 2, 4, 6, 8]:
            torch.manual_seed(42)
            m = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                          hot_experts=hot, cpu_delay_ms=0.0, n_threads=nt).cuda()
            t_sync = bench(m, x, False, None)[0]
            t1 = bench(m, x, True, "L1_松GIL")[0]
            t2 = bench(m, x, True, "L2_零CUDA+pinned")[0]
            print(f"{nt:>8}{t_sync:>10.3f}{t1:>11.3f}{t2:>11.3f}{t_sync / t2:>12.2f}x")


    # ============================================================
    # ✅ T5: tok_block × n_threads 二维扫描 —— 权重复用 vs 线程数
    #    tok_block=1 就是 T4 的老行为(只是改成按专家分桶)
    #    这一轮看的是绝对延迟 ms; sync 列同样会变快, 别盯着 speedup
    # ============================================================
    print(f"\n{'#' * 66}\n### T5 tok_block × 线程 扫描 (delay=0, L2 绝对延迟 ms)\n{'#' * 66}")
    for label, gpu_cnt, hot in configs:
        print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
        tbnt = "tb / nt"
        print(f"{tbnt:>8}" + "".join(f"{nt:>10}" for nt in (4, 6, 8)))
        grid = {}
        for tb in (1, 4, 8):
            row = []
            for nt in (4, 6, 8):
                torch.manual_seed(42)
                m = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                              hot_experts=hot, cpu_delay_ms=0.0,
                              n_threads=nt, tok_block=tb).cuda()
                t_sync = bench(m, x, False, None)[0]
                t2 = bench(m, x, True, "L2_零CUDA+pinned")[0]
                grid[(tb, nt)] = (t_sync, t2)
                row.append(t2)
            print(f"{tb:>8}" + "".join(f"{v:>10.3f}" for v in row))
        best = min(grid.items(), key=lambda kv: kv[1][1])
        (tb, nt), (ts, t2) = best
        print(f"  → 最优: tok_block={tb} threads={nt}  sync={ts:.3f}ms  L2={t2:.3f}ms")


    # ============================================================
    # ✅ T6: GPU 半边 —— 把"哪个 token 命中哪个热专家"从控制流降级成数据
    #    sparse 老写法每个热专家要 mask.any() + mask.nonzero() 各一次,
    #    两次都是强制 GPU->CPU 往返; gc=2 => 4 次往返 + 22 张 kernel。
    #    dense 之后: 固定形状一趟 GEMM, 5 张 kernel, 1 次往返。
    #    🚨 CPU kernel 一个字节都没动(T5 的 tok_block/线程照旧),
    #       所以这一列的变快【全部】来自 launch/往返的消失。
    # ============================================================
    T6_VARS = [("sparse(T5原状)",  "sparse", "off"),
               ("dense cat",       "cat",    "off"),
               ("dense bmm",       "bmm",    "off"),
               ("dense cat+hotG",  "cat",    "hot"),
               ("dense cat+fullG", "cat",    "full")]   # ✅ T7: 连路由+量化一起进图

    def gpu_half_stats(m, xx, ts_, ti_, st):
        """数 kernel 张数 / 主机往返次数 / GPU 真忙碌时间。profiler 挂了也不许拖垮主表"""
        try:
            for _ in range(10):
                m._gpu_hot_experts(xx, ts_, ti_, st)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as pr:
                for _ in range(20):
                    m._gpu_hot_experts(xx, ts_, ti_, st)
                torch.cuda.synchronize()
            ka = pr.key_averages()
            nl = sum(e.count for e in ka
                     if e.key in ("cudaLaunchKernel", "cudaGraphLaunch")) / 20
            nr = sum(e.count for e in ka if e.key in ("cudaStreamSynchronize",
                     "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy")) / 20
            bu = sum(e.device_time_total for e in ka if e.device_time_total > 0) / 20 / 1000
            return f"{nl:.0f}", f"{nr:.1f}", f"{bu:.3f}"
        except Exception as e:
            return "n/a", "n/a", "n/a"

    def t_half(m, xx, ts_, ti_, st, warm=20, iters=200, reps=5):
        import time as _t
        for _ in range(warm):
            m._gpu_hot_experts(xx, ts_, ti_, st)
        torch.cuda.synchronize(); best = 1e9
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = _t.perf_counter()
            for _ in range(iters):
                m._gpu_hot_experts(xx, ts_, ti_, st)
            torch.cuda.synchronize(); best = min(best, (_t.perf_counter() - t0) / iters * 1000)
        return best

    print(f"\n{'#' * 66}\n### T6 GPU 半边: kernel/主机往返 扫描 (delay=0)\n{'#' * 66}")
    for label, gpu_cnt, hot in configs:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"{'GPU 实现':<18}{'半边墙钟':>10}{'kernel':>8}{'主机往返':>9}{'GPU真忙':>9}"
              f"{'sync':>9}{'L1_松GIL':>10}{'L2_pinned':>11}")
        for name, impl, gr in T6_VARS:
            torch.manual_seed(42)
            m = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                          hot_experts=hot, cpu_delay_ms=0.0, n_threads=6, tok_block=8,
                          gpu_impl=impl, graph_scope=gr).cuda()
            ts_, ti_ = m.route(x)
            st = torch.cuda.Stream()
            if gr != "off":
                m._ensure_graphs(x)
            nl, nr, bu = gpu_half_stats(m, x, ts_, ti_, st)
            th = t_half(m, x, ts_, ti_, st)
            tsy = bench(m, x, False, None)[0]
            t1 = bench(m, x, True, "L1_松GIL")[0]
            t2 = bench(m, x, True, "L2_零CUDA+pinned")[0]
            print(f"{name:<18}{th:>10.3f}{nl:>8}{nr:>9}{bu:>9}"
                  f"{tsy:>9.3f}{t1:>10.3f}{t2:>11.3f}")

    # ============================================================
    # ✅ T9: 放置/容量 —— CPU 侧"每帧会被摸到的权重"一超 L3 就掉 DRAM
    #    两张脸: (a) gc 抬一个 => CPU 少一个【有流量】的专家, 整趟 2MB 没了;
    #            (b) 代价   => GPU 是 dense GEMM, 成本随 GPU 专家数线性涨。
    #    实测三档把机制钉死: gc=2 有流量4专家/8MB 1.051ms | gc=3 3专家/6MB 0.905ms
    #    (-13.8%) | gc=4 仍 3专家/6MB 0.941ms —— gc=3 与 gc=4 的 CPU 侧逐位等价,
    #    那 +3.4% 就是"多给 GPU 一个专家"的标价, 所以一达标就该停手。
    #    🚨 这一节同时是 _hot_gate 越界 bug 的回归: gc != top_k 以前从没跑过。
    # ============================================================
    print(f"\n{'#' * 66}\n### T9 放置: GPU 容量 x CPU 侧流量预算 (delay=0, 轮转 5 轮)\n{'#' * 66}")
    GCS = [2, 3, 4]
    for label, gpu_cnt, hot in configs:
        torch.manual_seed(42)
        probe = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                          hot_experts=hot, cpu_delay_ms=0.0).cuda()
        hits = measure_hit_rates(probe, x)
        rec, traf = plan_cpu_residency(hits, expert_size * hidden_size,
                                       gpu_expert_count=gpu_cnt)
        # ✅ T13: 两条规则并排印 —— v1(单腿, T9) / v2(两条腿, 下面这张表就是它的证据)
        rec2, pred2 = plan_gpu_capacity(hits, expert_size * hidden_size,
                                        4 * expert_size * hidden_size,
                                        gpu_expert_count=gpu_cnt)
        del probe
        print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
        print(f"命中率: {hits}")
        print(f"规则 v1(T9 单腿)给出 gc={rec} (CPU 侧被摸到的权重 {traf >> 20}MB, "
              f"预算 {int(L3 * 0.5) >> 20}MB = 0.5xL3)  "
              f"规则 v2(T13 两条腿)给出 gc={rec2} (预测关键路径 {pred2:.0f}us)")
        ms = {}
        for gc in GCS:
            torch.manual_seed(42)
            ms[gc] = HeteroMoE(hidden_size, num_experts, expert_size, gc, top_k,
                               hot_experts=hot, cpu_delay_ms=0.0, n_threads=6,
                               tok_block=8, gpu_impl="cat", graph_scope="full").cuda()
            for _ in range(20):
                ms[gc](x, async_mode=True, mode="L2_零CUDA+pinned")
        torch.cuda.synchronize()
        samp = {gc: [] for gc in GCS}
        for _ in range(5):
            for gc in GCS:
                samp[gc].append(bench(ms[gc], x, True, "L2_零CUDA+pinned")[0])
        base = statistics.median(samp[gpu_cnt])
        print(f"{'GPU专家':>8}{'CPU有流量专家':>16}{'被摸权重':>12}{'L2 med':>10}"
              f"{'σ%':>5}{'vs 原状':>11}")
        for gc in GCS:
            ntr = sum(1 for e in range(gc, num_experts) if hits[e] > 0)
            mb = ntr * expert_size * hidden_size >> 20
            med = statistics.median(samp[gc])
            sd = statistics.stdev(samp[gc]) / statistics.mean(samp[gc]) * 100
            star = "  ← 规则选的" if gc == rec else ""
            print(f"{gc:>8}{ntr:>16}{mb:>11}MB{med:>10.3f}{sd:>5.0f}"
                  f"{(med / base - 1) * 100:>+10.1f}%{star}")
        del ms, samp
        torch.cuda.empty_cache()


    # ============================================================
    # 总账: T4 -> +T5 -> +T6a -> +T6b 的累计效果 (L2 路径, 绝对 ms)
    # ============================================================
    # ============================================================
    # ✅ T7: 总账改成【同进程轮转】测量 —— 先建好全部模型, 再一轮一轮地交替量。
    #    顺序测量(建一个量一个)会被桌面背景负载(微信/Edge/VSCode, loadavg≈2)
    #    和 turbo 状态污染: 同一份二进制两次量到 0.288 / 0.231 都发生过。
    #    轮转之后漂移对每个变体同等作用, 相对差才可信; 括号里是 5 轮的相对标准差,
    #    判决列是中位数(不是 min)。
    # ============================================================
    # 第 6 列 auto = 用哪版放置规则: False(照抄配置里的 gc) | "v1"(T9 单腿) |
    #               "v2"(T13 两条腿)。🚨 同理, T13 之前的档位必须钉死 "v1",
    #                  否则 T13 换规则会把收益记到 T9 那档头上。
    # 第 8 列 = 前段实现(T11)。🚨 T11 之前的档位必须钉在 "torch":
    #   模块级默认已经是 "fused", 不钉死的话"这一档"会偷偷把 T11 的收益算到 T4 头上。
    # ✅ T12: 每档多了两列 (l2_stream, tail_graph)。
    #    🚨 T12 之前的档位必须钉死 ("sep", False) —— 和 T11 那条教训同一个道理:
    #        ctor 默认值现在是 single, 不钉住的话 T12 的收益会被算到 T4 头上。
    # ✅ T14: 第 11 列 cpu_sched (0=static,32 历史 / 1=guided)。
    #    🚨 同样地, T14 之前的档位全部钉 0 —— ctor 默认已经是 guided, 不钉住的话
    #       "T4 基线"那一档会偷偷把 T14 那 12~17% 算到 T4 头上。(第 4 次踩同一个坑:
    #       T11 pre_impl / T12 l2_stream / T13 auto / T14 cpu_sched)
    LADDER = [("T4 基线 (sparse,tb=1,nt=1)", "sparse", "off",  1, 1, "main",  False, "torch", "sep",  False,  0),
              ("+T5 tok_block=8",            "sparse", "off",  1, 8, "main",  False, "torch", "sep",  False,  0),
              ("+T5 线程=6",                 "sparse", "off",  6, 8, "main",  False, "torch", "sep",  False,  0),
              ("+T6a dense",                 "cat",    "off",  6, 8, "main",  False, "torch", "sep",  False,  0),
              ("+T6b hot-GEMM graph",        "cat",    "hot",  6, 8, "main",  False, "torch", "sep",  False,  0),
              ("+T7 full graph+D2H图",       "cat",    "full", 6, 8, "main",  False, "torch", "sep",  False,  0),
              # ✅ T8-②: 唯一一次主机等待挪进 worker。留档当反面教材 —— 终审只有
              #    A/delay0 那 -2.7%, delay=3 两配置全平(见 d2h_wait 的注释)
              ("T8 等D2H挪worker(否证)",     "cat",    "full", 6, 8, "worker", False, "torch", "sep",  False,  0),
              # ✅ T9-①: GPU 容量按"CPU 侧被摸到的权重 <= 0.5xL3"现算:
              #    配置A 会自己留在 gc=2(CPU 侧本来只有 6MB), 配置C 抬到 gc=3
              ("+T9 放置规则v1(流量预算)",   "cat",    "full", 6, 8, "main",  "v1", "torch", "sep",  False,  0),
              # ✅ T11: 前段 12 张 aten -> 2 张 triton(逐位等价, idx/x_q 完全相同)。
              #    它只动 graph_scope="full" 的前段图, 所以放在最后一档单独定价
              ("+T11 前段融合(2张kernel)",   "cat",    "full", 6, 8, "main",  "v1", "fused", "sep",  False,  0),
              # ✅ T12-①: 去掉那条【从来没并行过】的第二流(事件把它完全串行化了),
              #    热专家 replay 回默认流 => 每帧少 4 次主机调用。已转 ctor 默认。
              ("+T12 单流(去二流+Event)",    "cat",    "full", 6, 8, "main",  "v1", "fused", "single", False,  0),
              # ⚠️ T12-②: 尾图(H2D+add 折成 1 次 replay)。定价用, 不进默认:
              #    它把返回张量变成图内静态缓冲 => 下一帧覆写上一帧输出。
              ("+T12b 尾图(代价=覆写输出)",  "cat",    "full", 6, 8, "main",  "v1", "fused", "single", True,  0),
              # ✅ T13: 放置规则 v2 —— 给 GPU 半边定价(两条腿取 max), 配置A 从 gc=2
              #    抬到 gc=3。这一档就是"规则本身"的收益, 代码一行没动, 只换了选择。
              ("+T13 放置规则v2(两条腿)",    "cat",    "full", 6, 8, "main",  "v2", "fused", "single", False,  0),
              # ✅ T14: CPU kernel 的 OMP 行调度 static,32 -> guided。帧内主线程和绑在 core0 的
              #    OMP 线程0 同核抢 CUDA 提交 => static 预先平摊变成"整个并行区等最慢那条"。
              #    逐位等价(只换"谁算哪一行")。🚨 收益数字是本机专属(核数/桌面负载强相关),
              #    值钱的是结论, 见 OPTLOG.md 的 T14。与上一档只差这第 11 列
              ("+T14 OMP guided",           "cat",    "full", 6, 8, "main",  "v2", "fused", "single", False, 1)]
    H_BW = f"{CPU_BW_EFF / 1e9:.0f}/{GPU_BW_EFF / 1e9:.0f}GB/s"
    ROUNDS = 5
    for delay in (0.0, 3.0):
        print(f"\n{'#' * 66}\n### 总账: 各档累计 (L2_零CUDA+pinned, 绝对 ms, "
              f"轮转 {ROUNDS} 轮) | delay={delay}ms\n{'#' * 66}")
        # 放置决策的输入 = 路由器命中分布。生产里这是命中计数器的 EMA, 这里现量一次
        hits = {}
        for label, gpu_cnt, hot in configs:
            torch.manual_seed(42)
            probe = HeteroMoE(hidden_size, num_experts, expert_size, gpu_cnt, top_k,
                              hot_experts=hot, cpu_delay_ms=delay).cuda()
            hits[label] = measure_hit_rates(probe, x)
            del probe
        plans = {lb: plan_cpu_residency(hits[lb], expert_size * hidden_size,
                                        gpu_expert_count=gc)
                 for lb, gc, _ in configs}
        plans2 = {lb: plan_gpu_capacity(hits[lb], expert_size * hidden_size,
                                        4 * expert_size * hidden_size,
                                        gpu_expert_count=gc)
                  for lb, gc, _ in configs}
        print("放置规则: " + "  ".join(
            f"{lb[:5]} v1(T9)gc={plans[lb][0]}/CPU被摸{plans[lb][1] >> 20}MB "
            f"v2(T13)gc={plans2[lb][0]}@{plans2[lb][1]:.0f}us"
            for lb, _, _ in configs)
            + f"  (v1 预算 {int(L3 * 0.5) >> 20}MB=0.5xL3 | v2 带宽 {H_BW})")
        models = {}
        for name, impl, gr, nt, tb, dw, auto, pre, st, tg, cs in LADDER:
            for label, gpu_cnt, hot in configs:
                gc = (plans2 if auto == "v2" else plans)[label][0] if auto else gpu_cnt
                torch.manual_seed(42)
                models[(name, label)] = HeteroMoE(
                    hidden_size, num_experts, expert_size, gc, top_k,
                    hot_experts=hot, cpu_delay_ms=delay, n_threads=nt,
                    tok_block=tb, gpu_impl=impl, graph_scope=gr, d2h_wait=dw,
                    pre_impl=pre, l2_stream=st, tail_graph=tg,
                    cpu_sched=cs).cuda()
        samples = {k: [] for k in models}
        for _ in range(ROUNDS):
            for key in models:                       # 轮转, 不按档位成块
                samples[key].append(bench(models[key], x, True, "L2_零CUDA+pinned")[0])
        # 🚨 判决列用【中位数】不是 min: T8-② 教会我的 —— min-of-N 会奖励方差大的
        #    变体, 那个"−16%"就是这么造出来的(见 bench_dist_ab.py)。
        print(f"{'档位':<28}" + "".join(f"{lb[:5]:>18}" for lb, _, _ in configs))
        for name, *_ in LADDER:
            cells = []
            for label, _, _ in configs:
                v = samples[(name, label)]
                sd = statistics.stdev(v) / min(v) * 100 if len(v) > 1 else 0.0
                cells.append(f"{statistics.median(v):.3f}±{sd:.0f}%")
            print(f"{name:<28}" + "".join(f"{c:>18}" for c in cells))
        del models, samples
        torch.cuda.empty_cache()

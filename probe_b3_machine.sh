#!/usr/bin/env bash
# probe_b3_machine.sh —— B3 换机器轮的【候选机闸门探针】(§5.2i附3 / §5.2k)
# 用途：在任何一台候选机上跑一遍，自动判定「这台能不能接 B3(drifttrackN, ROUNDS=48 NP=5)」并打印缺什么。
# 纪律：本脚本**零装置改动**、不建臂、不计时、不读任何判决表 ⇒ 它的输出**不是判决表、不进记录**，
#       只用于路线裁决（同 scan_drift.py 的地位）。硬门槛全部来自已入库实测：
#       r5 单进程 7 臂峰值 RSS 3467MB / 显存 709MB；进程是【串行】跑的（等空载闸门 loadavg<0.8 才起下一个）；
#       AVX2 ext 编译 flags 钉死 -O3 -mavx2 -mfma -fopenmp；装置用 torch CUDA Graph 捕获。
set -u
P(){ printf '%-26s %s\n' "$1" "$2"; }
echo "===== B3 候选机闸门探测 $(date '+%F %T') ====="
# 🚨防误贴：这份输出必须来自【另一台】机器。基准机 = i5-10400 + RTX3060，machine-id 前 12 位 = da89a0d9d04f
#   （2026-11-24 本机自跑 = 硬门槛 6/6 过，但 6 条门槛的凭据全照本机写 ⇒ 本机不构成候选机，见 OPTLOG §5.2k）
ARCH=$(uname -m)
CPU=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)
NPROC=$(nproc)
CORES=$(LC_ALL=C lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')
L3=$(LC_ALL=C lscpu 2>/dev/null | awk -F: '/^L3/{gsub(/^ +/,"",$2);print $2}' | tr -s ' ')
AVX2=$(grep -om1 avx2 /proc/cpuinfo || true)
FMA=$(grep -om1 fma /proc/cpuinfo || true)
VNNI=$(grep -cE 'avx512_vnni' /proc/cpuinfo || true)
MEMA=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
MEMT=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
LA1=$(cut -d' ' -f1 /proc/loadavg)
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
GPUMIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPLUS=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GXX=$(g++ --version 2>/dev/null | head -1 || true)
NIN=$(ninja --version 2>/dev/null || true)
PY=$(python3 -V 2>&1)
TOR=$(python3 -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))' 2>/dev/null || echo 无torch)

MID=$(cut -c1-12 /etc/machine-id 2>/dev/null || echo 读不到)
P "机器身份 主机名/machine-id" "$(hostname 2>/dev/null || echo 无)/${MID}"
P "CPU 型号" "${CPU:-读不到}"
P "arch / 核" "$ARCH  逻辑=$NPROC 物理=${CORES:-?}"
P "L3 / AVX2 / FMA / VNNI" "${L3:-?} / ${AVX2:-无} / ${FMA:-无} / vnni行=$VNNI"
P "内存 总/可用" "${MEMT:-?}G / ${MEMA:-?}G"
P "loadavg(1min) 空载门槛<0.8" "$LA1"
P "GPU / 驱动 / 利用率" "${GPU:-无 nvidia-smi} / ${DRIVER:-?} / ${GPLUS:-?}"
P "g++ / ninja" "${GXX:-无 g++} / ${NIN:-无 ninja}"
P "python / torch" "$PY / $TOR"
echo "----- 硬门槛自动判定 -----"
pass=0; tot=6; fail=""
ok(){ pass=$((pass+1)); printf '  ✅ %s\n' "$1"; }
no(){ fail="$fail\n     ✗ $2"; printf '  ❌ %s\n' "$1"; }
[ "$ARCH" = "x86_64" ] && ok "arch = x86_64 (kernel 是 x86 intrinsics，ARM/Apple 直接不行)" || no "arch=$ARCH" "非 x86_64 ⇒ avx_gemv_int8.cpp 编不出来"
{ [ -n "$AVX2" ] && [ -n "$FMA" ]; } && ok "AVX2 + FMA 齐 (编译 flags 钉死 -mavx2 -mfma；无 VNNI 也没关系，走的仍是 maddubs)" || no "缺 AVX2/FMA" "缺 AVX2 或 FMA"
[ "${CORES:-0}" -ge 6 ] 2>/dev/null && ok "物理核 ${CORES} ≥ 6 (本机 6C/12T；OpenMP guided、已定稿 nt=6)" || no "物理核 ${CORES:-?} < 6" "核数 <6 ⇒ CPU 腿形状与本机差太远"
[ "${MEMA:-0}" -ge 8 ] 2>/dev/null && ok "可用内存 ${MEMA}G ≥ 8G (r5 单进程 7 臂峰值 RSS 3467MB；进程串行跑)" || no "可用内存 ${MEMA:-?}G < 8G" "可用内存 <8G"
[ -n "$GPU" ] && [ "${GPUMIB:-0}" -ge 2048 ] 2>/dev/null && ok "GPU ${GPU} (需 ≥2048MiB；r5 七臂显存 709MB) + CUDA Graph 需 Turing/CC≥7.5" || no "无 GPU 或显存 ${GPUMIB:-?}MiB < 2048" "无可用 NVIDIA GPU"
{ [ -n "$GXX" ] && [ -n "$NIN" ]; } && ok "g++ + ninja 齐 (torch cpp_extension 现场编 AVX2 ext)" || no "缺 g++ 或 ninja" "缺 g++/ninja"
echo "----- 不判但要记的事 -----"
if awk -v a="$LA1" 'BEGIN{exit !(a<0.8)}'; then echo "  ✅ 当场空载 (loadavg=$LA1 < 0.8)"; else echo "  ⚠️ 当场 loadavg=$LA1 ≥ 0.8：此刻不空载（正式跑要等到 <0.8；入库门槛就是这个数）"; fi
echo "  · 内存带宽/帧长/flip 价签 = 只能到了之后用冒烟轮量（本机 42GB/s、一帧 4.07ms、flip 一次性 19.9ms）；带宽差一大 ⇒ 刺激上限变，所有绝对值**只与新机自己比**"
echo "  · L3 ≥16MB 时工作集 12.7MB 不再越界(本机 12MB 越界是既定设定)；装置会自己印新的判定行，🚨不改代码"
echo "  ⚠️ 防误贴：若上面【机器身份】= 基准机 (i5-10400 / da89a0d9d04f) ⇒ 这不是 B3，只是同一实例再抽一次"
echo "  · 云机：burstable/共享 vCPU(t3/t4/e 类) 与被 MPS/MIG 分享的卡 = 不合格（那正是我们要消的噪声源）"
echo "===== 结论：硬门槛 $pass/$tot 过 ====="
if [ "$pass" = "$tot" ]; then
  echo "  ✅ 这台可以接 B3。下一步顺序：拷代码 → 建 env(或指认现成 env) → 冒烟轮 → 红线 → 正式 ROUNDS=48 NP=5"
  echo "     权重：能联网就自动下 SmolLM-135M(~270MB)；不能联网就从本机拷 ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M"
else
  echo -e "  ❌ 不合格，缺：$fail"
fi

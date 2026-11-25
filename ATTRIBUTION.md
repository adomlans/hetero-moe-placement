# 署名、边界与发布前自查

## 1. 上游与许可

代码树派生自 <https://github.com/labmlai/annotated_deep_learning_paper_implementations>（**MIT**，
Copyright (c) 2020 Varuna Jayasiri，全文见 `LICENSE`）。本仓库是把其中
`labml_nn/transformers/glu_variants/` 子树抽出来独立成仓，**目录相对位置保持不变** ——
这样 `writeups/` 与 `OPTLOG.md` 里每一条复现命令都能逐字照跑，不必重写路径。

| 内容 | 归属 |
|---|---|
| `hetero_moe_int8.py`、`avx_gemv_int8.cpp`、`pre_fused.py`、`dyn_placement.py`、`gpu_traffic.py`、`real_smollm.py`、`budget_check.py`、`combine_drifttrack.py`、`check_redlines.py`、`bench_*.py`(35 个，仓库共 46 个 .py)、`run_all.sh`、`sweep_aff.sh`、`scan_drift.py`、`probe_b3_machine.sh` | **本项目的工作** |
| `writeups/`（5 篇 + 2 张图 + `make_ledger.py`）、`results/`（**42 张判决表 + 5 份 r5 进程 dump = 47 个文件**）、`OPTLOG.md` | **本项目的工作** |
| `simple.py`、`simple.ipynb`、`experiment.py`、`benchmark_engine.py`、`__init__.py` | 上游教程文件，**本仓库有意未收录**（不是本项目的工作；且本地工作副本处于学习期改动状态，不适合发布） |
| `triton_moe_kernel.py` | 学习期试验件，与主线结论**无关**，保留只为完整 |
| `HANDOFF.md` | **内部交接台账**（协作协议 + 本机绝对路径 + 逐轮状态行），**有意未收录** ⇒ `OPTLOG.md` 里 23 处「HANDOFF」引用读作内部台账，**不影响任何复现命令** |

## 2. 口径声明（读这个仓库之前必须知道）

1. 🚨 本仓库**不含**「自适应 / 动态放置有效」这一类主张。可引用的收益全部来自**开局一次性放置**。
2. `OPTLOG.md` 是**原始实验日志**：里面有【**不进记录**】标记的内容（`scan_drift.py` 真语料预检、
   `probe_b3_machine.sh` 候选机探测、任何模型自跑的读数）**不构成主张、不是测量结果**，只用于路线裁决；
   里面被标注**作废**的数值是历史留痕，引用只许引「定值」。
3. 判决表（`results/`）全部由人自跑并贴全文，机器状态在头行（date / MHz / loadavg）= 保鲜凭据。
4. 所有常数**本机专属**（i5-10400 6C/12T、DDR4 实测 42GB/s、L3 12MB、无 AVX-VNNI、RTX3060 skinny 250GB/s）。
   **可迁移的是测法与结论，不是数字。**

## 3. 发布前自查清单（建议按顺序跑，都在仓库根目录）

```bash
# ① 自证：交付物里的每个数都能对回归档（对不上 make_ledger.py 会直接 SystemExit，不出图）
PY=/home/adomlans/miniconda3/envs/dl_lab/bin/python; $PY writeups/make_ledger.py

# ② 体积与文件数（当前：工作树 4.0MB / 110 个受版本控制文件，另有 .git 2.9MB，GitHub 无压力）
du -sh --exclude=.git . ; git ls-files | wc -l ; du -sh .git

# ③ 口径扫描：这三条都应当是 0 命中 / 只有「否证」语境的命中
grep -rn "自适应放置有效\|动态放置有效" --include="*.md" . | grep -v "不含\|不是"
grep -rn "3\.5x" --include="*.md" .            # 只允许出现在「实测是 1.0x」那句旁边

# ④ 隐私归一化（可选，但要自己看过 diff 再提交）：把机器用户名路径换成 ~
grep -rl "/home/adomlans" --include="*.md" . | tee /tmp/paths.txt   # 先看范围
# git diff  <!-- 若已 git init，改完用 git diff 逐行看过 -->
```

## 4. 我在这个仓库里刻意留下的东西

- **四次挂账**（`c299b2b9` / `96ecadf1` / `48499173` / `1f69a7cb`）：预设判决树 + 空对照不过 ⇒ 问句不许读。
- **作废记录**：被推翻的口径（迁移价签「稳态惩罚」读法、r4 的「臂身份固定效应」归属、r2 的伪红轮）
  都**保留原文**并写明被谁推翻。
- **`§4.1 没进正文的 10 张逐张记归属`**：不是漏引用，是显式记账。
- **未计量项**（真实文本上的自然漂移）写成「未计量项」，**不写成负结果**。

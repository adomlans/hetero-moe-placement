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

# ② 体积与文件数（工作树 4.0MB、`.git` 2.9MB，GitHub 无压力。🚨文件数**不在正文抄死** ⇒ 以 `git ls-files | wc -l` 现场输出为准）
du -sh --exclude=.git . ; git ls-files | wc -l ; du -sh .git

# ③ 口径扫描：这三条都应当是 0 命中 / 只有「否证」语境的命中
grep -rn "自适应放置有效\|动态放置有效" --include="*.md" . | grep -v "不含\|不是"
grep -rn "3\.5x" --include="*.md" .            # 只允许出现在「实测是 1.0x」那句旁边

# ④ 隐私归一化（可选，但要自己看过 diff 再提交）：把机器用户名路径换成 ~
grep -rl "/home/adomlans" --include="*.md" . | tee /tmp/paths.txt   # 先看范围
# git diff  <!-- 若已 git init，改完用 git diff 逐行看过 -->

# ⑤ 🚨隐私扫描必须扫**全历史全部 blob**，只扫工作树+commit 元数据是不够的（本仓库踩过 ⇒ 见 §5 差异表第 4 行）
#    模式串从交互/环境进来，**不许写进任何被提交的文件**（写进去 = 断言自伤）
read -rp "输入要扫的隐私字面串（邮箱/手机号，不留痕）> " P && git grep -I -l -F "$P" $(git rev-list main)   # 空输出 = 全历史 0 命中
git log --pretty=%ce          # 22 行逐行看，只应是 GitHub noreply
```

## 4. 我在这个仓库里刻意留下的东西

- **四次挂账**（`c299b2b9` / `96ecadf1` / `48499173` / `1f69a7cb`）：预设判决树 + 空对照不过 ⇒ 问句不许读。
- **作废记录**：被推翻的口径（迁移价签「稳态惩罚」读法、r4 的「臂身份固定效应」归属、r2 的伪红轮）
  都**保留原文**并写明被谁推翻。
- **`§4.1 没进正文的 10 张逐张记归属`**：不是漏引用，是显式记账。
- **未计量项**（真实文本上的自然漂移）写成「未计量项」，**不写成负结果**。

## 5. 发布副本与工作区台账的差异【脱敏与口径登记 · 逐条】

工作区还有一份**内部协作台账** `HANDOFF.md`（协作协议 / 进度指针 / 归档表），**有意不收录**本仓库
⇒ `OPTLOG.md` 里对它的 23 处引用在本仓库内**是悬空的**，全部指向这份未发布的内部件，不是笔误。

| # | 发布副本相对工作区的差异 | 理由 |
|---|---|---|
| 1 | `OPTLOG.md` §5.2n 里那一处**邮箱字符串**替换成「邮箱见仓库级 `.git/config`」（其余逐字同） | 公开仓库会被爬；联系方式**公开面一律不写**（简历页眉也只留 GitHub handle）。✅**现口径 = 三通道，不是当年那句「双通道」**：真实邮箱在 ① 当前 tip 全部正文 ② 全部 commit 元数据 ③ **全部历史全部 blob** 三处都 = 0，作者与 committer 邮箱都是 GitHub noreply。复核（命令都不含邮箱字面串 ⇒ 模式串从命令行/环境传入，写进文件就是自伤）：`git log --pretty=%aE` 与 `git log --pretty=%ce`（各 22 行）逐行都是 noreply，加上 §3 第⑤条的全历史扫描返回空。🚨**但这句在 2026-11-25 之前的若干小时里是假的**，第 4 行写了它为什么假、什么时候才真正成立 |
| 2 | `TODO_publish.md` 里两处同样的邮箱字符串脱敏；该文件在 push 后 `git rm` | 它是发布清单，不是交付物 |
| 3 | 两份都**不删历史轮日志**：被后续轮推翻的读数原文保留、只加「谁推翻了它」的注 | 「旧表一张不动」是这套测法的一部分 |
| 4 | 🚨**本仓库的历史被第三次改写，而且是首次 push 之后补做的**：前两次改写只处理了 tip 正文与 commit 元数据、**没扫历史 blob** ⇒ 首次 push 出去的历史里确有 **13 个文件×commit** 带真实邮箱；本次用 `git filter-branch --tree-filter` 把**全历史全部 blob** 里的该字面串一并换成同一个脱敏串，再 `--force-with-lease` 强推 | 不写这一行，本文件第 1 行和 `README.md` 末尾那句「不写邮箱」就是假话 —— 它们在补做改写之前**确实有一段时间是假的**。改写自证：① 改写前后 `HEAD^{tree}` 逐字相同（tip 树 = `a95496a6`）② `git diff 备份..main` = **0 字节**（= 除该字面串外没有任何内容变化）③ 22 条 commit 的「日期/作者/标题」与改写前备份 1:1、差异 0 字节 ④ 改写后全历史该字面串 = 0（改写前 13）。🚨**诚实边界（2026-11-26 实测，不是推测）**：强推之后 GitHub 侧**确实还留着旧对象** —— 显式 `git fetch origin <被甩掉的旧 tip 的 SHA>` 仍能把旧历史整个取回（实测 70 个不属于新历史的对象），而**新克隆取不到**（实测新克隆 23 条 commit 全历史 = 0）。⇒ 本行「全历史 = 0」的适用范围 = **强推之后的新克隆**；要让旧对象彻底消失只有两条路：向 GitHub Support 申请清除，或删仓重建。任何在强推前**已克隆**的副本照旧留有旧 blob |

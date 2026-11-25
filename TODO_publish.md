# 发布清单（TODO_publish.md）

> 这份文件是**发布动作清单**，不是交付物：**零新测量、零新数字**，里面每条命令都在仓库根目录逐字可跑。
> 发布做完就删掉它：`git rm TODO_publish.md && git commit -m "发布清单已执行完毕" && git push`
> 与之配套的口径/署名自查在 `ATTRIBUTION.md` §3；这里只列**还没按下的键**。

---

## 0 · 当前状态（一行）

主线 = **结项冻结**（不再产生新测量）；骨架 = 工作树 4.0MB / **110 个受版本控制文件** / `results/` 42 张判决表 + 5 份 r5 进程 dump；
**未加 remote、未 push**；`writeups/README.md` §4 是 42 张归档的逐张索引，§4.1 是「没进正文的 10 张」的显式记账。

---

## 1 · 必做三件（P0，全都要你动手或点头）

### P0-1 提交者身份【现在全部 commit 署名 `codex <codex@local>`】

**为什么必须先做**：直接 push 出去，GitHub 上每一条 commit 的作者都会显示成 `codex`，且不关联你的账号——
一个「你自己写的东西」在页面上看起来像别人代工的。本机没有全局 `user.name/user.email`，所以之前建的 commit 用了兜底身份。

```bash
cd ~/GV_public
git config user.name "真名" && git config user.email "真邮箱"
git rebase --root --exec 'git commit --amend --no-edit --reset-author'
git log --format='%h %an <%ae>'          # 校验：每一条都应是你自己
```
🚨 这会**重写全部 commit 的作者字段**（内容、顺序、文件都不变，但 hash 会变），所以要在 **push 之前**做完；push 之后再改就要强推。

### P0-2 建空仓库 + 认证 + push

本机现状：没有 `gh` CLI、`~/.ssh` 里没有私钥、没有 credential helper、没有全局 git 身份 ⇒ **四条认证路都是空的**，
所以这一步我替你做完不了，必须你点一次网页。

1. GitHub 网页**建一个空仓库**（🚨 不要勾「Add README」也不要勾 `.gitignore`，会和现有历史冲突）。
2. 认证二选一：
   - **省事**：Settings → Developer settings → Fine-grained tokens → 只勾那一个仓库、权限只给 `Contents: Write`；
   - **长期**：`ssh-keygen -t ed25519 -C "你的邮箱"` ⇒ 把 `~/.ssh/id_ed25519.pub` 贴到 GitHub → Settings → SSH and GPG keys。
3. 推送（HTTPS 就填用户名 + 把 PAT 当密码；SSH 把 URL 换成 `git@github.com:...`）：

```bash
cd ~/GV_public
git remote add origin https://github.com/<用户名>/<仓库名>.git
git push -u origin main
```
4. push 后**肉眼三查**（10 秒，专治「本地对、网页错」）：
   - README 首屏两张 png 有没有渲染出来（`writeups/fig1_ladder.png` / `fig2_exchange.png`）；
   - `results/` 网页里是不是 **42 个 .txt + `dt_r5_dumps/` 里 5 个 .json**；
   - 随便点进一张判决表，**头行**有没有 `date` / `MHz` / `loadavg`（那是全仓库可信度的来源）。

### P0-3 简历 7 处占位符

`RESUME_onepage.md`（57 行，负结果在第一块）。现在还剩 7 处占位：`【姓名】`、`【目标岗位：…】`、`【城市】`、
`【邮箱】`、`【GitHub 链接】`×2、`【在读/在职与年限】`。

```bash
grep -n "【" ~/GV_public/RESUME_onepage.md          # 填之前：3 行 / 7 处
grep -o "【[^】]*】" ~/GV_public/RESUME_onepage.md | wc -l   # 填完之后：应为 0
```
把 7 项发我，我写进去，并按「校招/实习」还是「社招 Infra」调段落顺序（两者的差别只在：负结果放第一块，还是单价判据放第一块）。

---

## 2 · 发布前最后一遍自查（5 条，我这轮全跑过；你只需再确认一次）

```bash
cd ~/GV_public
PY=/home/adomlans/miniconda3/envs/dl_lab/bin/python
$PY writeups/make_ledger.py                                            # ① 对账：对不上直接 SystemExit，不出图
git status --short | wc -l                                             # ② 应为 0（工作树干净）
ls results/*.txt | wc -l                                               # ③ 应为 42
grep -rn "自适应放置有效\|动态放置有效" --include="*.md" . | grep -v "不含\|不是"    # ④ 应无输出
grep -rn "3\.5x" --include="*.md" .                                    # ⑤ 只应出现在「实测是 1.0x」那句旁边
```

---

## 3 · 可选两件

| 项 | 我的建议 | 命令 |
|---|---|---|
| 抹掉机器用户名 `/home/adomlans`（共 5 处，全在**复现命令的解释器路径**上） | **不抹**：本仓库的复现硬约束就是「只有这一台机器、这一个解释器」，抹掉 ⇒ 命令不能逐字照跑；留着 ⇒ 暴露的只是本机用户名 | `grep -rl "/home/adomlans" --include="*.md" . \| xargs sed -i 's#/home/adomlans#~#g'` 然后 `git diff` 逐行看过再提交 |
| 删掉本文件 | **发布后删**（清单不是交付物） | `git rm TODO_publish.md && git commit -m "发布清单已执行完毕" && git push` |

---

## 4 · 挂着的候选（都要新资源；不是「没收尾」，是「二期」）

| 候选 | 买到什么 | 为什么现在不做 | 入口 |
|---|---|---|---|
| held-out ppl（wikitext-2 一类） | 质量段从「只报相对量」升级到绝对 ppl —— **唯一不需要新硬件的科学性加分** | 要一个语料文件 + 一轮判决表（数十分钟量级），不改任何已有结论 | `CORPUS=/path/to/wt2.txt`（`writeups/README.md` §7「语料」） |
| 同机对打已发布的 offload 路（llama.cpp / KTransformers 一类） | 唯一能**合法**改掉「不与已发布系统比数字」这条口径的做法 = 真跑 | 必须先定「比什么口径」，否则做成不可比的车祸现场；且要新写一套对照装置 | `README.md` 首屏三条不许之 #2 |
| 规模：E=64 / top-8 / 7B 级专家 | `u_e` 单价表（每专家一次 `gc=E−1` 的 KL，几乎免费）第一次拿到可迁移性凭据 | 要新显存 / 新 CPU | `writeups/README.md` §7「规模与并行外推」 |
| 多卡专家并行（all-to-all） | 工业界 MoE 的真难题 | 本机 0 可能 | 同上 |
| B3 换机器（**搁置**，不是判死） | 「这把尺不合格」到底是本机结论还是普适结论 | 手边没有第二台物理实例 | `probe_b3_machine.sh`（30 秒探针，随时可跑） |
| B6 第六次动尺子 | —— | 收益上限已被 `OPTLOG §5.2i附5` 封死；永久挂着等点头 | 同上节状态行 |

---

## 5 · 三条不许（防止把「还没结束」误读成「该继续」）

不许**第五次加发数**（买重复层已离线核算判死）；不许下调 `TARGET` / `T_LO` / `N_MIN` / `NP_MIN` / `thr_rel`；
P2 冻结项（int4、biased 打包、VNNI、寄存器分块、`down` 合图、大 bs 调优）不动手。旧表一张不动、不碰裁判。

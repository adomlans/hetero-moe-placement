# 发布清单（TODO_publish.md）

> 这份文件是**发布动作清单**，不是交付物：**零新测量、零新数字**，里面每条命令都在仓库根目录逐字可跑。
> 发布做完就删掉它：`git rm TODO_publish.md && git commit -m "发布清单已执行完毕" && git push`
> 与之配套的口径/署名自查在 `ATTRIBUTION.md` §3；这里只列**还没按下的键**。

---

## 0 · 当前状态（一行）

主线 = **结项冻结**（不再产生新测量）；骨架 = 工作树 4.0MB / **111 个受版本控制文件**（含本文件，发布后删 ⇒ 净 110）/ `results/` 42 张判决表 + 5 份 r5 进程 dump；
**remote 已加**（`origin = git@github.com:adomlans/hetero-moe-placement.git`，SSH 钥匙已注册、实测 `Hi adomlans!`）、**远端是空仓**（`git ls-remote origin` = 空输出 + exit 0）⇒ **仍未 push**；`writeups/README.md` §4 是 42 张归档的逐张索引，§4.1 是「没进正文的 10 张」的显式记账。

---

## 1 · 必做三件（P0，全都要你动手或点头）

### P0-1 提交者身份 —— ✅**已完成【2026-11-25】**

全部 commit 的作者已改写为 **曹志瑄 <noreply 邮箱>**（改写时 8 条，第二十三刀时 19 条；`git config` 已写进**仓库级** `.git/config`，
以后在这个仓库新建 commit 会自动带上正确身份）。🚨本文件**不抄 hash**：hash 已随两次改写变过两轮，抄进来必然陈旧 ⇒ 逐字以 `git log --oneline` 为准。

```bash
git -C ~/GV_public log --format='%ae' | sort -u   # 校验：应只剩 1 条 = 277908902+adomlans@users.noreply.github.com
```
🚨 硬约束的**真实边界** = 「**push 之后**不许动历史」（push 之前历史还只在本地，改了不需要 `--force`）。据此第二次改写已放在**第二十三刀**、push 之前 = 作者邮箱由真实 QQ 邮箱换成 GitHub noreply；**这是最后一次**，push 之后一律不许再动。

### P0-2 建空仓库 + 认证 + push

✅**认证已完成【2026-11-25 · 走 SSH 长期钥匙，PAT 那条没走】**：`ssh-keygen -t ed25519` 生成 `~/.ssh/id_ed25519_github`（注释 `github-push-key-2026-11-25`，
指纹 `SHA256:Ej730POeqiGDTLLbdpz2gdnznzLNS9htn5yxmfSahh4`）⇒ 公钥已贴进 GitHub → Settings → SSH and GPG keys。本机另有两个坑必须记着：

1. ✅**空仓已建【2026-11-25】**：`https://github.com/adomlans/hetero-moe-placement`（Public；三个初始化勾一个都没勾 ⇒ 远端 `refs = 0`，push 不会冲突）。
   🚨**这轮踩过的坑（如实记账）**：第一次建成了下划线名 `hetero_moe_placement` ⇒ 本地 `git ls-remote origin` 一直回 `Repository not found`——
   GitHub 只对**大小写**不敏感，`_` 与 `-` 是两个不同名字。修法 = **网页 Settings → Repository name 改名**（空仓改名零风险），而不是迁就它去改仓库里那 3 处链接（`grep -rn "hetero-moe-placement" --include="*.md" .`）。
   ① `/etc/hosts` 被 Steam++ 把 github 系 27 条域名写死到 `127.0.0.1`（现已全部注释掉）；② 路由器 DNS 发下来的 GitHub IP 实测不可用
   ⇒ `~/.ssh/config` 把 `github.com` 直接钉到**已核对官方指纹**（`SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU`）的 `20.205.243.160:443`，
   `IdentitiesOnly yes`；备用 `Host github-22` 走 22 端口。实测 `ssh -T git@github.com` ⇒ `Hi adomlans! You've successfully authenticated`。
3. 推送（`origin` 已加好，🚨**别再跑 `git remote add`**，重复加会直接报错）：

```bash
cd ~/GV_public
# origin 已存在，逐字应是：git@github.com:adomlans/hetero-moe-placement.git —— 校验用 `git remote -v`
git push -u origin main
```
4. push 后**肉眼三查**（10 秒，专治「本地对、网页错」）：
   - README 首屏两张 png 有没有渲染出来（`writeups/fig1_ladder.png` / `fig2_exchange.png`）；
   - `results/` 网页里是不是 **42 个 .txt + `dt_r5_dumps/` 里 5 个 .json**；
   - 随便点进一张判决表，**头行**有没有 `date` / `MHz` / `loadavg`（那是全仓库可信度的来源）。

### P0-3 简历 —— ✅**全部关闭【2026-11-25 第二刀】**（7 处占位 0 残留）

已按 **AI Infra 实习** 定稿：新增 `## 0. 教育背景 · 求职意向`（学校 / 网络安全与执法 / 大二在读 / base 南京 / 意向实习），
段落顺序**保持负结果在第一块**（实习岗最稀缺的信号就是「会否定自己」，且这与仓库 README 首屏一致）。

```bash
grep -o "【[^】]*】" ~/GV_public/RESUME_onepage.md | wc -l    # 应为 0
grep -c "江苏警官学院" ~/GV_public/RESUME_onepage.md          # 应为 2（页眉 + §0）
```
**四项确认的落定结果（用户 2026-11-25 逐条回答）**：
① ✅ GitHub = `adomlans`、仓库名 = `hetero-moe-placement` ⇒ 3 处链接已填真值；
② ✅ 校名 = **江苏警官学院**（原文「江苏景观学院」是 jǐngguān 同音误输入，用户确认改）；
③ ⚪ **可补强项（GPA / 排名 / 课程 / 获奖）：用户决定先不写** ⇒ §0 不留空占位，也不编；日后要加只加 2 行；
④ ✅ 实习可用时间已写进 §0：**仅寒暑假可到岗（假期内可全职）**
　 —— 面试被问「学期内能不能远程」就照这句答，**不要现场改口**（口径一致比口径漂亮值钱）。

🚨 别补的：任何「参与过 XX 项目 / 熟悉 XX 框架」这种没有仓库或归档支撑的条目。这份简历的溢价全在**可查**两个字。

---

### P0-4 PII 两条 —— ✅**已关闭【2026-11-25 第二十三刀】**（它挂了三轮，是唯一真正挡在 `git push` 前面的东西）

- **PII-A｜commit 元数据里的真实 QQ 邮箱**（GitHub 网页上是公开字段，会被爬虫抓走）⇒ 作者邮箱换成
  **`277908902+adomlans@users.noreply.github.com`**（`277908902` = `adomlans` 的用户 id，API 实测），并把仓库级 `.git/config` 一起换掉。
  复核 = `git -C ~/GV_public log --format='%ae' | sort -u` 只剩这一条，且 `git log --format='%an' | sort -u` 只剩 `曹志瑄`。
- **PII-B｜`RESUME_onepage.md` 第 3 行的真实邮箱**（全仓正文唯一一处）⇒ **删邮箱**，那行只留
  `江苏南京 · github.com/adomlans/hetero-moe-placement · 江苏警官学院 · 网络安全与执法 · 大二在读`，与第十八刀「公开面只挂 GitHub handle」对齐。
- **双通道断言（push 前必为 0）**：`E=<那个 QQ 邮箱的字面量>`（🚨本文件**故意不写出来**，写进来这条断言就自己失败了）
  ⇒ `git -C ~/GV_public grep -c "$E" $(git -C ~/GV_public ls-files)` 无输出，且 `git -C ~/GV_public log --format='%ae %an' | grep -c "$E"` 得 0。

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
| 把简历压成**严格一页 PDF** | 现在 65 行（渲染约 1.2 页）。要压：删 §5「三句不要引用」并把 §0 并进页眉 ⇒ 我 30 秒给你一版；不压也行（技术面看内容不看页数） | `wc -l RESUME_onepage.md` |

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

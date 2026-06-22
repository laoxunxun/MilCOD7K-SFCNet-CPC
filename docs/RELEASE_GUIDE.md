# 手把手发布教程（零基础）

从"什么都没准备"到"GitHub 上有一个公开项目 + 百度网盘里能下载数据集/权重"，一步步来。
假设你**从没用过 Git 和 GitHub**，跟着做就行。

---

## 总体顺序（先看这个）

```
① 注册/装好工具  →  ② 先把数据集+权重传百度网盘(拿到链接)  →  ③ 把链接填进项目
                  →  ④ 在 GitHub 建空仓库  →  ⑤ 用 git 把项目推上去  →  ⑥ 打标签、验证
```

为什么先传网盘？因为项目里 README 有"链接+提取码"的占位符，**必须等文件传上去、拿到分享链接之后才能填**。所以网盘在前，推代码在后。

> 小提示：国内访问 GitHub 有时会慢。push 卡住的话，挂个代理或换个时段再试即可，不影响最终结果。

---

## 第①步：准备账号和工具（一次性，以后都能复用）

### 1.1 注册 GitHub 账号
1. 浏览器打开 https://github.com ，点右上角 **Sign up**。
2. 填邮箱、设密码、取一个用户名（英文，比如 `yourname`），按提示验证邮箱。
3. 注册完记牢你的**用户名**，后面到处要用（项目里 `<your-user>` 都要换成它）。

### 1.2 安装 Git（Windows）
1. 打开 https://git-scm.com/download/win ，下载 64-bit Git for Windows 安装包。
2. 双击安装，**一路点 Next（全部默认）** 即可。它会装好 Git 和一个叫 **Git Bash** 的终端。
3. 安装完，在任意文件夹空白处**右键**，应该能看到 "Git Bash Here" 选项——说明装好了。

> Git for Windows 自带"凭据管理器"，第一次推代码时会自动弹浏览器让你登录 GitHub，**不用手动配密码**，对新手很友好。

### 1.3 设置一次你的身份（告诉 Git 你是谁）
打开 **Git Bash**（开始菜单搜 "Git Bash"），依次输入（把引号里换成你自己的），每行回车：

```bash
git config --global user.name  "你的GitHub用户名"
git config --global user.email "你注册GitHub的邮箱"
```

这一步一辈子做一次就够了。

---

## 第②步：把数据集和权重传到百度网盘（先做这个）

### 2.1 打包数据集

> **要打包、要上传的是哪个数据集？**
> 是 `F:\python\SFCNet-main\SFCNet-main\SFCNet\Dataset_multiclass_5class_new`（**原生格式**，代码 dataloader 直接读的那套，里面是 `Imgs / GT(*.npy) / Edge`）。
> 你另一个 `G:\work1\data\datasets` 是**最初的 YOLOseg 原始标注**，它是数据来源、只留在本地，**不上传**。
> 原生格式那套就是从 YOLOseg 转换来的——同一批图、同样的标注，只是换成了代码能直接用的格式。
> 所以下面命令里的路径**直接用、不用改**。

打开 Git Bash，进到项目文件夹：
```bash
cd /f/python/SFCNet-main/MilCOD7K-SFCNet-CPC
```
执行打包：
```bash
bash tools/pack_release.sh /f/python/SFCNet-main/SFCNet-main/SFCNet/Dataset_multiclass_5class_new
```
跑完会在项目下 `release/` 文件夹里生成 3 个文件：
- `MilCOD7K_native.zip`（数据集本体，最大）
- `metadata.csv`
- `SHA256SUMS.txt`

### 2.2 准备权重文件夹
新建一个文件夹，比如 `D:\release_weights`，把训练好的权重拷进去并**重命名**成下面这些名字（左边是最终名字，右边是它在原工程里的位置）：

| 最终文件名 | 原工程里的位置 |
|---|---|
| `sfcnet_cpc_t015.pth` | `cpts_ablation_F2_temp015/Net_multi_best_iou.pth` |
| `sfcnet_cpc_t007.pth` | `cpts_multiclass_5class_v4_cpc/Net_multi_best_iou.pth` |
| `sfcnet_fscpc.pth` | `cpts_fscpc_v1/Net_multi_best_iou.pth` |
| `sfcnet_cacpc.pth` | `cpts_cccpc_v1/Net_multi_best_iou.pth` |
| `sfcnet_baseline.pth` | `cpts_v3_baseline_retrain/Net_multi_best_iou.pth` |

（SMT 编码器初始化权重 `smt_tiny_imagenet1k.pth` 如果你想一起发，也放进来。）

### 2.3 上传到百度网盘
1. 打开 https://pan.baidu.com 登录（没有账号就注册一个）。
2. 点 "上传" → "上传文件夹/文件"：
   - 传 `release/` 里的 3 个文件（zip + csv + SHA256SUMS.txt）；
   - 传 `release_weights` 文件夹（或直接把权重和数据集放一个总文件夹也行，看你怎么方便分享）。
3. 等上传完。

### 2.4 分享，拿到"链接 + 提取码"
1. 在网盘里选中要分享的文件夹 → 点上方 "分享"。
2. **有效期选 "永久"**（重要，否则链接会过期）。
3. 提取方式选 "提取码"，生成后会得到：
   - 一条链接，形如 `https://pan.baidu.com/s/XXXXXX`
   - 一个 4 位提取码，形如 `abcd`
4. 把**链接**和**提取码**记下来（记事本存好），第③步要用。
5. **不要**在提取码之外再加"下载密码"，两层验证会烦到下载的人。

---

## 第③步：把链接填回项目

用任意编辑器（VS Code / 记事本都行）打开下面 3 个文件，把里面的占位符换成真实的链接和提取码：

1. **`README.md`** —— 找到这两段，替换：
   ```
   链接: https://pan.baidu.com/s/XXXXXXXX      <!-- TODO: paste share link -->
   提取码: xxxx                                 <!-- TODO: paste extraction code -->
   ```
   （数据集那段、权重那段各一组，都换成你刚拿到的。）
2. **`docs/dataset.md`** —— 第 1 节 "Download" 里同样的占位符，换成数据集链接。
3. **`checkpoints/README.md`** —— 权重那段占位符，换成权重链接。

同时，把项目里所有 `<your-user>` 换成你的 GitHub 用户名（可以用 VS Code 的"全局查找替换"一次搞定）。

---

## 第④步：在 GitHub 上建一个空仓库

1. 登录 GitHub，点右上角 **`+`** → **New repository**。
2. 填：
   - **Repository name**：`MilCOD7K-SFCNet-CPC`
   - **Description**：随便写一句，如 `Multi-class camouflaged military object segmentation`
   - 选 **Public**（公开）
   - **下面的三个勾全不要勾**（Add a README / .gitignore / license）——因为你本地已经有了，勾了反而冲突。
3. 点 **Create repository**。建好后页面会显示一段提示命令，**不用管它**，我们用自己的命令（第⑤步）。
4. 记下这个仓库的地址，形如 `https://github.com/你的用户名/MilCOD7K-SFCNet-CPC.git`

---

## 第⑤步：用 Git 把项目推上去（核心步骤）

在项目文件夹里**右键 → Git Bash Here**，然后**一条一条**输入下面的命令（每条输完回车）：

### 5.1 初始化 git 仓库
```bash
git init -b main
```
> 作用：在这个文件夹里建一个 git 仓库，主分支叫 `main`。

### 5.2 把所有该上传的文件加进来
```bash
git add .
```
> 作用：把当前目录下所有文件"标记为待提交"。`.gitignore` 会自动挡住 `data/`、`checkpoints/*.pth`、`*.npy` 等大文件，不用担心把几个 G 的数据集传上去。

**先检查一下别传错东西**（很重要）：
```bash
git status
```
> 会列出"将要提交的文件"。**确认里面没有 `data/MilCOD7K/...`、没有 `.pth`、没有 `*.npy`**（除了 sample 里的 6 个小 npy，那是故意要的）。如果有大文件混进来了，说明 `.gitignore` 没生效，先告诉我再继续。

### 5.3 提交（存一个本地快照）
```bash
git commit -m "Initial release: MilCOD7K + SFCNet-CPC"
```
> 作用：把这些文件存成本地的第一个版本。`-m` 后面是说明文字。

### 5.4 关联到 GitHub 上的仓库
```bash
git remote add origin https://github.com/你的用户名/MilCOD7K-SFCNet-CPC.git
```
> 作用：告诉本地 git "我的远程仓库在那儿"。把 `你的用户名` 换成真实用户名。

### 5.5 推送上去
```bash
git push -u origin main
```
> 作用：把本地代码传到 GitHub。

**第一次推送时**：
- 通常会**自动弹出浏览器窗口**让你登录 GitHub（这是 Git 自带的凭据管理器）。点 "Sign in with browser" → 授权 → 关掉。
- 授权完，终端会继续跑，看到类似 `main -> main` 就说明成功了。
- 如果它要你在终端里输用户名/密码：**密码那里不能填登录密码**，要填"个人访问令牌(PAT)"——见下面"常见问题 Q1"。

推送成功后，刷新你的 GitHub 仓库页面，就能看到代码和文件了。

---

## 第⑥步：打发布标签 + 完善页面

### 6.1 打一个版本标签 v1.0
在 Git Bash 里：
```bash
git tag -a v1.0 -m "v1.0 release"
git push origin v1.0
```
> 作用：标记一个正式版本。别人能清楚看到这是 v1.0。

### 6.2 完善 GitHub 仓库页面
1. **写一句描述 + 加标签(topic)**：在仓库页面右上角文件列表上方有个 **About** 区块，点旁边的齿轮 ⚙️：
   - Description 填一句项目简介。
   - Topics 加几个关键词（回车分隔）：`camouflaged-object-detection`、`semantic-segmentation`、`contrastive-learning`、`military`、`dataset`、`pytorch`。
2. 这些能让别人搜到你、看起来更正规。

---

## 第⑦步：验证（换个干净的视角测一遍）

理想情况下，换台电脑或让别人 clone 一份试：
1. 在 GitHub 仓库页面点绿色的 **Code** 按钮 → 复制 HTTPS 地址。
2. 别处打开 Git Bash：
   ```bash
   git clone https://github.com/你的用户名/MilCOD7K-SFCNet-CPC.git
   cd MilCOD7K-SFCNet-CPC
   pip install -r requirements.txt
   ```
3. 按 README 指引，从百度网盘下载数据集解压到 `data/`、权重放到 `checkpoints/`。
4. 跑一下评估：
   ```bash
   python test.py --config configs/sfcnet_cpc_t015.yaml \
                  --checkpoint checkpoints/sfcnet_cpc_t015.pth
   ```
   预期看到 **~84.93 mIoU / 91.71 mF1**，说明整套发布是通的。

---

## 常见问题（FAQ）

**Q1：push 时提示要密码，填了登录密码还报错？**
GitHub 从 2021 年起不让用登录密码 push 了，要用"个人访问令牌(PAT)"代替密码：
1. GitHub 右上角头像 → **Settings** → 左侧最下 **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**。
2. Note 随便填（如 `my-pc`），Expiration 选有效期，下面 **勾选 `repo`**（整组勾上），拉到底点 Generate。
3. 生成的令牌（一长串）**只显示一次，马上复制保存**。
4. 回到 push 的终端：用户名填 GitHub 用户名，密码处**粘贴这个令牌**。以后这台机器会记住，不用再输。

**Q2：push 报错说文件太大（超过 100MB）被拒绝？**
说明有大文件混进来了。检查：
```bash
git status            # 看有没有 data/ 或 .pth
git rm -r --cached data checkpoints   # 如果误加，从暂存区移除（不删本地文件）
git commit -m "exclude large files"
git push
```
确认 `.gitignore` 里有 `data/*`、`checkpoints/*`、`*.pth`、`*.npy`（项目里已配好）。

**Q3：出现一堆 `LF will be replaced by CRLF` 警告？**
无害，Windows 正常现象，忽略即可。想消掉的话在项目根建个 `.gitattributes` 写一行 `* text=auto`。

**Q4：以后想更新代码怎么办？**
改完文件后，三连：
```bash
git add .
git commit -m "说明改了什么"
git push
```
就更新到 GitHub 了。

**Q5：完全不熟命令行，有没有图形界面办法？**
有。装 **GitHub Desktop**（https://desktop.github.com ，图形界面），用 "Add existing repository" 打开本项目文件夹，然后点 "Publish repository" 一键上传，后续改动也是点几下提交。逻辑和命令行一样，只是不用敲命令。命令行用不顺手可以走这条。

---

## 发布前最终清单（过一遍再推）

- [ ] 作者姓名已填 `README.md`、`CITATION.cff`。
- [ ] 所有 `<your-user>` 已替换成真实用户名。
- [ ] 百度网盘分享设为**永久**，链接+提取码已填进 README / dataset.md / checkpoints README。
- [ ] `git status` 确认没有把 `data/`、`.pth`、大数据集 `*.npy` 加进去。
- [ ] 数据集压缩包里带了 `metadata.csv`。
- [ ] 代码没有残留绝对路径或个人信息：`grep -rnE "F:\\\\|C:\\\\|/home/" --include='*.py' .`

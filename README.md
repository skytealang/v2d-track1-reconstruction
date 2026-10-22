# V2D Challenge 2026 — Track 1

本项目只读取官方 Track 1 单目视频与元信息。禁止读取 Track 2 的网格、人体/物体轨迹、相机标定或对应测试序列的其他真值。

## 当前实际状态（2026-10-22，北京时间）

分割质量复核正在使用公开 SAM 2.1 Hiera Large，输出独立保存在 `artifacts/seg_large/masks/`，不会覆盖旧掩码。人工 RGB 关键帧纠错记录在 `configs/seg_large_corrections.json`，实际原分辨率提示在 `artifacts/seg_large/prompts/`。新版预览为 `artifacts/seg_large/episode_XXXXXX_masks_h264.mp4`。新版在用户视频复核前不标为冻结，也不自动触发下游重建。

```bash
.envs/v2d/bin/python scripts/run_seg_large.py
.envs/v2d/bin/python scripts/run_seg_large.py --episodes 17 18 19
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_masks.py --prompts artifacts/seg_large/prompts/episode_000012.json --checkpoint checkpoints/sam2-large/sam2.1_hiera_large.pt --sam-config configs/sam2.1/sam2.1_hiera_l.yaml --bidirectional --output artifacts/seg_large/masks
.envs/v2d/bin/python scripts/render_mask_video.py --episode 12 --masks-dir artifacts/seg_large/masks --output artifacts/seg_large/episode_000012_masks_h264.mp4
```

17–19 的复核配置增加了顶部、滚筒头、细柄的前景点和桌面/手部的背景点。18、19 另在 `crop` 指定的 RGB 区域运行 Large，保留原有全画面人体掩码，将局部物体掩码映射回原分辨率。`scripts/refine_mask_crop.py` 由批处理自动调用，临时视频使用无损 H.264 并在完成后删除。提示点、裁剪坐标和原始 RGB 均可复现，未使用隐藏真值。批处理支持选择序列，修改配置后仅相应提示哈希失效并重算。

清理清单 `artifacts/cleanup_manifest.json` 记录删除的过时试跑、失败跟踪和未评分候选；其有用 JSON 报告保留于 `artifacts/retired_reports/`。真实 Kaggle 评分包、最优归档、原视频、深度缓存及旧版掩码/预览保留。

### 清洗分割后的重跑与提交

`scripts/run_clean_batch.py` 将已复核的 Large 掩码接入全部下游步骤，独立输出到 `artifacts/clean_v3/`。它重算网格、MediaPipe Heavy 关键点、200 步 MHR 拟合、FoundationPose 和时序组装，复用相同 MoGe 2 深度缓存。每段记录输入掩码 SHA256、实际命令、耗时和返回码，已完成且哈希匹配的序列可恢复。只有整段 FoundationPose 少于两帧正轮廓重合时，才使用预测掩码/深度的平移锚定兜底；该修复另记在段报告，旋转仍有失败风险。此次对照因此不是严格的纯分割单变量实验。

```bash
.envs/v2d/bin/python scripts/run_clean_batch.py
.envs/v2d/bin/python scripts/audit_predictions.py --episodes artifacts/clean_v3/episodes --output artifacts/clean_v3/audit.json
.envs/v2d/bin/python scripts/pack_public_candidate.py --episodes artifacts/clean_v3/episodes --commit https://github.com/skytealang/v2d-track1-reconstruction/commit/REAL_SHA --output artifacts/submissions/track1_v3_clean.parquet
```

所有模型均使用当前已安装的对应版本：SAM 2.1 Large、MoGe 2、MediaPipe Heavy、MHR 和 FoundationPose。本次没有切回 SAM2 Small，也不把这些公开替代组件宣称为整体最优模型；物体网格仍是单目可见面近似，完整官方 CARI4D 尚未运行。

- 已获取官方仓库，版本 `7eb9721088615fe9201634a6be5cf48b049d1090`，并解包、逐文件校验 Submission Kit。
- 已创建 `.envs/v2d`：Python 3.11、PyTorch 2.5.1 / CUDA 12.4。`pip check` 通过。
- 已下载并校验全部 67 个 Track 1 文件，共 513,185,564 字节。数据版本 `5f68335f3acc802033d1e80728c1633197521de8`。
- 已完整解码 30 段视频、16,563 帧；分辨率 1536×1152，30fps；序列与提交模板一致。
- 已对真实 `episode_000000` 的全部 790 帧运行 MoGe 2，混合精度、逐帧推理、768×576、1200 tokens。全段用时 107.24 秒（复用首次试跑的 16 帧），PyTorch 峰值分配显存 2256.45MiB，峰值保留 2368MiB。原分辨率见各 NPZ 的 `source_size`；内参为归一化坐标。
- 官方 CARI4D 源码契约测试 12 项、自建时序优化测试 5 项通过。
- 原生重依赖环境 `.envs/cari4d` 已安装，FoundationPose 的 C++/CUDA、PyTorch3D、nvdiffrast 和 SAM2 已实际运行；不依赖 Docker。尚未运行完整 CARI4D。
- SAM2.1 Small 已完成首段全部 790 帧人体/物体掩码，58.512 秒，峰值分配显存 708.48MiB。呼啦圈网格从首帧 RGB、预测掩码和预测深度拟合，无 Track 2 输入。
- FoundationPose 完成首段 790 帧物体 6DoF 跟踪；基础版本出现漂移，加入轮廓低重合重新定位后，预测掩码平均 IoU 从 0.1252 升至 0.2502，59.061 秒，峰值分配 5700.32MiB。该 IoU 不是官方指标，也不证明真实位姿准确。
- 公共 MediaPipe Pose Heavy 在 CPU 上完成首段 790/790 帧关键点检测，87.271 秒。公开 MHR 的首版全段拟合已完成，217.391 秒、峰值 689.45MiB。人体外观仍有明显偏差；已另存头部约束全量对照，未据此宣称准确率提升。
- 本地 HF/Kaggle 凭据均可用，Kaggle 五个比赛的入赛检查全部通过。CARI4D 后续已获授权，Body/Objects 仍被拒绝；当前公开版不依赖这三个模型的权重。
- 已运行独立合成轨迹的时序/接触对照实验。它不使用比赛真值，也不构成官方验证成绩。
- **公开模型备选已完成全部30段、16,563帧人体MHR/物体轨迹，30个物体网格均闭合。官方打包器生成740,780行、10,405,694字节的完整Parquet，首份提交的五项Kaggle评分全部COMPLETE。没有运行完整官方CARI4D。**
- 产物 `artifacts/episodes/`、审计 `artifacts/prediction_audit.json`、打包 `artifacts/submissions/track1.parquet`。全量原始批处理日志跨度约1723秒，包含历史失败；单独重试后完成覆盖，不能将原始控制器的失败返回码当成最终审计结果。首次正式打包8.558秒，已覆盖全部9,877个计分帧。
- 代码仓库：https://github.com/skytealang/v2d-track1-reconstruction 。提交所用代码为真实已推送版本 `3619f262ac7d22e64de7d2be93329fa0d375dc00`，Parquet中的commit URL对应此版本。

首份真实公开成绩如下，全部越低越好。receipt：`artifacts/submissions/first-public-receipt.json`，文件SHA256：`5a786ebd9085463bc530bdea0ac220bf1c5e800a83d863649788a29e6fae5c7e`。这些是已验证可评分的初版，CD误差很大，尚未核实综合排名。很低的PEN不能单独证明接触正确，人体/物体分离也可能降低穿透。

| 指标 | Public score | Submission ID |
|---|---:|---:|
| CD-H cm | 37.67795 | 57514086 |
| CD-O cm | 151.42816 | 57514098 |
| ACC-H cm/frame² | 2.63350 | 57514108 |
| ACC-O cm/frame² | 0.31251 | 57514113 |
| PEN cm | 0.01007 | 57514123 |

## 第二版：MHR根平移初始化修复

第一版将MoGe估计的髋部位置直接作为MHR全局平移，但MHR在零全局平移下的髋部中点并不在原点，原生Y约90.558cm；这会使优化从明显偏移的位置开始。现在先根据每个批次的种子姿态计算未平移的髋部中点，再赋值全局平移，使模型髋部在初始化时精确落在预测目标上。MHR平移参数每单位为10cm，初始化同时处理相机坐标翻转。

首段在相同200步、batch64、13个对应关节设置下，预测关键点残差从0.177944m降到0.037573m，归一化重投影残差从0.037735降到0.010152；初始化髋部最大误差4.82e-7m。它们仍不是GT指标。第二版使用独立目录保留首份成绩与产物：

```bash
.envs/v2d/bin/python scripts/refit_public_humans.py
.envs/v2d/bin/python scripts/audit_predictions.py --episodes artifacts/episodes_v2 --output artifacts/prediction_audit_v2.json
.envs/v2d/bin/python scripts/pack_public_candidate.py --episodes artifacts/episodes_v2 --commit "$V2D_COMMIT_URL" --output artifacts/submissions/track1_v2.parquet
```

`refit_public_humans.py` 只重跑MHR拟合和组装，复用Track1预测关键点、深度、物体网格/轨迹；不读取真值。人参数在 `artifacts/public_human_v2/`，状态、命令、时长与显存在 `artifacts/public_human_v2_status.json`。每段初始化误差大于1e-5m会明确失败。第二版全量已完成：批处理378.704秒，峰值690.93MiB；全量审计与官方打包通过，包SHA256为 `8da4bede539c27e3e0666b011f9cd909e5c83b023ba2fce359fad0047d5f1027`，所用代码commit为 `187b02a3a68521219057ddcd4c721ab32db8fc4a`。

第二份提交的五项评分均COMPLETE，均优于初版，当前保留为最佳公开候选。`artifacts/checkpoints/best_public/` 包含全部30段NPZ/网格/报告、提交Parquet、receipt与SHA256清单 `manifest.json`；初版仍在 `artifacts/checkpoints/first_scored_public/`。这些大型结果被.gitignore排除，不推送至代码仓库。

| 指标 | v1 | v2 | 降幅 | v2 Submission ID |
|---|---:|---:|---:|---:|
| CD-H cm | 37.67795 | 33.44210 | 11.24% | 57514516 |
| CD-O cm | 151.42816 | 142.51805 | 5.88% | 57514521 |
| ACC-H cm/frame² | 2.63350 | 2.49748 | 5.16% | 57514531 |
| ACC-O cm/frame² | 0.31251 | 0.29662 | 5.08% | 57514534 |
| PEN cm | 0.01007 | 0.00941 | 6.55% | 57514536 |

这仅是两个已评分版本间的公开成绩比较，尚未核实综合排名。下一项主要问题是物体形状、单目尺度与遮挡漂移；均值人体形状、未拟合的手指与接触也仍限制精度。没有根据小PEN推断接触正确。

## 实际环境

工具执行环境实际为 Ubuntu 22.04，而非环境描述中的 PowerShell。检测到 8×RTX 3090（每卡 24GiB）、驱动 570.211.01、CUDA toolkit 12.8、约 251GiB 内存、数据盘初始约 2.6TiB 可用。GPU 6 上已有其他任务，本项目首段使用 GPU 0，全量使用 GPU 0、1、2、3、4、5、7。未找到 Docker 命令或 socket；官方容器构建已实际尝试并报 `FileNotFoundError: docker`。

## 官方要求

来源：

- https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- https://github.com/nvidia-isaac/video_to_data
- https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_1
- `external/submission_kit/v2d_submission_kit/tracks/track_1/README.md`
- `artifacts/official_faq.txt`：从官方 FAQ PDF 提取，明确禁止用 Track 2 资产恢复 Track 1 网格或相机参数。

截止时间：**2026-11-04 17:00 EST，即北京时间 2026-11-05 06:00**。内部按用户目标在 11 月 4 日前完成有效提交。官网限制每周每 track/tier 五次，最后三天无限制；实际上传前也应检查 Kaggle 的实时限制和截止设置。官网两轴等权的说明与 FAQ 各指标名次求和的描述需区别保留；不能自行推定最终汇总权重。

提交为一个 Parquet，上传至五个比赛：`v2d-challenge-track1-{cd-h,cd-o,acc-h,acc-o,pen}`。全部越低越好。官方依据第一个计分帧的人体对应顶点拟合一次 Sim(3)，对整段人体和物体统一应用；物体不单独对齐。CD-H/CD-O 为表面 Chamfer（cm），ACC-H 为 22 个关节二阶差分误差，ACC-O 为物体平移二阶差分误差（cm/frame²），PEN 为手部点在所提交物体网格内的穿透深度（cm）。遮挡帧也必须给出连续物体轨迹。

每段需要：

| 文件/字段 | 形状/含义 |
|---|---|
| `episode_XXXXXX.npz: pose` | `[T,136]`，原生 MHR 模型参数，不能直接填 CARI4D 内部连续旋转参数 |
| `scales`, `shape` | `[68]`, `[45]`，每段恒定 |
| `object_rotation`, `object_translation` | `[T,3,3]`, `[T,3]`，按视频原始帧编号 |
| `object_scale` | 正标量，`point_scene = scale * R @ point_mesh + translation` |
| `episode_XXXXXX_object.glb` | 自行重建的三角网格；官方打包器按模板压缩/填充至 4096 顶点/面预算 |

实际计分帧只能从公开 sample 读取，不得根据未知测试真值选择。需要真实可访问的 GitHub `/commit/<40位SHA>` URL；私有仓库需让 `Nvidia-V2D-Challenge` 账号可访问。

## 复现

```bash
bash scripts/bootstrap.sh
# 如果默认 PyPI 连接慢，可事先设置公开软件包镜像：
# export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

`configs/sources.json` 固定代码与模型版本，`requirements.lock.txt` 记录本次实际安装版本。公开数据主站本次连接超时，因此使用 `https://hf-mirror.com` 下载公开 Track 1 文件，并校验文件大小、LFS SHA256。受限模型应在官方 Hugging Face 完成授权，本项目没有借助镜像获取受限模型。

```bash
curl -fL 'https://hf-mirror.com/api/datasets/nvidia/video_to_data_challenge/tree/5f68335f3acc802033d1e80728c1633197521de8/track_1?recursive=true&limit=1000' -o artifacts/track1_remote_tree.json
.envs/v2d/bin/python scripts/fetch_track1.py --endpoint https://hf-mirror.com --revision 5f68335f3acc802033d1e80728c1633197521de8 --tree artifacts/track1_remote_tree.json
.envs/v2d/bin/python scripts/audit_data.py --decode-all
```

MoGe 2 权重为公开模型，约 1.3GB，首次下载需几分钟：

```bash
mkdir -p checkpoints/moge2
curl -fL --retry 2 'https://hf-mirror.com/Ruicheng/moge-2-vitl-normal/resolve/b135031bae30b5ac2ae141a0e68717795ce38340/model.pt?download=true' -o checkpoints/moge2/model.pt
sha256sum checkpoints/moge2/model.pt
# 280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01
CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python scripts/run_moge.py --episode 0 --max-frames 16
CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python scripts/run_moge.py --episode 0
```

`run_moge.py` 按输入和权重哈希复用缓存，每帧原子落盘；不同输入/配置拒绝复用旧缓存。当前运行产物是独立深度初始化缓存，不会自动成为官方 CARI4D 后续阶段的输入。发生单帧 OOM 时脚本给出降低尺寸/tokens 的重跑命令；更改配置需换输出目录。

```bash
.envs/v2d/bin/python -m pytest tests external/video_to_data/reconstruction/modules/v2d_cari4d/tests/test_contract.py -q
.envs/v2d/bin/python scripts/benchmark_temporal.py
.envs/v2d/bin/python scripts/check_readiness.py
```

## 完整重建的阻塞与接续

官方 CARI4D 不是 RGB-only 的完整入口：必须先自行生成同帧人/物掩码和物体网格。不能将其文档中提供扫描物体的 SQA 条件直接用于本赛道。

1. 获得 `nvidia/cari4d_commercial`（自动授权）和 `facebook/sam-3d-body-dinov3`（人工授权）的访问许可，在服务器安全配置 `HF_TOKEN`。使用 SAM3D Objects 生成网格还需单独申请 `facebook/sam-3d-objects`。仅存在 Token 不表示已获模型批准。
2. 原生重依赖移植已完成安装，并实际运行公开 FoundationPose 跟踪。仍需授权后执行完整 CARI4D，检查额外模型资源与各阶段兼容性。
3. 使用官方 SAM2 标注/传播工具生成 object ID 0（人体）与 1（物体）掩码；使用官方 `run_pack_masks` 保持逐帧编号/尺寸一致。随后用获批的单图重建模型生成物体网格，并通过视频深度/人体尺度校准，不能从 Track 2 获取尺度或标定。
4. 将原始视频链接/复制为 `artifacts/inputs/episode_000000.0.color.mp4`，掩码命名为 `episode_000000_masks_k0.h5`，重建网格命名为 `episode_000000_object.glb`，运行 `bash scripts/run_cari4d.sh 000000`。该脚本缺输入/运行时会明确失败，不生成替代预测。
5. 将输出的 MHR 解码网格通过官方 `tools/track1/mesh_to_mhr_params.py` 转为提交参数，检查共享 identity、坐标轴翻转和单位。官方 converter 文档提供 Apache-2.0 的 Meta MHR 公开模型，**不需要使用 SAM3D Body 附带的受限版本**。目前尚未验证真实 CARI4D bundle 到该转换器的集成。
6. 为全部 30 段生成真实 NPZ 和物体网格后，设置真实 `V2D_COMMIT_URL` 并运行 `bash scripts/pack_track1.sh`。当前打包脚本只封装官方 packer，不补齐预测。
7. 在服务器安全配置 Kaggle 凭据（`~/.kaggle/access_token`，权限 `600`），浏览器接受五个比赛规则，运行官方 `--check-entry`、`--dry-run` 后上传并保存 receipt。公开备选首份文件已按此流程上传至全部五项并取得评分。

凭据不能写入代码、日志或聊天。`scripts/check_readiness.py` 只记录存在性，不输出凭据内容。最新 Kaggle entry check 已确认五个比赛均已入赛。自动化无法取得模型方尚未批准的许可。

## 原生运行与公开模型备选

当前主线优先使用不依赖受限 Meta 权重的公开批处理：

```bash
bash scripts/bootstrap.sh
bash scripts/setup_native.sh
bash scripts/setup_pose.sh
.envs/v2d/bin/python scripts/run_public_batch.py --episodes 3 --gpus 0 --smoke
.envs/v2d/bin/python scripts/run_public_batch.py
.envs/v2d/bin/python scripts/pack_public_candidate.py
.envs/v2d/bin/python scripts/summarize_public_run.py
```

批处理默认在 GPU 0、1、2、3、4、5、7 上处理 episode 1–29，保留已完成的 episode 0，避开 GPU 6。`configs/public_baseline_prompts.json` 中的提示来自官方 Track 1 首帧 RGB，512×384 预览坐标按比例映射回原始图像。每段顺序运行 MoGe、SAM2、可见表面网格、目标人体裁剪后的 MediaPipe、公开 MHR 拟合、FoundationPose 和时序组装。完成阶段按报告跳过，失败段写入 `artifacts/public_batch_status.json`，可用 `--episodes ID ...` 接续；更改配置或提示后应清理对应的下游阶段报告再重跑。

`--status-file artifacts/retry_status.json` 可为重试指定独立状态文件；旧阶段日志会保留为 `*.previous-*.log`。物体遮挡恢复会拒绝无效重新注册，跟踪无效时保留上一有效姿态并降低置信度，默认恢复间隔10帧。网格简化破坏封闭性时自动降低观测表面的采样分辨率并重新生成，要求闭合且不超过3500面。`summarize_public_run.py` 输出阶段耗时/显存CSV、预测轮廓低重合列表以及首末帧投影拼图到 `artifacts/public_run_summary/`；这些是运行与预测一致性检查，不是GT评测。

呼啦圈首段保留已验证的圆环先验重建。其余序列的首版网格通过分割轮廓与预测深度构建可见表面，再沿视线方向做有限厚度补全；这是基于观测的 2.5D 近似，背面、凹陷和遮挡部分可能严重失真。它不使用受限单图重建模型、扫描资产或 Track 2 真值。第一版优先覆盖全部序列，不代表具有竞争力的形状精度。

`pack_public_candidate.py` 复用官方打包器的网格预算和数组检查。没有 GitHub commit URL 时，只生成 `artifacts/submissions/track1_candidate_not_for_upload.parquet`，刻意省略 `code_commit_url`，用于本地检查，**不能上传**。获得真实远程 commit URL 后，传入 `--commit URL`，调用官方原始写入器生成完整文件，随后必须执行官方 uploader 的 dry-run。没有伪造仓库地址。

完整检查和上传命令如下。先确认当前代码已推送到远程，再将该SHA写入提交；每次真实上传使用新的receipt路径，轮询已有receipt不会再次消耗上传配额。网络需要代理时，在服务器本地配置对应环境变量。

```bash
export V2D_COMMIT_URL="https://github.com/skytealang/v2d-track1-reconstruction/commit/$(git rev-parse HEAD)"
.envs/v2d/bin/python scripts/audit_predictions.py
.envs/v2d/bin/python scripts/pack_public_candidate.py --commit "$V2D_COMMIT_URL"
.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/tools/preflight.py --track track_1
.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/submit/v2d_submit.py artifacts/submissions/track1.parquet --track track_1 -m "Public-model Track 1 baseline" --dry-run
.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/submit/v2d_submit.py artifacts/submissions/track1.parquet --track track_1 -m "Public-model Track 1 baseline" --wait --receipt artifacts/submissions/first-public-receipt.json
.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/submit/v2d_submit.py --track track_1 --poll-receipt artifacts/submissions/first-public-receipt.json
```

主推理 conda 环境为项目目录下的 `.envs/cari4d`；轻量环境为 `.envs/v2d`。可用 `conda activate "$PWD/.envs/cari4d"`，也可直接调用环境内 Python，不需要激活。

```bash
bash scripts/setup_native.sh
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_masks.py --output artifacts/inputs
.envs/cari4d/bin/python scripts/reconstruct_hoop.py
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_object_pose.py --max-frames 0 --recover-iou 0.10 --output artifacts/object_pose_recovery
.envs/v2d/bin/python scripts/model_access.py --proxy http://127.0.0.1:17890 --download-approved
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_native_cari4d.py --proxy http://127.0.0.1:17890
```

`setup_native.sh` 记录已使用的原生构建步骤与锁定依赖；尚未在第二个全新目录复装验收。需要已有 CUDA toolkit、C++/CMake 和 conda。`patches/cari4d-native-paths.patch` 将官方容器绝对路径改为可配置的源代码路径。`warp-lang==1.9.1` 兼容本机驱动；新版 1.18 不能在该驱动使用 CUDA。SAM2 可选小孔填充 CUDA 算子在本机崩溃，脚本关闭此后处理。decord 0.6.0 的 ctypes wheel 错误标为 cp36，修复脚本仅在匹配该元数据且运行时可导入时修正标签。

公开权重补充获取命令（网络需要时在本地设置 `HTTPS_PROXY`/`HTTP_PROXY`）：

```bash
mkdir -p checkpoints/sam2-small checkpoints/mhr
curl -fL --retry 2 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -o checkpoints/sam2-small/sam2.1_hiera_small.pt
PATH="$PWD/.envs/cari4d/bin:$PATH" .envs/cari4d/bin/python -m v2d.foundation_pose.lib.download_weights --backend nvlabs_pytorch --output_dir checkpoints/cari4d/foundationpose
curl -fL --retry 2 -H 'Accept: application/octet-stream' https://api.github.com/repos/facebookresearch/MHR/releases/assets/443665434 -o checkpoints/mhr/assets.zip
unzip -p checkpoints/mhr/assets.zip assets/mhr_model.pt > checkpoints/mhr/mhr_model.pt
.envs/v2d/bin/python scripts/record_provenance.py
```

来源、版本和预期哈希见 `configs/sources.json`；`artifacts/provenance.json` 记录实际权重哈希。MHR 权重来自公开 release v1.0.1，许可 Apache-2.0。MediaPipe Heavy 从其官方公开模型地址自动下载到独立 `.envs/pose` 包内，哈希记录在关键点 `run.json`。

公开备选人体关键点环境与命令：

```bash
.envs/v2d/bin/python -m venv .envs/pose
.envs/pose/bin/python -m pip install mediapipe==0.10.21 numpy==1.26.4 opencv-contrib-python==4.11.0.86
.envs/pose/bin/python scripts/run_pose_landmarks.py
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/fit_mhr_landmarks.py
CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/fit_mhr_landmarks.py --max-frames 0 --output artifacts/mhr_landmarks_full
.envs/v2d/bin/python scripts/assemble_episode.py
```

已有权重和环境后，`bash scripts/run_first_public.sh` 顺序执行首段公开流程并保存日志。它会重新计算关键点、人体拟合及物体跟踪；MoGe 复用校验缓存，已有 SAM2 掩码保留。首段 NPZ 检查不等于完整官方 packer/preflight 验收，打包全量提交仍需30段和真实远程 commit URL。

MediaPipe 输出是预测的髋部中心关键点，不是场景标定人体网格。备选拟合使用公开 Apache-2.0 MHR、MediaPipe 的 2D/3D 关键点与预测 MoGe 髋部深度；不读取受限 SAM 模型，也不使用 Track 2。全量结构与打包已验证，其人体形状、手指、接触与遮挡准确率仍需实际评分。可视化显示部分物体明显漂移、2.5D网格失真；低预测掩码IoU只用于定位问题，不代表GT指标。

## 本地优化实验

`src/v2d_challenge/temporal.py` 用稀疏最小二乘联合观测置信度、二阶时序先验和已知局部接触点推导的平移目标。遮挡位置可填 NaN，置信度必须为零；无足够锚点时明确报错。真实接触目标必须由预测的手部、物体局部接触和旋转计算，并进行几何门控，不能注入未知真值。

已运行实验使用解析合成轨迹，包含 45 帧与 20 帧遮挡；调参 seed 0–9，独立测试 seed 100–119。选中的时序权重为 100。线性填补 / 时序 / 时序+合成接触的平均位置误差分别为 6.00 / 2.82 / 0.94cm，遮挡区为 11.34 / 6.97 / 0.64cm。接触目标是加噪模拟观测；实验没有人模型、物体旋转、网格或接触估计误差。**不能据此推断真实序列或 Kaggle 提升**。后续应在独立公开 HOI 数据/合成 MHR 场景上复核，并做同权重无接触、不同遮挡长度、错误接触门控和运动突变对照。

## 产物

- `artifacts/hardware.json`：硬件检测。
- `artifacts/download_manifest.json`：67 文件哈希与缓存信息。
- `artifacts/data_audit.json`：全部序列与计分帧、完整解码记录。
- `artifacts/moge2/episode_000000/{000000.npz,...,000789.npz}`：深度、有效 mask、估计内参。
- `artifacts/moge2/episode_000000/{first_depth.jpg,identity.json,run.json}`：预览、缓存身份与耗时/显存。
- `artifacts/temporal_ablation.json`：合成实验结果。
- `artifacts/readiness.json`：当前阻塞。
- `artifacts/episodes/episode_000000.npz` 与 `artifacts/episodes/episode_000000_object.glb`：首段真实预测参数与网格。
- `artifacts/object_pose_recovery/`、`artifacts/mhr_landmarks_full/`：预测、关键帧投影和逐阶段资源报告。
- `logs/`：安装、下载、测试、真实推理、Kaggle/构建失败日志。
- `EXPERIMENTS.md`：可版本追溯的阶段记录。

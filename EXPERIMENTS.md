# 实验与工程记录

日期：2026-10-22（Asia/Shanghai）。早期阶段记录没有Kaggle分数；全量提交后的真实评分见后文。

| 阶段 | 实际命令 | 结果/产物 | 耗时 | 显存 | 后续 |
|---|---|---|---|---|---|
| 硬件 | `nvidia-smi`，`df -h`，`free -h`，`/usr/local/cuda/bin/nvcc --version` | `artifacts/hardware.json`；8×3090，Docker 缺失 | 单次检测约秒级，未独立计时 | 未运行 CUDA 计算 | 使用 GPU 0，避开 6 |
| 环境 | `conda create -p "$PWD/.envs/v2d" python=3.11 pip -y`；安装 requirements、torch、MoGe | Python3.11 / torch2.5.1+cu124；`pip check` 无冲突 | 分步骤日志，未记录统一墙钟 | 安装无模型计算 | 完整 CARI4D 环境待授权后处理 |
| 数据 | `python scripts/fetch_track1.py --endpoint https://hf-mirror.com --revision 5f68335f3acc802033d1e80728c1633197521de8 --tree artifacts/track1_remote_tree.json` | `data/official/track_1`；67 文件，513,185,564B，LFS 哈希通过 | 逐文件时间见 manifest；未记录统一墙钟 | 无 GPU | 完整解码 |
| 数据审计 | `.envs/v2d/bin/python scripts/audit_data.py --decode-all` | 30 段、16563 帧均解码，模板一致 | 37.615s | 无 GPU | 真实首段推理 |
| 官方源码测试 | `.envs/v2d/bin/python -m pytest external/video_to_data/reconstruction/modules/v2d_cari4d/tests/test_contract.py -q` | 12 passed；不是完整容器验收 | 0.13s | 无 GPU | 完整环境后运行容器测试 |
| MoGe 首跑 | `CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python scripts/run_moge.py --episode 0 --max-frames 16` | 16 帧真实深度缓存 | 11.646s | PyTorch allocated 2256.45MiB，reserved2368MiB | 整段 |
| MoGe 整段 | `CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python scripts/run_moge.py --episode 0` | 790/790；复用16帧；`artifacts/moge2/episode_000000/run.json` | 107.238s | 同上 | 预测掩码、网格、MHR |
| 合成消融 | `.envs/v2d/bin/python scripts/benchmark_temporal.py` | 调参10条、测试20条；不使用挑战真值 | 0.130s | 无 GPU | 加入真实预测接触与独立公开验证 |
| 官方构建尝试 | `.envs/v2d/bin/python -m v2d.cari4d.docker.build` | `FileNotFoundError: docker` | 未独立计时 | 未启动 GPU | 容器运行时或原生移植 |
| Kaggle 入口 | `.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/submit/v2d_submit.py --track track_1 --check-entry` | 缺认证；未检查账号参赛规则，未上传 | 未独立计时 | 无 GPU | 用户本地认证/接受规则 |
| 集成检查 | `.envs/v2d/bin/python -m pytest tests external/video_to_data/reconstruction/modules/v2d_cari4d/tests/test_contract.py -q` | 17 passed | 0.37s | 无 GPU | 授权后推进真实 HOI |

## 合成平移消融

单独为时序+接触分支在训练种子选择权重100，然后三分支在独立测试种子使用同一权重比较；因此并非分别调参后的各分支最优结果。观测噪声2.5cm，合成接触目标噪声1.2cm。接触分支额外使用模拟接触观测。

| 分支 | 位置误差 cm | 遮挡位置误差 cm | 二阶差分误差 cm/frame² |
|---|---:|---:|---:|
| 线性填补 | 5.9962 | 11.3427 | 7.1122 |
| 时序正则 | 2.8167 | 6.9743 | 0.08488 |
| 时序+合成接触 | 0.9406 | 0.6444 | 0.06074 |

以上不包含物体旋转、人体/手部预测误差、物体形状或真实接触推断误差，不是官方 ACC/PEN/CD 分数。

## 已处理的一般错误

- 工具实际使用 Bash：首次 PowerShell 命令失败后改用 Linux 检测。
- HF 主站超时：公开数据改走公开镜像，固定 revision，核验 LFS SHA256。
- 镜像对 Python 默认 HTTP 客户端返回403：使用明确的 curl User-Agent；没有获取受限资源。
- Git 大量无关机器人 LFS 资源下载停滞：停止该进程，仅恢复缺失文件的 LFS 指针，保留 Track1 所需源码与已解包 kit。
- 新版 Gradio 自动升级 huggingface-hub：固定 Gradio5.49.1/hub0.36.0并移除残余 hf-gradio，最终 `pip check` 通过。

## 凭据配置后的实际进展

公开全量版：`scripts/run_public_batch.py` 使用 GPU 0、1、2、3、4、5、7，按序列隔离日志，并支持按完成报告接续。启动命令 `.envs/v2d/bin/python -u scripts/run_public_batch.py > logs/public-batch-all.log 2>&1`。不调用 CARI4D、SAM 3D Body 或 SAM 3D Objects 权重。

非呼啦圈 smoke：`.envs/v2d/bin/python scripts/run_public_batch.py --episodes 3 --gpus 0 --smoke`，总计50.612秒；16帧深度、SAM2、红碗可见表面网格、裁剪后人体关键点、公开MHR拟合及FoundationPose均实际运行。网格1752顶点/3500面且闭合；MHR拟合11.792秒、峰值687.94MiB；FoundationPose3.725秒、峰值5695.62MiB，预测掩码IoU0.3380。形状背面和厚度为启发式补全，不能将这些预测一致性指标视为GT精度。

全量产物组装后使用 `scripts/audit_predictions.py` 检查30段帧数、数组有限值、旋转与网格，再通过 `scripts/pack_public_candidate.py --commit <实际URL>` 调用官方打包器。尚未获得完整重建时不上传；不使用虚构GitHub URL。

| 阶段 | 实际命令 | 产物与结果 | 耗时 | 峰值分配显存 |
|---|---|---|---|---|
| 原生环境 | 安装官方 Dockerfile 对应依赖、编译 PyTorch3D / nvdiffrast / FoundationPose / SAM2 | `.envs/cari4d`，安装日志在 `logs/*build*`、`logs/cari4d-native-dependencies.log`；FoundationPose 真实注册/跟踪验证通过 | 多步安装，未统一计时 | 编译未统一采样 |
| SAM2 全段 | `CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_masks.py --output artifacts/inputs` | `artifacts/inputs/episode_000000_masks_k0.h5`，790 帧；目视检查关键帧 | 58.512s | 708.48MiB |
| 公开 MHR | `CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python scripts/check_mhr_model.py` | `artifacts/mhr_forward_check.json`，合成8帧模型前向通过 | 3.175s | 713.69MiB |
| 官方参数转换器 | `CUDA_VISIBLE_DEVICES=0 .envs/v2d/bin/python external/submission_kit/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py --model checkpoints/mhr/mhr_model.pt --input artifacts/mhr_synthetic_vertices.npy --output artifacts/mhr_synthetic_params.npz --precision float32 --model-batch 16 --keyframes 3` | 合成8帧自重建平均顶点误差 0.0000918845mm；仅验证参数/单位闭环，不是视频精度 | 27.5s | 未独立采样 |
| 呼啦圈拟合 | `.envs/cari4d/bin/python scripts/reconstruct_hoop.py` | `artifacts/inputs/episode_000000_object.glb`；1152顶点/2304面，闭合网格，主半径0.34457m、管半径0.00540m；单目尺度不确定 | 0.190s | CPU |
| FoundationPose 试跑 | `CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_object_pose.py` | `artifacts/object_pose_smoke/poses.npy`；真实16帧 | 4.246s | 5695.84MiB |
| FoundationPose 基础 | `CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_object_pose.py --max-frames 0 --output artifacts/object_pose_full` | 790帧；抬圈后漂移，预测掩码平均轮廓IoU 0.12525 | 44.377s | 5695.83MiB |
| FoundationPose 恢复 | `CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/run_object_pose.py --max-frames 0 --recover-iou 0.10 --output artifacts/object_pose_recovery` | 790帧；平均IoU 0.25023，预览见该目录；该值不是GT指标 | 59.061s | 5700.32MiB |
| 人体公开备选 | `.envs/pose/bin/python scripts/run_pose_landmarks.py` | `artifacts/pose_landmarks/episode_000000/landmarks.npz`；790/790帧检测 | 87.271s | CPU |
| Kaggle 复查 | `.envs/v2d/bin/python external/submission_kit/v2d_submission_kit/submit/v2d_submit.py --track track_1 --check-entry` | `logs/kaggle-entry-latest.log`；五项全部已入赛；未上传 | 未独立计时 | CPU |
| HF 复查 | `.envs/v2d/bin/python scripts/model_access.py --proxy http://127.0.0.1:17890 --download-approved` | `artifacts/model_access.json`；CARI4D未授权，Body/Objects被作者拒绝；未下载受限模型 | 各API约1秒 | CPU |

原生环境已实际完成 `pip check`；重安装脚本尚未在空环境独立复验。所有可视化、IoU与关键点残差只检查预测内部一致性，不替代未知真值评测。

首版公开 MHR 拟合命令：`CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/fit_mhr_landmarks.py --max-frames 0 --output artifacts/mhr_landmarks_full`。790帧、217.391秒、峰值689.45MiB；12个人体关键点的预测间残差11.42cm、归一化重投影残差0.02316。它们不是GT误差。关键帧可见头部、体型偏差，因此后续加入耳部中点到头部关节的弱对应。

单段组装：`.envs/v2d/bin/python scripts/assemble_episode.py`，0.022秒、CPU，生成 `artifacts/episodes/episode_000000.npz` 和网格，完整790帧、官方旋转检查通过。物体平移二阶差分RMS从0.11670降至0.004023m/frame²，平均平移改动3.07cm；这里只表示更平滑，不能推断官方ACC提升。该文件不包含另外29段，尚未打包上传。

头部约束为可选 `--fit-head`，默认保留12关节首版便于复现。实验命令 `CUDA_VISIBLE_DEVICES=0 .envs/cari4d/bin/python scripts/fit_mhr_landmarks.py --fit-head --output artifacts/mhr_landmarks_head_confirm`，先验证前16帧。增加的第13个对应是预测耳部中点到MHR头部关节，存在解剖对应偏差；不能直接比较不同关节集合的平均残差。尚未替换首版全段产物。

## 当前必须解决的输入

完整视频 MHR 与物体轨迹联合验证；扩展全部30段并完成正式打包和Kaggle评分。公开模型主线已实际运行，不依赖SAM3D Body/Objects授权。原生运行时、Kaggle认证/入赛和公开GitHub远程均已完成。

## 全量运行中的恢复修复

代码已实际推送至公开仓库 `https://github.com/skytealang/v2d-track1-reconstruction`，首个全量实现提交 `da8b20d76322b80a5f43b1c92d682b6b40ab5394`。授权使用GitHub CLI设备流程，凭据保存在项目目录之外，未写入仓库。

- 物体严重遮挡时，FoundationPose重新注册可能返回无效矩阵。现在在渲染前检查候选，拒绝无效恢复；跟踪无效时保留上个有效姿态并将该帧置信度置零。默认恢复间隔10帧，避免低轮廓重合时每帧重复注册。episode6/7实际重跑成功，完整816/777帧，包含组装分别耗时163.114/83.283秒。详见 `artifacts/public_retry_status.json` 和逐帧物体报告。遮挡期间保持姿态是一种估计，不是真值恢复。
- 闭合表面的二次简化可能产生孔洞：episode9从24912面简化至3500面后不再闭合。现在在简化破坏封闭性或超过预算时，逐步降低轮廓采样分辨率并重新构建表面，直到同时满足闭合与3500面预算。episode9/11/19真实输入重建通过，分别0.376/0.639/0.260秒，均为CPU，1758/1752/1752顶点、3500面。产物在 `artifacts/inputs/episode_XXXXXX_object.glb`。
- 批处理可通过 `--status-file` 隔离并行重试记录；重跑保留上一版阶段日志。恢复命令：`.envs/v2d/bin/python -u scripts/run_public_batch.py --episodes 9 11 19 --gpus 2 3 4 --status-file artifacts/public_mesh_retry_status.json`。
- 修复后17项现有测试通过（0.39秒）；还需完成全部序列及官方提交文件检查。此处仍无Kaggle分数。

首段头部约束全量对照已实际运行：`CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .envs/cari4d/bin/python -u scripts/fit_mhr_landmarks.py --episode 0 --max-frames 0 --steps 200 --batch-size 64 --fit-head --output artifacts/public_human/episode_000000`。790帧、82.522秒、峰值690.99MiB；预览在 `artifacts/public_run_summary/episode0_head_comparison.jpg`。它与原版的迭代数/批大小/对应关节集合不同，无法归因头部约束的独立效果，未替换首段已组装版本；没有宣称质量提升。

新增 `scripts/setup_pose.sh` 将Pose依赖安装到独立venv并执行pip check，已在现有环境实际运行通过，日志 `logs/setup-pose-reproducibility.log`；尚未声称全新机器复装验收。`scripts/summarize_public_run.py` 根据真实逐阶段JSON生成 `artifacts/public_run_summary/stages.csv`、`summary.json` 及物体首末帧拼图；阶段时长相加不是多卡并发墙钟时长。可视化仍显示若干类别的明显漂移、形状变形，低IoU列表只作为后续排查线索，不作为官方准确率。

## 首份全量Kaggle提交

30段、16,563帧全部组装，9,877个计分帧覆盖；30个输入网格均闭合。原始主批处理日志出生/末次修改时间差1723秒（`stat`），包括失败的尝试；控制器保留历史失败，因此返回2，独立重试完成后以全量审计与官方打包结果为准。episode10的早期网格7520面，虽然闭合但官方简化无法满足4096面预算。重新采样成3500面并重新跟踪/组装，重跑60.749秒，报告 `artifacts/public_budget_retry_status.json`；没有修改官方打包器。

| 阶段 | 命令/记录 | 真实结果与产物 | 用时/峰值 |
|---|---|---|---|
| 全量审计 | `.envs/v2d/bin/python scripts/audit_predictions.py` | `artifacts/prediction_audit.json`，30/30、16563帧、有限值/旋转通过、30个闭合网格 | CPU；未单独记录墙钟 |
| 正式打包 | `.envs/v2d/bin/python scripts/pack_public_candidate.py --commit https://github.com/skytealang/v2d-track1-reconstruction/commit/3619f262ac7d22e64de7d2be93329fa0d375dc00` | `artifacts/submissions/track1.parquet`，740780行，10405694B，官方检查通过 | 8.558秒，CPU |
| 官方dry-run | `v2d_submit.py artifacts/submissions/track1.parquet --track track_1 -m "Public-model Track 1 baseline 3619f26" --dry-run` | `logs/submission-public-first-dry-run.log`，身份/规则/格式通过，各项5次剩余 | 未单独计时，CPU |
| 真实上传/等待 | 同文件使用 `--wait --receipt artifacts/submissions/first-public-receipt.json` | 五项全部COMPLETE，无评分错误 | 网络及Kaggle服务端耗时；本地CPU |

成功阶段报告的累计用时（不是并发墙钟）：深度2452.113秒、分割1588.874秒、网格8.167秒、关键点2103.785秒、MHR1943.780秒、物体1840.046秒、组装0.642秒；峰值分配显存分别2256.45、708.48、0、CPU、691.07、5701.03、0MiB。统计只包含最新成功阶段，不含失败重试的全部成本。详细记录 `artifacts/public_run_summary/stages.csv`。

| 指标 | Public score | Submission ID | 状态 |
|---|---:|---:|---|
| CD-H cm | 37.67795 | 57514086 | COMPLETE |
| CD-O cm | 151.42816 | 57514098 | COMPLETE |
| ACC-H cm/frame² | 2.63350 | 57514108 | COMPLETE |
| ACC-O cm/frame² | 0.31251 | 57514113 | COMPLETE |
| PEN cm | 0.01007 | 57514123 | COMPLETE |

分数证明提交链路已打通，不代表重建精度足够。CD-O尤其差；低PEN不能区分真实接触与人/物分离，不能据此宣称接触优化成功。首份Parquet与receipt保留不覆盖，后续候选另存。尚未核实综合排名。

## MHR根平移初始化修复对照

直接运行公开MHR前向确认：零全局平移时，左右髋部原生坐标分别为 `[8.20465,90.55811,0.53257]` 与 `[-8.20465,90.55811,0.53257]` cm；全局Y平移参数增加1时，所有关节增加10cm。原有初始化没有减去这个髋部偏移。修复先对当前种子姿态做骨架前向，扣除未平移的髋部中点后再设置全局平移，涵盖批次种子旋转。

同设置首段对照：均790帧、200步、batch64、13个对应关节；修复前 `artifacts/public_human/episode_000000/run.json`，修复后 `artifacts/public_human_v2/episode_000000/run.json`。预测3D关节平均残差0.1779439→0.0375731m，归一化2D重投影残差0.0377352→0.0101523；初始化髋部最大误差4.8147e-7m。修复后85.811秒、峰值690.85MiB。真实对照图 `artifacts/public_run_summary/episode0_root_fix_comparison.jpg`。这些残差只表明与同一组预测观测更一致，尚不能代替Kaggle准确率。

全量命令 `.envs/v2d/bin/python -u scripts/refit_public_humans.py > logs/public-human-v2-all.log 2>&1`，使用GPU0/1/2/3/4/5/7，复用物体结果，在 `artifacts/episodes_v2/` 组装，不覆盖初版。状态 `artifacts/public_human_v2_status.json` 逐段记录初始化检查、真实命令、时长、峰值。修复后现有17项测试通过（0.35秒）；全量和第二份评分进行中。

第二版全量完成：378.704秒并发墙钟，拟合成功阶段用时合计2125.587秒（包含已先行计算的首段85.811秒，不等于批处理墙钟），峰值690.93MiB。30段最大初始化髋部误差4.8834e-7m。按帧数加权的预测关键点残差0.183969→0.040437m，归一化重投影残差0.043327→0.012419。对照的episode0为同设置13关节试验，不是首份评分使用的12关节版本；因此官方v1/v2成绩对照还包含首段的对应关节/迭代配置差异。详细JSON `artifacts/root_fix_ablation.json`。

另以公开MHR实际骨架前向检查4个不同旋转种子、合成目标髋部位置，最大位置误差1.1921e-7m，CPU0.557秒，无挑战GT，输出 `artifacts/mhr_root_initialization_roundtrip.json`。

严格审计 `.envs/v2d/bin/python scripts/audit_predictions.py --episodes artifacts/episodes_v2 --output artifacts/prediction_audit_v2.json`：30/30、16563帧通过。第二版官方打包8.642秒，输出 `artifacts/submissions/track1_v2.parquet`，740780行、10430161字节，SHA256 `8da4bede539c27e3e0666b011f9cd909e5c83b023ba2fce359fad0047d5f1027`，真实代码commit `187b02a3a68521219057ddcd4c721ab32db8fc4a`。官方dry-run通过，各项4次剩余。第二次上传使用独立receipt `artifacts/submissions/v2-public-receipt.json`，评分等待中；第一份成功包另备份到 `artifacts/checkpoints/first_scored_public/`。

## 第二份实际评分与最优归档

第二次实际命令：`HTTPS_PROXY=http://127.0.0.1:17890 HTTP_PROXY=http://127.0.0.1:17890 .envs/v2d/bin/python -u external/submission_kit/v2d_submission_kit/submit/v2d_submit.py artifacts/submissions/track1_v2.parquet --track track_1 -m "Correct MHR hip initialization 187b02a" --wait --receipt artifacts/submissions/v2-public-receipt.json`。五项全部COMPLETE，无评分错误，日志 `logs/submission-public-v2-upload.log`。

| 指标 | v1 | v2 | 公开分数降幅 | v2 ID |
|---|---:|---:|---:|---:|
| CD-H cm | 37.67795 | 33.44210 | 11.24% | 57514516 |
| CD-O cm | 151.42816 | 142.51805 | 5.88% | 57514521 |
| ACC-H cm/frame² | 2.63350 | 2.49748 | 5.16% | 57514531 |
| ACC-O cm/frame² | 0.31251 | 0.29662 | 5.08% | 57514534 |
| PEN cm | 0.01007 | 0.00941 | 6.55% | 57514536 |

v2在五项测得的公开指标上均优于v1，因此完整归档到 `artifacts/checkpoints/best_public/`，包括30段NPZ/网格/报告、Parquet、打包报告和receipt，93个文件SHA256记录于 `manifest.json`。初版不删除。仍未核实综合排名；物体142.5cm误差说明远未达到高质量重建目标。后续优先研究预测掩码/深度锚定以修复物体漂移、改进完整网格与尺度、人体时序与接触约束。

# 实验与工程记录

日期：2026-10-22（Asia/Shanghai）。这些记录没有 Kaggle 分数。

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

获批 CARI4D / SAM3D Body 权重或验证公开人体备选；完整视频 MHR 与物体轨迹联合验证；扩展全部30段；真实生产代码 GitHub commit URL。原生运行时、首段预测掩码/网格、Kaggle认证/入赛已完成。公开 Apache-2.0 MHR 模型已下载并校验，它本身不是人体视频估计器。

# 咪咕乒乓球赛题：E 组实验与验收工具

本目录保存 E 组已完成的只读数据检查、模型检查、公开集基线复现和平台性能验收脚本。尚未完成 GPU 训练


| 脚本 | 用途 |
|---|---|
| `audit.py` | 检查训练标注、异常框、分辨率，并在本地导出球心 |
| `review_data_split.py` | 检查临时按目录分组划分与视频重复 |
| `prepare_samples.py`、`check_samples.py` | 制作和检查三帧正样本 |
| `inspect_benchmark.py`、`evaluate_model_samples.py` | 检查官方 ONNX 结构与小样本球心误差 |
| `evaluate_public_cpu.py` | 在 CPU 上复现公开验证集完整落点流程 |
| `score_public_predictions.py` | 按公开阈值统计预测文件的总体及三视角 F1 |
| `platform_profile.py` | 在可用 4090 环境中运行未修改的官方框架并记录耗时 |

本机依赖见 `requirements-local.txt`。GPU 性能验收须在比赛指定镜像中运行，先检查 `nvidia-smi` 和 `torch.cuda.is_available()`，并按平台实际文件路径调整命令。所有分数以官方评测为准。

# 人体运动预测项目

本项目实现基于时空注意力、离散余弦变换（DCT）和图卷积模块的人体 3D 运动预测。当前代码支持 Human3.6M 和 CMU 两个数据集，并提供训练、整体测试、按动作测试、可视化以及逐样本误差显著性检验脚本。

> 说明：本 README 按当前仓库中的实际代码整理。项目中仍有部分历史代码、硬编码路径和参数命名不一致的问题，因此当前版本更准确的定位是“可复现运行说明和整理清单”，而不是完全一键化的发布版本。

## 1. 项目结构

```text
.
├── main_h36m_3d.py             # Human3.6M 训练和整体测试入口
├── main_h36m_3d_eval.py       # Human3.6M 按动作评估入口
├── main_cmu_3d.py             # CMU 训练入口
├── main_cmu_3d_eval.py        # CMU 按动作评估入口
├── significance_test.py       # 配对显著性检验
├── export_h36m_prediction.py  # 导出 Human3.6M 预测结果
├── visualize_*.py             # 可视化脚本
├── model/
│   ├── AttModel.py            # Human3.6M 模型
│   ├── AttModel_CMU.py        # CMU 模型
│   ├── GCN.py
│   └── GCN_CMU.py
├── utils/
│   ├── opt.py                 # 命令行参数
│   ├── h36motion3d.py         # Human3.6M 数据加载
│   ├── CMU_motion_3d.py       # CMU 数据加载
│   ├── data_utils.py          # 数据预处理和运动学转换
│   └── log.py                 # 日志和 checkpoint 保存
├── datasets/
│   ├── h3.6m/
│   └── cmu/
├── checkpoint/                 # checkpoint、配置和测试结果
└── outputs/                    # 可视化和导出的预测结果
```

## 2. 环境依赖

代码主要依赖以下 Python 包：

- Python
- PyTorch（当前训练和数据预处理代码主要使用 CUDA）
- NumPy
- SciPy
- h5py
- pandas
- matplotlib
- seaborn
- six

显著性检验使用 `scipy.stats`。可视化和图片处理脚本可能还需要 Pillow。

建议在服务器上根据 CUDA 版本安装匹配的 PyTorch，然后检查依赖：

```bash
python -c "import torch, numpy, scipy, h5py, pandas, matplotlib, seaborn; print('imports ok'); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

当前仓库没有统一的 `requirements.txt` 或 `environment.yml`，因此正式提交前建议把服务器上的实际环境导出，例如：

```bash
pip freeze > requirements_server.txt
```

## 3. 数据准备

所有训练和评估命令都建议从项目根目录执行，因为当前数据加载器使用相对路径。

### 3.1 Human3.6M

当前代码期望的数据结构如下：

```text
datasets/
└── h3.6m/
    ├── S1/
    ├── S5/
    ├── S6/
    ├── S7/
    ├── S8/
    ├── S9/
    └── S11/
```

每个 subject 目录中应包含类似以下文件：

```text
S5/walking_1.txt
S5/walking_2.txt
S5/eating_1.txt
...
```

当前 `utils/h36motion3d.py` 中的主体划分为：

| `split` | 当前代码使用的主体 | 用途 |
|---|---|---|
| `0` | S1、S6、S7、S8、S9 | 训练 |
| `1` | S11 | 验证 |
| `2` | S5 | 测试 |

数据加载时默认每隔 2 帧采样，并将指数映射转换为 3D 关节坐标。Human3.6M 的动作列表在数据加载器中默认为：

```text
walking, eating, smoking, discussion, directions,
greeting, phoning, posing, purchases, sitting,
sittingdown, takingphoto, waiting, walkingdog,
walkingtogether
```

### 3.2 CMU

当前代码期望的数据结构如下：

```text
datasets/
└── cmu/
    ├── train/
    │   ├── basketball/
    │   ├── basketball_signal/
    │   ├── directing_traffic/
    │   ├── jumping/
    │   ├── running/
    │   ├── soccer/
    │   ├── walking/
    │   └── washwindow/
    └── test/
        ├── basketball/
        ├── basketball_signal/
        ├── directing_traffic/
        ├── jumping/
        ├── running/
        ├── soccer/
        ├── walking/
        └── washwindow/
```

每个动作目录中应包含类似以下文件：

```text
datasets/cmu/train/walking/walking_1.txt
datasets/cmu/test/walking/walking_1.txt
```

CMU 数据加载器会在读取 `.txt` 后转换为 3D 坐标，并在相应动作目录中写出 `.npy` 缓存文件。因此运行账号需要对数据目录具有写权限，或者提前准备好可写的数据副本。

## 4. 默认实验设置

当前 `utils/opt.py` 的默认参数并不完全适配所有模型入口。由于模型内部要求 `kernel_size == 10`，正式实验应显式指定与 checkpoint 一致的参数。

常用设置如下：

| 参数 | Human3.6M 常用值 | CMU 常用值 |
|---|---:|---:|
| 输入帧数 `input_n` | 50 | 50 |
| 预测帧数 `output_n` | 10 或 25 | 10 |
| 输入维度 `in_features` | 66 | 75 |
| `kernel_size` | 10 | 10 |
| `d_model` | 256 | 256 |
| `num_stage` | 12 | 12 |
| `dct_n` | 20 | 20 |
| batch size | 32 | 32 |
| test batch size | 16 | 32 |
| 初始学习率 | 0.0005 | 0.0005 |
| 训练 epoch | 50 | 50 |

Human3.6M 评估代码中的 `dim_used` 固定为 66 维，CMU 评估代码中的 `dim_used` 固定为 75 维。因此不要只依赖 `utils/opt.py` 的默认 `in_features=54`。

## 5. Human3.6M 训练

建议使用与当前模型结构匹配的参数：

```bash
python main_h36m_3d.py --ckpt checkpoint/repro_h36m --in_features 66 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 10 --dct_n 20 --lr_now 0.0005 --epoch 50 --batch_size 32 --test_batch_size 16
```

训练时 `utils/opt.py` 会根据入口脚本和参数自动生成实验子目录。上面命令的 checkpoint 通常会保存到类似以下目录：

```text
checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20/
```

主要输出包括：

```text
ckpt_best.pth.tar
ckpt_last.pth.tar
option.json
test.csv
```

恢复训练时，应继续使用相同的模型参数和 `--ckpt` 父目录，并增加 `--is_load`：

```bash
python main_h36m_3d.py --is_load --ckpt checkpoint/repro_h36m --in_features 66 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 10 --dct_n 20
```

## 6. Human3.6M 整体测试

`main_h36m_3d.py` 的评估分支加载 `ckpt_best.pth.tar`，并将每个测试样本的平均 MPJPE 保存为 `.npy`，用于后续显著性检验：

```bash
python main_h36m_3d.py --is_eval --ckpt checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20 --in_features 66 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 10 --dct_n 20
```

输出文件：

```text
checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20/test_walking.csv
checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_errors.npy
```

`test_sample_errors.npy` 的第一个维度对应测试样本顺序。比较两个模型时，两个模型必须使用完全相同的测试数据、窗口生成方式和 DataLoader 顺序。

重新运行 Human3.6M 的 `main_h36m_3d.py --is_eval` 后，还会生成：

```text
checkpoint/repro_h36m/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_ids.csv
```

该文件一行对应一个误差值，记录动作、子序列、下采样后窗口起点、原始帧起点及稳定的 `sample_id`。Human3.6M 的当前测试协议为 S5、15 个动作、每个动作 2 个子序列、每个子序列固定抽取 128 个窗口，因此预期共 `15 x 2 x 128 = 3840` 行。

在 HisRepItself 也导出同样格式的 `sample_id` 后，使用以下命令确认两个 `.npy` 可配对：

```bash
python verify_paired_samples.py \
  --a_errors checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_errors.npy \
  --a_ids checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/test_sample_ids.csv \
  --b_errors /path/to/hisrepitself_h36m_sample_errors.npy \
  --b_ids /path/to/hisrepitself_h36m_sample_ids.csv \
  --name_a STRN \
  --name_b HisRepItself
```

只有脚本输出 `PAIRABLE` 时，才可以对这两个数组做 paired t-test 或 Wilcoxon 检验。

## 7. Human3.6M 按动作评估

`main_h36m_3d_eval.py` 使用迭代预测方式生成较长未来序列，加载的是：

```text
<checkpoint目录>/ckpt_last.pth.tar
```

当前脚本中的动作列表被写死为：

```python
acts = ["waiting"]
```

因此默认只评估 `waiting`。如果需要评估全部动作，需要在运行前修改 `main_h36m_3d_eval.py` 中的 `acts` 列表。

示例：

```bash
python main_h36m_3d_eval.py --ckpt checkpoint/main_h36m_3d_in50_out10_ks10_dctn20 --in_features 66 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 25 --dct_n 20 --test_batch_size 16
```

输出文件包括：

```text
<checkpoint目录>/test_pre_action.csv
<checkpoint目录>/waiting_sample_errors.npy
<checkpoint目录>/test_pre_action_sample_errors.npy
```

这里的 `output_n=25` 表示最终评估 25 个未来帧；脚本内部以每次 10 帧、共 3 次迭代的方式进行预测。

## 8. CMU 训练

CMU 训练入口使用 `datasets/cmu/train/` 作为训练数据，并使用 `datasets/cmu/test/` 进行验证和测试：

```bash
python main_cmu_3d.py --ckpt checkpoint/repro_cmu --in_features 75 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 10 --dct_n 20 --lr_now 0.0005 --epoch 50 --batch_size 32 --test_batch_size 32
```

常见输出目录：

```text
checkpoint/repro_cmu/main_cmu_3d_in50_out10_ks10_dctn20/
```

主要输出包括：

```text
ckpt_best.pth.tar
ckpt_last.pth.tar
option.json
test.csv
```

## 9. CMU 按动作评估

推荐使用 `main_cmu_3d_eval.py` 进行 CMU 测试。脚本当前评估以下 8 个动作：

```text
basketball
basketball_signal
directing_traffic
jumping
running
soccer
walking
washwindow
```

示例：

```bash
python main_cmu_3d_eval.py --ckpt checkpoint/repro_cmu/main_cmu_3d_in50_out10_ks10_dctn20 --in_features 75 --num_stage 12 --d_model 256 --kernel_size 10 --input_n 50 --output_n 10 --dct_n 20 --test_batch_size 32
```

输出文件包括：

```text
<checkpoint目录>/test_pre_action.csv
<checkpoint目录>/basketball_sample_errors.npy
<checkpoint目录>/basketball_signal_sample_errors.npy
...
<checkpoint目录>/washwindow_sample_errors.npy
<checkpoint目录>/test_pre_action_sample_errors.npy
```

注意：`main_cmu_3d.py` 中的 `--is_eval` 分支当前存在 `dataset` 变量未定义的问题，因此 CMU 独立测试应使用 `main_cmu_3d_eval.py`，不要直接依赖 `main_cmu_3d.py --is_eval`。

## 10. 显著性检验

### 10.1 统计对象

评估脚本保存的 `*_sample_errors.npy` 是逐样本误差。每个样本的误差为预测区间内、所有关节的平均 MPJPE。

因此，比较两个模型时应使用：

- 同一个数据集；
- 同一个动作或同一组动作；
- 同一个输入长度和预测长度；
- 相同的测试窗口；
- 相同的样本顺序；
- 两个模型分别得到的逐样本误差文件。

不能把一个模型的整体平均值和另一个模型的逐样本数组直接进行配对检验。

### 10.2 运行命令

Linux 服务器上：

```bash
python significance_test.py --a path/to/ours_sample_errors.npy --b path/to/baseline_sample_errors.npy --name_a STRN --name_b Baseline
```

如果输入数组是二维数组，例如形状为 `[样本数, 预测帧数]`，默认会对每个样本沿预测帧求平均：

```bash
python significance_test.py --a path/to/ours_errors.npy --b path/to/baseline_errors.npy --name_a STRN --name_b Baseline --reduce mean --alpha 0.05 --json_out significance_result.json
```

脚本会输出：

- 样本数、均值和标准差；
- 配对 t 检验；
- Wilcoxon 符号秩检验；
- Cohen's d 效应量；
- JSON 格式的结果（如果指定 `--json_out`）。

### 10.3 论文报告建议

显著性检验应在同一测试样本上进行配对比较。建议在论文中同时报告：

1. 两个模型的平均 MPJPE；
2. 平均误差差值和 95% 置信区间；
3. 配对 t 检验或 Wilcoxon 检验的 p 值；
4. Cohen's d 等效应量；
5. 显著性水平，例如 `alpha=0.05`。

如果同时比较多个动作或多个预测时刻，应额外考虑多重比较校正。当前 `significance_test.py` 不自动执行多重比较校正，需要在汇总结果时另行处理。

## 11. 训练曲线、误差分布和平均排名

审稿人要求补充训练/验证曲线、误差分布图和平均排名分析。当前仓库提供 `analyze_results.py` 统一生成这些结果。

### 11.1 训练和验证曲线

输入文件为训练过程中生成的 `test.csv`。该文件通常包含 `epoch`、`m_p3d_h36`、`valid_m_p3d_h36` 和各预测时刻的测试误差。

```bash
python analyze_results.py curves \
  --log H36M=checkpoint/main_h36m_3d_in50_out10_ks10_dctn20/test.csv \
  --log CMU=checkpoint/main_cmu_3d_in50_out10_ks10_dctn20/test.csv \
  --out_dir outputs/analysis \
  --filename training_curves.png
```

输出：

```text
outputs/analysis/training_curves.png
```

该图左侧为训练/验证 MPJPE 曲线，右侧为不同预测时刻的测试 MPJPE 曲线。

### 11.2 逐样本误差分布图

输入文件为评估脚本生成的 `*_sample_errors.npy`。比较多个模型时，每个模型应使用同一批测试样本。

```bash
python analyze_results.py distribution \
  --model STRN=checkpoint/strn/test_sample_errors.npy \
  --model Baseline=checkpoint/baseline/test_sample_errors.npy \
  --out_dir outputs/analysis \
  --filename error_distribution.png
```

输出：

```text
outputs/analysis/error_distribution.png
outputs/analysis/error_distribution_summary.csv
```

如果输入是二维数组，形状为 `[样本数, 预测帧数]`，脚本会同时画整体逐样本误差分布和逐预测时刻平均误差曲线。如果输入是一维数组，脚本会画整体逐样本误差分布。

### 11.3 平均排名表

输入文件为最终评估结果 CSV，例如 `test_pre_action.csv` 或训练日志 `test.csv`。

```bash
python analyze_results.py ranking \
  --result 'Human3.6M|STRN|checkpoint/strn_h36m/test_pre_action.csv' \
  --result 'Human3.6M|Baseline|checkpoint/baseline_h36m/test_pre_action.csv' \
  --result 'CMU|STRN|checkpoint/strn_cmu/test_pre_action.csv' \
  --result 'CMU|Baseline|checkpoint/baseline_cmu/test_pre_action.csv' \
  --out_dir outputs/analysis
```

输出：

```text
outputs/analysis/ranking_by_horizon.csv
outputs/analysis/average_ranking.csv
outputs/analysis/average_ranking_wide.csv
```

排名规则为：在同一数据集、同一预测时刻下，MPJPE 越低排名越靠前，然后对所有数据集和预测时刻的排名取平均。只有一个模型时平均排名恒为 1，因此论文中应至少包含本文方法和一个 baseline。

如果输入是 `test.csv`，默认使用最后一行结果；也可以使用验证集误差最低的 epoch：

```bash
python analyze_results.py ranking \
  --result 'Human3.6M|STRN|checkpoint/strn_h36m/test.csv' \
  --result 'Human3.6M|Baseline|checkpoint/baseline_h36m/test.csv' \
  --row best_valid \
  --out_dir outputs/analysis
```

## 12. 预测结果导出和可视化

项目中包含多种历史可视化脚本，常用脚本包括：

```text
export_h36m_prediction.py
visualize_h36m_walking_gt.py
visualize_h36m_walking_ours.py
viz_grid_compare.py
viz_grid_compare_fixed.py
viz_multi_action_overlay_2d.py
```

例如，导出 Human3.6M 预测结果前，应先检查脚本中的 checkpoint 路径和参数是否与当前模型一致：

```bash
python export_h36m_prediction.py --help
```

部分旧脚本内部仍包含固定的 checkpoint 路径、动作列表或设备设置，使用前需要检查脚本顶部的参数和路径。生成的 `.npy`、图片和文本结果通常保存在 `outputs/` 或指定输出目录。

## 13. 当前复现注意事项

以下问题是当前代码整理时需要明确记录的事项：

1. 数据加载器中的数据路径目前主要写死为 `./datasets/h3.6m` 和 `./datasets/cmu/`。虽然 `utils/opt.py` 提供了 `--data_dir`，但当前两个主要数据加载器没有统一使用该参数。
2. 多数训练、数据预处理和模型代码直接调用 `.cuda()`，推荐使用有 CUDA 的服务器运行。
3. `model/AttModel.py` 和 `model/AttModel_CMU.py` 都断言 `kernel_size == 10`。
4. Human3.6M 和 CMU 的输入维度不同，分别常用 66 和 75，不能混用 checkpoint。
5. `main_h36m_3d_eval.py` 当前默认只评估 `waiting`，按动作实验前需要确认并记录 `acts` 列表。
6. CMU 测试数据转换过程中会生成 `.npy` 文件，需要数据目录写权限。
7. 不同入口对 checkpoint 的加载文件不同：部分训练/整体评估入口加载 `ckpt_best.pth.tar`，Human3.6M 按动作评估入口加载 `ckpt_last.pth.tar`。
8. `option.json` 记录的是运行参数，但历史 checkpoint 目录中可能存在旧实验遗留的参数，正式实验应以实际命令和模型结构为准。
9. Human3.6M 的 `split` 语义以 `utils/h36motion3d.py` 当前实现为准，当前实现为 `0=训练、1=验证、2=测试`。
10. 进行显著性检验时，必须保存并对齐两个模型的逐样本误差，而不是只比较 CSV 中的最终均值。

## 14. 面向审稿人代码整理的建议

为了提高代码可复现性，后续建议继续完成以下整理：

1. 增加统一的 `requirements.txt` 或 `environment.yml`。
2. 将数据根目录改为所有入口统一使用的命令行参数。
3. 将动作列表和主体划分移到配置文件或命令行参数中。
4. 统一 `ckpt_best` 和 `ckpt_last` 的评估规则，并在 README 中固定说明。
5. 移除调试脚本中的个人绝对路径和无关实验代码。
6. 增加随机种子、CUDA 版本、PyTorch 版本和 GPU 型号记录。
7. 为每次实验保存完整命令、参数、checkpoint、测试结果和逐样本误差。
8. 对不同预测时刻和不同动作的统计检验增加多重比较校正。
9. 将历史脚本中的乱码注释、重复导入和未使用变量逐步清理。

## 15. 推荐的复现实验记录

每次服务器实验建议至少记录以下信息：

```text
实验名称：
运行命令：
数据集：
训练/验证/测试划分：
输入帧数：
预测帧数：
kernel_size：
dct_n：
num_stage：
batch size：
epoch：
随机种子：
GPU 型号：
PyTorch 版本：
checkpoint 路径：
最终 CSV 路径：
逐样本误差路径：
```

这样可以同时满足代码整理、实验复现和审稿回复中的可追溯性要求。

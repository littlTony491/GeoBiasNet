# GeoBiasNet

3D 人体动作预测的 H3.6M 训练与测试实现，包含四阶段模型、SBCU 模块、数据准备、检查点恢复和按动作评测。

## 环境安装

```powershell
git clone https://github.com/littlTony491/GeoBiasNet.git
cd GeoBiasNet
conda env create -f environment.yml
conda activate geobiasnet
```

运行依赖为 PyTorch 和 NumPy。已验证 Python 3.10.22、PyTorch 2.11.0+cu128、NumPy 2.2.6。本机已有 `barnet-train` 环境时也可直接复用。

需要 CUDA 12.8 构建时：

```powershell
python -m pip install -r requirements-gpu-cu128.txt
```

## 下载数据

使用 H3.6M 的 99 维指数映射 TXT 数据。下载器采用 [LearnTrajDep 作者提供的镜像](https://github.com/wei-mao-2019/LearnTrajDep#get-the-data)，下载 ZIP 后自动整理目录，并校验 7 个主体、15 个动作、210 个文本文件及其坐标格式。

```powershell
python -m pip install -r requirements-download.txt
python download_data.py
```

完成后的目录可直接用于训练：

```text
dataset/h3.6m/
  S1/
    walking_1.txt
    walking_2.txt
    ...
  S5/
  S6/
  S7/
  S8/
  S9/
  S11/
```

下载支持续传和缓存；重复执行时会检查已有数据。数据使用须遵守 [Human3.6M 数据集许可](https://vision.imar.ro/human3.6m/eula.php)。

已有压缩包时，无需安装下载依赖，也无需联网：

```powershell
python download_data.py --archive /path/to/h36m.zip
python download_data.py --check-only
```

可通过 `--output-dir` 指定数据目录，通过 `--url` 使用其他 ZIP 下载地址，网络需要代理时可传入 `--proxy`。已有数据也可直接用训练入口的 `--data-dir` 指定。

## 训练

默认输入 10 帧、预测 50 帧，DCT 长度 60，模型宽度 16，批量 16，训练 50 轮。

```powershell
python train.py --device cuda:0 --output-n 50 --output-dir checkpoints/h36m_10in50out
```

预测 25 帧时使用 `--output-n 25`，并指定对应的输出目录。CPU 运行可使用 `--device cpu`。

每轮保存 `history.jsonl`、逐时刻指标、`last.pt`、`best.pt` 和编号检查点。恢复时保持原参数及计划轮数：

```powershell
python train.py --device cuda:0 --output-n 50 --epochs 50 --resume latest --output-dir checkpoints/h36m_10in50out
```

学习率默认按 `0.1 ** (1 / epochs)` 在每轮开始前衰减；可用 `--lr-decay` 显式指定。其他参数可通过 `python train.py --help` 查看。

## 测试

测试自己的模型：

```powershell
python test.py --device cuda:0 --output-n 50 --checkpoint checkpoints/h36m_10in50out/best.pt --output-dir results/h36m_10in50out
```

测试仓库提供的发布权重：

```powershell
python test.py --device cuda:0 --output-n 50 --checkpoint checkpoint/ckpt_best.pth.tar --output-dir results/published_checkpoint
```

结果保存为 `test_metrics.json` 和 `test_metrics.csv`。预测长度和模型宽度需要与检查点一致。

## 数据划分与指标

- 训练主体：S1/S6/S7/S8/S9。
- 默认验证主体：S5，对应 `--validation-split 2`；使用 S11 验证时指定 `--validation-split 1`。
- 测试主体：S5，分别计算 15 个动作，再对动作等权平均。
- 指标：32 关节 MPJPE，单位毫米；训练/验证按样本平均，测试按动作平均。
- 训练目标：最终输出和四级输出的五个 MPJPE 之和。

## 代码检查

```powershell
python -m unittest discover -s tests -v
```

测试覆盖数据安装、模型结构、训练计算、指标口径和检查点恢复。实现来源记录于 `provenance.json`。

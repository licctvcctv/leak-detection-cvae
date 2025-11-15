# 跑冒滴漏异常检测系统 (Leak Detection System)

基于条件变分自编码器-UNet (Conditional VAE-UNet) 的生成式异常检测模型，用于检测工业场景中的油液积聚、油液渗漏和积水等异常情况。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目实现了一个基于条件 VAE-UNet 的跑冒滴漏异常检测系统，采用生成式建模方法学习异常区域的分布特征，并通过后处理将概率热力图转换为 YOLO 格式的检测框。

### 检测类别

- **Oil_accumulation** (油液积聚)
- **Oil_seepage** (油液渗漏)  
- **Standing_water** (积水)

### 主要特点

- ✨ **条件 VAE 架构**：结合先验编码器和后验编码器，实现更稳定的生成式建模
- 🎯 **UNet 骨干网络**：利用跳跃连接保留多尺度特征信息
- 🔥 **混合损失函数**：BCE + Dice + KL散度，平衡重建质量和生成约束
- 📊 **热力图转检测框**：通过阈值分割、轮廓检测和 NMS 后处理生成精确检测框
- 🚀 **数据增强**：支持水平翻转、亮度调整、噪声注入等增强策略

## 🏗️ 模型架构

```
输入图像 (288×512×3)
    ↓
┌─────────────────────────────────────────┐
│  Image Encoder (UNet Encoder)           │
│  - ConvBlock (32)                       │
│  - DownBlock (64, 128, 256, 512)        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Latent Branch                          │
│  - Prior Encoder (仅依赖图像)            │
│  - Posterior Encoder (图像+标签)         │
│  - Reparameterization (潜变量采样)       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Decoder (UNet Decoder + Skip)          │
│  - UpBlock (256, 128, 64, 32)           │
│  - Output Conv (3 channels)             │
└─────────────────────────────────────────┘
    ↓
输出热力图 (288×512×3)
```

## 📦 环境配置

### 依赖要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (推荐使用 GPU)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/licctvcctv/leak-detection-cvae.git
cd leak-detection-cvae

# 安装依赖
pip install -r requirements.txt
```

## 📂 数据集结构

```
dataset_work_final/
├── images/
│   ├── train/          # 训练图片 (2836张)
│   └── test/           # 测试图片 (120张)
├── labels/
│   └── train/          # YOLO格式标签文件
├── artifacts/
│   ├── split.json      # 训练/验证集划分
│   └── cvae_detector.pt # 训练好的模型权重
└── results_cvae/       # 检测结果输出目录
```

### 标签格式 (YOLO)

每行格式：`class_id center_x center_y width height`

```
0 0.512 0.345 0.123 0.089
1 0.678 0.567 0.098 0.076
```

## 🚀 使用方法

### 训练模型

打开 `vae_detector.ipynb` 并按顺序执行所有单元格：

```python
# 主要超参数
IMAGE_SIZE = (288, 512)  # (H, W)
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 3e-4
LATENT_DIM = 64
KL_WEIGHT = 1e-4
```

### 推理预测

模型会自动对 `images/test/` 中的 120 张测试图片进行推理，并将结果保存到 `results_cvae/` 目录。

```python
# 推理参数
threshold = 0.45        # 热力图阈值
min_score = 0.3         # 最小置信度
nms_thresh = 0.4        # NMS IoU阈值
min_area_ratio = 1e-4   # 最小区域面积比例
```

## 📊 训练流程

1. **数据准备**：解析 YOLO 标签，构建热力图 mask
2. **数据划分**：90% 训练集 + 10% 验证集
3. **模型训练**：
   - 损失函数：`Loss = BCE + Dice + β·KL`
   - 优化器：AdamW (lr=3e-4, weight_decay=1e-4)
   - 学习率调度：CosineAnnealingLR
4. **模型评估**：监控验证集 IoU，保存最优 checkpoint
5. **推理输出**：生成 120 个 YOLO 格式的结果文件

## 🎯 性能指标

- **验证集 IoU**：监控指标，用于选择最优模型
- **BCE Loss**：二值交叉熵损失
- **Dice Loss**：Dice 系数损失
- **KL Divergence**：KL 散度（生成式约束）

## 📝 文件说明

- `vae_detector.ipynb` - 主要代码文件（包含完整训练和推理流程）
- `artifacts/split.json` - 数据集划分配置
- `artifacts/cvae_detector.pt` - 训练好的模型权重（需自行训练生成）
- `.gitignore` - Git 忽略规则（排除大型数据文件）

## 🔧 技术细节

### 损失函数

```python
BCE = binary_cross_entropy_with_logits(logits, masks)
Dice = 1 - (2 * intersection + ε) / (union + ε)
KL = 0.5 * Σ(exp(σ_post - σ_prior) + (μ_prior - μ_post)² / exp(σ_prior) - 1 + σ_prior - σ_post)
Total Loss = BCE + Dice + β·KL  (β = 1e-4)
```

### 后处理流程

1. 对每个类别的概率图应用高斯模糊
2. 阈值二值化 (threshold=0.45)
3. 轮廓检测提取候选框
4. 过滤小面积区域
5. NMS 去除重叠框
6. 归一化坐标并输出 YOLO 格式

## 📄 License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

[@licctvcctv](https://github.com/licctvcctv)

## 🙏 致谢

本项目为模式识别与机器学习课程大作业，感谢课程组提供的数据集和技术指导。




# 视网膜血管分割优化研究报告：从 LFA-Net 复现到几何感知架构的探索

## 1. 项目概述与背景 (Introduction)

视网膜血管的形态学变化是糖尿病视网膜病变、高血压和青光眼等多种系统性疾病的重要生物标志物。然而，视网膜血管结构复杂，包含大血管、微血管以及视盘（Optic Disc, OD）和病灶等强干扰背景，使得自动化分割极具挑战性。

本项目基于 **LFA-Net (Lightweight Network with LiteFusion Attention)** [1] 展开研究。LFA-Net 是一种极轻量级网络（约 0.11M 参数），结合了 Vision Mamba 的动态特性和多尺度特征提取能力，在资源受限环境下表现优异。本项目旨在复现 LFA-Net，并针对其在视盘区域误判和血管断裂问题，探索引入先验知识（视盘掩膜）和几何感知机制（DCN/Strip Pooling + 混合损失）的改进方案。

## 2. 数据集与预处理 (Datasets & Preprocessing)

### 2.1 数据集
实验主要基于 **DRIVE 数据集** 进行：
*   **训练集**：20 张图像（及其对应的金标准掩膜）。
*   **测试集**：20 张图像。
*   **分辨率**：原始分辨率 $565 \times 584$，实验中统一调整为 $560 \times 560$ 以适应网络下采样。

### 2.2 精细化预处理与增强
为了克服训练数据稀缺的问题，我们实施了严格的预处理和增强策略：
1.  **通道选择**：提取 RGB 图像的 **绿色通道 (Green Channel)**，因其在视网膜图像中血管对比度最高。
2.  **对比度增强**：应用 **CLAHE** (限制对比度自适应直方图均衡化) 增强局部血管细节。
3.  **形态学处理**：使用 **TopHat 变换** 抑制光照不均匀，突出暗细血管。
4.  **数据扩增**：训练时在线应用随机旋转、翻转、弹性形变（Elastic Transform）和亮度对比度调整，以增强模型泛化能力。

## 3. 基准模型复现 (Baseline Reproduction)

我们首先复现了原始 LFA-Net 模型架构，使用 Dice Loss 进行训练。

*   **训练设置**：100 Epochs, Adam Optimizer, LR=1e-3.
*   **基准结果 (Baseline Results)**：
    *   **Jaccard (IoU)**: 0.6342
    *   **F1-Score**: 0.7759
    *   **Sensitivity (Recall)**: 0.7595
    *   **Precision**: 0.7982
    *   **Accuracy**: 0.9619
    *   **Specificity**: 0.9814

**分析**：基准模型表现出良好的平衡性，但在处理复杂背景时仍存在误判，且部分微细血管存在断裂现象。

## 4. 阶段一：视盘先验信息的引入与分析 (Phase 1: Optic Disc Integration)

### 4.1 动机
视盘（OD）是眼底图像中亮度极高的区域，且血管在此处汇聚。我们假设：引入明确的视盘位置信息作为先验知识，可以帮助网络抑制视盘边界的假阳性（False Positives）。

### 4.2 方法
1.  利用预训练 SegFormer 生成视盘掩膜（OD Mask）。
2.  修改网络输入层，将 RGB 图像与 OD Mask 进行 **通道拼接 (Channel Concatenation)**，输入维度变为 $(B, 4, H, W)$。

### 4.3 结果与失败分析
*   **实验结果 (LFA-Net + OD Input)**：
    *   **Jaccard**: 0.6240 (**↓ 1.02%**)
    *   **Sensitivity**: 0.7490 (**↓ 1.05%**)
    *   **Precision**: 0.7945 (**↓ 0.37%**)

*   **原因深度剖析**：
    该尝试并未带来预期提升，反而导致性能下降，主要原因包括：
    1.  **硬约束引入噪声 (Hard Constraint Noise)**：自动生成的 OD Mask 并非绝对准确，错误的先验信息直接误导了网络。
    2.  **语义混淆 (Semantic Confusion)**：视盘区域内部本身包含大量粗血管。将 OD Mask 作为输入，网络可能难以区分“视盘背景（需抑制）”和“视盘内血管（需保留）”，导致视盘区域血管被错误抑制。
    3.  **特征同质化**：简单的输入拼接在深层网络中被稀释，无法在 Bottleneck 阶段提供有效的空间注意力指导。

## 5. 阶段二：几何感知架构与混合损失函数 (Phase 2: Geometric & Topological Optimization)

鉴于显式引入 OD Mask 的失败，我们转向**“隐式几何感知”**，旨在通过改进卷积算子和损失函数，强化网络对血管形态的捕捉能力。

### 5.1 损失函数重构 (Loss Function Engineering)
为了解决微血管断裂和类别不平衡问题，我们设计了联合损失函数 `VesselSegmentationLoss`：

$$ L_{total} = \lambda_1 L_{FocalTversky} + \lambda_2 L_{Boundary} $$

1.  **Focal Tversky Loss** ($\alpha=0.7, \beta=0.3, \gamma=0.75$)：
    *   相比 Dice Loss，Tversky Loss 引入 $\alpha, \beta$ 参数调节 FP 和 FN 的惩罚权重。
    *   **重点**：我们设置了较高的 $\alpha$ (0.7)，这意味着我们**更严厉地惩罚假阳性 (False Positives)**。
2.  **Boundary Loss**：
    *   基于距离变换（Distance Transform）或梯度边界，惩罚预测边界与真实边界的偏差。这能有效提升血管边缘的锐度。

### 5.2 架构创新 (Architectural Innovations)
针对血管“细长、弯曲”的特性，我们在 LFA-Net 基础上引入了两个变体：

*   **变体 A (DCN + Strip Pooling)**：
    *   **Deformable Convolution (DCN v2)**：替代 Encoder 中的普通卷积，使卷积核采样点能自适应血管弯曲形状。
    *   **Strip Pooling (SP)**：在 Bottleneck 引入长条形池化核，捕捉长距离血管依赖。
*   **变体 B (ASPP + CBAM)**：
    *   **ASPP**：利用不同膨胀率的空洞卷积捕捉多尺度特征（粗/细血管）。
    *   **CBAM**：引入通道注意力，筛选对血管敏感的特征图。

### 5.3 实验结果对比 (Results Comparison)

| Method | Jaccard (IoU) | F1-Score | Sensitivity (Recall) | **Precision** | **Specificity** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (LFA-Net)** | **0.6342** | **0.7759** | **0.7595** | 0.7982 | 0.9814 |
| **New Loss Only** | 0.6119 | 0.7581 | 0.6812 | **0.8654** | **0.9897** |
| **Loss + DCN + SP** | 0.6077 | 0.7550 | 0.6799 | 0.8579 | 0.9891 |
| **Loss + ASPP** | 0.6120 | 0.7587 | 0.6790 | **0.8665** | **0.9899** |

### 5.4 深入分析 (Critical Analysis)

虽然改进后的 Jaccard 指数略有下降，但仔细观察各分项指标，我们可以发现**模型行为发生了本质变化**：

1.  **精度与特异性的显著提升**：
    *   改进方案（Loss + ASPP）的 **Precision 从 79.8% 飙升至 86.6%**。
    *   **Specificity 提升至 98.99%**。
    *   **结论**：新的损失函数（特别是 Boundary Loss 和 Tversky 的参数设置）极大地抑制了背景噪声。Baseline 模型可能把很多背景噪点预测为血管（从而虚高了 Recall），而新模型非常“干净”。

2.  **灵敏度下降的代价**：
    *   Sensitivity 从 75.9% 降至 67.9%。
    *   **结论**：模型变得过于“保守”。Boundary Loss 在训练初期可能对微小血管的偏移惩罚过重，导致网络倾向于忽略那些不确定的微血管以降低 Loss。

3.  **架构的有效性**：
    *   在同样的 Loss 下，**ASPP 版本的性能 (0.6120) 优于 DCN+SP 版本 (0.6077)**，且非常接近仅改 Loss 的版本。这说明在极小样本（20张训练图）下，引入过多参数（如 DCN 的偏移生成器）可能导致过拟合，而经典的 ASPP 结构更加稳健。

## 6. 后续工作计划与数据补充 (Future Work)

为了在最终报告中展示更全面的结论，并尝试恢复 Sensitivity，需补充以下工作：

1.  **可视化对比 (Qualitative Analysis)**：
    *   **需求**：生成对比图，展示 Baseline vs. New Method。
    *   **预期现象**：New Method 的图像背景应该非常干净，视盘边界无误判，但可能丢失了末端微血管。这能证明 Precision 提升的来源。

2.  **损失权重调优**：
    *   目前 Tversky 的 $\alpha=0.7$ 可能过高（过度抑制 FP）。下一步将尝试 $\alpha=0.3, \beta=0.7$，以鼓励模型召回更多微血管，平衡 Precision 和 Recall。

3.  **消融实验图表**：
    *   绘制 Training Loss 和 Validation Jaccard 曲线，检查新架构是否收敛较慢，是否需要更多 Epochs。

## 7. 结论 (Conclusion)

本研究表明，单纯依靠先验掩膜（OD Mask）的硬融合在视网膜血管分割中效果有限。通过引入几何感知损失（Boundary Loss）和注意力机制，我们成功构建了一个**高精度、高特异性**的分割模型，有效解决了背景误判问题。虽然在召回率上有所妥协，但这为需要极低误诊率的临床筛查场景提供了有价值的参考架构。未来的工作将聚焦于在保持高精度的同时，通过动态损失加权策略恢复对微血管的捕获能力。
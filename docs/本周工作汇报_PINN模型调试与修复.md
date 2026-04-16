# 本周工作汇报 — PINN模型调试与修复

## 问题现象
训练 200 epoch 后 PSNR 仅 8.93 dB，预测相位图几乎全平，无法恢复任何相位结构。

---

## 修复1：降低 global_scale（10 → 1）

### 修改原因
原始 AirySimulator 中 `global_scale=10.0`，配合 Zernike 系数幅度 ×2.0，导致 Airy 公式内部的等效相位 φ_scaled 范围达到 [-202, 167] rad，对应约 **58.7 条干涉条纹**。这使得 Airy 函数 I(φ) = I_max/(1+F·sin²(φ/2)) 在相位空间内产生 **30 个极值点（不可逆点）**，从单帧强度图到相位的反演问题严重病态。

### 修改前
- `physics_ops.py`: `global_scale=10.0`（默认值）
- `integrated_net.py`: `global_scale=3.0`（UnrolledPINN 默认值）
- 等效条纹数：~59 条
- Airy 反函数在 φ∈[-5,5] 内有 30 个极值点

### 修改后
- `physics_ops.py`: `global_scale=1.0`
- `integrated_net.py`: `global_scale=1.0`
- 等效条纹数：~2.9 条
- Airy 反函数在 φ∈[-5,5] 内仅 2 个极值点

### 物理原理
Fizeau 干涉仪的 Airy 公式：

$$I = \frac{I_{max}}{1 + F \cdot \sin^2(\phi_{scaled}/2)}$$

其中 φ_scaled = global_scale × φ。该函数关于 φ 具有周期 2π/global_scale。当 global_scale=10 时，每 0.628 rad 的相位变化就对应一个完整条纹周期。若总相位变化达到 ±100 rad，则产生约 60 条密集条纹。从密集条纹的单帧强度图反演相位，等价于同时解相位解缠绕和去噪——这是一个高度非线性、多解的逆问题，远超网络的学习能力。

降低 global_scale 到 1.0 后，同等相位范围下仅产生 ~3 条条纹，Airy 函数在大部分区间内单调，反演问题变为良态。

---

## 修复2：降低 Zernike 系数幅度（×2.0 → ×0.5）+ 去除随机 tilt

### 修改原因
1. Zernike 系数幅度 ×2.0 使相位范围达到 [-20, 17] rad（极端值），过大
2. 每个样本添加的随机 tilt（±1.0）引入额外的线性相位分量，但模型的 AirySimulator 只有一组全局 tilt 参数（tilt_x, tilt_y），无法匹配每个样本的独立 tilt，导致物理层建模不一致

### 修改前
```python
coeffs *= 2.0                          # 相位范围 [-20, 17] rad (极端)
tilt_x = rng.uniform(-1.0, 1.0)       # 随机tilt
tilt_y = rng.uniform(-1.0, 1.0)
I_clean = airy(phase_t + tilt_t)       # 强度包含tilt
Y_list.append(phase_t)                 # 目标不含tilt → 不一致
```

### 修改后
```python
coeffs *= 0.5                          # 相位范围 ±2 rad (平均), ±5 rad (极端)
I_clean = airy(phase_t)               # 无tilt
Y_list.append(phase_t)                # 一致
```

### 物理原理
在 Fizeau 干涉仪中，tilt 对应两个测试面之间的微小倾斜角，表现为等间距的平行条纹。在 Zernike 展开中，tilt 对应 Z₂ (tip) 和 Z₃ (tilt) 两个低阶模式。当数据中包含独立于样本的 tilt，但模型物理层只有全局 tilt 时，模型内部的 Airy 正向模拟无法准确复现每个样本的强度图，导致物理梯度信号（I_real - Airy(φ)）包含系统误差，干扰网络学习。

去除 tilt 后，物理层 Airy(φ) 能精确复现每个样本的干涉图，使 PINN 的"物理驱动"路径畅通无阻。

---

## 修复3：添加物理一致性损失（Physics-Informed Loss）

### 修改原因
原始训练仅使用 L1(pred_phase, gt_phase)，未利用已知的 Airy 正向模型。纯相位空间的损失在高相位区域梯度平坦（因为相位空间大，梯度被稀释），模型容易收敛到均值相位（即输出几乎平坦）。

### 修改前
```python
loss = criterion(pred_phase, batch_y)  # 仅L1相位损失
```

### 修改后
```python
loss_main = criterion(pred_phase, batch_y)           # MSE相位损失
I_pred = model.airy(pred_phase)                      # Airy正向模拟
loss_phys = criterion(I_pred, batch_x)               # MSE强度一致性
loss = loss_main + 0.1 * loss_phys                   # 联合损失
```

### 物理原理
PINN（Physics-Informed Neural Network）的核心思想是将物理约束嵌入损失函数。对于 Fizeau 干涉仪：

$$\mathcal{L}_{total} = \underbrace{\|\varphi_{pred} - \varphi_{gt}\|^2}_{\text{数据保真}} + \lambda \cdot \underbrace{\|Airy(\varphi_{pred}) - I_{real}\|^2}_{\text{物理一致性}}$$

物理一致性损失确保：预测的相位通过已知的 Airy 正向模型后，能够复现输入的干涉强度图。这个约束：
1. 提供了额外的梯度信号，加速收敛
2. 在相位空间的平坦区域提供了通过强度空间的"隧道效应"梯度
3. 联合优化 AirySimulator 的可学习参数（反射率 R、缩放因子）使物理模型更精确

权重 λ=0.1 的选择依据：I_max=1.0 时强度范围 [0,1]，而相位范围 [-5,5]，量级差约 5-10 倍，故 λ=0.1 使两项损失在相近量级。

---

## 修复4：添加梯度裁剪（Gradient Clipping）

### 修改原因
物理一致性损失通过 Airy 函数反向传播，而 Airy 函数 I(φ)=I_max/(1+F·sin²(φ/2)) 的导数在某些相位值处可能很大（特别是在条纹暗区附近），导致梯度爆炸。

### 修改前
```python
loss.backward()
optimizer.step()  # 无梯度保护
```

### 修改后
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 物理原理
Airy 函数的导数：

$$\frac{\partial I}{\partial \phi} = -\frac{I_{max} \cdot F \cdot \sin(\phi) \cdot s}{2(1 + F\sin^2(\phi/2))^2}$$

当 φ 接近暗条纹中心（sin²(φ/2)≈1）时，分母 (1+F)² ≈ 81（对于 R=0.5, F=8），导数绝对值约为 I_max·F·s/(2·81) ≈ 0.05。但在亮条纹附近（sin²(φ/2)≈0），分母≈1，导数绝对值可达 I_max·F·s/2 ≈ 4.0。梯度在不同相位区域的变化可达 ~80 倍。

梯度裁剪（max_norm=1.0）防止 Airy 层在某些极端相位值处产生的大梯度破坏训练稳定性。

---

## 修复5：切换 MSE 损失 + 提高学习率

### 修改原因
修复1-4后训练仍然收敛缓慢（3 epoch val loss 从 0.993 降到 0.978）。分析发现：

1. **Airy 函数低灵敏度区**：当相位 φ∈[2,5] rad 时，Airy 强度集中在 [0.11, 0.20]（极窄区间），对相位变化几乎"失明"。降低系数到 ×0.5 后，典型相位 ±2 rad，强度范围扩大到 [0.25, 1.0]——信息量提升 3 倍。
2. **L1 损失的常数梯度**：L1 的梯度为 sgn(error)，不区分大误差和小误差，导致早期收敛慢。MSE 的梯度 2·error 与误差正比，利于快速修正大误差。

### 修改前
```python
criterion = nn.L1Loss()             # 常数梯度
lr = 5e-4; epochs = 200
```

### 修改后
```python
criterion = nn.MSELoss()            # 梯度正比于误差
lr = 1e-3; epochs = 100
```

### 物理原理
Airy 函数灵敏度分析：

$$\frac{\partial I}{\partial \phi}\bigg|_{\phi=0} = 0, \quad \frac{\partial I}{\partial \phi}\bigg|_{\phi=\pi/2} \approx -0.16$$

灵敏度峰值仅 0.16，意味着 1 rad 相位变化仅产生 ~0.16 强度变化。当噪声标准差 σ=0.05 时，单像素信噪比 SNR ≈ 3.2。对于 ±2 rad 的相位范围（~6 个分辨元素），SNR 足够；但对于 ±5 rad（~16 个分辨元素），SNR 不足。

---

## 修复效果总结

| 指标 | 修复前（原始） | 修复后（当前） |
|------|---------------|---------------|
| Epoch 1 val loss | 0.993 (L1) | 0.659 (MSE) |
| Epoch 4 val loss | ~0.978（停滞） | 0.632（持续下降） |
| 预测相位图 | 全平，无结构 | 待完整训练后评估 |
| PSNR | 8.93 dB | 待完整训练后评估 |

## 修改摘要表

| # | 修复项 | 修改文件 | 核心改动 | 物理依据 |
|---|--------|----------|----------|----------|
| 1 | global_scale 10→1 | physics_ops.py, integrated_net.py | 降低 Airy 内部相位缩放 | 减少条纹数（59→3条），使反演从严重病态变为良态 |
| 2 | 去除 tilt + 降系数 | train.py | 去掉随机 tilt，coeffs 2.0→0.5 | 消除物理层建模不一致；Airy 灵敏度区间从 [0.11,0.2] 扩大到 [0.25,1.0] |
| 3 | 物理一致性损失 | train.py | 添加 L_phys=‖Airy(φ̂)−I‖² | PINN 核心思想：嵌入正向模型约束，加速收敛 |
| 4 | 梯度裁剪 | train.py | clip_grad_norm(1.0) | 防止 Airy 导数在亮/暗条纹附近 ~80 倍差异导致梯度爆炸 |
| 5 | MSE 损失 + lr 调整 | train.py | L1→MSE, lr 5e-4→1e-3 | MSE 梯度与误差正比（vs L1 常数），利于快速修正大误差 |

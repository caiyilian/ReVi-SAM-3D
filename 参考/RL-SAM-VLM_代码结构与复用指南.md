# ReVi-SAM-3D 代码结构与功能映射

本文档用于详细记录 ReVi-SAM-3D 训练框架中各个阶段、每个具体步骤的核心功能，以及涉及的对应代码具体位置信息，以便后续拓展到视频分割任务时快速定位和检索。

---

## 阶段 0：工程初始化与基础骨架 (Foundation)

### 步骤 1：创建工程目录结构
- **功能**：初始化整个算法工程的物理骨架，确立“高内聚、低耦合”的模块化划分，为后续的数据加载、大模型注入、强化学习环境等子系统预留清晰的独立目录。
- **涉及的具体位置信息**：
  - **项目根目录**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\`
  - **数据预处理与加载器存放处**：`ReVi-SAM-3D\data\`
  - **核心模型架构存放处**：`ReVi-SAM-3D\models\`
  - **强化学习智能体（策略网络、DQN、经验回放池）**：`ReVi-SAM-3D\models\rl_agent\`
  - **通用网络组件（如冻结的LLM特征提取层）**：`ReVi-SAM-3D\models\common\`
  - **强化学习交互闭环环境（Gym Env）**：`ReVi-SAM-3D\env\`
  - **通用工具（评价指标、日志等）**：`ReVi-SAM-3D\utils\`

### 步骤 2：引入 SAM 2 基础代码与权重
- **功能**：将 Meta 官方的 `segment-anything-2` 核心源码作为基础模块引入，提供未经修改的原始 Image Encoder、Memory Attention 等结构。将其作为后续魔改（如注入 DCNv4 和 LLM 层）的基础“手术台”，无需从头实现视觉特征提取框架。
- **涉及的具体代码位置**：
  - **SAM 2 魔改基座代码库**：`ReVi-SAM-3D\models\sam2_modified\sam2-main\`

---

## 阶段 1：数据流与预处理 (Data & VLM Mock)

### 步骤 3：编写基础 3D 图像读取模块
- **功能**：实现对医学图像 `.nii.gz` 格式文件的读取，并完成必要的数据预处理，如 Z-score 归一化和图像缩放 (Resize)，为网络提供标准化的输入张量。
- **涉及的具体代码位置**：
  - **数据集处理代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\data\dataset.py` 中的 `load_medical_image()` 函数等。

### 步骤 4：实现 3D 体素到 2D 序列的转换逻辑
- **功能**：将 3D 的体素数据沿 Z 轴切片，转化为连续的 2D 图像序列，并将其封装为标准的 PyTorch `Dataset` 格式，以适配 SAM 2 的 2D 输入要求以及时空追踪器的序列化处理要求。
- **涉及的具体代码位置**：
  - **序列转换与 Dataset 封装代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\data\dataset.py` 内部的 Dataset 类。

### 步骤 5：实现 VLM 文本先验的 Mock 机制
- **功能**：在正式的 LLaVA-Med 离线推理接入前，通过模拟 (Mock) 机制为每张切片自动生成固定的解剖学文本描述，以保证数据流和网络联调不被阻塞。
- **涉及的具体代码位置**：
  - **Mock 文本生成逻辑**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\data\dataset.py` 中的 `get_mock_vlm_text()` 函数。

### 步骤 6：集成 CLIP/BERT 文本特征提取
- **功能**：引入轻量级的文本编码器（如 CLIP 或 BERT），将步骤 5 产生的 Mock 文本转化为 1D 的语义特征向量 (Semantic Embedding)，以便后续注入到视觉大模型中。
- **涉及的具体代码位置**：
  - **文本编码器代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\sam2_modified\text_encoder.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step6_text_encoder.py`

---

## 阶段 2：魔改 SAM 2 (创新点 1 & 2)

### 步骤 7：实现跨维度注意力投影模块 (创新点 1 核心)
- **功能**：构建 `DeformableProjectionModule`，包含 Cross-Attention (文本向视觉投影) 并融合 DCNv4 算子，实现 1D 文本语义向 2D 视觉特征的变形注入。
- **涉及的具体代码位置**：
  - **核心模块代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\sam2_modified\projection.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step7_projection.py`

### 步骤 8：将投影模块注入 SAM 2 Image Encoder
- **功能**：修改 SAM 2 原生的 `image_encoder`，在指定的 ViT Block 之后插入步骤 7 编写的投影模块，完成文本先验对图像特征的融合。
- **涉及的具体代码位置**：
  - **魔改文件代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\sam2_modified\sam2-main\sam2\modeling\sam2_base.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step8_sam2_injection.py`

### 步骤 9：提取并冻结 LLaMA 单层 (创新点 2 & 3 基础)
- **功能**：从预训练的 LLaMA 模型中剥离出单层，冻结其参数，并构建适配的投影层 (Adapter) 和 2D RoPE 位置编码，形成“三明治”结构，用于提供强大的序列匹配能力。
- **涉及的具体代码位置**：
  - **LLM提取器代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\common\llm_extractor.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step9_llm_extractor.py`

### 步骤 10：重写 SAM 2 Memory Attention 模块 (创新点 2 核心)
- **功能**：动态拦截并重写原版 `MemoryAttention`。引入全局分支（通过提取的 LLM 层寻找时空对应关系）和局部分支（通过 DCNv4 进行微观几何对齐），从而替换掉原生纯视觉的 Transformer。
- **涉及的具体代码位置**：
  - **魔改文件代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\sam2_modified\sam2-main\sam2\modeling\memory_attention.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step10_memory_attention.py`

### 步骤 11：替换 SAM 2 的原生追踪器并实现双向 Wrapper
- **功能**：在原生 `SAM2VideoPredictor` 中注入双向传播机制 (`bidirectional_propagation`)，使模型能够分别沿 Z 轴正反两个方向传播，最终拼装完整的 3D 预测结果。
- **涉及的具体代码位置**：
  - **魔改文件代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\sam2_modified\sam2-main\sam2\sam2_video_predictor.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step11_bidirectional_wrapper.py`

---

## 阶段 3：构建强化学习闭环 (创新点 3)

### 步骤 12：构建大模型赋能的轻量级 RL 策略网络
- **功能**：设计 `LLMDrivenPromptAgentDDQN` 网络。该网络将 SAM 2 提取的 2D 视觉特征图与当前边界框 (Bbox) 转化为 Token 并拼接，送入复用的冻结 LLM 层（提取全局上下文），最后通过轻量级 MLP 输出 9 个离散动作（调整 Bbox）的 Q 值。
- **涉及的具体代码位置**：
  - **策略网络代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\rl_agent\llm_policy.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step12_llm_policy.py`

### 步骤 13：移植并适配 DQN 核心算法
- **功能**：实现深度 Q 网络 (DQN) 的核心控制逻辑，包括经验回放池 (Replay Buffer)、$\epsilon$-Greedy 动作探索机制以及网络参数的软更新 (Soft Update)。将其状态空间适配为 `(视觉特征, 当前Bbox)` 的双元组。
- **涉及的具体代码位置**：
  - **DQN 智能体代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\models\rl_agent\dqn_agent.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step13_dqn_agent.py`

### 步骤 14：编写 Dice 评估与辅助指标函数
- **功能**：实现 Dice Similarity Coefficient (DSC) 的计算逻辑。该函数用于评估 SAM 2 预测的 Mask 与 Ground Truth 之间的重合度，是后续强化学习环境中计算 Reward (奖励) 的绝对基石。
- **涉及的具体代码位置**：
  - **指标计算代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\utils\metrics.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step14_metrics.py`

### 步骤 15：搭建 RL 交互闭环环境 (ClosedLoopEnvironment)
- **功能**：继承 `gym.Env`，封装标准的强化学习交互环境。在内部 `step` 函数中实现：接收智能体动作 -> 调整 Bbox -> 调用魔改版 SAM 2 预测当前帧 Mask -> 计算 Dice 变化量 -> 返回新的状态和 Reward。
- **涉及的具体代码位置**：
  - **交互环境代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\env\sam_closed_loop.py`
  - **功能验证测试程序**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\test_step15_env.py`

---

## 阶段 4：系统集成与训练流水线 (Integration)

### 步骤 16：组装主训练循环 (`train_rl.py` 上半部)
- **功能**：作为整个框架的入口点，负责模块导入、命令行参数解析 (`argparse`)，以及 `Dataset`、`DataLoader`、RL Agent、Gym Environment 等核心组件的初始化和实例化。
- **涉及的具体代码位置**：
  - **训练脚本代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\train_rl.py` 中的 `setup_training()` 函数。

### 步骤 17：编写闭环训练逻辑 (`train_rl.py` 下半部)
- **功能**：实现端到端的强化学习训练大循环。包含：获取初始状态 -> Agent 预测动作 -> 环境执行动作返回 Reward 和新状态 -> 存入经验回放池（并使用 `.detach().cpu()` 防止显存泄漏） -> 触发 DQN 的 `learn()` 梯度更新。
- **涉及的具体代码位置**：
  - **训练脚本代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\train_rl.py` 中的 `train_loop()` 函数。

### 步骤 18：全链路空跑测试 (Dry Run)
- **功能**：通过注入极小的 Batch Size 和 Tensor 尺寸参数，进行端到端的代码跑通测试。验证数据流、Mock 文本特征、DCNv4 形变、SAM 2 追踪、RL 交互闭环以及 Loss 计算是否完全畅通无报错。
- **涉及的具体代码位置**：
  - **测试入口代码**：`e:\projects\大模型分割\方案\ReVi-SAM-3D\train_rl.py` 文件底部的 `if __name__ == "__main__":` 代码块。

---
**至此，3D医疗影像分割版本的 ReVi-SAM-3D 核心代码结构映射已全部记录完毕。未来开发“视频分割版本”时，可直接参考本指南快速定位需要复用或修改的代码模块。**

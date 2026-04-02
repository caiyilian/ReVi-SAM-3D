# 关于当前代码现状与创新点2稳妥方案的代码核对结论

## 1. 问题背景与本次代码核查范围

这次判断不是基于设想图，而是直接对当前训练主线做了代码核查，入口是 `SAM-Med3D-main/train.sh`，并进一步核对了 `train.py`、`train_core/factories.py`、`train_core/base_trainer.py`、`utils/data_loader.py`、`utils/click_method.py`。

先给结论：**founder 对当前整体进度的判断大方向是成立的，但需要加两个限定词：创新点3不是“论文层面已经完全收尾”，而是“代码主链已经成形”；创新点1不是“只差一根线就自然闭环”，而是“RL 核心骨架已经搭好，但还缺创新点2提供的语义先验，因此现在更像可运行的 prompt policy scaffold，而不是最终版本”。**

同时，本次代码核查也明确了一个关键事实：**当前真实训练预处理并不是简单口述版“只对 D 维裁剪/补零”，而是 `tio.ToCanonical()` 后接 `tio.CropOrPad(mask_name='label', target_shape=(128,128,128))`。** 这意味着创新点2如果继续按“原始全切片离线文本特征 + 训练时再做全量融合”的思路推进，会在坐标对齐、无效切片处理、裁剪映射和 teacher/student 使用边界上都埋下问题。

下面分开说。

---

## 2. 对当前完成情况的判断

## 2.1 创新点3：代码主线基本已经成形

如果 founder 现在说“创新点3基本搞定了”，**从代码主线角度看，这个判断大致成立，而且不是空口成立，是能在训练链路里看到的。**

我核到的关键点有：

1. `train.sh` 已经显式传入 `--semi_supervised_labeled_ratio 0.5`，说明当前默认训练入口已经在走半监督设置，而不是仍停留在纯监督草稿阶段。
2. `train_core/factories.py` 里，训练集会按 labeled / unlabeled 进行拆分，并通过 `is_labeled` 明确标记样本身份。
3. `base_trainer.py` 的 `train_epoch()` 中，labeled 样本和 unlabeled 样本已经走不同分支：
   - **labeled**：SAM 分支正常用 GT 参与交互监督；student 用 GT 主监督，可选叠加少量 pseudo supervision。
   - **unlabeled**：SAM 不吃监督损失；先由 student logits 产生类别预测，再走 `generate_pseudo_labels_without_gt()` 生成伪标签，最后 student 只用 pseudo supervision 学习。
4. 伪监督并不是裸用 mask，而是已经有**confidence threshold / confidence weight / pseudo loss weighting** 这一套质量控制链路。
5. teacher-student 的角色也已经比较清楚：SAM 更像 teacher 侧伪标签生成器，student 是最终吸收半监督信号的主体。

所以更准确的话应该是：

> **创新点3的“半监督 teacher-student + 无标注伪监督 + 置信度筛选”代码链路已经基本具备，属于主框架已落地；但是否“完全搞定”仍取决于后续实验能否证明无标注伪监督真的稳定增益。**

也就是说，**代码上基本成形，论文上还不能提前宣告彻底收官。**

## 2.2 创新点1：接近完成，但确实还没有闭环

founder 说“创新点1几乎搞定了，就差和创新点2对接”，这个判断也**大体成立**，但需要说清“差的不是一个接口细节，而是差它最重要的上游先验输入”。

当前代码里，RL 相关核心部件已经存在：

1. 有独立的 replay buffer；
2. 有 Q network / target network；
3. 有 state vector 构造；
4. 有动作空间（沿 D/H/W 移动 + STOP）；
5. 有 reward 设计；
6. 有无标注样本上的 proxy reward 版本；
7. 有 unlabeled 分支里 RL prompt 入口，而不是只在 labeled 样本上写了个壳。

但它还没有闭环，主要是因为下面两点：

### （1）labeled 主监督分支仍主要依赖 GT prompt

`interaction()` 里的主监督 SAM 分支，核心还是 `_run_supervised_prompt_episode_for_class()`，里面调用的是 `get_points(..., use_gt_prompt=True)`。这说明**当前真正承担 SAM 监督损失的主链，依旧主要是 GT 派生 prompt。**

RL 在 labeled 数据上目前更像一个并行学习分支：它在 `_run_rl_prompt_episode_for_class()` 中学策略、记 transition、优化 agent，但**还没有接管 labeled 主监督 prompt 生成本身**。

### （2）RL state 里还没有创新点2的语义先验

当前 `_build_state_vector()` 主要基于：

- 当前 mask 统计量；
- GT/pseudo 的体素比例；
- 当前点位置；
- 点击步数；
- 上一步 dice。

这说明**RL 目前看见的还是“局部分割状态 + 当前位置 + 过程进度”**，还没有接入“体级语义先验”。也就是说，现在它会学“如何动”，但还不会利用“这个类在这个 volume 里大概率在哪些 slice 更值得先搜、哪些类可能根本不存在”这类更高层信息。

所以更准确的判断应是：

> **创新点1的 RL 骨架、训练循环和无标注应用入口都已经有了，确实已接近完成；但它还缺创新点2提供的 volume-level semantic prior，因此目前仍属于“可训练的策略 scaffold”，不是最终版策略网络。**

---

## 3. 原始创新点2思路的问题在哪里

founder 原始思路是“两条分支”：

- 分支A：对每个 3D volume 的所有切片离线生成文本，再编码成每 slice 文本特征；
- 分支B：训练时学一个 slice attention，再和全部 slice 文本特征做融合。

这个思路并不是完全不能做，但**在当前代码基础和真实预处理约束下，不适合作为主推荐方案**。问题主要有五个。

## 3.1 它默认“原始切片坐标”和“训练输入坐标”可以自然对齐，但当前并不是

现在真实训练前会做 `ToCanonical + CropOrPad(mask_name='label', target_shape=(128,128,128))`。这意味着训练真正看到的不是原始 volume 的完整切片序列，而是**经过 canonical 对齐后、再围绕 label 对齐裁/补后的 128 深度坐标系**。

如果离线文本特征仍按“原始全部切片”来存，那么训练时 attention 分支看到的是裁后 volume，而文本分支对应的却可能是裁前 slice index。**这会导致最基础的 slice-to-slice 对齐不稳。**

## 3.2 它把“补零切片”理解成简单空白切片，但当前问题本质是“有效坐标域变化”

D < 128 时，确实会出现补齐；但这不是简单一句“空白切片 attention=0”就结束。真正的问题是：

- 哪些 slice 是真实体数据，哪些是补出来的；
- 裁后 128 个位置里，每个位置对应原始 canonical volume 的哪个 index；
- 哪些 slice 压根没有对应原始图像内容。

如果不显式维护这些信息，只靠网络自己学“空白切片没用”，训练会把“无内容”与“正常低信息 slice”混在一起。

## 3.3 它一上来就想做“全切片大融合”，但当前创新点2真正需要服务的是 RL，不是独立做一个重模型

当前代码现状里，创新点2最合理的位置不是替代主干 encoder，也不是额外造一个复杂大分支去输出最终分类裁决，而是**给创新点1的策略网络提供稳定、轻量、可解释的先验偏置**。

因此，如果一开始就上：

- 全切片文本特征堆叠；
- 学习式 attention；
- 再做复杂 cross-attention / dual-branch fusion；

那会让创新点2本身过重，而且很容易先被“坐标对齐和无效 slice”问题卡死。

## 3.4 它容易把 prior 做成“最终 presence 判决器”，这在当前阶段风险很大

医疗文本先验在这个方案里更适合做的是：

- 类别存在倾向；
- slice 搜索优先级；
- 起始点偏置；
- absent 类别上的探索抑制。

不适合一开始就直接做“该类存在/不存在”的硬裁决，再强行控制 RL 不许探索。否则一旦先验出错，会直接把后续交互堵死。

## 3.5 离线文本特征如果不带映射元数据，后面基本无法补救

如果离线时只存 “slice index -> embedding”，而不存：

- canonical 后索引；
- crop 起止位置；
- pad 区间；
- 是否有效 slice；

那么训练时几乎无法稳定知道“当前第 k 个训练 slice”到底对应原始哪一层。**这不是后处理能优雅修补的问题。**

---

## 4. 当前预处理约束下的关键设计原则

结合代码现状，我认为创新点2必须遵守下面六条原则。

## 4.1 先对齐真实训练坐标系，再谈先验

先验必须定义在**训练真正看到的 128×128×128 坐标系**上，而不是原始任意深度坐标系上。

也就是说，离线文本特征处理不能只记录原始 D 维切片，而要能映射到当前 `CropOrPad(mask_name='label', target_shape=(128,128,128))` 之后的 slice 坐标。

## 4.2 必须显式维护 valid-slice mask

对于裁后 128 个 slice，要明确知道：

- 该 slice 是否对应真实 canonical 图像内容；
- 该 slice 是否是 pad 产生的无效位置。

这个 valid-slice mask 不是可有可无的小辅助，而是创新点2能否稳定工作的基础元数据。

## 4.3 裁后只保留对应 slice 特征，不再拿“原始全切片全集”直接进训练

训练时应该使用的是**裁后 128 坐标系下、且只保留有效映射后的 slice feature 序列**。对于无对应原始内容的位置，不是生成假的文本特征，而是保留空位并由 valid mask 屏蔽。

## 4.4 prior 只服务 RL，不做最终 presence 硬裁决

创新点2应作为 policy prior，而不是 presence oracle。它可以给：

- 起始 slice 分布；
- 搜索优先级；
- absent 风险提示；
- 策略状态补充。

但不建议直接输出“该类一定不存在，所以 RL 不需要看”。

## 4.5 优先“代表切片选择 + masked pooling”，不要一开始就上重型双分支注意力融合

当前最稳的做法不是把 128 个 slice 全拿来做复杂融合，而是：

1. 先在有效 slice 里做一个轻量打分；
2. 选少量代表切片；
3. 对代表切片文本特征做 masked pooling / weighted pooling；
4. 输出体级先验向量和若干 slice-level 提示。

这样更贴近现在创新点1需要的输入，也更容易调试。

## 4.6 离线文本特征必须带 slice index / crop 映射元数据

离线阶段至少要保存：

- 原始 canonical slice index；
- 该 slice 映射到裁后 128 坐标系的 index（若存在）；
- 是否被裁掉；
- 是否是 pad 对应的空位；
- volume 级别的 crop 起止信息。

没有这些元数据，创新点2后面几乎一定会反复返工。

---

## 5. 推荐的创新点2最终方案（单一主推荐）

我的单一主推荐是：

> **把创新点2收束为“基于真实裁后坐标系的代表切片语义先验模块”：离线生成并保存 canonical slice 文本特征与 crop 映射元数据；在线阶段仅对裁后 128 坐标系中的有效 slice 做轻量 slice scoring，选取少量代表切片，对其文本特征做 masked weighted pooling，输出面向 RL 的体级语义先验与 slice-level 搜索提示。**

这不是多方案之一，而是我认为当前代码基础下最稳、最容易落地、也最符合你现有创新点1/3结构的主路线。

### 方案拆成三层：

### 第一层：离线语义准备层

对每个 volume：

1. 先按真实训练前的一致方向做 canonical 对齐；
2. 以 canonical volume 的 slice 为单位生成文本；
3. 用冻结文本编码器得到每 slice 文本 embedding；
4. 同时保存 slice 到训练裁后 128 坐标系的映射信息。

这一步的输出不是单纯一个 embedding 文件，而是一份**“slice 特征 + 映射元数据包”**。

### 第二层：在线 prior 聚合层

训练/推理时，针对当前 128-depth 输入：

1. 根据 crop 映射，从离线特征包中取出**裁后仍有效**的 slice 文本特征；
2. 构建 `valid_slice_mask`；
3. 用轻量 slice scorer（可以很简单）对有效 slice 打分；
4. 选 top-k 代表切片；
5. 对其文本特征做 masked weighted pooling；
6. 得到一个体级 prior 向量和一个 slice-level priority 分布。

### 第三层：RL 使用层

把创新点2输出接入创新点1，但**只作为 policy guidance**：

- 改善初始点/初始 slice 选择；
- 给 state vector 增加体级先验项；
- 对动作偏置或 exploration 做软约束；
- 对 absent 倾向高的类别减少盲搜。

而不是让创新点2直接替代 RL 决策。

---

## 6. D>128 / D<128 / valid-slice mask / crop 映射的具体处理建议

这是 founder 这次问题里最关键的细节，我单独展开。

## 6.1 先统一口径：处理对象不是“原始 D”，而是“canonical 后再进入 CropOrPad 的深度坐标”

因为当前训练代码是 `ToCanonical + CropOrPad(mask_name='label', target_shape=(128,128,128))`，所以讨论 D>128 / D<128 时，严格说应当指的是：

> **canonical 后的 volume 深度，相对于最终 128-depth 裁后坐标系的映射关系。**

不要在文档里继续把它写成一个脱离真实 transform 的简化口述规则，否则后面实现时会对不上。

## 6.2 当“有效深度 > 128”时：不是简单删文本，而是保留 crop 内映射、显式标记 crop 外失效

处理建议：

1. 先依据真实 `CropOrPad(mask_name='label', ...)` 过程确定深度方向 crop 区间；
2. 只保留落在 crop 区间内的 canonical slice 文本特征；
3. 给每个保留 slice 记录其映射后的 `cropped_depth_index`；
4. crop 外 slice 特征不参与训练期 prior 聚合。

这样做的含义不是“把没用的切片粗暴丢掉”，而是：

- **训练输入只看 crop 后体素域；**
- **prior 也只在这个有效体素域内定义。**

这才与当前模型真实输入一致。

## 6.3 当“有效深度 < 128”时：不要给 pad slice 伪造文本特征，而要保留空位 + valid mask

处理建议：

1. 对有真实图像内容的 slice，照常保留其文本特征；
2. 对 pad 出来的位置，不生成“空白文本 embedding”去冒充真实切片；
3. 在 128-depth 序列上显式建立 `valid_slice_mask`：真实 slice 为1，pad slice 为0；
4. 所有 slice scoring、top-k 选择、pooling 都必须做 mask；
5. pad slice 在 attention / 打分中应被屏蔽，而不是仅仅期望网络学到它没用。

这比“空白切片 attention=0”更系统，因为这里不仅控制了权重，还明确保证：

- pad slice 不会进入 top-k；
- pad slice 不会污染 pooling；
- pad slice 不会和低信息真实 slice 混淆。

## 6.4 `valid_slice_mask` 应该怎么定义

推荐定义在**裁后 128-depth 坐标系**上，长度固定为128：

- `1`：该 depth 位置映射到 canonical volume 中某个真实 slice；
- `0`：该位置是 pad 产生的无效 depth 位置。

如果有进一步需要，还可以再加一个辅助信息：

- `mapped_original_depth_index`：记录该位置对应 canonical volume 的哪个 slice；
- 对于 pad 位置，设为 `-1`。

## 6.5 crop 映射至少要保存哪些信息

建议每个 volume 保存：

1. `canonical_depth`；
2. `crop_start_depth` / `crop_end_depth`；
3. `cropped_to_original_depth_index[128]`；
4. `valid_slice_mask[128]`；
5. 每个有效 slice 对应的文本 embedding。

如果后面还要做 H/W 方向更细的空间提示，可以再扩展，但当前阶段先不用把问题做重。

---

## 7. 创新点2应向创新点1输出哪些信息

这里不能再笼统写“给一个特征”。当前最合理的输出应该分成四类。

## 7.1 体级 prior 向量（volume-level prior embedding）

这是 masked weighted pooling 后得到的全局语义向量，作用是补充 RL state，让策略知道：

- 这个 volume 对当前类别的语义支持强不强；
- 当前任务更像哪一类解剖/病灶上下文。

## 7.2 slice-level priority 分布

不是最终裁决，而是一个长度为128的优先级分布（只在 valid slice 上有效），告诉 RL：

- 应优先从哪些深度区域开始搜；
- 哪些 slice 更值得继续探索。

这个量非常适合用来影响初始点或初始 slice 选择。

## 7.3 presence tendency score（软存在倾向分数）

注意是**soft tendency**，不是 hard decision。它的作用是：

- 如果某类语义支持很弱，RL 可以减少无效探索；
- 但仍保留少量探索余地，避免先验误杀真实目标。

## 7.4 top-k representative slice indices

除了连续分布，还建议显式输出少量代表 slice 的 index，用于：

- 初始化点时优先落在这些 slice；
- 在 RL 前几步提供明确的深度候选区间。

综合起来，我建议创新点2对创新点1提供的是：

> **`{volume_prior_embedding, slice_priority_distribution, presence_tendency_score, representative_slice_indices, valid_slice_mask}`**

这比“一个融合特征”更清楚，也更容易落到当前 RL 框架里。

---

## 8. 为什么这个版本最适合当前代码基础

这个版本最适合当前代码，不是因为它最华丽，而是因为它和现有基础最对路。

## 8.1 它顺着当前创新点1/3的接口关系走

当前代码已经有：

- 半监督 teacher-student 主线；
- RL prompt policy scaffold；
- unlabeled 数据上的 RL proxy reward 应用入口。

所以创新点2最合适的角色不是另起炉灶，而是**成为创新点1的上游先验模块**。

## 8.2 它优先解决真实痛点：坐标对齐和无效 slice

现在最大的真实风险不是“attention 够不够高级”，而是**slice 文本特征到底和训练输入是不是同一坐标系**。我给的方案优先把这个问题钉死，而不是先堆复杂网络。

## 8.3 它避免把创新点2做成过重副主线

如果创新点2一开始就做成全切片双分支大融合，它很容易喧宾夺主，甚至把论文重心从“先验约束 RL 提示”变成“多模态大融合”。这和你当前代码主线不一致。

## 8.4 它对后续迭代友好

先做“代表切片选择 + masked pooling”，以后如果实验支持，再升级为更复杂的 cross-attention 都可以；但反过来，从重型架构退回轻量方案通常更痛苦。

所以这个版本的优点是：**先把链路走通，再保留以后加复杂度的空间。**

---

## 9. 现在不建议怎么做

当前阶段，我明确不建议下面这些方向。

1. **不建议继续按“原始全切片文本特征直接和训练时 attention 全量融合”推进。** 先验与真实裁后坐标不对齐，后面问题会很多。
2. **不建议把 pad 出来的 slice 当成普通空白切片，并给它们分配可学习文本特征。** 这会污染 slice 级语义序列。
3. **不建议让创新点2直接输出 presence 的硬裁决，然后强行禁止 RL 探索。** 先验应做软约束，不应做最终裁定。
4. **不建议一开始就上重型双分支注意力 / 大规模 cross-attention 融合。** 当前最先该解决的是坐标系、映射、valid mask 和 RL 接口。
5. **不建议忽略 crop 映射元数据。** 如果离线特征包不带映射关系，后续几乎必返工。

---

## 10. 最终结论

最后收束成一句话：

> **从代码角度看，founder 说“创新点3基本完成、创新点1接近完成，只差与创新点2对接”这个判断大方向成立；但更精确地说，创新点3是半监督代码主链已经成形，创新点1是 RL scaffold 已经具备却尚未接入体级语义先验。**

对应地，创新点2不建议继续走“原始全切片离线文本 + 训练时全量注意力大融合”的原始版本，而应改为：

> **先对齐真实 `ToCanonical + CropOrPad(mask_name='label', target_shape=(128,128,128))` 坐标系，离线保存 slice 文本特征及 crop 映射元数据；在线只保留裁后有效 slice，显式维护 `valid_slice_mask`，用代表切片选择 + masked weighted pooling 构造体级语义先验，再把 `volume prior + slice priority + soft presence tendency + representative slice indices` 提供给创新点1的策略网络。**

这样处理后：

- D>128 时，prior 只定义在 crop 内有效 slice 上；
- D<128 时，pad 位置不伪造语义，只通过 `valid_slice_mask` 屏蔽；
- 创新点2输出的是**对 RL 有用的先验信息包**，而不是一个笼统特征，更不是最终 presence 判决器。

如果当前只选一个最稳、最贴代码现状、最容易继续推进的方案，我建议就按这一版收束。

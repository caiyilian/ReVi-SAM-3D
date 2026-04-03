代码：强化学习分割论文代码\Seg-R1-main
# Summary of *Seg-R1: Segmentation Can Be Surprisingly Simple with Reinforcement Learning*
## 1. Problems to Be Solved
- The mainstream paradigm for endowing large multimodal models (LMMs) with pixel-level segmentation capabilities relies on introducing specialized segmentation tokens and custom decoder architectures, which disrupts the continuity of the LMMs' native causal autoregressive structure.
- Supervised Fine-Tuning (SFT) for segmentation tasks demands large-scale pixel-level image-text paired annotations and extensive training computation, severely limiting the scalability of model development.
- SFT on pixel-level tasks typically causes catastrophic forgetting, degrading the original general-purpose multimodal understanding capabilities of the base LMMs.
- Existing prompt-guided segmentation methods deliver unsatisfactory performance on challenging tasks such as Camouflaged Object Detection (COD), and exhibit weak open-world generalization on zero-shot referring segmentation and reasoning segmentation.
- Traditional reinforcement learning (RL) methods for LMMs incur high computational overhead (e.g., requiring an additional critic model), and the potential of advanced RL algorithms (especially Group Relative Policy Optimization, GRPO) in pixel-level segmentation tasks remains largely unexplored.

## 2. Proposed Method: Pipeline and Key Innovations (Focus on Reinforcement Learning)
### Overall Framework and Pipeline
Seg-R1 is a simple yet effective RL-based segmentation framework built on frozen Qwen-2.5-VL (as the policy model) and SAM2 (as the mask generator). Its core design reformulates dense pixel-level segmentation into a sparse, autoregressive next-token mask prompt prediction task, which naturally aligns with the causal architecture of LLMs. The full pipeline is as follows:
1. Given an input image (and optional natural language query for open-world tasks), the Qwen-2.5-VL policy model autoregressively generates structured output sequences, including explicit reasoning processes, bounding boxes, point prompts, and corresponding labels.
2. The generated prompts are fed into the frozen SAM2 to produce the final segmentation mask.
3. During training, the model is optimized via the GRPO algorithm with a custom reward function that evaluates the quality of generated prompts and segmentation masks, updating the policy to maximize cumulative reward.

Two training paradigms are explored:
- **SFT + RL**: The model first undergoes 1-epoch cold-start SFT on the proposed Foreground Chain-of-Thought (FCoT) dataset to learn basic output formatting and prompt-guided segmentation logic, followed by RL fine-tuning on downstream segmentation datasets.
- **Pure RL from scratch (core paradigm)**: A two-stage RL training strategy is adopted. First, pre-RL training is performed on the high-resolution DIS5K dataset to learn fundamental segmentation structure knowledge. The model is then further RL fine-tuned on COD10K/CAMO (for COD) or DUTS (for Salient Object Detection, SOD) to enhance segmentation precision and reasoning ability.

### Key Innovations (Focus on Reinforcement Learning)
1. **Pioneering introduction of GRPO into pixel-level segmentation**: Unlike traditional RL algorithms that require an additional critic model for baseline estimation, GRPO estimates the baseline from group-level scores, drastically reducing memory consumption and computational overhead during training. This enables efficient RL optimization for segmentation tasks on large-scale LMMs without prohibitive resource costs.
2. **Pure RL-driven segmentation with no LMM architectural modifications**: Unlike existing methods that introduce custom segmentation tokens and decoder heads, Seg-R1 requires no changes to the native structure of Qwen-2.5-VL or additional special tokens. The model autonomously learns to construct annotation trajectories and generate high-quality SAM2 prompts purely through RL optimization, fully preserving the integrity of the LMM's causal autoregressive architecture.
3. **Task-specific dual-component reward mechanism for RL-based segmentation**: The total reward consists of two tailored parts:
    - **Format reward**: A binary reward (1.0 for fully compliant output, 0 otherwise) that ensures the model's output strictly follows the predefined tag format for reasoning processes and prompts, guaranteeing valid input to SAM2.
    - **Segmentation reward**: A weighted combination of 70% IoU and 30% S-measure, which balances global mask accuracy and fine-grained structural fidelity. This design mitigates the reward hacking issue caused by using S-measure alone, while avoiding the loss of structural details from using IoU alone.
4. **Superior generalization and capability preservation via pure RL**: Trained solely on 7,040 foreground segmentation image-mask pairs without any textual supervision, Seg-R1 achieves state-of-the-art zero-shot performance on out-of-domain tasks: 71.4 cIoU on the RefCOCOg test set for referring segmentation, and 56.7 gIoU on the ReasonSeg test set for reasoning segmentation, outperforming models fully supervised on these datasets. Meanwhile, pure RL training maintains the base LMM's performance on general multimodal benchmarks (MMBench, MME, POPE, AI2D) on par with the original Qwen-2.5-VL, eliminating the catastrophic forgetting problem caused by SFT.
5. **FCoT dataset for rigorous RL vs. SFT comparison**: The proposed FCoT dataset contains 1,500 image-mask pairs with step-by-step human-like annotation reasoning trajectories, and SAM2-compatible bounding box and point prompts. It provides a standardized SFT baseline to verify the advantages of RL over SFT in segmentation tasks.

## 3. Input, Output and Required Annotations
### Input
- **Inference phase**: The core input is an RGB image. For referring segmentation and reasoning segmentation tasks, an additional natural language query/expression is provided alongside the image.
- **Training phase**: For the core pure RL paradigm, the input only includes images and their corresponding ground truth segmentation masks. For the SFT baseline (for comparative experiments), additional prompt and chain-of-thought annotations from the FCoT dataset are used.

### Output
The direct output of the Seg-R1 policy model is an autoregressive sequence of structured content, including the reasoning process wrapped in `` tags, bounding boxes in `<bbox></bbox>` tags, point coordinates in `<points></points>` tags, and category labels in `<labels></labels>` tags. These structured prompts are then fed into SAM2 to generate the final output: the pixel-level segmentation mask of the target object.

### Required Annotations
For the core pure RL training paradigm of Seg-R1, **no additional annotations are required beyond the ground truth segmentation mask labels**. The model learns to generate valid prompts and reasoning processes purely through RL optimization using only image-mask pairs, with no need for manual prompt annotations, textual supervision, or chain-of-thought labels. Only the SFT baseline (for comparative experiments) relies on the additional prompt and CoT annotations from the FCoT dataset, which are not required for the core Seg-R1 method.
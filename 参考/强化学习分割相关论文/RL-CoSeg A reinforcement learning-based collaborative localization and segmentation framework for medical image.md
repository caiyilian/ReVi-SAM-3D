# Summary of RL-CoSeg: A reinforcement learning-based collaborative localization and segmentation framework for medical image
## 1. Problems to Be Solved
This paper targets two core challenges in medical image segmentation, especially for small targets (e.g., pancreas) in abdominal CT scans, along with critical limitations of existing mainstream solutions:
- **Inherent challenges of small medical target segmentation**: Small ROIs occupy a tiny fraction of CT images and are easily overwhelmed by background information, leading to missed detection and inaccurate predictions. Meanwhile, large inter-patient and inter-slice variations in target size and shape severely hinder model generalization.
- **Key limitations of conventional two-stage segmentation frameworks**:
  1. **Unresolvable trade-off between localization accuracy and target preservation**: Existing methods use manually fixed enlarged bounding boxes to avoid target missing, which lacks sample-specific adaptability. Insufficient expansion fails to cover the full target, while excessive expansion introduces redundant background and degrades segmentation precision.
  2. **Lack of collaborative optimization between two stages**: Localization and segmentation modules are trained independently in a decoupled pipeline. Localization errors directly propagate to the segmentation stage, while segmentation results cannot provide feedback to correct or optimize localization, severely limiting overall performance.
- **Defects of existing RL-based localization methods**: Most rely on a single IoU-based reward signal, which fails to capture the multi-dimensional requirements of medical detection. They also ignore feedback from downstream segmentation tasks, making it impossible to correct detection errors using subsequent task information.

## 2. Proposed Method: RL-CoSeg Framework
### 2.1 Overall Workflow
The proposed Reinforcement Learning-based Collaborative Localization and Segmentation (RL-CoSeg) framework consists of three core sub-networks, forming a closed-loop optimization pipeline:
1. **Initial coarse-to-fine localization via Localization Network (LN)**: The LN formulates the target localization process as a Markov Decision Process (MDP) solved by Deep Q-Learning (DQL). Taking the full abdominal CT image as input, the agent iteratively adjusts the bounding box through 12 discrete actions (7 shrinkage, 4 shifting, 1 termination) to generate a compact, target-complete candidate ROI, guided by a multi-factor reward mechanism.
2. **Collaborative bounding box refinement via Localization-Segmentation Collaboration Network (LSCN)**: The bounding box output by LN is enlarged by 1.5× to crop the local image, which is fed into the LSCN. The LSCN inherits the MDP framework of the LN, and performs fine-grained adjustment of the bounding box through 8 discrete actions (4 inward shrinkage, 4 outward expansion for each edge). Critically, it introduces real-time segmentation results from the Segmentation Network (SN) as a core reward signal, enabling bidirectional information interaction and collaborative optimization between localization and segmentation.
3. **Fine-grained segmentation via Segmentation Network (SN)**: The SN takes the final refined ROI from LSCN as input, outputs the pixel-level segmentation mask of the target ROI, and provides segmentation performance feedback to the LSCN during training to guide its policy learning. The SN is architecture-agnostic and can be flexibly integrated with mainstream segmentation backbones (e.g., UNet, TransUNet).

### 2.2 Key Innovations (Focus on Reinforcement Learning)
The core innovations of RL-CoSeg are centered on the design of reinforcement learning mechanisms to solve the above limitations, with the following key breakthroughs:
1. **Multi-factor reward mechanism for LN to balance localization precision and target preservation**
    This design fundamentally solves the misalignment between the single IoU-based reward and rational agent behavior in traditional RL detection methods. The step-wise reward function integrates four complementary components:
    - **IoU-based reward**: The core metric to evaluate the spatial overlap between the predicted box and the ground truth mask, rewarding IoU improvement and penalizing degradation.
    - **Center distance reward**: Addresses the problem that reasonable target-approaching actions receive no reward, with a lower weight (0.5 vs. 1 for IoU reward) to encourage shrinkage actions over translation for more compact localization.
    - **Recall reward**: Applies a heavy penalty (-2) when the bounding box fails to fully cover the target (Recall < 1), even if the IoU increases. This strongly constrains the agent from over-shrinking the box and losing critical target information, which is the top priority for medical imaging tasks.
    - **Time penalty term**: A linearly increasing penalty with the number of time steps, which mitigates the "reward hacking" phenomenon (the agent takes redundant paths to avoid termination and accumulate extra rewards), and improves localization efficiency.
    In addition, a **dynamic termination reward mechanism** is proposed to replace the static threshold-based reward in traditional DQL. It lowers the reward threshold in early training episodes to avoid frequent negative rewards that suppress exploration, effectively solving the agent's termination avoidance problem and stabilizing policy learning.

2. **Collaborative RL policy learning via LSCN to break the decoupled two-stage paradigm**
    This is the pioneering innovation of the paper, which establishes a closed-loop collaborative optimization between localization and segmentation for the first time via reinforcement learning. The core RL designs include:
    - **Segmentation-guided reward function**: The step-wise reward abandons IoU and center distance metrics that are inconsistent with segmentation performance, and consists of three task-aligned components: 1) localization recall reward to ensure full target coverage; 2) **segmentation performance reward (r_dsc)**, which uses the change in Dice Similarity Coefficient (DSC) of the SN's prediction as the core reward signal. A piecewise function is designed to achieve high sensitivity to small DSC variations, guiding the agent to adjust the bounding box toward directions that improve segmentation accuracy; 3) time penalty term to encourage efficient refinement.
    - **Sample-adaptive dynamic termination reward**: The termination reward is tied to the cumulative DSC improvement during the refinement process, avoiding premature termination on hard samples and insufficient optimization on easy samples, which adapts to the inherent variability of medical image segmentation difficulty.

3. **Heuristic-guided exploration strategy for stable RL training**
    To address the limitation that the standard ε-greedy strategy struggles to explore promising action sequences for segmentation tasks (where DSC presents discontinuous jumps), the proposed strategy prioritizes actions that improve DSC, then those that improve recall, then termination actions with positive rewards, and only takes random negative actions when no positive options exist. It is combined with the ε-greedy strategy to balance exploration quality and policy diversity, effectively avoiding local optima and improving training stability and convergence.

4. **Optimized state space design for the RL agent**
    The state space integrates global/local image features (from pre-trained segmentation encoders), historical action information, and a Fourier-based randomized positional encoding module (inspired by SAM), which explicitly transforms the spatial information of the bounding box into learnable features, enhancing the agent's spatial awareness and target perception.

## 3. Input, Output and Required Labels
### 3.1 Input
The core input is preprocessed 2D abdominal CT axial slices: the intensity values are clipped to [-100, 240], linearly normalized to [0, 255], resized to 224×224 pixels, and duplicated across three channels to form a 3-channel input. For volumetric CT data, the framework processes each axial slice individually.

### 3.2 Output
The framework has two main outputs:
1. The final pixel-level segmentation mask of the small target ROI (e.g., pancreas, pancreatic tumor);
2. The optimized bounding box of the target ROI generated by the collaborative localization pipeline (LN + LSCN).

### 3.3 Required Labels
- **Mandatory label**: Only the pixel-level segmentation ground truth mask is required. This mask is used for both the training of the segmentation network and the calculation of all reward signals in the RL framework (including IoU, recall, and DSC metrics in the reward functions).
- **No additional manual labels are needed**: All supervision signals for the RL policy learning (bounding box-related) are automatically derived from the segmentation ground truth mask. There is no requirement for manually annotated bounding box labels or any other extra annotation work beyond the segmentation mask.
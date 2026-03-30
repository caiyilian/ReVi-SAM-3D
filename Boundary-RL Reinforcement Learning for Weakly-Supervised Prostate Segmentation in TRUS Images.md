# Summary of Boundary-RL: Reinforcement Learning for Weakly-Supervised Prostate Segmentation in TRUS Images
## (1) Problem to Be Solved
This paper addresses two core critical challenges in medical image segmentation, with a specific focus on transrectal ultrasound (TRUS) prostate segmentation:
1.  **High annotation cost of fully supervised segmentation**: State-of-the-art fully supervised segmentation methods rely on pixel-level expert annotations, which are time-consuming to acquire, demand specialized clinical knowledge, and suffer from significant inter-observer variability, limiting the scalability of model training.
2.  **Performance bottlenecks of existing weakly supervised semantic segmentation (WSSS) methods**:
    - Most mainstream WSSS methods (e.g., multiple instance learning, MIL) frame segmentation as a pixel-level classification task, which delivers poor performance when noise or shadow artifacts exist within the region of interest (ROI), a common issue in TRUS images.
    - Sliding window-based WSSS approaches require exhaustive forward passes over the entire image, leading to low inference efficiency and increased false positives/negatives.
    - TRUS prostate segmentation itself faces inherent difficulties: indistinct soft tissue boundaries, missing boundary segments from shadow artifacts, high inter-patient variability in prostate shape and size, uneven intensity distribution, and insufficient informative supervision signals under weak label settings.
3.  **Lack of generalizable WSSS solutions**: Many prior weakly supervised methods are application-specific and fail to provide a generalizable framework for ROI boundary delineation with low annotation costs.

## (2) Proposed Method
### Workflow
The proposed Boundary-RL is a novel WSSS framework that redefines segmentation as a sequential boundary detection task via reinforcement learning (RL), rather than pixel-level classification. The entire workflow consists of two core cascaded modules, trained end-to-end with only patch-level weak labels:
1.  **Pre-training of Boundary Presence Classifier**
    - A binary classifier built on the EfficientNet backbone, trained to predict whether an image patch contains the prostate ROI (output 1) or not (output 0), using binary cross-entropy loss.
    - After pre-training, the classifier’s weights are fixed, and its prediction probability serves as the sole supervision signal for the subsequent RL controller, with a probability threshold of 0.9 for binary prediction.
2.  **RL-based Boundary Detection Controller (Core Module)**
    The boundary detection task is formalized as a Markov Decision Process (MDP), where the controller learns to sequentially move the image patch to localize the prostate boundary, optimized via the Proximal Policy Optimization (PPO) algorithm. The key components of the MDP are defined as follows:
    - **State**: The observed state at each time step includes the full input TRUS image and the current image patch localized by the controller.
    - **Action Space**: A discrete action space with four valid actions (up, down, left, right), where each action translates the patch’s coordinates along a single axis to update its position in the next time step.
    - **Reward Function**: A two-component reward function that balances dense reward shaping and sparse task-specific reward:
      1.  *Movement reward (r_mov)*: A dense shaping reward, set to +1 if the patch moves closer to the prostate centroid (estimated via pixel intensity or intensity-based registration, no manual annotation required) and -1 if it moves away.
      2.  *Termination reward (r_term)*: A sparse task reward directly derived from the pre-trained boundary presence classifier, equal to 1 if the current patch contains the prostate boundary/ROI, and 0 otherwise.
      3.  Total reward: \( r(s_t, a_t) = r_{mov} + 100 \cdot r_{term} \), where the scalar 100 weights the priority of the boundary detection task goal.
    - **Policy & Optimization**: The policy is parameterized by a neural network with 3 convolutional layers followed by 2 fully connected layers, which predicts the action distribution. The PPO algorithm is used to optimize the policy parameters via gradient ascent, with the goal of maximizing the expected cumulative discounted reward.
    - **Episodic Training**: For each input image, the controller runs M episodes. Each episode starts from a random edge pixel of the image, and terminates either when the classifier detects a boundary (triggering the termination reward) or when the maximum step limit (T=1000) is reached. Gaussian noise is added to the patch after each successful termination to encourage the controller to explore unique boundary points.
    - **Segmentation Generation**: The center points of all M patches detected as boundary points are collected, with outliers removed via distance-based filtering. A polygon is fitted to the remaining points to form the final pixel-level segmentation mask of the prostate.
    - **Inference**: The workflow is identical to training, except that no reward is calculated and the controller parameters are not updated.

### Key Innovations (Focus on Reinforcement Learning)
1.  **Paradigm shift for WSSS**: For the first time, weakly supervised medical image segmentation is reformulated as a sequential boundary detection task solved via RL, instead of the conventional pixel-level classification paradigm. This design inherently avoids performance degradation caused by intra-ROI artifacts, which is a major limitation of existing WSSS methods.
2.  **Weak supervision-driven RL reward design**: The framework uses the output of a patch-level classifier (trained only with weak labels) as the exclusive task reward for RL training, eliminating the need for pixel-level segmentation masks or any other strong supervision signals. This bridges the gap between weak annotation and the sparse reward requirements of RL for medical image segmentation.
3.  **Efficient and stable RL training mechanism**:
    - The two-component reward function combines dense movement reward shaping and sparse termination reward, effectively solving the exploration difficulty and training instability caused by sparse rewards in RL, and guiding the controller to converge to the ROI boundary efficiently.
    - The episodic training strategy with random edge initialization and noise-driven exploration enables dense and uniform sampling of ROI boundary points without exhaustive sliding window scanning over the entire image.
4.  **Superior inference efficiency and robustness**: Unlike sliding window-based MIL methods, the RL controller only performs forward passes on the classifier for patches along the movement trajectory, drastically reducing the number of inference computations and lowering false positive/negative predictions. It also maintains robust boundary localization even when intra-ROI artifacts are present in TRUS images.

### Input and Output
- **Input**: Center-cropped 360×360 2D TRUS prostate images (original size 403×361).
- **Output**: Pixel-level segmentation mask of the prostate gland, generated from the polygon fitted to the boundary points localized by the RL controller.

### Required Annotations
- The entire framework is trained **using only patch-level binary labels** (indicating the presence or absence of the prostate ROI within a 90×90 image patch), which is the only annotation required.
- **No pixel-level segmentation mask labels are needed** for training.
- No additional manual annotations are required: the prostate centroid used for movement reward calculation can be automatically estimated via pixel intensity statistics or intensity-based image registration, with no manual labeling involved.
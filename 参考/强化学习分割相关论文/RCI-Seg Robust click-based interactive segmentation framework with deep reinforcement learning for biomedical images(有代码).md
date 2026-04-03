代码：强化学习分割论文代码\DRL-Enhanced-Interactive-Segmentation-PyTorch-Implementation--main
# Summary of *RCI-Seg: Robust click-based interactive segmentation framework with deep reinforcement learning for biomedical images*
This paper, published in **Neurocomputing 2024**, proposes a novel two-stage interactive segmentation framework for multi-target biomedical images, with a core focus on mitigating the vulnerability of existing click-based interactive segmentation models to low-quality user interactions.

## 1. Problem to Be Solved
1.  **Critical sensitivity to interaction quality**: State-of-the-art click-based interactive segmentation models (e.g., NuClick) rely heavily on accurate, high-quality user click points. Segmentation performance degrades drastically when users click non-central or boundary positions of the region of interest (ROI), rather than the exact ROI center.
2.  **Practical limitations in clinical multi-target scenarios**: Multi-target biomedical images (e.g., nuclei images with numerous small, compact targets) make it extremely difficult and time-consuming for users to consistently provide precise center clicks, leading to unstable segmentation outputs and high user interaction burden.
3.  **Drawbacks of fully automatic segmentation**: Fully automatic segmentation methods require large-scale pixel-level annotated data for training. However, biomedical image annotation demands expert knowledge, is time-consuming and costly, and faces data privacy constraints, resulting in poor model generalization on unseen datasets.
4.  **Unrealistic assumption of existing interactive methods**: Nearly all existing interactive segmentation methods assume user-provided interaction information is perfectly accurate, without considering the impact of user interaction errors on segmentation performance, which limits their clinical practicability.

## 2. Proposed Method
### 2.1 Overall Workflow
The proposed **RCI-Seg** is a two-stage framework that decouples robust interaction point optimization and pixel-level segmentation, with independently trainable deep reinforcement learning (DRL) and convolutional neural network (CNN) modules.
1.  **Stage 1: Robust interaction point optimization via DRL**
    This stage takes the user’s initial click interaction point as input, models the point movement process as a Markov Decision Process (MDP), and uses a DRL model based on the Double DQN (DDQN) algorithm to simulate the movement of the interaction point. The agent learns a policy to move toward the ROI center, and outputs an optimized, high-quality interaction point named the *clue point*, which is more conducive to accurate segmentation.
2.  **Stage 2: CNN-based biomedical image segmentation**
    The clue point generated in the first stage is fused with the original biomedical image via multi-channel fusion. The fused feature map is fed into an encoder-decoder CNN segmentation network (modified from NuClick, with multi-scale convolution and residual modules) for pixel-level category prediction, and finally outputs the ROI segmentation mask.

### 2.2 Key Innovations (Focus on Deep Reinforcement Learning)
1.  **Pioneering DRL application for interaction robustness**
    This work is the first to introduce DRL to study the robustness of interaction points in interactive segmentation. The proposed two-stage framework fundamentally solves the problem of segmentation performance degradation caused by inaccurate user clicks, rather than requiring users to provide precise center clicks.
2.  **Task-specific MDP modeling for interaction point optimization**
    The authors designed a complete MDP framework tailored for 2D biomedical image ROI center localization, which is the core innovation of the DRL module:
    - **State definition**: The state at each timestep is a 32×32 image patch cropped with the agent’s current coordinate as the center. To capture continuous movement features, the final input state fuses the current state and the three most recent historical states (initial state replicas are used for the first three timesteps).
    - **Action space**: Four discrete pixel-level movement actions (up, down, left, right) are defined for the 2D biomedical image environment, which simplifies the agent’s decision space and improves movement efficiency.
    - **Customized reward function**: A distance-based reward function is designed as \(R=D(p^{t}, c)-D(p^{t+1}, c)\), where \(D(\cdot)\) calculates the Euclidean distance between two points, \(p^t\) and \(p^{t+1}\) are the agent’s coordinates at the current and next timestep, and \(c\) is the ground truth center coordinate of the ROI. The agent receives a positive reward when moving toward the ROI center and a negative penalty when moving away, directly driving the agent to learn a center-oriented movement policy.
    - **Practical termination condition**: A step-based termination rule is defined (the agent stops moving after a predefined number of steps), which solves the problem that the ROI center coordinate is unknown during the inference phase, ensuring the feasibility of the model in clinical deployment.
3.  **Stable and efficient DRL network design**
    The DRL network adopts the DDQN algorithm to eliminate the overestimation problem of the traditional DQN algorithm. Meanwhile, an experience replay mechanism with a fixed-capacity replay buffer is introduced to store the agent’s exploration experience, enabling batch random sampling for training and significantly improving the stability and convergence of the network training.
4.  **Additional framework innovations**
    - A three-channel interactive information fusion strategy is proposed (original image, current target map, other target maps), which avoids the interference of non-target objects on the current ROI segmentation in multi-target biomedical images.
    - The DRL and CNN modules can be trained independently, which adapts to the scenario of insufficient biomedical image data and endows the framework with excellent generalization ability across multiple segmentation tasks (nuclei, liver tumor, and vertebrae segmentation).

### 2.3 Input and Output
#### Overall Framework
- **Input**: 1) Original 2D biomedical image (including histopathology nuclei images, abdominal CT for liver tumors, and spine CT for vertebrae); 2) A single initial click interaction point provided by the user for each ROI (the point can be located at the internal non-central or boundary position of the ROI, without the need for precise center clicking).
- **Output**: Pixel-level binary segmentation mask of the target ROI.

#### DRL Module (Stage 1)
- **Input**: 32×32 image patches centered at the agent’s coordinates, fused with the current state and the previous three historical states; the initial coordinate of the user’s click point.
- **Output**: Optimized clue point coordinate, which is the final position of the agent after movement and is close to the real center of the ROI.

#### CNN Segmentation Module (Stage 2)
- **Input**: 128×128 image patches cropped with the clue point as the center, fused with three channels: the original RGB image, the current target map (pixel value 1 at the clue point position, 0 elsewhere), and the other target maps (pixel value 1 at the clue points of all targets except the current one, 0 elsewhere).
- **Output**: Pixel-level segmentation mask of the target ROI.

## 3. Required Labels
1.  **Mandatory basic label**: Pixel-level ground truth segmentation masks of ROIs, which are required for training the CNN segmentation module, consistent with standard supervised segmentation tasks.
2.  **Additional label for DRL training**: The **ground truth center coordinate of each ROI** is required to calculate the reward function and supervise the training of the DRL module.
    - Critical note: This ROI center coordinate does not require additional manual annotation. It can be directly and automatically calculated from the existing segmentation mask (e.g., via the centroid of the mask), which means the framework does not impose extra annotation burden on users.
3.  **Inference phase**: No additional labels are required. The model only needs the original image and the user’s initial click point to generate the final segmentation result.
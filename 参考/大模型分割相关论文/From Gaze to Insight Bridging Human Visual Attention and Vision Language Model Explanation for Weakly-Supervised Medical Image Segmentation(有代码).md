代码（但是大模型是在预处理阶段离线处理，所以代码里面没有大模型相关代码）：大模型分割代码\FGI-main
# Detailed Summary of *From Gaze to Insight: Bridging Human Visual Attention and Vision Language Model Explanation for Weakly-Supervised Medical Image Segmentation*
This paper is accepted by *IEEE Transactions on Medical Imaging*, and proposes a novel weakly-supervised medical image segmentation framework that synergizes clinician gaze data and vision-language model (VLM) outputs to address the annotation bottleneck in clinical medical image analysis.

## 1. Problem to Be Solved
The work targets three core, interconnected challenges in the field:
1.  **Prohibitive annotation cost of fully-supervised segmentation**: State-of-the-art medical image segmentation models rely on pixel-level manual annotations, which demand extensive clinical expertise, extremely long annotation time (18.7 hours for standard datasets in the study), and hinder large-scale clinical deployment of segmentation models.
2.  **Intrinsic limitations of single-source weak supervision signals**:
    -   **Clinician gaze data**: As a low-cost weak supervision signal (collected with zero additional workflow overhead during routine diagnosis via eye trackers), gaze points naturally indicate diagnostically relevant regions. However, it suffers from inherent sparsity, noise (spurious fixations from visual search or distraction), and lack of clear lesion boundary definitions, leading to incomplete and inaccurate segmentation when used alone.
    -   **Vision-language models (VLMs)**: VLMs can generate semantic context and morphological descriptions of lesions to explain the clinical significance of regions, but they face severe domain gaps when transferred from natural images to medical imaging, suffer from semantic hallucinations, and lack the pixel-level precision required for fine-grained segmentation tasks.
3.  **Unresolved gap between spatial localization and semantic interpretability**: Existing methods fail to effectively fuse the complementary strengths of gaze and VLM signals. Gaze only answers *where* clinicians focus during diagnosis, while VLMs only explain *why* those regions are clinically significant. Neither signal alone can balance annotation efficiency, segmentation accuracy, and clinical interpretability for real-world clinical scenarios.

## 2. Proposed Method
### 2.1 Overall Framework Overview
The authors propose a **teacher-student framework** that unifies human visual attention (gaze) and VLM-derived semantic explanations for weakly-supervised medical image segmentation. The teacher model learns robust cross-modal representations from high-reliability, sparse gaze annotations enhanced by VLM semantic cues, then transfers the refined knowledge to a student model. The student model is trained on broader, noisier gaze data, with three tailored optimization strategies to suppress label noise and align with the teacher’s reasoning.

### 2.2 End-to-End Workflow Pipeline
The workflow is divided into four sequential core stages:
1.  **Dual Gaze Pseudo-Mask Generation**
    From raw clinician gaze points collected by eye trackers, two types of pseudo-masks are generated for each medical image:
    -   *High-Confidence Mask (Mₕc)*: A small, meticulously verified set of masks capturing core diagnostic fixations, with precise but partial spatial coverage, used for teacher model supervision.
    -   *Broad-Coverage Mask (Mᵦc)*: A more extensive mask covering potential lesion regions, but containing noise (e.g., ambiguous or off-target fixations), used for student model training.
2.  **Structured VLM Prompting and Textual Embedding Extraction**
    Each medical image is fed into a large VLM (e.g., Doubao-1.5-Vision-Pro, GPT-4o) with a **clinically constrained structured prompt**, which forces the VLM to output a standardized JSON object describing abnormal regions, with predefined fields: *Location, Boundary, Characteristics, Area Percentage, Confidence, Remarks*. Each field has restricted standardized descriptors to minimize domain gap and semantic hallucinations. The structured text is then tokenized and encoded into a 768-dimensional semantic embedding (Fₜ) via a RoBERTa encoder, encapsulating lesion morphology, boundary, and contextual attributes.
3.  **Teacher Model Training with Multi-Modal Fusion**
    The teacher model is built on a UNet architecture, with a core **multi-scale text-vision fusion module** applied at each encoder layer. This module uses multi-head cross-attention to inject VLM-derived textual semantic cues into visual features at every scale, with a residual connection and learnable fusion weight to stabilize training. The teacher model is optimized with a **Partial Cross-Entropy (pCE) Loss** on Mₕc: only pixels with high-confidence gaze annotations contribute to the loss, while unlabeled pixels are excluded, allowing the model to leverage textual semantics to fill spatial gaps in sparse gaze signals.
4.  **Teacher-to-Student Knowledge Distillation and Student Optimization**
    The student model uses a lightweight UNet architecture trained on Mᵦc, with three core mechanisms to transfer the teacher’s cross-modal knowledge and suppress label noise:
    1.  **Angular Feature Consistency (AFC) Loss**: Enforces multi-scale feature-level alignment between the teacher and student at 4 encoder stages, measuring feature similarity via normalized inner product to distill refined cross-modal representations.
    2.  **Confidence-Weighted Consistency (CWC) Loss**: Defines confident positive/negative regions where both the teacher and student have high prediction certainty, then applies a confidence-weighted consistency regularization to align predictions in these reliable regions, avoiding interference from noisy low-confidence areas.
    3.  **Disagreement-Aware Random Masking (DARM)**: Selectively masks local patches in regions with high prediction disagreement between the teacher and student before computing the cross-entropy loss on Mᵦc. This prevents the student from overfitting to noisy gaze annotations and forces it to leverage global contextual information.
    The final student loss is the weighted sum of the masked cross-entropy loss on Mᵦc, AFC loss, and CWC loss.

### 2.3 Key Innovations
#### 2.3.1 Core Technical Innovations
1.  The first weakly-supervised medical image segmentation framework that fuses expert gaze patterns and VLM outputs, which overcomes the inherent limitations of single weak supervision signals and narrows the performance gap with fully-supervised methods while drastically reducing annotation costs.
2.  A novel teacher-student architecture that decouples high-reliability sparse gaze supervision (for the teacher) and broad noisy gaze supervision (for the student), enabling robust cross-modal knowledge transfer without increasing annotation burden.
3.  Three tailored technical modules: multi-scale text-vision feature alignment, confidence-aware consistency regularization, and disagreement-aware adaptive masking, which collectively address the sparsity and noise of gaze data and the spatial-semantic misalignment of VLM outputs.
4.  Superior empirical performance: the method achieves Dice scores of 80.78%, 80.53%, and 84.22% on the KvasirSEG, NCI-ISBI, and ISIC datasets respectively, delivering a 3–5% Dice improvement over state-of-the-art gaze-based baselines, while maintaining the same low annotation cost (only 2.2 hours, ~1/8 of fully-supervised methods).

#### 2.3.2 Critical Innovations in VLM Utilization
1.  **Clinical domain adaptation via structured prompt engineering**: Unlike generic unconstrained prompts, the predefined structured prompt with standardized field constraints minimizes the domain gap of general VLMs in medical imaging, effectively mitigates semantic hallucinations, and ensures the generated textual descriptions are clinically relevant, precise, and compatible with downstream segmentation tasks.
2.  **Upgrading VLMs from passive feature extractors to active supervisory oracles**: Most existing works only use VLMs for static visual feature extraction. This work leverages VLM-generated semantic embeddings as a core complementary supervision signal, which provides clinical rationale for lesion regions and fills the semantic gap of purely spatial gaze signals, realizing the unification of "where to look" and "why it matters".
3.  **Fine-grained multi-scale cross-modal alignment**: The multi-head cross-attention fusion module injects textual semantics into visual features at every encoder scale, rather than simple single-level feature concatenation. This design achieves fine-grained alignment between high-level semantic descriptions (e.g., lesion boundary, shape) and pixel-level spatial features, solving the core challenge of aligning VLM semantics with fine-grained spatial details for segmentation.
4.  **Enhancing clinical interpretability via cross-modal alignment**: The framework preserves the correlation between model predictions, gaze data, and VLM-generated lesion descriptions, turning the black-box segmentation model into a clinically interpretable system that aligns with clinicians' diagnostic reasoning logic.

### 2.4 Inputs, Outputs and Required Annotations
#### Inputs
1.  Raw medical images (supporting multiple modalities: endoscopic, MRI, dermoscopic images);
2.  Clinician gaze (eye-tracking) points collected during routine diagnosis;
3.  Structured lesion description text automatically generated by the VLM from the input medical image.

#### Outputs
1.  Pixel-level lesion segmentation masks for the input medical images;
2.  Clinically interpretable structured semantic descriptions of the segmented lesions (from the VLM);
3.  Cross-modal representations aligned with clinician visual attention, semantic context, and model predictions.

#### Required Annotations (Beyond Pixel-Level Segmentation Masks)
-   The framework **does NOT require pixel-level ground truth segmentation masks** for training, which is the core of its weakly-supervised design.
-   The **only required annotation is clinician gaze (eye-tracking) data**, which can be collected with near-zero additional cost during routine clinical diagnosis, with no need for extra manual labeling work.
-   No additional manual labels (e.g., bounding boxes, scribbles, image-level tags, or manual text descriptions) are needed. The VLM’s textual outputs are generated automatically via the predefined prompt, with no manual annotation burden.
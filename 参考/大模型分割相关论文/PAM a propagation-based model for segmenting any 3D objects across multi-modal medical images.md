# Detailed Summary of *PAM: a propagation-based model for segmenting any 3D objects across multi-modal medical images*
## 1. Problems to be Solved
This paper addresses critical unmet challenges in 3D volumetric medical image segmentation, which is a cornerstone task for clinical diagnosis, surgical planning, treatment response monitoring, and prognosis evaluation. The core problems are summarized as follows:

First, existing deep learning-based segmentation methods are inherently task-specific. These models require large, expert-annotated datasets and retraining for every new anatomical structure, lesion type, or imaging modality (CT, MRI, PET-CT, SRX, etc.). This paradigm incurs prohibitive labor and time costs, and is impractical for rare pathologies or emergent clinical scenarios with limited annotated data.

Second, prevailing Segment Anything Model (SAM)-derived medical foundation models have critical limitations when applied to 3D medical imaging, falling into two flawed categories:
- **Type I models (e.g., MedSAM)**: Perform 2D slice-by-slice segmentation, ignoring the inherent inter-slice structural and semantic continuity of 3D medical volumes. They require dense multi-slice or multi-view prompts to achieve coherent 3D segmentation, leading to high user interaction burden and discontinuous segmentation results.
- **Type II models (e.g., SegVol)**: Replace 2D convolutions with 3D counterparts to directly process volumetric data, but suffer from explosive parameter size, excessive computational overhead, and poor generalization to unseen objects/modalities. Their patch-by-patch prediction strategy causes fine-grained information loss, especially for irregular lesions with variable shapes.

Third, the direct transplantation of SAM’s natural image prompt-to-mask alignment paradigm to medical imaging is fundamentally unsuitable. Medical images have subtle pixel intensity and texture differences between foreground and background, with high semantic ambiguity, making it hard for SAM-like models to achieve robust cross-scenario generalization. These models fail to leverage the universal characteristic of 3D medical images: the continuous flow of anatomical information across adjacent slices.

Fourth, there is an urgent clinical need for a highly efficient, generalizable segmentation tool with minimal user interaction, fast inference, and robust performance for irregular, complex anatomical structures and lesions, which are the most challenging cases in real-world clinical practice.

## 2. Proposed Method
### 2.1 Overall Workflow
The authors propose **PAM (Propagating Anything Model)**, a novel Type III propagation-based framework for universal 3D medical image segmentation. Its workflow centers on two core modules (Box2Mask and PropMask) and an iterative bidirectional propagation inference pipeline, detailed as follows:
1. **User Interaction and Prompt Standardization**
   Users upload a 3D medical volume and provide a 2D prompt on a single guiding slice (the slice with the largest target cross-section, consistent with the RECIST clinical guideline). PAM supports two prompt types: 2D bounding boxes (PAM-2DBox) and freehand 2D contour masks (PAM-2DMask). For bounding box inputs, the Box2Mask module converts the coarse prompt into a standardized 2D binary mask for downstream processing.
2. **Box2Mask Module**
   This U-Net-based CNN module is trained on over 10 million medical images to perform foreground segmentation within the user-provided bounding box. It outputs multi-resolution probability maps optimized via deep supervision with a soft Dice loss. During inference, bilinear upsampling and soft voting are applied to generate the final 2D mask corresponding to the bounding box prompt.
3. **PropMask Module (Core Propagation Component)**
   This hybrid CNN-Transformer module models inter-slice continuity and propagates prompt information across the 3D volume:
   - A shared image encoder extracts multi-scale features from the guiding slice and adjacent slices, generating Key (K) and Query (Q) embeddings respectively.
   - A dedicated mask encoder transforms the guiding prompt into multi-scale Value (V) embeddings.
   - A cross-attention mechanism models long-range inter-slice structural and semantic relationships, transferring the guiding prompt information to adjacent slices.
   - A prompt-guided decoder with skip connections fuses attention-enhanced features and multi-scale local features to predict the segmentation mask for adjacent slices.
4. **Iterative Bidirectional Propagation Inference**
   - Initial segmentation is performed on adjacent slices using the user-provided guiding slice and prompt.
   - The outermost predicted slices from the previous step are iteratively used as new guiding slices for bidirectional parallel propagation.
   - The propagation terminates when the 3D volume boundary is reached, or no valid foreground is predicted in subsequent slices.
   - All slice-level predictions are aggregated to generate the final complete 3D volumetric segmentation mask.

### 2.2 Core Innovations (Especially for Foundation Model Application in Medical Imaging)
PAM introduces paradigm-shifting innovations for applying foundation models to 3D medical imaging, addressing the core limitations of existing SAM-derived models:
1. **Novel Paradigm for Medical Foundation Model Design**
   PAM breaks the direct prompt-to-mask transplantation paradigm of SAM-like models, and proposes the first Type III propagation-based framework. Instead of learning object-specific semantic features, PAM focuses on modeling the universal continuous information flow across slices in 3D medical images. This fundamental shift eliminates the overfitting problem of existing medical foundation models to limited object patterns in training data, and unlocks strong zero-shot generalization across diverse modalities, objects, and datasets.
2. **Efficient Hybrid Architecture with Targeted Innovations**
   - The hybrid CNN-Transformer design combines the strengths of CNNs for intra-slice local feature extraction and Transformer attention for inter-slice long-range dependency modeling. With only 32.48M parameters for PropMask (53.1M total with Box2Mask), PAM achieves significantly faster inference than pure Transformer-based models (MedSAM, SegVol) while delivering superior segmentation accuracy.
   - Decoupled image and mask encoders: Ablation studies confirm that this design captures complementary image and prompt information, delivering substantial performance gains over shared encoder architectures.
   - Multi-scale prompt-guided decoder: Progressive multi-scale feature fusion progressively improves segmentation precision, especially for object boundaries and irregular structures.
   - Dynamic normalization strategy: Adapts to the varying intensity distributions of medical images from different anatomical regions and modalities, outperforming fixed normalization strategies in general-purpose segmentation tasks.
3. **Groundbreaking Advancements in Medical Foundation Model Usability and Generalization**
   - **Minimal prompt dependency**: PAM only requires a single 2D prompt from one view to generate full 3D segmentation, while MedSAM and SegVol mandate two-view 3D bounding box prompts. This reduces user interaction time by 63.6%, drastically lowering the clinical operation burden.
   - **State-of-the-art zero-shot performance**: Across 44 diverse datasets, 168 object types, and 4 imaging modalities, PAM-2DBox improves average DSC by 23.1% over MedSAM and 19.3% over SegVol, outperforming both models on 31/34 internal datasets and all 10 external datasets.
   - **Superior few-shot adaptability**: With minimal fine-tuning on limited annotated data, PAM rapidly adapts to novel objects and outperforms task-specific models trained from scratch. Even when trained from scratch on small datasets, it recovers over 75.33% of the full model’s performance, solving the core challenge of scarce annotated medical data.
   - **Exceptional performance for irregular objects**: PAM’s performance gains are negatively correlated with object regularity (r < -0.1249), meaning it delivers the largest improvements for complex, irregular lesions and anatomical structures— the most challenging cases in clinical practice.
   - **Unmatched inference efficiency**: Its bidirectional parallel propagation strategy eliminates the need for overlapping window sliding, achieving significantly faster inference than MedSAM’s slice-by-slice prediction and SegVol’s patch-by-patch processing (Wilcoxon rank-sum test, P < 0.001).
4. **Clinical Translation Innovation**
   PAM enables accurate 3D tumor reconstruction using only physician-provided 2D RECIST measurements as prompts, without any fine-tuning. The 3D features derived from PAM’s segmentation significantly differentiate high- and low-risk groups in gastric cancer patients (Log-rank P = 0.013), while traditional 2D RECIST measurements show no significant prognostic value (Log-rank P = 0.14). This validates PAM’s direct clinical utility for prognosis evaluation and treatment planning.

## 3. Input, Output and Label Requirements
### 3.1 Input
PAM has two core input components, with no strict limitations on imaging modality or target object type:
1. **3D volumetric medical images**: Supports all mainstream medical imaging modalities, including CT, MRI, PET-CT, and synchrotron radiation X-ray (SRX).
2. **User-provided 2D prompt**: Defined on a single guiding slice in one anatomical view (axial, sagittal, or coronal), with two optional types:
   - 2D bounding box (for PAM-2DBox): A minimal interaction prompt enclosing the target object on a single slice.
   - 2D freehand contour mask (for PAM-2DMask): A hand-drawn outline of the target, providing more precise guidance for objects with ambiguous boundaries or irregular shapes.
   - PAM also supports 3D bounding box prompts (PAM-3DBox) for head-to-head comparison with MedSAM and SegVol, with further improved segmentation performance.

### 3.2 Output
The primary output is a **voxel-level 3D volumetric binary segmentation mask**, which is spatially aligned with the input 3D medical volume and precisely delineates the foreground region of the user-specified target object (organ, lesion, tissue, etc.). Intermediate outputs include the 2D segmentation mask of the guiding slice (from Box2Mask) and slice-level 2D segmentation masks generated during each propagation step, which are aggregated to form the final 3D result.

### 3.3 Label Requirements
- **Training phase**: PAM only requires **voxel-level 3D segmentation ground-truth masks** for model training, with no need for any additional labels. All training samples are generated exclusively from these segmentation masks:
  - For the Box2Mask module, 19 million+ 2D image-mask training pairs are generated by simulating bounding boxes from the 3D ground-truth masks, with no extra annotations required.
  - For the PropMask module, 1.34 million+ propagation tasks (guiding slice, guiding prompt, adjacent slice ground truth) are constructed directly from the 3D segmentation masks, without any additional labels such as category tags, text descriptions, modality labels, or anatomical prior labels.
- **Inference/clinical application phase**: No labels or pre-annotations are required. PAM performs zero-shot 3D segmentation using only the user’s minimal 2D prompt, with no need for pre-defined target category information or prior annotations.
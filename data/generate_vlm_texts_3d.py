import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

# 将 models/LLaVA-Med-main 添加到 sys.path，以便导入其模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
llava_path = os.path.join(project_root, 'models', 'LLaVA-Med-main')
if llava_path not in sys.path:
    sys.path.append(llava_path)

# 导入 LLaVA-Med 模块
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    LLAVA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LLaVA-Med dependencies not found. Please ensure it is installed correctly. Error: {e}")
    LLAVA_AVAILABLE = False


def normalize_intensity(x):
    """
    归一化图像亮度到 0-1
    """
    b = np.percentile(x, 99.5)
    t = np.percentile(x, 0.5)
    x = np.clip(x, t, b)
    if np.max(x) == np.min(x):
        return np.zeros_like(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def extract_middle_slice_as_pil(nii_path):
    """
    从 3D nii.gz 文件中提取中间切片，并转换为 RGB PIL Image 供大模型推理
    """
    img_nii = nib.load(nii_path)
    img_array = img_nii.get_fdata()
    
    # 确保通道顺序为 (D, H, W)
    if img_array.shape[2] < img_array.shape[0] and img_array.shape[2] < img_array.shape[1]:
        img_array = np.transpose(img_array, (2, 1, 0))
        
    # 提取 Z 轴的中间切片
    mid_idx = img_array.shape[0] // 2
    slice_array = img_array[mid_idx, :, :]
    
    # 归一化到 0-255 并转为 PIL Image
    slice_array = normalize_intensity(slice_array)
    slice_array = (slice_array * 255).astype(np.uint8)
    image = Image.fromarray(slice_array).convert('RGB')
    return image

def main(args):
    if not LLAVA_AVAILABLE:
        print("LLaVA is not available. Exiting...")
        return

    disable_torch_init()
    
    print(f"Loading LLaVA-Med Model from {args.model_path} ...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, False, False, device=args.device
    )
    print("Model Loaded Successfully!")

    # 查找所有的 .nii.gz 文件 (这里为了演示，遍历 data_dir 下所有的 .nii.gz)
    # 在真实项目中，你可能需要根据实际的 train/val 列表来筛选
    image_paths = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.nii.gz"), recursive=True))
    
    # 过滤掉标签文件（简单的基于路径包含 "labelsTr" / "labelsTs" 的判断）
    image_paths = [p for p in image_paths if "labelsTr" not in p and "labelsTs" not in p]
    
    print(f"Found {len(image_paths)} medical images to process.")

    results = {}
    
    # 固定的 Prompt，引导模型描述解剖学结构
    prompt_text = "Please describe the anatomical structures and potential lesions in this medical image in detail."

    for img_path in tqdm(image_paths, desc="Generating VLM Texts"):
        filename = os.path.basename(img_path)
        
        try:
            # 1. 提取中间切片
            image = extract_middle_slice_as_pil(img_path)
            
            # 2. 处理图像特征
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            # 3. 构建大模型对话 Prompt
            conv = conv_templates["mistral_instruct"].copy()
            if getattr(model.config, 'mm_use_im_start_end', False):
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            
            stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            # 4. 执行推理
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=512,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                    stopping_criteria=[stopping_criteria])

            # 5. 解析输出结果
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            prompt_text_plain = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if outputs.startswith(prompt_text_plain):
                outputs = outputs[len(prompt_text_plain):].strip()
                
            results[filename] = outputs
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            results[filename] = f"Error generating description: {str(e)}"

    # 6. 保存为离线 JSON 供 Dataset 读取
    output_json = os.path.join(os.path.dirname(__file__), "vlm_texts_3d.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccessfully saved {len(results)} descriptions to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 根据之前的测试脚本，设定好默认权重路径
    parser.add_argument("--model-path", type=str, default=r"e:\projects\大模型分割\预训练大模型\LLaVA-Med-main\llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "Task02_Heart"), help="Dataset root directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
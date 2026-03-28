import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image

def main(args):
    disable_torch_init()
    
    # 1. 加载模型和处理器
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device
    )

    # 2. 读取并处理医学图像切片
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # 3. 设置对话模板
    conv = conv_templates["mistral_instruct"].copy()

    # --- 这里是核心修改点：固定指令 Prompt ---
    prompt_text = "Please describe the anatomical structures in this medical image in detail." 
    
    if getattr(model.config, 'mm_use_im_start_end', False):
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

    # 4. 构建模型输入
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # 5. 执行推理
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=False, # 建议设为 False 获取最确定性的解剖学描述
            temperature=None,
            max_new_tokens=512,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids),
            stopping_criteria=[stopping_criteria])

    # 6. 打印输出结果
    # 对于传入 inputs_embeds 的 generate，往往返回的只包含生成的 token，或者由于版本原因包含 prompt
    # 最稳妥的方法是直接 batch_decode 然后如果发现包含 prompt 就将其切除，但通常直接 decode 即可
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # 如果模型输出自带了 prompt，我们可以把 prompt 的文本去掉
    prompt_text_plain = prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
    if outputs.startswith(prompt_text_plain):
        outputs = outputs[len(prompt_text_plain):].strip()

    print(f"\n[解剖学描述输出]:\n{outputs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认指向刚才提到的权重
    parser.add_argument("--model-path", type=str, default=r"/public/cyl/fourth_works/pretrained_weights/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True, help="你的 2D 医学切片路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)
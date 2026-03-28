import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextPromptEncoder(nn.Module):
    """
    基于 CLIP 的轻量级文本特征提取器，用于将 VLM 生成的 Mock 文本
    转化为 1D 的 Semantic Embedding，供后续的跨维度投影模块使用。
    """
    def __init__(self, model_name="/public/cyl/fourth_works/pretrained_weights/clip-vit-base-patch32", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        # 1. 加载 Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # 2. 加载 CLIP 文本模型
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        
        # 3. 冻结所有参数 (拿来主义，纯特征提取)
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        self.text_model.eval()

    @torch.no_grad()
    def forward(self, text_list):
        """
        :param text_list: List[str] - 输入的文本列表 (例如 Batch Size 个 Mock 文本)
        :return: text_embeddings [B, N, C] - B是Batch大小，N是序列长度(通常77)，C是特征维度(如512)
                 pooled_output [B, C] - 聚合后的全局特征
        """
        # Tokenize 文本
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)
        
        # 提取特征
        outputs = self.text_model(**inputs)
        
        # outputs.last_hidden_state: [B, seq_len, hidden_size] (例如: [B, 77, 512])
        # outputs.pooler_output: [B, hidden_size]
        return outputs.last_hidden_state, outputs.pooler_output

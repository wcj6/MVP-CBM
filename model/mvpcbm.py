import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from torchvision import transforms
from .utils import FFN
from einops import rearrange
import pdb


class SparseContribution(nn.Module):
    def __init__(self):
        super(SparseContribution, self).__init__()
        # 初始化可学习的缩放参数 τ，初始值为 0.2
        self.tau = nn.Parameter(torch.tensor(0.2))
        # 初始化可学习的游标参数 κ，初始值为 0
        self.kappa = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_concepts]
        
        Returns:
            sparse_alpha (torch.Tensor): 稀疏化后的贡献度
            mask (torch.Tensor): 稀疏化掩码
        """
        # 计算每个概念的绝对重要性
        abs_importance = torch.abs(x)  # 形状: [batch_size, num_concepts]
        
        # 计算每个样本的最大值和最小值
        p_max = torch.max(abs_importance, dim=1, keepdim=True).values  # [batch_size, 1]
        p_min = torch.min(abs_importance, dim=1, keepdim=True).values  # [batch_size, 1]
        
        # 计算阈值 t
        threshold = torch.sigmoid(self.kappa) * (p_max - p_min) + p_min  # [batch_size, 1]
        
        # 计算调整因子
        adjustment = torch.exp(self.tau * (abs_importance - threshold))  # [batch_size, num_concepts]
        
        # 计算最终的贡献度 α
        alpha = adjustment * x  # [batch_size, num_concepts]
        
        # 创建掩码，筛选出大于等于阈值的元素
        mask = (abs_importance >= threshold).float()  # [batch_size, num_concepts]
        
        # 应用掩码，得到稀疏化后的贡献度
        sparse_alpha = alpha * mask  # [batch_size, num_concepts]
          # 计算 L1 正则化损失，鼓励稀疏性
    
        
        return sparse_alpha, mask

class SparseConceptSpecificAttention(nn.Module):
    def __init__(self, num_concepts=34):
        super(SparseConceptSpecificAttention, self).__init__()
        self.num_concepts = num_concepts
        
        # 线性层用于计算每个概念的注意力分数
        self.attention_fc = None  # 将在前向传播中初始化
        
        # 稀疏化模块
        self.sparse = SparseContribution()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_layers, feature_dim]
        
        Returns:
            compressed (torch.Tensor): 稀疏化后的压缩特征，形状 [batch_size, feature_dim]
            mask (torch.Tensor): 稀疏化掩码，形状 [batch_size, num_concepts]
        """
        batch_size, num_layers, feature_dim = x.size()
        
        # 初始化线性层（仅在第一次前向传播时初始化）
        if self.attention_fc is None:
            self.attention_fc = nn.Linear(feature_dim, self.num_concepts).to(x.device)
        
        # 重塑 x 为 [batch_size * num_layers, feature_dim]
        x_reshaped = x.view(-1, feature_dim)
        
        # 计算原始注意力分数
        attention_scores = self.attention_fc(x_reshaped)  # [batch_size * num_layers, num_concepts]
        
        # 重塑回 [batch_size, num_layers, num_concepts]
        attention_scores = attention_scores.view(batch_size, num_layers, self.num_concepts)  # [batch_size, num_layers, num_concepts]
        # print(attention_scores.shape)
        # [128, 12, 34]
        # 对层维度应用 softmax，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_layers, num_concepts]
        # print(attention_weights.shape)
        
        # 对每个概念应用稀疏化
        # 先对 attention_weights 进行求和，得到每个概念的总重要性
        concept_importance = torch.sum(attention_weights, dim=1)  # [batch_size, num_concepts]
        # print(concept_importance.shape)
        
        # 应用稀疏化模块    
        sparse_alpha, mask = self.sparse(concept_importance)  # [batch_size, num_concepts], [batch_size, num_concepts]
        
        # 重新调整 attention_weights，使其与稀疏化后的 α 对齐
        # 首先，将 attention_weights 与稀疏化后的 mask 进行逐元素相乘
        sparse_attention = attention_weights * mask.unsqueeze(1)  # [batch_size, num_layers, num_concepts]
        
        # 重新归一化注意力权重，使其在层维度上和为 1（防止稀疏化后总和小于1）
        sparse_attention = sparse_attention / (sparse_attention.sum(dim=1, keepdim=True) + 1e-8)  # [batch_size, num_layers, num_concepts]
        
        # 使用稀疏化后的注意力权重对输入进行加权求和
        compressed = torch.sum(x * sparse_attention, dim=1)  # [batch_size, feature_dim]
        
        return compressed, mask

class SparseContributionPreAgg(nn.Module):
    def __init__(self):
        super(SparseContributionPreAgg, self).__init__()
        # 初始化可学习的缩放参数 τ，初始值为 0.2
        self.tau = nn.Parameter(torch.tensor(0.2))
        # 初始化可学习的游标参数 κ，初始值为 0
        self.kappa = nn.Parameter(torch.zeros(1))

        self.beta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_layers, num_concepts]
        
        Returns:
            sparse_alpha (torch.Tensor): 稀疏化后的贡献度
            mask (torch.Tensor): 稀疏化掩码
        """
        batch_size, num_layers, num_concepts = x.size()
        
        # 计算每个层概念激活的绝对重要性
        abs_importance = torch.abs(x)  # [batch_size, num_layers, num_concepts]
        
        # 计算每个样本每个层的最大值和最小值
        p_max = torch.max(abs_importance, dim=1, keepdim=True).values  # [batch_size, 1, num_concepts]
        p_min = torch.min(abs_importance, dim=1, keepdim=True).values  # [batch_size, 1, num_concepts]
        
        # 计算阈值 t
        threshold = torch.sigmoid(self.kappa) * (p_max - p_min) + p_min  # [batch_size, 1, num_concepts]
        # print(threshold.shape)
        # print(f"稀疏贡献度阈值: {threshold[0,0,:]}") 
        adjustment =self.tau*torch.exp( (abs_importance - threshold))  # [batch_size, num_layers, num_concepts]
        
        # 计算最终的贡献度 α
        alpha = adjustment * x  # [batch_size, num_layers, num_concepts]
        # 创建掩码，筛选出大于等于阈值的元素
        mask = (abs_importance >= threshold).float()  # [batch_size, num_layers, num_concepts]
        # softmask
        # mask = torch.sigmoid(self.beta * (abs_importance - threshold))
        reg_loss = torch.mean(mask) 
        # 应用掩码，得到稀疏化后的贡献度
        sparse_alpha = alpha * mask  # [batch_size, num_layers, num_concepts]
        
        return sparse_alpha, mask,reg_loss, threshold

class SparseConceptSpecificAttentionPreAgg(nn.Module):
    def __init__(self, num_concepts=34):
        super(SparseConceptSpecificAttentionPreAgg, self).__init__()
        self.num_concepts = num_concepts
    
        
        # 线性层用于计算每个概念的注意力分数
        self.attention_fc = None  # 将在前向传播中初始化

        self.attention_fc = nn.LazyLinear(self.num_concepts)
        
        # 稀疏化模块
        self.sparse = SparseContributionPreAgg()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_layers, feature_dim]
        
        Returns:
            compressed (torch.Tensor): 稀疏化后的压缩特征，形状 [batch_size, feature_dim]
            mask (torch.Tensor): 稀疏化掩码，形状 [batch_size, num_layers, num_concepts]
            
        """
        batch_size, num_layers, feature_dim = x.size()

        # 重塑 x 为 [batch_size * num_layers, feature_dim]
        x_reshaped = x.view(-1, feature_dim)
        
        # 计算原始注意力分数
        attention_scores = self.attention_fc(x_reshaped)  # [batch_size * num_layers, num_concepts]
        
        # 重塑回 [batch_size, num_layers, num_concepts]
        attention_scores = attention_scores.view(batch_size, num_layers, self.num_concepts)  # [batch_size, num_layers, num_concepts]

        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_layers, num_concepts]
        # 对每一层的注意力权重进行稀疏化
        sparse_alpha, mask,reg_loss,threshold = self.sparse(attention_weights)  # [batch_size, num_layers, num_concepts], [batch_size, num_layers, num_concepts]
        
        # sparsity = calculate_overall_sparsity(mask)
        # print(f"整体稀疏度: {sparsity:.2f}")  # 输出: 整体稀疏度: 0.33
        # layerwise_sparsity = calculate_layerwise_sparsity(mask)
        # print(f"层级稀疏度: {layerwise_sparsity}")  # 输出: 层级稀疏度: [0.25, 0.25]
        # conceptwise_sparsity = calculate_conceptwise_sparsity(mask)
        # print(f"概念级稀疏度: {conceptwise_sparsity}")  # 输出: 概念级稀疏度: [0.0, 0.25, 0.25]

        # 重新调整 attention_weights，使其与稀疏化后的 α 对齐
        # 重新归一化注意力权重，使其在层维度上和为 1（防止稀疏化后总和小于1）
        sparse_attention = sparse_alpha / (sparse_alpha.sum(dim=1, keepdim=True) + 1e-8)  # [batch_size, num_layers, num_concepts]
        # 使用稀疏化后的注意力权重对输入进行加权求和
        compressed = torch.sum(x * sparse_attention, dim=1)  # [batch_size, feature_dim]
        # print(compressed.shape)
        
        return compressed, mask,reg_loss,threshold

def calculate_overall_sparsity(mask):
    """
    计算整体稀疏度。
    
    Args:
        mask (torch.Tensor): 稀疏化掩码，形状为 [batch_size, num_layers, num_concepts]
    
    Returns:
        float: 稀疏度（0到1之间）
    """
    total_elements = mask.numel()
    zero_elements = (mask == 0).sum().item()
    sparsity = zero_elements / total_elements
    return sparsity


def calculate_layerwise_sparsity(mask):
    """
    计算每一层的稀疏度。
    
    Args:
        mask (torch.Tensor): 稀疏化掩码，形状为 [batch_size, num_layers, num_concepts]
    
    Returns:
        list of float: 每一层的稀疏度
    """
    # 计算每个样本每层的稀疏度
    layer_sparsity = (mask == 0).float().mean(dim=2)  # [batch_size, num_layers]
    # 平均所有样本的每层稀疏度
    average_layer_sparsity = layer_sparsity.mean(dim=0)  # [num_layers]
    return average_layer_sparsity.tolist()

def calculate_conceptwise_sparsity(mask):
    """
    计算每个概念的稀疏度。
    
    Args:
        mask (torch.Tensor): 稀疏化掩码，形状为 [batch_size, num_layers, num_concepts]
    
    Returns:
        list of float: 每个概念的稀疏度
    """
    # 计算每个概念的稀疏度
    concept_sparsity = (mask == 0).float().mean(dim=(0, 1))  # [num_concepts]
    return concept_sparsity.tolist()

class AvgPoolProjector(nn.Module):
    def __init__(
        self,
        layer_num: int = 2,
        mm_hidden_size: int = 768,  # Same as input feature dimension
        llm_hidden_size: int = 512,  # Desired output feature dimension
        num_attributes: int = 7
    ):
        super().__init__()
        self.layer_num = layer_num
        self.mm_hidden_size = mm_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.num_attributes = num_attributes
        self.build_net()

    def build_net(self):
        # We want to reduce the sequence length (196) to 7
        self.sampler = nn.AdaptiveAvgPool1d(self.num_attributes)  # Reduce the sequence length to 7

        # Define the MLP projector
        modules = [nn.Linear(self.mm_hidden_size, self.llm_hidden_size)]
        for _ in range(1, self.layer_num):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        self.mlp_projector = nn.Sequential(*modules)

    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, h_dim = visual_feat.shape  # 128, 196, 768
        reshaped_visual_feat = rearrange(visual_feat, "b n d -> b d n")  # [128, 768, 196]
        
        # Apply adaptive average pooling along the sequence dimension
        pooled_visual_feat = self.sampler(reshaped_visual_feat)  # [128, 768, 7]
        
        # Apply MLP projector to reduce feature dimension to 512
        output_feat = self.mlp_projector(pooled_visual_feat.transpose(1, 2))  # [128, 7, 512]
        
        return output_feat

class AdaptiveWeightedAggregation(nn.Module):
    def __init__(self, num_layers, feature_dim):
        super(AdaptiveWeightedAggregation, self).__init__()
        # 可训练的权重参数，用来为每一层加权，权重的形状是 [num_layers]
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)  # 初始化为均匀分布

        # 如果需要分类器，可以取消注释以下两行并根据需要调整输出类别数
        # self.classifier = nn.Linear(feature_dim, num_classes)  # 输出类别数

    def forward(self, layer_features):
        """
        layer_features: 张量，形状为 (batch_size, num_layers, feature_dim)
        """
        # 确保输入的形状正确
        assert layer_features.dim() == 3, "Input tensor must be 3-dimensional (batch_size, num_layers, feature_dim)"
        batch_size, num_layers, feature_dim = layer_features.size()
        
        # 验证权重的数量是否匹配
        assert num_layers == self.weights.size(0), "Number of layers in input must match number of weights"

        # 先对每个层的特征进行加权
        # 将权重扩展到 (1, num_layers, 1) 以便广播
        weights = self.weights.view(1, num_layers, 1)  # 形状: (1, 13, 1)
        weighted_features = layer_features * weights  # 形状: (128, 13, 34)

        # 将加权后的特征沿着层的维度进行聚合
        aggregated_features = torch.sum(weighted_features, dim=1)  # 聚合后形状: (128, 34)

        # 如果需要分类器，可以取消注释以下两行
        # logits = self.classifier(aggregated_features)  # 形状: (128, num_classes)
        # return logits

        return aggregated_features  # 返回形状: (128, 34)

class EntropyWeightedAggregation(nn.Module):
    def __init__(self, feature_dim, num_classes):
        """
        初始化 EntropyWeightedAggregation 模块。

        Args:
            feature_dim (int): 特征向量的维度（例如，34）。
            num_classes (int): 分类任务的类别数（例如，7）。
        """
        super(EntropyWeightedAggregation, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.epsilon = 1e-6  # 防止除零的小常数

    def compute_entropy(self, probabilities):
        """
        计算概率分布的熵值。

        Args:
            probabilities (Tensor): 形状为 [batch_size, num_layers, feature_dim] 的张量。

        Returns:
            Tensor: 形状为 [batch_size, num_layers] 的熵值。
        """
        epsilon = 1e-8  # 防止 log(0)
        probabilities = torch.clamp(probabilities, min=epsilon, max=1 - epsilon)
        entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)  # [batch_size, num_layers]
        return entropy

    def forward(self, scores):
        """
        前向传播过程。

        Args:
            scores (Tensor): 输入得分，形状为 [batch_size, num_layers, feature_dim]。

        Returns:
            logits (Tensor): 分类的 logits，形状为 [batch_size, num_classes]。
            weighted_confidence (Tensor): 加权聚合后的置信度得分，形状为 [batch_size, feature_dim]。
            max_confidence (Tensor): 每个特征维度上的最大置信度得分，形状为 [batch_size, feature_dim]。
        """
        # 1. Sigmoid 归一化
        confidence_scores = torch.sigmoid(scores)  # [batch_size, num_layers, feature_dim]

        # 2. 熵计算
        entropy_scores = self.compute_entropy(confidence_scores)  # [batch_size, num_layers]
        # print(entropy_scores)

        # 3. 计算权重（逆熵）
        weights = 1.0 / (entropy_scores + self.epsilon)  # [batch_size, num_layers]

        # 4. 加权平均聚合
        weighted_confidence = (confidence_scores * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)  # [batch_size, feature_dim]

        # 5. 最大得分选择
        max_confidence, _ = torch.max(confidence_scores, dim=1)  # [batch_size, feature_dim]

        # 6. 分类
        logits = self.classifier(weighted_confidence)  # [batch_size, num_classes]

        return logits, weighted_confidence, max_confidence

class FeatureSignificanceSelector(nn.Module):
    def __init__(self, input_dim=768, reduced_dim=512, top_k=7):
        super(FeatureSignificanceSelector, self).__init__()
        self.input_dim = input_dim
        self.reduced_dim = reduced_dim
        self.top_k = top_k
        
        self.linear = nn.Linear(input_dim, reduced_dim)

        self.score_layer = nn.Linear(reduced_dim, 1)
        
    def forward(self, x):
        """
        x: 输入张量，形状 (batch_size=96, seq_length=196, input_dim=768)
        返回:
            selected_features: 形状 (batch_size=96, top_k=7, reduced_dim=512)
        """
        # 降维
        x_reduced = self.linear(x)  # 形状: (96, 196, 512)
        
        # 计算显著性得分
        scores = self.score_layer(x_reduced).squeeze(-1)  # 形状: (96, 196)
        
        # 选择得分最高的 top_k 特征
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)  # 形状: (96, 7)
        
        # 根据索引选择特征
        batch_size = x_reduced.size(0)
        selected_features = x_reduced[torch.arange(batch_size).unsqueeze(1), topk_indices]  # 形状: (96, 7, 512)
        
        return selected_features

    
class AttentionWeightedAggregation(nn.Module):
    def __init__(self, num_layers, feature_dim, num_concepts, num_classes=1, num_heads=2):
        """
        初始化注意力加权聚合模块。

        参数：
        - num_layers (int): 输入特征的层数（例如11）。
        - feature_dim (int): 每个特征的维度（例如34）。
        - num_concepts (int): 概念数量（例如34）。
        - num_classes (int): 每个概念的分类类别数量（默认为1，即每个概念输出一个标量）。
        - num_heads (int): 多头注意力的头数（默认为2）。确保 feature_dim 能被 num_heads 整除。
        """
        super(AttentionWeightedAggregation, self).__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim  # 34
        self.num_concepts = num_concepts  # 34
        self.num_classes = num_classes  # 1
        self.num_heads = num_heads  # 注意力头数

        self.query = nn.Parameter(torch.randn(num_concepts, feature_dim))

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        前向传播。

        参数：
        - x (tensor): 输入张量，形状为 [B, num_layers, feature_dim]，例如 [128, 11, 34]。

        返回：
        - logits (tensor): 形状为 [B, num_concepts] 的分类结果。
        - aggregated_features (tensor): 形状为 [B, num_concepts, feature_dim] 的聚合特征。
        - attn_weights (tensor): 形状为 [B, num_concepts, num_layers] 的注意力权重。
        """
        B, num_layers, F = x.shape  # B=128, num_layers=11, F=34
        assert F == self.feature_dim, f"Feature dimension mismatch: expected {self.feature_dim}, got {F}"

        query = self.query.unsqueeze(0).expand(B, -1, -1)  # [128,34,34]
        attn_output, attn_weights = self.multihead_attn(query=query, key=x, value=x)  # [128,34,34], [128,34,11]
        logits = self.classifier(attn_output)  # [128,34,1]

        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # [128,34]
        else:
            logits = logits  # 保持 [128,34,num_classes]

        aggregated_features = attn_output  # [128,34,34]

        return logits

class ConceptSpecificAttention(nn.Module):
    def __init__(self, num_concepts=34):
        super(ConceptSpecificAttention, self).__init__()
        self.num_concepts = num_concepts
        
        # Linear layer to compute attention scores for each concept
        self.attention_fc = None  # We will initialize this in the forward pass
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_layers, feature_dim)
        """
        batch_size, num_layers, feature_dim = x.size()
        
        # Initialize the linear layer if it hasn't been initialized yet
        if self.attention_fc is None:
            self.attention_fc = nn.Linear(feature_dim, self.num_concepts).to(x.device)
        
        # Reshape x to (batch_size * num_layers, feature_dim)
        x_reshaped = x.view(-1, feature_dim)
        
        # Compute raw attention scores
        attention_scores = self.attention_fc(x_reshaped).to(x.device)  # (batch_size * num_layers, num_concepts)
        
        # Reshape back to (batch_size, num_layers, num_concepts)
        attention_scores = attention_scores.view(batch_size, num_layers, self.num_concepts).to(x.device)
        
        # Apply softmax on the layer dimension
        attention_weights = F.softmax(attention_scores, dim=1).to(x.device)  # (batch_size, num_layers, num_concepts)
        
        # Weighted sum over layers
        compressed = torch.sum(x * attention_weights, dim=1).to(x.device)  # (batch_size, feature_dim)
        
        return compressed

class ImportanceThresholding(torch.nn.Module):
    def __init__(self):
        super(ImportanceThresholding, self).__init__()
        
        # 初始化可学习的参数
        self.tau = torch.nn.Parameter(torch.tensor( 0.2)
         )
         # 缩放参数 tau
        
        # 如果 kappa 需要被使用，请取消下面这行的注释：
        # self.kappa = Parameter(torch.zeros(1))  # 可学习的阈值缩放参数 kappa
      
    def forward(self, p_l):
    
        global_feat_importance = p_l/ self.tau 
        alpha_l = F.softmax(global_feat_importance, dim=-1)
        return alpha_l
    
def count_keys_and_values(data_dict):
    num_keys = len(data_dict)
    total_values = sum(len(values) for values in data_dict.values())
    
    # 打印详细信息
    # for key, values in data_dict.items():
    #     print(f"Key '{key}' has {len(values)} values.")
    
    return num_keys, total_values


class mvpcbm(nn.Module):  
    def __init__(self, concept_list, model_name='openclip', config=None):
        super().__init__()

        self.concept_list = concept_list
        self.model_name = model_name
        self.config = config
       
        self.ImportanceThresholding = ImportanceThresholding()
        self.SparseContribution = SparseContribution()
        if self.model_name in ['biomedclip', 'openclip','bioclip']:
            if self.model_name == 'biomedclip':
                self.model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            elif self.model_name == 'openclip': 
                self.model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
                self.tokenizer = get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            elif self.model_name == 'bioclip':
                self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
                self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
           
            config.preprocess = preprocess
            self.model.cuda()
            
            concept_keys = list(concept_list.keys())
            num_attributes, num_concepts = count_keys_and_values(concept_list)

            self.concept_token_dict = {}
            self.global_attr_concepts = {}
            
            for key in concept_keys:
                if config.dataset == 'isic2018':
                    prefix = f"this is a dermoscopic image, the {key} of the lesion is "
                if config.dataset == 'cmmd':
                    prefix = f"this is a x-ray image, the {key} of the breast cancer is "
                if config.dataset == 'mimic_cxr':
                    prefix = f"this is a x-ray image, the {key} of the lesion is "
                if config.dataset == 'idrid':
                    prefix = f"this is a Diabetic Retinopathy, the {key} of the lesion is "
                if config.dataset == 'busi':
                    prefix = f"this is a ultrasound image, the {key} of the lesion is "
                if config.dataset == 'cm':
                    prefix = f"this is a x-ray image, the {key} of the lesion is "
                if config.dataset == 'nct':
                    prefix = f"this is a histopathological image, the {key} of the lesion is "
                if config.dataset == 'siim':
                    prefix = f"this is a x-ray image, the {key} of the pneumothorax  lesion is "
                attr_concept_list = concept_list[key]
                prefix_attr_concept_list = [prefix + concept for concept in attr_concept_list]
                global_attr_concept = prefix+ f"{' '.join(attr_concept_list)}"
                
                tmp_global_attr_concept = self.tokenizer(global_attr_concept).cuda()
                _, tmp_global_attr_concept_feats, global_logit_scale = self.model(None, tmp_global_attr_concept)
                self.global_attr_concepts[key] = tmp_global_attr_concept_feats.detach()

                tmp_concept_text = self.tokenizer(prefix_attr_concept_list).cuda()
                _, tmp_concept_feats, logit_scale = self.model(None, tmp_concept_text)
                self.concept_token_dict[key] = tmp_concept_feats.detach()

            self.logit_scale = logit_scale.detach()
            self.global_logit_scale = global_logit_scale.detach()
        
        self.visual_features = []
        self.Avg=AvgPoolProjector(num_attributes)

        self.hook_list = []
        def hook_fn(module, input, output):
            self.visual_features.append(output) # detach to aboid saving computation graph
                                                 # might need to remove if finetune the full model
                                                 
        for i in range(len(self.model.visual.trunk.blocks)):  # i 从 0 到 11
            layer = self.model.visual.trunk.blocks[i]  # 获取第 i 层
            self.hook_list.append(layer.register_forward_hook(hook_fn))  # 注册钩子
       
        
        self.fc_confidence = nn.Linear(768, 512) 
    
        self.visual_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(7, 768)))
        self.attention_fc = None  # 将在前向传播中初始化

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        self.ffn = FFN(768, 768*4)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(in_features=768, out_features=512, bias=False)

        self.cls_head = nn.Linear(in_features=num_concepts, out_features=config.num_class)
        self.SparseConceptSpecificAttention = SparseConceptSpecificAttention(num_concepts)
        self.SparseConceptSpecificAttentionPreAgg = SparseConceptSpecificAttentionPreAgg(num_concepts)
        
        self.AdaptiveWeightedAggregation = AdaptiveWeightedAggregation(num_layers = len(self.model.visual.trunk.blocks),feature_dim = num_concepts)
        self.AttentionWeightedAggregation = AttentionWeightedAggregation(num_layers = len(self.model.visual.trunk.blocks),feature_dim = num_concepts,num_concepts=num_concepts,num_classes=1,num_heads=2)
        
        for param in self.model.text.parameters():
            param.requires_grad = False
        for param in self.model.visual.trunk.parameters():
            param.requires_grad = True
        
        self.visual_tokens.requires_grad = True
    
    def get_backbone_params(self):
        return self.model.visual.trunk.parameters()
    def get_bridge_params(self):
        param_list = []
        
        param_list.append(self.visual_tokens)
        for param in self.cross_attn.parameters():
            param_list.append(param)
        for param in self.ffn.parameters():
            param_list.append(param)
        for param in self.norm.parameters():
            param_list.append(param)
        for param in self.proj.parameters():
            param_list.append(param)
        for param in self.cls_head.parameters():
            param_list.append(param)
        return param_list


    def forward(self, imgs):
        
        self.visual_features.clear()
        img_feats, _, _ = self.model(imgs, None)
        
        layer_feat = []
        global_feat = []

        cpm = []
        for i in range(len(self.model.visual.trunk.blocks)):
            # 获取第 i 层的特征
            img_feat_map = self.visual_features[i][:, 1:, :]  # 第 i 层的特征图
            B, _, _ = img_feat_map.shape

            # global class token
            class_token = self.visual_features[i][:, 0, :] 
            class_feat = self.fc_confidence(class_token).unsqueeze(1)
            # print(class_feat.shape)
            # [128, 1, 512]

            # Preference
            globa_feat_important = []
            for key in self.global_attr_concepts.keys():
                global_sim = (class_feat @ self.global_attr_concepts[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
                globa_feat_important.append(global_sim)
            
            globa_feat_important = torch.cat(globa_feat_important, dim=1)
            globa_feat_important = torch.sigmoid(globa_feat_important)
 
            cpm.append(globa_feat_important)
            img_feat_map = self.Avg(img_feat_map)

            # Scaling
            globa_Preference = self.ImportanceThresholding(globa_feat_important)

            image_logits_dict = {}
            idx = 0
            for key in self.concept_token_dict.keys():     
                image_logits_dict[key] =globa_Preference[:, idx:idx+1]*(self.logit_scale *img_feat_map[:, idx:idx+1, :] @ self.concept_token_dict[key].repeat(B, 1, 1).permute(0, 2, 1)).squeeze(1)
                idx += 1
        
            image_logits_list = []
            for key in image_logits_dict.keys():
                image_logits_list.append(image_logits_dict[key])
            image_logits = torch.cat(image_logits_list, dim=-1)
            layer_feat.append(image_logits)
    

        layer_score = torch.stack(layer_feat, dim=1) 
        image_logits,mask,reg_loss,threshold = self.SparseConceptSpecificAttentionPreAgg(layer_score)
       
        image_logits_dict = {}
        idx = 0
        for key in self.concept_token_dict.keys():                
            image_logits_dict[key] = image_logits[:, idx]
            idx += 1
        
        cls_logits = self.cls_head(image_logits)

        return cls_logits, image_logits_dict, reg_loss
           
  
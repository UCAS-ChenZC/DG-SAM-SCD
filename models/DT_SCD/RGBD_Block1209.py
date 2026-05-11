"""
DFormerv2 Attention Module - Standalone Version
将DFormerv2的核心注意力机制提取为独立模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class DWConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """input (b h w c)"""
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


def angle_transform(x, sin, cos):
    """旋转位置编码变换"""
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    """几何先验生成器"""
    def __init__(self, embed_dim, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        # 1. 位置编码的频率参数（类似Transformer正弦编码）
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # shape: [embed_dim//num_heads]
        
        # 2. 可学习权重：平衡空间距离和深度距离
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        # weight[0]: 空间距离权重
        # weight[1]: 深度距离权重
        
        # 3. 衰减系数：不同注意力头的衰减速度不同
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        # shape: [num_heads]
        # 例如8个头：[-2.0, -2.57, -3.14, -3.71, -4.29, -4.86, -5.43, -6.0]
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """生成基于深度差异的注意力偏置"""
        B, _, H, W = depth_grid.shape
        
        # 1. 展平深度图
        grid_d = depth_grid.reshape(B, H * W, 1)
        
        # 2. 计算任意两点的深度差
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        
        # 3. 应用衰减系数（深度差越大，注意力权重越低）
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        # shape: [B, num_heads, H*W, H*W]
        return mask_d
        #物理意义：深度差为0 → mask=0 → 注意力增强  深度差大 → mask负值大 → 注意力抑制

    def generate_pos_decay(self, H: int, W: int):
        """生成基于欧式距离的注意力偏置"""
        # 1. 生成坐标网格
        index_h = torch.arange(H).to(self.decay)    # [0, 1, ..., H-1]
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  
        
         # 2. 计算曼哈顿距离（L1距离）
        mask = grid[:, None, :] - grid[None, :, :]  # [H*W, H*W, 2]
        mask = (mask.abs()).sum(dim=-1) # [H*W, H*W]
        
        # 3. 应用衰减
        mask = mask * self.decay[:, None, None] # [num_heads, H*W, H*W]
        return mask
        #设计优势：使用曼哈顿距离而非欧式距离（计算更快）    不同头关注不同空间范围

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """生成行/列方向的深度衰减（用于分离式注意力）"""
        # depth_grid: [B, C, H, W]
        
         # 计算同一行/列内不同位置的深度差
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        # shape: [B, C, H, W, W] 或 [B, C, W, H, H]
        
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        return mask
        #用途：支持分离式轴向注意力（Axial Attention）  降低计算复杂度：O(HW)² → O(H²W + HW²)

    def generate_1d_decay(self, l: int):
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        # 1. 调整深度图尺寸
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:    # 分离式注意力
            # 2a. 生成位置编码 (2D--> 1D sin/cos 拆分)
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            # reshape成 (H, W, embed_dim//2)

            # 3a. 生成行/列方向的注意力偏置 (结合空间和深度信息)    分别计算水平和垂直方向的mask
            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            # 4a. 融合空间和深度的注意力偏置（信息）
            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))
            
        else:   # 标准全局注意力
            # 2b. 生成2D位置编码
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            
            # 3b. 生成空间距离的注意力偏置（计算全局mask）
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            
            # 4b. 融合空间和深度的注意力偏置（信息）    加权融合
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior


class DFormerv2Attention(nn.Module):
    """
    DFormerv2 几何自注意力模块 - 独立版本
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        use_decomposed: 是否使用分解注意力 (True=Decomposed_GSA, False=Full_GSA)
        initial_value: 几何先验初始值
        heads_range: 头范围
        value_factor: 值投影的扩展因子
    """
    def __init__(
        self, 
        embed_dim=128,  # 特征维度
        num_heads=8,    # 注意力头数
        use_decomposed=True,    # 是否使用分解注意力
        initial_value=2,    # 几何先验初始衰减值
        heads_range=4,  # 不同头的衰减范围      改为6？？
        value_factor=1,  # Value通道扩展因子
        allow_dynamic=True  # 🆕 新增参数：是否允许动态切换
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.default_mode = use_decomposed  # 🆕 保存默认模式
        self.allow_dynamic = allow_dynamic   # 🆕 动态模式开关
        self.value_factor = value_factor
        
        # 1. 维度分配
        self.head_dim = self.embed_dim * self.value_factor // num_heads     # 每个头的Value维度
        self.key_dim = self.embed_dim // num_heads                          # 每个头的Key维度
        self.scaling = self.key_dim ** -0.5                                 # 缩放因子
        
        # 2. Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.value_factor, bias=True)
        
        # 位置编码
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)     # 可学习位置编码 (LEPE)
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)  # 卷积位置编码
        # 为什么用DWConv做位置编码？ 因为深度可分离卷积可以捕捉局部空间信息，增强位置感知能力。 传统的正弦位置编码是全局的，而DWConv可以更好地适应图像的局部结构变化。
        # 参数量小但能有效补充全局注意力的局部感知能力
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim * self.value_factor, embed_dim, bias=True)
        
        # 几何先验生成器
        self.geo_prior_gen = GeoPriorGen(embed_dim, num_heads, initial_value, heads_range)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def decomposed_attention(self, x, rel_pos):
        """分解的几何自注意力"""
        bsz, h, w, _ = x.size()
        (sin, cos), (mask_h, mask_w) = rel_pos

        # ========== 1. QKV投影 ==========
        q = self.q_proj(x)  # [B, H, W, C]
        k = self.k_proj(x)  # [B, H, W, C]
        v = self.v_proj(x)  # [B, H, W, C*value_factor]
        # 可学习位置编码 (对Value)
        lepe = self.lepe(v) # [B, H, W, C*value_factor]

        # ========== 2. 重塑为多头格式 ==========
        k = k * self.scaling    # 缩放Key
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, H, W, key_dim]
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, H, W, key_dim]
        
        # ========== 3. 旋转位置编码 (RoPE) ==========
        qr = angle_transform(q, sin, cos)   # 根据位置旋转Query
        kr = angle_transform(k, sin, cos)   # 根据位置旋转Key
        # 作用: 使得QK内积天然包含相对位置信息

        # ========== 4. 宽度方向注意力 ==========
        qr_w = qr.transpose(1, 2)   # [B, H, num_heads, W, key_dim]
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) # [B, H, num_heads, W, key_dim]

        
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)    # [B, H, num_heads, W, W]
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)    # 加入深度先验
        qk_mat_w = torch.softmax(qk_mat_w, -1)
        v = torch.matmul(qk_mat_w, v)   # [B, H, num_heads, W, head_dim]

        # ========== 5. 高度方向注意力 ==========
        qr_h = qr.permute(0, 3, 1, 2, 4)    # [B, W, num_heads, H, key_dim]
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)    # [B, W, num_heads, H, key_dim]

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)    # [B, W, num_heads, H, H]
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)    # 加入深度先验
        qk_mat_h = torch.softmax(qk_mat_h, -1)
        output = torch.matmul(qk_mat_h, v)  # [B, W, num_heads, H, head_dim]

        # ========== 6. 合并输出 ==========
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # [B, H, W, C*value_factor]
        output = output + lepe  # 加入LEPE
        output = self.out_proj(output)   # 投影回原始维度
        return output
        # 分解注意力的优势：降低计算复杂度  O((HW)²) → O(H²W + HW²)  适合高分辨率输入
        # 但可能略微损失全局上下文捕捉能力
        # 标准全局注意力: O(H²W²) = O(N²)   分解轴向注意力: O(H·W² + W·H²) = O(HW(H+W)) ≈ O(N√N)
        # 例如 128×128 输入:    - 全局: 268M FLOPS  - 分解: 8.4M FLOPS (节省97%)
        
    def full_attention(self, x, rel_pos):
        """完整的几何自注意力"""
        bsz, h, w, _ = x.size()
        (sin, cos), mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        qr = angle_transform(q, sin, cos)
        kr = angle_transform(k, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = vr.flatten(2, 3)
        
        qk_mat = qr @ kr.transpose(-1, -2)
        qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)
        
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

    def forward(self, x, depth_map,  force_mode=None):
        """
        前向传播
        
        Args:
            x: RGB特征 (B, C, H, W)
            depth_map: 深度图 (B, 1, H, W)
            
        Returns:
            输出特征 (B, C, H, W)
        """
        # 🆕 动态模式选择逻辑
        if self.allow_dynamic and force_mode is not None:
            use_decomposed = force_mode
        else:
            use_decomposed = self.default_mode

        # (B,C,H,W) → 转换为 (B, H, W, C) 格式
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        depth_map = depth_map[:, 0, :, :].unsqueeze(1)
        
        # 添加卷积位置编码
        x = x + self.cnn_pos_encode(x)  # 先注入局部空间信息
        
        # 生成几何先验
        geo_prior = self.geo_prior_gen((H, W), depth_map, split_or_not=use_decomposed)
        # 返回: ((sin, cos), mask) 或 ((sin, cos), (mask_h, mask_w))
        # 为什么先加位置编码？ 因为位置编码可以帮助模型更好地理解空间结构，尤其是在处理图像数据时。 先注入局部信息，再通过注意力机制捕捉全局关系，可以提升模型的表现力。
        # 在QKV投影前注入位置信息，让后续变换感知空间布局
        
        # 🆕 根据动态模式执行注意力
        if use_decomposed:
            x = self.decomposed_attention(x, geo_prior)
        else:
            x = self.full_attention(x, geo_prior)
        
        # 转换回 (B, C, H, W) 格式
        x = x.permute(0, 3, 1, 2).contiguous()
        
        return x

class AdaptiveDFormer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=16):
        super().__init__()  # ✅ 必须第一行
        self.attn = DFormerv2Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_decomposed=True ) # 默认模式)
        self.mode_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),                    # (B, C, 1, 1) -> (B, C)
            nn.Linear(embed_dim, 64),        # 🆕 添加隐藏层
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)                 # 输出2分类logits
        )
    
    def forward(self, x, depth):
        # 根据特征复杂度选择模式
        mode_logits = self.mode_selector(x)
        # 2. 选择模式（批次级决策）
        # argmax(dim=1) 返回 (B,)，取平均判断主要模式
        use_decomposed = mode_logits.argmax(dim=1).float().mean() > 0.5
        # 3. 执行注意力计算
        output = self.attn(x, depth, force_mode=use_decomposed)
        
        return output

class MultiStageRGBDFusion(nn.Module):
    """多阶段RGBD融合模块"""
    def __init__(self, embed_dim=128, num_heads=8, num_stages=3):
        super().__init__()
        self.num_stages = num_stages
        
        # 多个RGBD注意力模块
        self.rgbd_blocks = nn.ModuleList([
            DFormerv2Attention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                use_decomposed=True,
                initial_value=2,
                heads_range=4
            ) for _ in range(num_stages)
        ])
        
        # 每个阶段后的特征精炼卷积
        self.refine_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_stages)
        ])
        
        # 自适应融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(num_stages) / num_stages)
        
    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: (B, C, H, W)
            depth_feat: (B, C, H, W) 深度特征
        """
        outputs = []
        x = rgb_feat
        
        for i, (rgbd_block, refine_conv) in enumerate(zip(self.rgbd_blocks, self.refine_convs)):
            # RGBD融合
            fused = rgbd_block(x, depth_feat)
            
            # 残差连接
            fused = x + fused
            
            # 特征精炼
            fused = refine_conv(fused)
            
            outputs.append(fused)
            x = fused  # 级联到下一阶段
        
        # 多阶段特征加权融合
        weighted_output = sum(w * out for w, out in zip(self.fusion_weights, outputs))
        
        return weighted_output

class EnhancedRGBDFusion(nn.Module):
    """前后加卷积的增强融合"""
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        
        # 前置特征对齐卷积
        self.pre_align = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),  # DW卷积
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),  # PW卷积
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 深度特征增强
        self.depth_enhance = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.Sigmoid()  # 生成注意力权重
        )
        
        # RGBD注意力
        self.rgbd_attn = DFormerv2Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_decomposed=True,
            initial_value=2,
            heads_range=4
        )
        
        # 后置特征精炼
        self.post_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 1, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: (B, C, H, W)
            depth_feat: (B, 1, H, W) 深度图
        """
        # 1. 前置特征对齐
        rgb_aligned = self.pre_align(rgb_feat)
        
        # 2. 深度特征增强（生成空间注意力）
        # depth_attention = self.depth_enhance(depth_feat)
        # rgb_weighted = rgb_aligned * depth_attention
        
        # 3. RGBD注意力融合
        fused = self.rgbd_attn(rgb_aligned, depth_feat)
        
        # 4. 后置特征精炼
        refined = self.post_refine(fused)
        
        # 5. 门控融合原始特征和精炼特征
        gate_weight = self.gate(torch.cat([rgb_feat, refined], dim=1))
        output = gate_weight * refined + (1 - gate_weight) * rgb_feat
        
        return output
# 使用示例
if __name__ == "__main__":
    # 创建模型
    # attention = DFormerv2Attention(
    #     embed_dim=128,
    #     num_heads=8,
    #     use_decomposed=True,  # True使用分解注意力, False使用完整注意力
    #     initial_value=2,
    #     heads_range=4
    # )
    # attention = MultiStageRGBDFusion(embed_dim=128, num_heads=8, num_stages=3)
    attention = EnhancedRGBDFusion(embed_dim=128, num_heads=8)
    # 创建输入
    rgb_features = torch.randn(1, 128, 64, 64)  # (B, C, H, W)
    depth_map = torch.randn(1, 3, 512, 512)       # (B, 1, H, W)
    
    # 前向传播
    output = attention(rgb_features, depth_map)
    
    print(f"Input shape: {rgb_features.shape}")
    print(f"Depth shape: {depth_map.shape}")
    print(f"Output shape: {output.shape}")
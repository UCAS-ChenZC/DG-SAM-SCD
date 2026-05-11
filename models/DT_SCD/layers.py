import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.DT_SCD.Dysample import DySample_UP


class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=1)
        )
    def forward(self, x):
        return self.dwconv(x)


class CBA1x1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class CBA3x3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.cba = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.cba(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ECA(nn.Module):
    def __init__(self, kernal=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernal, padding=(kernal - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# class task_interaction_module(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')

#     def forward(self, old_sem1, old_sem2, old_change, change_result):
#         # old_sem1, old_sem2: 语义特征图
#         # old_change: 原始变化检测结果?????(暂不确定是啥)
#         # change_result: 变化检测网络的输出（未经过softmax）
#         # ----- 1. 基于语义特征差异生成语义变化图 -----
#         sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
#         sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
#         sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1)))
        
#         # ----- 2. 融合语义变化图到变化检测结果 -----
#         new_change = old_change * sem_out

#         # ----- 3. 语义一致性约束（软掩码版本） -----
#         b, c, h, w = old_sem1.size()
#         fea_sem1 = torch.reshape(old_sem1.permute(0, 2, 3, 1), [b*h*w, c])
#         fea_sem2 = torch.reshape(old_sem2.permute(0, 2, 3, 1), [b*h*w, c])

#         # 使用 softmax 获取变化/未变化概率
#         probs = F.softmax(change_result, dim=1)
#         p_unchange = probs[:, 0, :, :]   # 未变化概率
#         p_change = probs[:, 1, :, :]     # 变化概率

#         # 生成软标签 target in [-1, 1]
#         target = p_unchange - p_change   # 未变化高→正数，变化高→负数
#         target = target.reshape(b * h * w)

#         # CosineEmbeddingLoss 支持软标签，但需保证其范围 [-1, 1]
#         similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)

#         return new_change, similarity_loss

class New_DySampleFusion(nn.Module):
    def __init__(self,
                 in_channels_high,
                 in_channels_low,
                 out_channels,
                 scale=2,
                 style='pl',
                 groups=4,
                 dyscope=False,
                 mid_channels=128,
                 guide_low_for_upsample=True):  # 新增：是否用低层细节指导上采样
        super().__init__()

        self.mid_channels = mid_channels
        if mid_channels is None:
            mid_channels = min(in_channels_high, in_channels_low)

        self.guide_low_for_upsample = guide_low_for_upsample

        # 通道对齐
        self.proj_high = nn.Conv2d(in_channels_high, mid_channels, 1)
        self.proj_low  = nn.Conv2d(in_channels_low,  mid_channels, 1)

        # 如果要用低层引导上采样，则需要一个将 [high, low_down] -> mid 的融合层
        if guide_low_for_upsample:
            # high_p: [B, mid, H, W]
            # low_down: [B, mid, H, W]
            # concat -> [B, 2*mid, H, W] -> [B, mid, H, W]
            self.upsample_cond_proj = nn.Sequential(
                nn.Conv2d(mid_channels * 2, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )

        # DySample_UP 的输入通道数仍是 mid_channels
        self.upsampler = DySample_UP(in_channels=mid_channels,
                                     scale=scale,
                                     style=style,
                                     groups=groups,
                                     dyscope=dyscope)

        # 门控
        self.gate = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_high, x_low):
        # 1) 通道对齐
        if x_high.shape[1] == self.mid_channels:
            x_high_p = x_high
            x_low_p  = self.proj_low(x_low)
        elif x_low.shape[1] == self.mid_channels:
            x_high_p = self.proj_high(x_high)
            x_low_p  = x_low
        else:
            x_high_p = self.proj_high(x_high)
            x_low_p  = self.proj_low(x_low)

        B, C, H, W = x_high_p.shape
        _, _, H2, W2 = x_low_p.shape  # H2 = 2H, W2 = 2W

        # 2) 如果需要，先把低分辨率特征下采到 H×W 做“细节指导”
        if self.guide_low_for_upsample:
            # 下采到与 x_high_p 尺寸匹配
            x_low_down = F.interpolate(
                x_low_p, size=(H, W),
                mode='bilinear', align_corners=False
            )  # [B, mid, H, W]

            # 组合 high 语义 和 low 细节，引导上采样 offset
            cond_feat = torch.cat([x_high_p, x_low_down], dim=1)  # [B, 2*mid, H, W]
            cond_feat = self.upsample_cond_proj(cond_feat)        # [B, mid, H, W]

            # 用“语义+细节”的条件特征做 DySample 上采样
            x_high_up = self.upsampler(cond_feat)                 # [B, mid, 2H, 2W]
        else:
            # 原来的做法：只用 high 自己做 offset 预测
            x_high_up = self.upsampler(x_high_p)                  # [B, mid, 2H, 2W]

        # 3) 门控融合（和你原来一样）
        gate_input = torch.cat([x_high_up, x_low_p], dim=1)  # [B, 2*mid, 2H, 2W]
        alpha = self.gate(gate_input)                        # [B, mid, 2H, 2W]

        fused = alpha * x_high_up + (1.0 - alpha) * x_low_p  # [B, mid, 2H, 2W]
        out = self.out_conv(fused)

        return out

class New_task_interaction_module(nn.Module):
    def __init__(self, sampling_ratio=0.05):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')
        self.alpha = nn.Parameter(torch.tensor(0.3))       # gating（可学习）
        self.sampling_ratio = sampling_ratio

    def forward(self, x1, x2, xc, change_logits):
        # ---------- Sem → Change ----------
        sem_max = torch.max(torch.abs(x1 - x2), dim=1, keepdim=True)[0]
        sem_avg = torch.mean(torch.abs(x1 - x2), dim=1, keepdim=True)
        sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max, sem_avg], dim=1)))

        # gating 后的交互
        new_xc = xc + self.alpha * (xc * sem_out)

        # ---------- Change → Sem ----------
        B, C, H, W = x1.shape
        fea1 = F.normalize(x1.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)
        fea2 = F.normalize(x2.permute(0, 2, 3, 1).reshape(-1, C), dim=-1)

        # soft target（更稳定）
        change_prob = change_logits[:, 1].reshape(-1)
        unchange_prob = change_logits[:, 0].reshape(-1)
        target = unchange_prob - change_prob

        # 稀疏采样
        sample_mask = torch.rand_like(target) < self.sampling_ratio

        loss = self.loss_f(fea1[sample_mask], fea2[sample_mask], target[sample_mask])

        return new_xc, loss


class task_interaction_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sem2Change = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')
        # self.CBA = CBA3x3(64, 128)

    def forward(self, old_sem1, old_sem2, old_change, change_result):
        sem_max_out, _ = torch.max(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_avg_out = torch.mean(torch.abs(old_sem1 - old_sem2), dim=1, keepdim=True)
        sem_out = self.sigmoid(self.Sem2Change(torch.cat([sem_max_out, sem_avg_out], dim=1)))
        new_change = old_change * sem_out
        # new_change = self.CBA(new_change)

        b, c, h, w = old_sem1.size()
        fea_sem1 = torch.reshape(old_sem1.permute(0,2,3,1), [b*h*w, c])
        fea_sem2 = torch.reshape(old_sem2.permute(0,2,3,1), [b*h*w, c])

        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        target = unchange_mask.float()
        target = target - change_mask.float()
        target = torch.reshape(target, [b * h * w])
        similarity_loss = self.loss_f(fea_sem1, fea_sem2, target)
        return new_change, similarity_loss

class Dep_Change_Specific_Transfer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel*2, in_channel)
        self.eca = ECA()
        
        # 深度差分编码：1通道 -> in_channel
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, in_channel // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.resblock = self._make_layer(ResBlock, 384, 128, 6, stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        """
        x1, x2: (B, 128, 64, 64)   RGB-D backbone提取后的双时相特征
        x3, x4: (B, 1, 512, 512)   原始深度图
        """
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        # 生成两个融合特征，这么做是为了引入时序敏感性（防止x1,x2的位置交换影响结果）
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        # 深度差分
        x3_ds = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x4_ds = F.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)
        depth_diff = torch.abs(x3_ds - x4_ds)         # (B,1,64,64)
        depth_feat = self.depth_encoder(depth_diff)   # (B,128,64,64)
        

        # 三者融合
        change = torch.cat([change, diff, depth_feat], dim=1)  # (B,384,64,64)
        change = self.resblock(change)
        return change

class advanced_depth_change_interaction(nn.Module):
    def __init__(self, depth_channels=64, change_channels=64):
        super().__init__()
        
        # === 1. 多尺度空间金字塔深度感知 ===
        self.pyramid_depths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(depth_channels, depth_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(depth_channels),
                nn.ReLU(inplace=True)
            )
            for k in [3, 5, 7]
        ])
        
        # 金字塔特征融合
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(depth_channels * 3, depth_channels, kernel_size=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True)
        )
        
        # === 2. Cross-Attention: 深度引导变化特征 ===
        self.depth_to_change_attn = nn.MultiheadAttention(
            embed_dim=depth_channels, 
            num_heads=8, 
            batch_first=True
        )
        
        # === 3. 几何一致性感知模块 ===
        self.geometry_filter = nn.Sequential(
            nn.Conv2d(depth_channels * 2 + change_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # === 4. 自适应特征融合 ===
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(change_channels * 2, change_channels, kernel_size=1),
            nn.BatchNorm2d(change_channels),
            nn.ReLU(inplace=True)
        )
        
        # === 5. 损失函数 ===
        self.depth_consistency_loss = nn.CosineEmbeddingLoss(margin=0., reduction='mean')
        self.geometry_loss = nn.L1Loss(reduction='mean')
        self.contrastive_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
    def forward(self, depth_fea1, depth_fea2, old_change, change_result):
        """
        Args:
            depth_fea1: [B, C, H, W] - 时序1的深度特征
            depth_fea2: [B, C, H, W] - 时序2的深度特征
            old_change: [B, C_change, H, W] - 原始变化特征
            change_result: [B, 2, H, W] - 变化检测结果(logits)
        Returns:
            refined_change: [B, C_change, H, W] - 细化后的变化特征
            total_loss: 标量 - 约束损失
        """
        b, c, h, w = depth_fea1.size()
        
        # ======== 策略1: 多尺度深度差异提取 ========
        depth_diff = torch.abs(depth_fea1 - depth_fea2)
        multi_scale_depth = []
        for pyramid_conv in self.pyramid_depths:
            multi_scale_depth.append(pyramid_conv(depth_diff))
        
        # 融合多尺度特征
        depth_diff_fused = self.pyramid_fusion(torch.cat(multi_scale_depth, dim=1))
        
        # ======== 策略2: Transformer全局交互 ========
        # 将特征展平为序列
        depth_tokens = depth_diff_fused.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        change_tokens = old_change.flatten(2).permute(0, 2, 1)       # [B, H*W, C]
        
        # Cross-Attention: 深度特征作为Key/Value，变化特征作为Query
        enhanced_change_tokens, attn_weights = self.depth_to_change_attn(
            query=change_tokens,
            key=depth_tokens,
            value=depth_tokens
        )
        
        # 恢复空间维度
        enhanced_change = enhanced_change_tokens.permute(0, 2, 1).reshape(b, c, h, w)
        
        # ======== 策略3: 几何一致性过滤 ========
        geometry_input = torch.cat([depth_fea1, depth_fea2, old_change], dim=1)
        geometry_mask = self.geometry_filter(geometry_input)
        
        # ======== 策略4: 自适应融合 ========
        # 融合原始变化特征和增强后的特征
        fused_change = self.adaptive_fusion(
            torch.cat([old_change, enhanced_change], dim=1)
        )
        
        # 应用几何mask
        refined_change = fused_change * geometry_mask
        
        # ======== 损失计算 ========
        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        
        # Loss 1: 深度特征一致性损失
        fea_depth1 = depth_fea1.permute(0,2,3,1).reshape(b*h*w, c)
        fea_depth2 = depth_fea2.permute(0,2,3,1).reshape(b*h*w, c)
        target = unchange_mask.float() - change_mask.float()
        target = target.reshape(b * h * w)
        depth_consistency_loss = self.depth_consistency_loss(fea_depth1, fea_depth2, target)
        
        # Loss 2: 几何差异显著性损失
        depth_diff_map = torch.norm(depth_fea1 - depth_fea2, p=2, dim=1)
        change_mask_float = change_mask.float()
        geometry_loss = self.geometry_loss(
            depth_diff_map * change_mask_float,
            torch.ones_like(depth_diff_map) * change_mask_float
        )
        
        # Loss 3: 注意力对齐损失（新增）
        # 确保attention weights与变化区域对齐
        attn_map = attn_weights.mean(dim=1).reshape(b, h, w)  # [B, H, W]
        alignment_loss = F.binary_cross_entropy(
            attn_map,
            change_mask.float(),
            reduction='mean'
        )
        
        # Loss 4: 对比学习损失（可选，增强判别性）
        # 采样变化/未变化区域的特征进行对比
        if self.training:
            # 简化版：仅在训练时计算
            change_indices = change_mask.nonzero(as_tuple=False)
            unchange_indices = unchange_mask.nonzero(as_tuple=False)
            
            if len(change_indices) > 0 and len(unchange_indices) > 0:
                # 随机采样
                num_samples = min(256, len(change_indices), len(unchange_indices))
                change_samples = change_indices[torch.randperm(len(change_indices))[:num_samples]]
                unchange_samples = unchange_indices[torch.randperm(len(unchange_indices))[:num_samples]]
                
                # 提取特征
                anchor = enhanced_change[change_samples[:, 0], :, change_samples[:, 1], change_samples[:, 2]]
                positive = depth_diff_fused[change_samples[:, 0], :, change_samples[:, 1], change_samples[:, 2]]
                negative = depth_diff_fused[unchange_samples[:, 0], :, unchange_samples[:, 1], unchange_samples[:, 2]]
                
                contrastive_loss = self.contrastive_loss(anchor, positive, negative)
            else:
                contrastive_loss = torch.tensor(0.0, device=depth_fea1.device)
        else:
            contrastive_loss = torch.tensor(0.0, device=depth_fea1.device)
        
        # 总损失
        total_loss = (depth_consistency_loss + 
                     0.5 * geometry_loss + 
                     0.3 * alignment_loss + 
                     0.2 * contrastive_loss)
        
        return refined_change, total_loss

class depth_change_interaction_module(nn.Module):
    def __init__(self, depth_channels=64, change_channels=64):
        super().__init__()
        
        # === 1. 深度引导的变化增强 ===
        # 深度差异到变化注意力的转换
        self.Depth2Change = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=False)
        )
        
        # === 2. 几何一致性感知 ===
        # 利用深度特征的几何先验过滤误检
        self.GeometryFilter = nn.Sequential(
            nn.Conv2d(depth_channels * 2 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # === 3. 深度-变化联合注意力 ===
        self.DepthChangeAttention = nn.Sequential(
            nn.Conv2d(change_channels + depth_channels * 2, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, change_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # === 4. 损失函数 ===
        # 深度一致性损失（未变化区域深度应相似）
        self.depth_consistency_loss = nn.CosineEmbeddingLoss(margin=0., reduction='mean')
        # 几何结构损失（保持空间关系）
        self.geometry_loss = nn.L1Loss(reduction='mean')
        
    def forward(self, depth_fea1, depth_fea2, old_change, change_result):
        """
        Args:
            depth_fea1: [B, C, H, W] - 时序1的深度特征
            depth_fea2: [B, C, H, W] - 时序2的深度特征
            old_change: [B, C_change, H, W] - 原始变化特征
            change_result: [B, 2, H, W] - 变化检测结果(logits)
        Returns:
            refined_change: 细化后的变化特征
            total_loss: 约束损失
        """
        b, c, h, w = depth_fea1.size()
        
        # ======== 策略1: 深度差异引导的变化增强 ========
        # 计算深度特征的显著性差异
        depth_max_diff, _ = torch.max(torch.abs(depth_fea1 - depth_fea2), dim=1, keepdim=True)
        depth_avg_diff = torch.mean(torch.abs(depth_fea1 - depth_fea2), dim=1, keepdim=True)
        
        # 生成深度变化注意力图
        depth_attention = self.sigmoid(
            self.Depth2Change(torch.cat([depth_max_diff, depth_avg_diff], dim=1))
        )
        
        # ======== 策略2: 几何一致性过滤 ========
        # 融合双时相深度特征和当前变化预测，识别几何一致区域
        change_prob = torch.softmax(change_result, dim=1)[:, 1:2, :, :]  # 变化概率
        geometry_input = torch.cat([depth_fea1, depth_fea2, change_prob], dim=1)
        geometry_mask = self.GeometryFilter(geometry_input)
        
        # ======== 策略3: 深度-变化联合注意力细化 ========
        # 将深度几何先验融入变化特征
        joint_input = torch.cat([old_change, depth_fea1, depth_fea2], dim=1)
        attention_weight = self.DepthChangeAttention(joint_input)
        
        # 多层次融合
        refined_change = old_change * attention_weight * depth_attention * geometry_mask
        
        # ======== 损失计算 ========
        # 提取变化/未变化mask
        change_mask = torch.argmax(change_result, dim=1)
        unchange_mask = ~change_mask.bool()
        
        # Loss 1: 未变化区域的深度特征一致性约束
        fea_depth1 = depth_fea1.permute(0,2,3,1).reshape(b*h*w, c)
        fea_depth2 = depth_fea2.permute(0,2,3,1).reshape(b*h*w, c)
        target = unchange_mask.float() - change_mask.float()
        target = target.reshape(b * h * w)
        depth_consistency_loss = self.depth_consistency_loss(fea_depth1, fea_depth2, target)
        
        # Loss 2: 变化区域的几何差异应显著
        depth_diff_map = torch.norm(depth_fea1 - depth_fea2, p=2, dim=1)
        change_mask_float = change_mask.float()
        # 变化区域应有高深度差异
        geometry_loss = self.geometry_loss(
            depth_diff_map * change_mask_float,
            torch.ones_like(depth_diff_map) * change_mask_float
        )
        
        # Loss 3: 深度注意力与变化预测的对齐损失
        alignment_loss = F.binary_cross_entropy(
            depth_attention.squeeze(1),
            change_mask.float(),
            reduction='mean'
        )
        
        total_loss = depth_consistency_loss + 0.5 * geometry_loss + 0.3 * alignment_loss
        
        return refined_change, total_loss

class DySampleFusion(nn.Module):
    def __init__(self,
                 in_channels_high,   # x1 的通道数，例：128
                 in_channels_low,    # x1_low 的通道数，例：64
                 out_channels,       # 融合后输出通道数，例：64
                 scale=2,
                 style='pl',         # 或 'pl'
                 groups=4,
                 dyscope=False,
                 mid_channels=128): # 中间统一通道数
        super().__init__()

        self.mid_channels = mid_channels
        if mid_channels is None:
            mid_channels = min(in_channels_high, in_channels_low)

        # 把高分辨率、低分辨率特征都投影到同一通道数 mid_channels
        self.proj_high = nn.Conv2d(in_channels_high, mid_channels, 1)
        self.proj_low  = nn.Conv2d(in_channels_low,  mid_channels, 1)

        # DySample_UP 的输入通道数要等于 proj_high 的输出通道数
        self.upsampler = DySample_UP(in_channels=mid_channels,
                                     scale=scale,
                                     style=style,
                                     groups=groups,
                                     dyscope=dyscope)

        # 简单门控：根据高、低两种特征，学习一个融合权重
        self.gate = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 1),
            nn.Sigmoid()
        )

        # 最后的 3x3 conv 输出到 out_channels
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x_high, x_low):
        """
        x_high: [B, C_high, H,   W  ]  e.g. [8, 128, 64, 64]
        x_low:  [B, C_low,  2H,  2W ]  e.g. [8, 64,  128,128]
        """

        # 1) 通道对齐
        if x_high.shape[1] == self.mid_channels:
            x_high_p = x_high   # [B, mid, H, W]
            x_low_p  = self.proj_low(x_low)     # [B, mid, 2H, 2W]
        elif x_low.shape[1] == self.mid_channels:
             x_high_p = self.proj_high(x_high)   # [B, mid, H, W]
             x_low_p  = x_low     # [B, mid, 2H, 2W]
        else:
            x_high_p = self.proj_high(x_high)   # [B, mid, H, W]
            x_low_p  = self.proj_low(x_low)     # [B, mid, 2H, 2W]

        # 2) 用 DySample_UP 上采样高层特征到 2H×2W
        x_high_up = self.upsampler(x_high_p)   # [B, mid, 2H, 2W]

        # 3) 计算门控权重
        #    组合两种特征，学习一个权重 α ∈ [0,1]，表示“高层特征的占比”
        gate_input = torch.cat([x_high_up, x_low_p], dim=1)  # [B, 2*mid, 2H, 2W]
        alpha = self.gate(gate_input)                        # [B, mid, 2H, 2W]

        # 4) 融合：F = α * x_high_up + (1 - α) * x_low_p
        fused = alpha * x_high_up + (1.0 - alpha) * x_low_p  # [B, mid, 2H, 2W]

        # 5) 输出到目标通道数
        out = self.out_conv(fused)                           # [B, out_channels, 2H, 2W]

        return out

class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        #upsample
        #   ConvTranspose2d --> 转秩卷积层（内核大小为7） --> 作用为上采样
        self.upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.catconv = CBA3x3(out_channel * 2, out_channel)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, up, skip):
        #upsample
        up = self.upconv(up)
        up = torch.cat([up, skip], dim=1)
        up = self.catconv(up)
        return up

# BCFE + Stacked Resblocks
class Change_Specific_Transfer(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = CBA1x1(in_channel*2, in_channel)
        self.eca = ECA()
        self.resblock = self._make_layer(ResBlock, in_channel*2, in_channel, 6, stride=1)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        xc1 = self.conv(torch.cat([x1, x2], dim=1))
        xc2 = self.conv(torch.cat([x2, x1], dim=1))
        # 生成两个融合特征，这么做是为了引入时序敏感性（防止x1,x2的位置交换影响结果）
        change = self.eca(xc1 + xc2)
        diff = torch.abs(x1 - x2)
        change = torch.cat([change, diff], dim=1)   # 从而保留变化特征和原始差分信息
        change = self.resblock(change)  # 对变化特征进行进一步提取（堆叠 6 个 ResBlock）
        return change


class Multi_Level_Feature_Aggreagation(nn.Module):

    def __init__(self,):
        super(Multi_Level_Feature_Aggreagation, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)
        # self.proj_x1 = DWConv(512, 128)

        self.cat_conv = CBA1x1(384, 128)


    def forward(self, x1, x2, x3):
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)
        # x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        # x3 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3], dim=1)
        # x = F.interpolate(x1, 64, mode='bilinear', align_corners=False)
        x = self.cat_conv(x)
        return x

class MLFA(nn.Module):

    def __init__(self,):
        super(MLFA, self).__init__()

        self.proj1 = DWConv(512, 128)
        self.proj2 = DWConv(256, 128)
        # self.proj_x1 = DWConv(512, 128)

        self.cat_conv = CBA1x1(384, 128)

        self.Dysample3 = DySample_UP(in_channels=512,scale=4,style='pl',groups=4)
        self.Dysample2 = DySample_UP(in_channels=256,scale=2,style='pl',groups=4)

    def forward(self, x1, x2, x3):
        x3 = self.Dysample3(x3)
        x2 = self.Dysample2(x2)
        x3 = self.proj1(x3)
        x2 = self.proj2(x2)

        
        x = torch.cat([x1, x2, x3], dim=1)
        # x = F.interpolate(x1, 64, mode='bilinear', align_corners=False)
        x = self.cat_conv(x)
        return x
    
class Progressive_Feature_Aggregation(nn.Module):
    def __init__(self,):
        super(Progressive_Feature_Aggregation, self).__init__()
        """渐进式特征融合：从高层到低层逐步融合"""
        # Stage 1: 高层特征处理
        self.high_proj = nn.Sequential(
            DWConv(512, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: 中层融合
        self.mid_fusion = nn.Sequential(
            DWConv(512, 256),  # 256(x2) + 256(x3_proj)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DWConv(256, 128)
        )
        
        # Stage 3: 低层融合
        self.low_fusion = nn.Sequential(
            DWConv(256, 128),  # 128(x1) + 128(mid_fused)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2, x3):
        # 从高到低逐步融合
        x3_proj = self.high_proj(x3)  # 512->256
        
        # 融合中层
        x2_fused = torch.cat([x2, x3_proj], dim=1)  # 256+256=512
        x2_fused = self.mid_fusion(x2_fused)  # 512->128
        
        # 融合低层
        x_final = torch.cat([x1, x2_fused], dim=1)  # 128+128=256
        x_final = self.low_fusion(x_final)  # 256->128
        
        return x_final
    
class Boundary_Decoder(nn.Module):
    def __init__(self,in_chanels=64,out_chanels=64):
        super(Boundary_Decoder, self).__init__()
        self.sobel_x, self.sobel_y = get_sobel(in_chanels, 1)
        self.conv = nn.Conv2d(in_chanels, out_chanels, 1, 1, 0)

    def forward(self, x, size):
        x = F.upsample(x, size, mode='bilinear')
        x = run_sobel(self.sobel_x, self.sobel_y, x)
        x = self.conv(x)
        return x


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


### 1208
class Progressive_Decoder(nn.Module):
    """渐进式解码器"""
    def __init__(self, in_channels=128, skip_channels=64):
        super().__init__()
        
        # 多阶段上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 跳跃连接融合
        self.skip_fusion = Skip_Connection_Fusion(in_channels//2, skip_channels)
        
        # 特征细化
        self.refine = nn.Sequential(
            ResBlock(in_channels//2, in_channels//2),
            ResBlock(in_channels//2, in_channels//2)
        )
        
        # 边界增强
        self.boundary_enhance = Boundary_Enhancement(in_channels//2)
    
    def forward(self, x, skip):
        # 上采样
        x = self.up1(x)
        
        # 跳跃连接
        x = self.skip_fusion(x, skip)
        
        # 特征细化
        x = self.refine(x)
        
        # 边界增强
        x = self.boundary_enhance(x)
        
        return x

class Skip_Connection_Fusion(nn.Module):
    """智能跳跃连接融合"""
    def __init__(self, up_channels, skip_channels):
        super().__init__()
        # 通道对齐
        self.align = nn.Conv2d(skip_channels, up_channels, 1)
        
        # 注意力门控
        self.gate = nn.Sequential(
            nn.Conv2d(up_channels*2, up_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, up, skip):
        skip = self.align(skip)
        gate = self.gate(torch.cat([up, skip], dim=1))
        out = up + gate * skip
        return out

class Boundary_Enhancement(nn.Module):
    """边界增强模块"""
    def __init__(self, channels):
        super().__init__()
        # Sobel算子
        self.sobel_x = nn.Conv2d(channels, channels, 3, padding=1, bias=False, groups=channels)
        self.sobel_y = nn.Conv2d(channels, channels, 3, padding=1, bias=False, groups=channels)
        
        # 初始化Sobel核
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        for i in range(channels):
            self.sobel_x.weight.data[i, 0] = sobel_kernel_x
            self.sobel_y.weight.data[i, 0] = sobel_kernel_y
        
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        # 边界融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        out = self.fusion(torch.cat([x, edge_x, edge_y], dim=1))
        return out
    

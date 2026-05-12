from torchvision import models
import time
from models.DT_SCD.layers import *
from models.DT_SCD.RGBD_Block1207 import DFormerv2Attention, MultiStageRGBDFusion
from models.DT_SCD.DG_funcs import *
from sam2.build_sam import build_sam2
import random

class IdentityHook(nn.Module):
    def forward(self, x):
        return x

def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),     #bias= False是后加的
    #添加：BN RELU
        nn.BatchNorm2d(out_channels),  # 先进行 BN 归一化
        nn.LeakyReLU(negative_slope=0.1, inplace=True),  # LeakyReLU 代替 ReLU

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #添加：Dropuot
        nn.Dropout(0.3)  # 增加 Dropout，减少过拟合
    )
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
#         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out
class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out
class Conv1Relu(nn.Module):  # 1*1卷积用来降维
    def __init__(self, in_ch, out_ch):
        super(Conv1Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class SAM_Feature_Aggreagation(nn.Module):
    """
    输入:
        x1: (B, 144, 128, 128)
        x2: (B, 288,  64,  64)
        x3: (B, 576,  32,  32)
        x4: (B,1152,  16,  16)

    输出:
        out_high: (B, 128, 64, 64)
        out_low : (B,  64,128,128)
    """
    def __init__(self):
        super(SAM_Feature_Aggreagation, self).__init__()

        # 低层特征聚合，输出 (B,64,128,128)
        self.low_proj_x1 = ConvBNReLU(144, 48, kernel_size=1, padding=0)
        self.low_proj_x2 = ConvBNReLU(288, 16, kernel_size=1, padding=0)

        self.low_fuse = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1)
        )

        # 高层特征聚合，输出 (B,128,64,64)
        self.high_proj_x2 = ConvBNReLU(288, 64, kernel_size=1, padding=0)
        self.high_proj_x3 = ConvBNReLU(576, 32, kernel_size=1, padding=0)
        self.high_proj_x4 = ConvBNReLU(1152, 16, kernel_size=1, padding=0)
        self.high_proj_low = ConvBNReLU(64, 16, kernel_size=1, padding=0)

        self.high_fuse = nn.Sequential(
            ConvBNReLU(128, 128, kernel_size=3, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2, x3, x4):
        # =========================
        # 1) 生成低层特征 out_low
        # target size: 128x128
        # =========================
        x1_low = self.low_proj_x1(x1)  # (B,48,128,128)

        x2_up = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x2_low = self.low_proj_x2(x2_up)  # (B,16,128,128)

        low_cat = torch.cat([x1_low, x2_low], dim=1)  # (B,64,128,128)
        out_low = self.low_fuse(low_cat)  # (B,64,128,128)

        # =========================
        # 2) 生成高层特征 out_high
        # target size: 64x64
        # =========================
        x2_high = self.high_proj_x2(x2)  # (B,64,64,64)

        x3_up = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3_high = self.high_proj_x3(x3_up)  # (B,32,64,64)

        x4_up = F.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x4_high = self.high_proj_x4(x4_up)  # (B,16,64,64)

        low_down = F.interpolate(out_low, size=x2.shape[2:], mode='bilinear', align_corners=False)
        low_high = self.high_proj_low(low_down)  # (B,16,64,64)

        high_cat = torch.cat([x2_high, x3_high, x4_high, low_high], dim=1)  # (B,128,64,64)
        out_high = self.high_fuse(high_cat)  # (B,128,64,64)

        return out_high, out_low
    
class DepthResidualInputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_to_rgb = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=False)
        )

        # 初始为 0，保证一开始等价于原始 RGB 输入，不破坏 SAM 预训练先验
        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.depth_to_rgb[-1].weight)
        
        # self.to3 = nn.Conv2d(4, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, rgb, depth):
        depth_residual = self.depth_to_rgb(depth)
        rgb_guided = rgb + self.gamma * depth_residual
        # rgb_guided = self.to3(rgb_guided)
        return rgb_guided

class DepthGuidedFeatureModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.modulator = nn.Sequential(
            nn.Conv2d(2, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, kernel_size=1, bias=True)
        )

        # 零初始化，保证初始时不破坏 SAM 特征
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, feat, depth, depth_diff):
        """
        feat:       B,C,H,W
        depth:      B,1,H0,W0
        depth_diff: B,1,H0,W0
        """
        h, w = feat.shape[2:]

        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
        depth_diff = F.interpolate(depth_diff, size=(h, w), mode='bilinear', align_corners=False)

        prior = torch.cat([depth, depth_diff], dim=1)
        gamma_beta = self.modulator(prior)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # FiLM-like modulation
        out = feat * (1 + torch.tanh(gamma)) + beta
        return out
    
class BTSCD_WUSU(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, ratio = 0.5, checkpoint_path=None):
        super(BTSCD_WUSU, self).__init__()
    ### SAM 模型的加载和修改
        model_cfg = 'sam2_hiera_l.yaml'
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        # old_proj = self.encoder.patch_embed.proj

        # new_proj = nn.Conv2d(
        #     4,
        #     old_proj.out_channels,
        #     kernel_size=old_proj.kernel_size,
        #     stride=old_proj.stride,
        #     padding=old_proj.padding,
        #     bias=(old_proj.bias is not None),
        # )

        # with torch.no_grad():
        #     new_proj.weight[:, :3].copy_(old_proj.weight)
        #     new_proj.weight[:, 3:4].copy_(old_proj.weight.mean(dim=1, keepdim=True) * 0.1)
        #     if old_proj.bias is not None:
        #         new_proj.bias.copy_(old_proj.bias)

        # self.encoder.patch_embed.proj = new_proj

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embed_dims = [144, 288, 576, 1152]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = 256
        self.drop_path_rate = 0.1
        self.SFA = SAM_Feature_Aggreagation()
        # self.sam_input_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        self.sam_input_adapter = DepthResidualInputAdapter()
        self.depth_mods = nn.ModuleList([
                            DepthGuidedFeatureModulation(144),
                            DepthGuidedFeatureModulation(288),
                            DepthGuidedFeatureModulation(576),
                            DepthGuidedFeatureModulation(1152),
                        ])
        # with torch.no_grad():
        #     self.sam_input_adapter.weight.zero_()
        #     self.sam_input_adapter.weight[:, :3, 0, 0] = torch.eye(3)
    ###

        self.change_specific_transfer = Change_Specific_Transfer(128) # BCFE
        # self.change_specific_transfer2 = Change_Specific_Transfer(64) # BCFE
        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)
        self.Dec3 = decoder(128, 64)
        # self.DecCD = Progressive_Decoder(128, 64)
        # self.Dec1 = Progressive_Decoder(128, 64)
        # self.Dec2 = Progressive_Decoder(128, 64)

        self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem3 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)
        # self.classifierCD_final = nn.Conv2d(64, 1, 1, 1, 0, bias=False)
        # self.classifierSem1 = ConvLayer(64, num_classes, kernel_size=3, stride=1, padding=1)
        # self.classifierSem2 = ConvLayer(64, num_classes, kernel_size=3, stride=1, padding=1)
        # self.classifierCD = ConvLayer(64, 2, kernel_size=3, stride=1, padding=1)

        self.boundary_decoder = Boundary_Decoder()
        self.eca = ECA()
        self.boundary_classifier = nn.Sequential(
            CBA3x3(64, 32),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )
#自己加的如下
        self.SBRM_128 = SBRM(128)
        self.SBRM_64 = SBRM(64)
        
        self.Trans = Trans(dim=128)
        # self.low_conv = nn.ConvTranspose2d(
        #                 in_channels=256,
        #                 out_channels=64,
        #                 kernel_size=4,
        #                 stride=2,
        #                 padding=1
        #             )
        # self.low_conv = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #                 nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        #             )
        
# 定义：Grad-Cam hooks
        # self.id_x1_after_FCN = IdentityHook()
        # self.id_x2_after_FCN = IdentityHook()
        # self.id_x1_after_trans = IdentityHook()
        # self.id_x2_after_trans = IdentityHook()
        # self.id_x1_fuse = IdentityHook()
        # self.id_x2_fuse = IdentityHook()
        # self.id_x1_after_fuse = IdentityHook()
        # self.id_x2_after_fuse = IdentityHook()
        # self.id_out1 = IdentityHook()
        # self.id_out2 = IdentityHook()
        # self.BCD_1_hook = IdentityHook()
        # self.BCD_2_hook = IdentityHook()
        # self.xc_fuse_hook = IdentityHook()
        # self.xc_hook = IdentityHook()
        # self.xc2_hook = IdentityHook()
        # self.id_x1_dep_after_trans = IdentityHook()
        # self.id_x2_dep_after_trans = IdentityHook()
        
        self.conv_diff = conv_diff(128, 64)
        self.depth_change_interaction_module = depth_change_interaction_module()
        self.RGBD = DFormerv2Attention(
                                    embed_dim=128,
                                    num_heads=16,
                                    use_decomposed=True,  # True使用分解注意力, False使用完整注意力
                                    initial_value=2,
                                    heads_range=4
                                )
        # self.RGBD_multi = MultiStageRGBDFusion(embed_dim=128, num_heads=16, num_stages=3)
        self.conv4 = Conv1Relu(256, 128)
        self.dep_change_specific_transfer = Dep_Change_Specific_Transfer(128) # BCFE
    
    def _set_depth_lora_context(self, depth, depth_diff):
        """
        给 encoder 内所有 DepthGuidedLoRA_qkv 设置深度上下文。
        不依赖外部 LoRA_sam 包装器。
        """
        for m in self.encoder.modules():
            if hasattr(m, "set_depth_context"):
                m.set_depth_context(depth, depth_diff)
        
    def forward(self, x1, x2, x3, d1, d2, d3):
    # def forward(self, x1, x2):
        x_size = x1.size()
        
    # 提取RGB特征
        # x1 = torch.cat([x1, x3], dim=1)  # (B, 4, H, W)
        # x2 = torch.cat([x2, x4], dim=1)  # (B, 4, H, W)
        # x1_FCN, x1_low = self.FCN(x1)   #Backbone输出： x(8 128 64 64) x_low(8 64 128 128)
        # x2_FCN, x2_low = self.FCN(x2)
        # # x1_FCN = self.id_x1_after_FCN(x1_FCN)     #可视化代码
        # # x2_FCN = self.id_x2_after_FCN(x2_FCN)     #可视化代码
        
        #SAM-LORA输出： (8 144 128 128) (8 288 64 64) (8 576 32 32) (8 1152 16 16)
        depth_diff13 = torch.abs(d1 - d3)
        # depth_diff12 = torch.abs(d1 - d2)
        
        x1_sam = self.sam_input_adapter(x1, d1)
        self._set_depth_lora_context(d1, depth_diff13)
        x11, x12, x13, x14 = self.encoder(x1_sam)
        
        x2_sam = self.sam_input_adapter(x2, d2)
        self._set_depth_lora_context(d2, depth_diff13)
        x21, x22, x23, x24 = self.encoder(x2_sam)
        
        x3_sam = self.sam_input_adapter(x3, d3)
        self._set_depth_lora_context(d3, depth_diff13)
        x31, x32, x33, x34 = self.encoder(x3_sam)
        
        x11 = self.depth_mods[0](x11, d1, depth_diff13)
        x12 = self.depth_mods[1](x12, d1, depth_diff13)
        x13 = self.depth_mods[2](x13, d1, depth_diff13)
        x14 = self.depth_mods[3](x14, d1, depth_diff13)

        x21 = self.depth_mods[0](x21, d2, depth_diff13)
        x22 = self.depth_mods[1](x22, d2, depth_diff13)
        x23 = self.depth_mods[2](x23, d2, depth_diff13)
        x24 = self.depth_mods[3](x24, d2, depth_diff13)

        x31 = self.depth_mods[0](x31, d3, depth_diff13)
        x32 = self.depth_mods[1](x32, d3, depth_diff13)
        x33 = self.depth_mods[2](x33, d3, depth_diff13)
        x34 = self.depth_mods[3](x34, d3, depth_diff13)

        x1_FCN, x1_low = self.SFA(x11, x12, x13, x14)
        x2_FCN, x2_low = self.SFA(x21, x22, x23, x24)
        x3_FCN, x3_low = self.SFA(x31, x32, x33, x34)
        
    # 进行 Transformer-based 增强和交互
        x1t, x2t, x3t = self.Trans(x1_FCN, x2_FCN, x3_FCN)  # Transformer增强和交互  x1t(8 128 64 64) x2t(8 128 64 64) x3t(8 128 64 64)
        x1 = x1_FCN + x1t
        x2 = x2_FCN + x2t
        x3 = x3_FCN + x3t

        # xc = self.change_specific_transfer(x1, x2)  #融合x1,x2特征 --> xc(8 128 64 64)
        xc = self.dep_change_specific_transfer(x1, x3, d1, d3)  # 引入深度信息进行变化特定特征提取
        
        xc = self.SBRM_128(xc)
        
        x1 = self.Dec1(x1, x1_low)  # x(8 128 64 64) x_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合
        x2 = self.Dec2(x2, x2_low)
        x3 = self.Dec3(x3, x3_low)
        
        # xc_low = torch.abs(x1 - x2) # (B, 64, 128, 128) 融合以后的特征在128*128尺度 直接做差取绝对值
        xc_low = torch.cat([x1, x3], dim=1)  # (B, 128, 128, 128)
        xc_low = self.conv_diff(xc_low)  # (B, 64, 128, 128)
        xc_low = self.SBRM_64(xc_low)
        
        xc = self.DecCD(xc, xc_low)  # xc(8 128 64 64) xc_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合   !!!又用了一次多尺度融合!!!
        # xc1 = self.xc_hook(xc1)   #    xc_fuse可视化代码
    # Classifier
        init_BCD_change = self.classifierCD(xc)  #(8 2 128 128)  先初步预测变化图 用于后续的任务交互模块 即BCD任务
        # BCD_change = self.BCD_1_hook(BCD_change)   #可视化代码
        # new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, init_BCD_change)  #new_xc(8 64 128 128)
        new_xc, pixel_sim_loss = self.depth_change_interaction_module(x1, x3, xc, init_BCD_change)  #new_xc(8 64 128 128)
        
        
        final_BCD_change = self.classifierCD(new_xc)  #new_change(8 2 128 128)  最终变化图预测 交互后，又输出了一次BCD任务的结果？？？？
        # new_change = self.BCD_2_hook(new_change)   #可视化代码

        out1 = self.classifierSem1(x1)  #out1(8 7 128 128)
        out2 = self.classifierSem2(x2)
        out3 = self.classifierSem3(x3)
        
        out1 = F.upsample(out1, x_size[2:], mode='bilinear')    #(8 7 128 128) --> (8 7 512 512)
        # # # out1 = self.id_out1(out1)
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        # # # out2 = self.id_out2(out2)
        out3 = F.upsample(out3, x_size[2:], mode='bilinear')
        
        change_out = F.upsample(final_BCD_change, x_size[2:], mode='bilinear')
        # change_out = torch.sigmoid(change_out)
        # out1 = self.Dysample_out(out1)    #(8 7 128 128) --> (8 7 512 512)
        # out2 = self.Dysample_out(out2)
        # change_out = self.Dysample_chage_out(final_BCD_change)

        # return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change
        return out1, out2, out3, change_out, pixel_sim_loss

class BTSCD_WUSU_random(BTSCD_WUSU):
    """
    BTSCD_WUSU 的 random-pair 版本。

    输入:
        x1, x2, x3: 三个时相 RGB 图像, shape = [B, 3, H, W]
        d1, d2, d3: 三个时相深度图, shape = [B, 1, H, W]
        pair: 可选，指定变化检测时相对。
              None:
                  train 模式下随机选择 [0,1] / [0,2] / [1,2]
                  eval 模式下默认固定 [0,2]
              [0,1]: T1-T2
              [0,2]: T1-T3
              [1,2]: T2-T3

    输出:
        outs: [out1, out2, out3]
              三个时相语义预测 logits，每个 shape = [B, num_classes, H, W]
        change_out:
              当前 pair 的二值变化预测 logits，shape = [B, 2, H, W]
        pixel_sim_loss:
              深度变化交互模块产生的辅助损失
        pair:
              当前用于变化检测的时相对，例如 [0, 2]
    """

    def __init__(self, in_channels=3, num_classes=7, ratio=0.5, checkpoint_path=None):
        super(BTSCD_WUSU_random, self).__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            ratio=ratio,
            checkpoint_path=checkpoint_path
        )

    def _choose_pair(self, pair=None):
        """
        训练阶段默认随机 pair；
        验证/测试阶段默认固定 [0, 2]，避免验证指标随机波动。
        """
        if pair is None:
            # if self.training:
            #     pair = sorted(random.sample(range(3), 2))
            # else:
            #     pair = [0, 2]
            pair = sorted(random.sample(range(3), 2))

        if isinstance(pair, tuple):
            pair = list(pair)

        if not isinstance(pair, list):
            raise TypeError(f"pair 应为 list/tuple，例如 [0, 2]，但得到: {type(pair)}")

        if len(pair) != 2:
            raise ValueError(f"pair 长度必须为 2，但得到: {pair}")

        pair = sorted([int(pair[0]), int(pair[1])])

        if pair not in ([0, 1], [0, 2], [1, 2]):
            raise ValueError(f"pair 只能是 [0,1] / [0,2] / [1,2]，但得到: {pair}")

        return pair

    def _encode_one_phase(self, rgb, depth, depth_diff):
        """
        对单个时相执行:
        RGB + depth residual adapter -> SAM2 encoder -> depth-guided feature modulation -> SFA

        返回:
            high_feat: [B,128,64,64]
            low_feat : [B, 64,128,128]
        """
        x_sam = self.sam_input_adapter(rgb, depth)

        self._set_depth_lora_context(depth, depth_diff)
        x1, x2, x3, x4 = self.encoder(x_sam)

        x1 = self.depth_mods[0](x1, depth, depth_diff)
        x2 = self.depth_mods[1](x2, depth, depth_diff)
        x3 = self.depth_mods[2](x3, depth, depth_diff)
        x4 = self.depth_mods[3](x4, depth, depth_diff)

        high_feat, low_feat = self.SFA(x1, x2, x3, x4)

        return high_feat, low_feat

    def forward(self, x1, x2, x3, d1, d2, d3, pair=None):
        x_size = x1.size()

        pair = self._choose_pair(pair)

        rgb_list = [x1, x2, x3]
        depth_list = [d1, d2, d3]

        p0, p1 = pair

        # 当前随机 pair 对应的深度差异。
        # 例如 pair=[0,1] 时，为 |d1-d2|；
        # pair=[0,2] 时，为 |d1-d3|；
        # pair=[1,2] 时，为 |d2-d3|。
        depth_diff_pair = torch.abs(depth_list[p0] - depth_list[p1])

        # =========================================================
        # 1. 三个时相分别提取 SAM2 + depth-guided 特征
        # =========================================================
        high_feats = []
        low_feats = []

        for rgb, depth in zip(rgb_list, depth_list):
            high_feat, low_feat = self._encode_one_phase(
                rgb=rgb,
                depth=depth,
                depth_diff=depth_diff_pair
            )
            high_feats.append(high_feat)
            low_feats.append(low_feat)

        x1_high, x2_high, x3_high = high_feats
        x1_low, x2_low, x3_low = low_feats

        # =========================================================
        # 2. 三时相 Transformer 交互
        # 注意：这里要求你的 self.Trans 支持三输入三输出。
        # 你的 BTSCD_WUSU 当前 forward 中已经按三输入调用 self.Trans。
        # =========================================================
        x1t, x2t, x3t = self.Trans(x1_high, x2_high, x3_high)

        x1_feat = x1_high + x1t
        x2_feat = x2_high + x2t
        x3_feat = x3_high + x3t

        feat_list = [x1_feat, x2_feat, x3_feat]

        # =========================================================
        # 3. 当前 pair 的高层变化特征
        # 原 BTSCD_WUSU 固定用 x1,x3,d1,d3；
        # random 版改成 feat_list[p0], feat_list[p1], depth_list[p0], depth_list[p1]
        # =========================================================
        xc = self.dep_change_specific_transfer(
            feat_list[p0],
            feat_list[p1],
            depth_list[p0],
            depth_list[p1]
        )

        xc = self.SBRM_128(xc)

        # =========================================================
        # 4. 三个时相分别做语义 decoder
        # =========================================================
        x1_dec = self.Dec1(x1_feat, x1_low)
        x2_dec = self.Dec2(x2_feat, x2_low)
        x3_dec = self.Dec3(x3_feat, x3_low)

        dec_list = [x1_dec, x2_dec, x3_dec]

        # =========================================================
        # 5. 当前 pair 的低层变化特征
        # 原 BTSCD_WUSU 固定 cat([x1, x3])；
        # random 版改成 cat([dec_list[p0], dec_list[p1]])
        # =========================================================
        xc_low = torch.cat([dec_list[p0], dec_list[p1]], dim=1)
        xc_low = self.conv_diff(xc_low)
        xc_low = self.SBRM_64(xc_low)

        xc = self.DecCD(xc, xc_low)

        # =========================================================
        # 6. 当前 pair 的二值变化预测 + 变化语义交互
        # =========================================================
        init_BCD_change = self.classifierCD(xc)

        new_xc, pixel_sim_loss = self.depth_change_interaction_module(
            dec_list[p0],
            dec_list[p1],
            xc,
            init_BCD_change
        )

        final_BCD_change = self.classifierCD(new_xc)

        # =========================================================
        # 7. 三个时相语义输出
        # =========================================================
        out1 = self.classifierSem1(x1_dec)
        out2 = self.classifierSem2(x2_dec)
        out3 = self.classifierSem3(x3_dec)

        out1 = F.interpolate(out1, size=x_size[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=x_size[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=x_size[2:], mode='bilinear', align_corners=False)

        change_out = F.interpolate(
            final_BCD_change,
            size=x_size[2:],
            mode='bilinear',
            align_corners=False
        )

        return [out1, out2, out3], change_out, pixel_sim_loss, pair


class BTSCD_WUSU_every(BTSCD_WUSU):
    """
    BTSCD_WUSU 的 random-pair 版本。

    输入:
        x1, x2, x3: 三个时相 RGB 图像, shape = [B, 3, H, W]
        d1, d2, d3: 三个时相深度图, shape = [B, 1, H, W]
        pair: 可选，指定变化检测时相对。
              None:
                  train 模式下随机选择 [0,1] / [0,2] / [1,2]
                  eval 模式下默认固定 [0,2]
              [0,1]: T1-T2
              [0,2]: T1-T3
              [1,2]: T2-T3

    输出:
        outs: [out1, out2, out3]
              三个时相语义预测 logits，每个 shape = [B, num_classes, H, W]
        change_out:
              当前 pair 的二值变化预测 logits，shape = [B, 2, H, W]
        pixel_sim_loss:
              深度变化交互模块产生的辅助损失
        pair:
              当前用于变化检测的时相对，例如 [0, 2]
    """

    def __init__(self, in_channels=3, num_classes=7, ratio=0.5, checkpoint_path=None):
        super(BTSCD_WUSU_every, self).__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            ratio=ratio,
            checkpoint_path=checkpoint_path
        )

    def _choose_pair(self, pair=None):
        """
        训练阶段默认随机 pair；
        验证/测试阶段默认固定 [0, 2]，避免验证指标随机波动。
        """
        if pair is None:
            # if self.training:
            #     pair = sorted(random.sample(range(3), 2))
            # else:
            #     pair = [0, 2]
            pair = sorted(random.sample(range(3), 2))

        if isinstance(pair, tuple):
            pair = list(pair)

        if not isinstance(pair, list):
            raise TypeError(f"pair 应为 list/tuple，例如 [0, 2]，但得到: {type(pair)}")

        if len(pair) != 2:
            raise ValueError(f"pair 长度必须为 2，但得到: {pair}")

        pair = sorted([int(pair[0]), int(pair[1])])

        if pair not in ([0, 1], [0, 2], [1, 2]):
            raise ValueError(f"pair 只能是 [0,1] / [0,2] / [1,2]，但得到: {pair}")

        return pair

    def _encode_one_phase(self, rgb, depth, depth_diff):
        """
        对单个时相执行:
        RGB + depth residual adapter -> SAM2 encoder -> depth-guided feature modulation -> SFA

        返回:
            high_feat: [B,128,64,64]
            low_feat : [B, 64,128,128]
        """
        x_sam = self.sam_input_adapter(rgb, depth)

        self._set_depth_lora_context(depth, depth_diff)
        x1, x2, x3, x4 = self.encoder(x_sam)

        x1 = self.depth_mods[0](x1, depth, depth_diff)
        x2 = self.depth_mods[1](x2, depth, depth_diff)
        x3 = self.depth_mods[2](x3, depth, depth_diff)
        x4 = self.depth_mods[3](x4, depth, depth_diff)

        high_feat, low_feat = self.SFA(x1, x2, x3, x4)

        return high_feat, low_feat

    def forward(self, x1, x2, x3, d1, d2, d3, pair=None):
        x_size = x1.size()

        # pair = self._choose_pair(pair)
        pair_01 = [0, 1]
        pair_02 = [0, 2]
        pair_12 = [1, 2]
        
        rgb_list = [x1, x2, x3]
        depth_list = [d1, d2, d3]

        p0, p1 = pair

        # 当前随机 pair 对应的深度差异。
        # 例如 pair=[0,1] 时，为 |d1-d2|；
        # pair=[0,2] 时，为 |d1-d3|；
        # pair=[1,2] 时，为 |d2-d3|。
        depth_diff_pair = torch.abs(depth_list[p0] - depth_list[p1])

        # =========================================================
        # 1. 三个时相分别提取 SAM2 + depth-guided 特征
        # =========================================================
        high_feats = []
        low_feats = []

        for rgb, depth in zip(rgb_list, depth_list):
            high_feat, low_feat = self._encode_one_phase(
                rgb=rgb,
                depth=depth,
                depth_diff=depth_diff_pair
            )
            high_feats.append(high_feat)
            low_feats.append(low_feat)

        x1_high, x2_high, x3_high = high_feats
        x1_low, x2_low, x3_low = low_feats

        depth_diff13 = torch.abs(d1 - d3)
        depth_diff12 = torch.abs(d1 - d2)
        x1_sam = self.sam_input_adapter(x1, d1)
        self._set_depth_lora_context(d1, depth_diff13)
        x11, x12, x13, x14 = self.encoder(x1_sam)
        
        x2_sam = self.sam_input_adapter(x2, d2)
        self._set_depth_lora_context(d2, depth_diff12)
        x21, x22, x23, x24 = self.encoder(x2_sam)
        
        x3_sam = self.sam_input_adapter(x3, d3)
        self._set_depth_lora_context(d3, depth_diff13)
        x31, x32, x33, x34 = self.encoder(x3_sam)
        
        x11 = self.depth_mods[0](x11, d1, depth_diff13)
        x12 = self.depth_mods[1](x12, d1, depth_diff13)
        x13 = self.depth_mods[2](x13, d1, depth_diff13)
        x14 = self.depth_mods[3](x14, d1, depth_diff13)

        x21 = self.depth_mods[0](x21, d2, depth_diff12)
        x22 = self.depth_mods[1](x22, d2, depth_diff12)
        x23 = self.depth_mods[2](x23, d2, depth_diff12)
        x24 = self.depth_mods[3](x24, d2, depth_diff12)

        x31 = self.depth_mods[0](x31, d3, depth_diff13)
        x32 = self.depth_mods[1](x32, d3, depth_diff13)
        x33 = self.depth_mods[2](x33, d3, depth_diff13)
        x34 = self.depth_mods[3](x34, d3, depth_diff13)

        x1_FCN, x1_low = self.SFA(x11, x12, x13, x14)
        x2_FCN, x2_low = self.SFA(x21, x22, x23, x24)
        x3_FCN, x3_low = self.SFA(x31, x32, x33, x34)
        # =========================================================
        # 2. 三时相 Transformer 交互
        # 注意：这里要求你的 self.Trans 支持三输入三输出。
        # 你的 BTSCD_WUSU 当前 forward 中已经按三输入调用 self.Trans。
        # =========================================================
        x1t, x2t, x3t = self.Trans(x1_high, x2_high, x3_high)

        x1_feat = x1_high + x1t
        x2_feat = x2_high + x2t
        x3_feat = x3_high + x3t

        feat_list = [x1_feat, x2_feat, x3_feat]

        # =========================================================
        # 3. 当前 pair 的高层变化特征
        # 原 BTSCD_WUSU 固定用 x1,x3,d1,d3；
        # random 版改成 feat_list[p0], feat_list[p1], depth_list[p0], depth_list[p1]
        # =========================================================
        xc = self.dep_change_specific_transfer(
            feat_list[p0],
            feat_list[p1],
            depth_list[p0],
            depth_list[p1]
        )

        xc = self.SBRM_128(xc)

        # =========================================================
        # 4. 三个时相分别做语义 decoder
        # =========================================================
        x1_dec = self.Dec1(x1_feat, x1_low)
        x2_dec = self.Dec2(x2_feat, x2_low)
        x3_dec = self.Dec3(x3_feat, x3_low)

        dec_list = [x1_dec, x2_dec, x3_dec]

        # =========================================================
        # 5. 当前 pair 的低层变化特征
        # 原 BTSCD_WUSU 固定 cat([x1, x3])；
        # random 版改成 cat([dec_list[p0], dec_list[p1]])
        # =========================================================
        xc_low = torch.cat([dec_list[p0], dec_list[p1]], dim=1)
        xc_low = self.conv_diff(xc_low)
        xc_low = self.SBRM_64(xc_low)

        xc = self.DecCD(xc, xc_low)

        # =========================================================
        # 6. 当前 pair 的二值变化预测 + 变化语义交互
        # =========================================================
        init_BCD_change = self.classifierCD(xc)

        new_xc, pixel_sim_loss = self.depth_change_interaction_module(
            dec_list[p0],
            dec_list[p1],
            xc,
            init_BCD_change
        )

        final_BCD_change = self.classifierCD(new_xc)

        # =========================================================
        # 7. 三个时相语义输出
        # =========================================================
        out1 = self.classifierSem1(x1_dec)
        out2 = self.classifierSem2(x2_dec)
        out3 = self.classifierSem3(x3_dec)

        out1 = F.interpolate(out1, size=x_size[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=x_size[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=x_size[2:], mode='bilinear', align_corners=False)

        change_out = F.interpolate(
            final_BCD_change,
            size=x_size[2:],
            mode='bilinear',
            align_corners=False
        )

        return [out1, out2, out3], change_out, pixel_sim_loss, pair
    
    
class SpectralDepthResidualInputAdapter(nn.Module):
    """
    适配 RGB 或 RGB+NIR 输入到 SAM2 的 3 通道输入空间。

    输入:
        image:
            [B,3,H,W] -> RGB
            [B,4,H,W] -> RGB+NIR
        depth:
            [B,1,H,W]

    输出:
        image_guided:
            [B,3,H,W]，可直接送入 SAM2 encoder

    设计:
        RGB 作为主干输入；
        NIR 通过 nir_to_rgb 映射为 3 通道残差；
        depth 通过 depth_to_rgb 映射为 3 通道残差；
        gamma_nir 和 gamma_depth 零初始化，保证初始状态等价于纯 RGB 输入。
    """
    def __init__(self, in_channels=4):
        super().__init__()
        assert in_channels in [3, 4], \
            f"in_channels 只能是 3 或 4，但当前为 {in_channels}"

        self.in_channels = in_channels

        self.depth_to_rgb = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=False)
        )

        self.nir_to_rgb = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=False)
        )

        # 初始为 0：一开始等价于 RGB 输入，不破坏 SAM 预训练先验
        self.gamma_depth = nn.Parameter(torch.zeros(1))
        self.gamma_nir = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.depth_to_rgb[-1].weight)
        nn.init.zeros_(self.nir_to_rgb[-1].weight)

    def forward(self, image, depth):
        """
        image:
            [B,3,H,W] RGB
            [B,4,H,W] RGB+NIR
        depth:
            [B,1,H,W]
        """
        if image.dim() != 4:
            raise RuntimeError(f"image 应为 [B,C,H,W]，但得到 {image.shape}")

        if depth.dim() != 4 or depth.shape[1] != 1:
            raise RuntimeError(f"depth 应为 [B,1,H,W]，但得到 {depth.shape}")

        if image.shape[1] == 3:
            rgb = image
            nir_residual = 0.0

        elif image.shape[1] == 4:
            rgb = image[:, :3, :, :]
            nir = image[:, 3:4, :, :]
            nir_residual = self.nir_to_rgb(nir)

        else:
            raise RuntimeError(
                f"输入图像通道数必须为 3 或 4，但当前为 {image.shape[1]}"
            )

        depth_residual = self.depth_to_rgb(depth)

        image_guided = (
            rgb
            + self.gamma_depth * depth_residual
            + self.gamma_nir * nir_residual
        )

        return image_guided

class SpectralDepthGuidedFeatureModulation(nn.Module):
    """
    使用 depth、depth_diff、NIR、NIR_diff 共同调制 SAM 特征。
    """
    def __init__(self, channels):
        super().__init__()

        # prior = [depth, depth_diff, nir, nir_diff] -> 4 通道
        self.modulator = nn.Sequential(
            nn.Conv2d(4, channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, kernel_size=1, bias=True)
        )

        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, feat, depth, depth_diff, nir, nir_diff):
        h, w = feat.shape[2:]

        depth = F.interpolate(depth, size=(h, w), mode='bilinear', align_corners=False)
        depth_diff = F.interpolate(depth_diff, size=(h, w), mode='bilinear', align_corners=False)
        nir = F.interpolate(nir, size=(h, w), mode='bilinear', align_corners=False)
        nir_diff = F.interpolate(nir_diff, size=(h, w), mode='bilinear', align_corners=False)

        prior = torch.cat([depth, depth_diff, nir, nir_diff], dim=1)

        gamma_beta = self.modulator(prior)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        out = feat * (1 + torch.tanh(gamma)) + beta
        return out

class BTSCD_WUSU_NIR(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, ratio = 0.5, checkpoint_path=None):
        super(BTSCD_WUSU_NIR, self).__init__()
    ### SAM 模型的加载和修改
        model_cfg = 'sam2_hiera_l.yaml'
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embed_dims = [144, 288, 576, 1152]
        self.depths = [3, 3, 4, 3]
        self.embedding_dim = 256
        self.drop_path_rate = 0.1
        self.SFA = SAM_Feature_Aggreagation()
        # self.sam_input_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        # self.sam_input_adapter = DepthResidualInputAdapter()
        self.sam_input_adapter = SpectralDepthResidualInputAdapter(in_channels=in_channels)
        self.depth_mods = nn.ModuleList([
                            DepthGuidedFeatureModulation(144),
                            DepthGuidedFeatureModulation(288),
                            DepthGuidedFeatureModulation(576),
                            DepthGuidedFeatureModulation(1152),
                        ])
        self.depth_mods = nn.ModuleList([
                            SpectralDepthGuidedFeatureModulation(144),
                            SpectralDepthGuidedFeatureModulation(288),
                            SpectralDepthGuidedFeatureModulation(576),
                            SpectralDepthGuidedFeatureModulation(1152),
                        ])
        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)
        self.Dec3 = decoder(128, 64)

        self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem3 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)

        self.boundary_decoder = Boundary_Decoder()
        self.eca = ECA()
        self.boundary_classifier = nn.Sequential(
            CBA3x3(64, 32),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )
#自己加的如下
        self.SBRM_128 = SBRM(128)
        self.SBRM_64 = SBRM(64)
        
        self.Trans = Trans(dim=128)
        
        self.conv_diff = conv_diff(128, 64)
        self.depth_change_interaction_module = depth_change_interaction_module()
        self.conv4 = Conv1Relu(256, 128)
        self.dep_change_specific_transfer = Dep_Change_Specific_Transfer(128) # BCFE
    
    def _set_depth_lora_context(self, depth, depth_diff):
        """
        给 encoder 内所有 DepthGuidedLoRA_qkv 设置深度上下文。
        不依赖外部 LoRA_sam 包装器。
        """
        for m in self.encoder.modules():
            if hasattr(m, "set_depth_context"):
                m.set_depth_context(depth, depth_diff)
    def get_nir(self, x):
        if x.shape[1] == 4:
            return x[:, 3:4, :, :]
        else:
            return torch.zeros_like(x[:, :1, :, :])
    def forward(self, x1, x2, x3, d1, d2, d3):
    # def forward(self, x1, x2):
        x_size = x1.size()
        nir1 = self.get_nir(x1)
        nir2 = self.get_nir(x2)
        nir3 = self.get_nir(x3)
        nir_diff13 = torch.abs(nir1 - nir3)
        
        #SAM-LORA输出： (8 144 128 128) (8 288 64 64) (8 576 32 32) (8 1152 16 16)
        depth_diff13 = torch.abs(d1 - d3)
        # depth_diff12 = torch.abs(d1 - d2)
        
        x1_sam = self.sam_input_adapter(x1, d1)
        self._set_depth_lora_context(d1, depth_diff13)
        x11, x12, x13, x14 = self.encoder(x1_sam)
        
        x2_sam = self.sam_input_adapter(x2, d2)
        self._set_depth_lora_context(d2, depth_diff13)
        x21, x22, x23, x24 = self.encoder(x2_sam)
        
        x3_sam = self.sam_input_adapter(x3, d3)
        self._set_depth_lora_context(d3, depth_diff13)
        x31, x32, x33, x34 = self.encoder(x3_sam)
        
        # x11 = self.depth_mods[0](x11, d1, depth_diff13)
        # x12 = self.depth_mods[1](x12, d1, depth_diff13)
        # x13 = self.depth_mods[2](x13, d1, depth_diff13)
        # x14 = self.depth_mods[3](x14, d1, depth_diff13)

        # x21 = self.depth_mods[0](x21, d2, depth_diff13)
        # x22 = self.depth_mods[1](x22, d2, depth_diff13)
        # x23 = self.depth_mods[2](x23, d2, depth_diff13)
        # x24 = self.depth_mods[3](x24, d2, depth_diff13)

        # x31 = self.depth_mods[0](x31, d3, depth_diff13)
        # x32 = self.depth_mods[1](x32, d3, depth_diff13)
        # x33 = self.depth_mods[2](x33, d3, depth_diff13)
        # x34 = self.depth_mods[3](x34, d3, depth_diff13)
        
        #   添加NIR通道Adapter
        x11 = self.depth_mods[0](x11, d1, depth_diff13, nir1, nir_diff13)
        x12 = self.depth_mods[1](x12, d1, depth_diff13, nir1, nir_diff13)
        x13 = self.depth_mods[2](x13, d1, depth_diff13, nir1, nir_diff13)
        x14 = self.depth_mods[3](x14, d1, depth_diff13, nir1, nir_diff13)

        x21 = self.depth_mods[0](x21, d2, depth_diff13, nir2, nir_diff13)
        x22 = self.depth_mods[1](x22, d2, depth_diff13, nir2, nir_diff13)
        x23 = self.depth_mods[2](x23, d2, depth_diff13, nir2, nir_diff13)
        x24 = self.depth_mods[3](x24, d2, depth_diff13, nir2, nir_diff13)

        x31 = self.depth_mods[0](x31, d3, depth_diff13, nir3, nir_diff13)
        x32 = self.depth_mods[1](x32, d3, depth_diff13, nir3, nir_diff13)
        x33 = self.depth_mods[2](x33, d3, depth_diff13, nir3, nir_diff13)
        x34 = self.depth_mods[3](x34, d3, depth_diff13, nir3, nir_diff13)
        
        x1_FCN, x1_low = self.SFA(x11, x12, x13, x14)
        x2_FCN, x2_low = self.SFA(x21, x22, x23, x24)
        x3_FCN, x3_low = self.SFA(x31, x32, x33, x34)
        
    # 进行 Transformer-based 增强和交互
        x1t, x2t, x3t = self.Trans(x1_FCN, x2_FCN, x3_FCN)  # Transformer增强和交互  x1t(8 128 64 64) x2t(8 128 64 64) x3t(8 128 64 64)
        x1 = x1_FCN + x1t
        x2 = x2_FCN + x2t
        x3 = x3_FCN + x3t

        # xc = self.change_specific_transfer(x1, x2)  #融合x1,x2特征 --> xc(8 128 64 64)
        xc = self.dep_change_specific_transfer(x1, x3, d1, d3)  # 引入深度信息进行变化特定特征提取
        
        xc = self.SBRM_128(xc)
        
        x1 = self.Dec1(x1, x1_low)  # x(8 128 64 64) x_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合
        x2 = self.Dec2(x2, x2_low)
        x3 = self.Dec3(x3, x3_low)
        
        # xc_low = torch.abs(x1 - x2) # (B, 64, 128, 128) 融合以后的特征在128*128尺度 直接做差取绝对值
        xc_low = torch.cat([x1, x3], dim=1)  # (B, 128, 128, 128)
        xc_low = self.conv_diff(xc_low)  # (B, 64, 128, 128)
        xc_low = self.SBRM_64(xc_low)
        
        xc = self.DecCD(xc, xc_low)  # xc(8 128 64 64) xc_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合   !!!又用了一次多尺度融合!!!
        # xc1 = self.xc_hook(xc1)   #    xc_fuse可视化代码
    # Classifier
        init_BCD_change = self.classifierCD(xc)  #(8 2 128 128)  先初步预测变化图 用于后续的任务交互模块 即BCD任务
        # BCD_change = self.BCD_1_hook(BCD_change)   #可视化代码
        # new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, init_BCD_change)  #new_xc(8 64 128 128)
        new_xc, pixel_sim_loss = self.depth_change_interaction_module(x1, x3, xc, init_BCD_change)  #new_xc(8 64 128 128)
        
        
        final_BCD_change = self.classifierCD(new_xc)  #new_change(8 2 128 128)  最终变化图预测 交互后，又输出了一次BCD任务的结果？？？？
        # new_change = self.BCD_2_hook(new_change)   #可视化代码

        out1 = self.classifierSem1(x1)  #out1(8 7 128 128)
        out2 = self.classifierSem2(x2)
        out3 = self.classifierSem3(x3)
        
        out1 = F.upsample(out1, x_size[2:], mode='bilinear')    #(8 7 128 128) --> (8 7 512 512)
        # # # out1 = self.id_out1(out1)
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        # # # out2 = self.id_out2(out2)
        out3 = F.upsample(out3, x_size[2:], mode='bilinear')
        
        change_out = F.upsample(final_BCD_change, x_size[2:], mode='bilinear')
        # change_out = torch.sigmoid(change_out)
        # out1 = self.Dysample_out(out1)    #(8 7 128 128) --> (8 7 512 512)
        # out2 = self.Dysample_out(out2)
        # change_out = self.Dysample_chage_out(final_BCD_change)

        # return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change
        return out1, out2, out3, change_out, pixel_sim_loss
# if __name__ == '__main__':
#     x1 = torch.randn(1, 3, 512, 512).cuda().float()
#     x2 = torch.randn(1, 3, 512, 512).cuda().float()

#     model = BTSCD(3, num_classes=7).cuda()
#     model.eval()  # 将模型设置为推理模式
#     from fvcore.nn import FlopCountAnalysis
#     flops = FlopCountAnalysis(model, (x1, x2))
#     total = sum([param.nelement() for param in model.parameters()])
#     print("Params_Num: %.2fM" % (total/1e6))
#     print("FLOPs: %.2fG" % (flops.total()/1e9))

#     with torch.no_grad():
#         for _ in range(10):
#             _ = model(x1, x2)

#     # 正式计时
#     start_time = time.time()
#     with torch.no_grad():
#         output = model(x1, x2)
#     end_time = time.time()

#     inference_time = end_time - start_time
#     print(f"Inference time: {inference_time * 1000:.2f} ms")



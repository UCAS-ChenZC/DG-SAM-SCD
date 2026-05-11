from torchvision import models
import time
from models.DT_SCD.layers import *
from einops import rearrange
from models.DT_SCD.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from models.DT_SCD.RGBD_Block1207 import DFormerv2Attention, MultiStageRGBDFusion
from models.DT_SCD.RGBD_Block1209 import AdaptiveDFormer


class CNN_base(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(CNN_base, self).__init__()
        resnet = models.resnet34(pretrained)    #resnet34 --> 64 64 128 256 512
        # resnet = models.resnet50(pretrained)    #resnet50 --> 64 256 512 1024 2048
        # resnet = models.resnet18(pretrained)    #resnet50 --> 64 256 512 1024 2048
        newconv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        
        # 替换 layer3, layer4 中的 stride 为 1，并设置对应的 dilation
        # def _nostride_dilate(m, dilate):
        #     classname = m.__class__.__name__
        #     if classname.find('Conv') != -1:
        #         if m.stride == (2, 2):
        #             m.stride = (1, 1)
        #             if m.kernel_size == (3, 3):
        #                 m.dilation = (dilate // 2, dilate // 2)
        #                 m.padding = (dilate // 2, dilate // 2)
        #         elif m.kernel_size == (3, 3):
        #             m.dilation = (dilate, dilate)
        #             m.padding = (dilate, dilate)
        # self.layer3 = resnet.layer3.apply(lambda m: _nostride_dilate(m, dilate=2))
        # self.layer4 = resnet.layer4.apply(lambda m: _nostride_dilate(m, dilate=2))
        
        self.mlfa = Multi_Level_Feature_Aggreagation()
        # self.CC = CBA1x1(256,64)
        # self.upsamplex2 = nn.Upsample(scale_factor=2)
        # self.conv_pred = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.PFA = Progressive_Feature_Aggregation()
        
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

    def forward(self, x):
        x = self.layer0(x)  # x(8 64 256 256)
        x = self.maxpool(x)  # x(8 64 128 128)
        x_low = self.layer1(x)  # x(8 64 128 128)
        x1 = self.layer2(x_low)  # x1(8 128 64 64)
        
        x2 = self.layer3(x1)    # x2(8 256 32 32)
        x3 = self.layer4(x2)    # x3(8 512 16 16)
        
        x = self.mlfa(x1, x2, x3)
        # x = self.PFA(x1, x2, x3)
        return x, x_low     #x(8 128 64 64) x_low(8 64 128 128)


class Trans(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=64, dropout=0.):
        super(Trans, self).__init__()
        self.dim = dim
        self.token_len = 4
        self.conv_a = nn.Conv2d(self.dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.with_pos = 'learned'
        if self.with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*2, self.dim))
        decoder_pos_size = 256//4
        self.with_decoder_pos = None
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder =nn.Parameter(torch.randn(1, 32,
                                                                 decoder_pos_size,
                                                                 decoder_pos_size))
        self.transformer = Transformer(dim=self.dim, depth=1, heads=8,
                                       dim_head=64,
                                       mlp_dim=64, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=self.dim, depth=8,
                            heads=8, dim_head=64, mlp_dim=64, dropout=0,
                                                      softmax=False,similarity_type='tanh')
        
    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens
    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x
    
    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)  #x(1 4096 128)  (1 16384 64)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x
    
    def forward(self, x1, x2):
        # extract tokens
        token1 = self._forward_semantic_tokens(x1)  #token1(8 4 128)    (1 4 64)
        token2 = self._forward_semantic_tokens(x2)
        # token1_dep = self._forward_semantic_tokens(x1_dep)
        # token2_dep = self._forward_semantic_tokens(x2_dep)
        tokens_ = torch.cat([token1, token2], dim=1)    # cat --> tokens --> transformer encoder
        # tokens_dep_ = torch.cat([token1_dep, token2_dep], dim=1)    # cat --> tokens --> transformer encoder
        # forward transformer Encoder
        tokens = self._forward_transformer(tokens_)
        # tokens_dep = self._forward_transformer(tokens_dep_)
        token1, token2 = tokens.chunk(2, dim=1)    #split into two parts
        # token1_dep, token2_dep = tokens_dep.chunk(2, dim=1)
        # forward transformer Decoder
        x1 = self._forward_transformer_decoder(x1, token1)  #x1(8 32 64 64)     Q:x1 K,V: token1(new)
        x2 = self._forward_transformer_decoder(x2, token2)  #x2(8 32 64 64)     Q:x2 K,V: token2(new)
        return x1, x2


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
class BTSCD(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, ratio = 0.5):
        super(BTSCD, self).__init__()
        # self.FCN = FCN(in_channels, pretrained=True)
        # self.FCN = CNN_base(in_channels, pretrained=True, model='resnet34')
        self.FCN = CNN_base(in_channels, pretrained=True)

        self.change_specific_transfer = Change_Specific_Transfer(128) # BCFE
        # self.change_specific_transfer2 = Change_Specific_Transfer(64) # BCFE
        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)
        # self.DecCD = Progressive_Decoder(128, 64)
        # self.Dec1 = Progressive_Decoder(128, 64)
        # self.Dec2 = Progressive_Decoder(128, 64)

        self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)
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
        self.RGBD_multi = MultiStageRGBDFusion(embed_dim=128, num_heads=16, num_stages=3)
        self.conv4 = Conv1Relu(256, 128)
        # self.AdaptiveDFormer = AdaptiveDFormer(128,16)
        # self.convd2x = UpsampleConvLayer(64, 64, kernel_size=4, stride=2)
        # self.dense_2x = nn.Sequential( ResidualBlock(64))
        # self.convd1x = UpsampleConvLayer(64, 64, kernel_size=4, stride=2)
        # self.dense_1x = nn.Sequential( ResidualBlock(64))
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
    # def forward(self, x1, x2):
        x_size = x1.size()
        
    # 提取RGB特征
        x1 = torch.cat([x1, x3], dim=1)  # (B, 4, H, W)
        x2 = torch.cat([x2, x4], dim=1)  # (B, 4, H, W)
        x1_FCN, x1_low = self.FCN(x1)   #Backbone输出： x(8 128 64 64) x_low(8 64 128 128)
        x2_FCN, x2_low = self.FCN(x2)
        # x1_FCN = self.id_x1_after_FCN(x1_FCN)     #可视化代码
        # x2_FCN = self.id_x2_after_FCN(x2_FCN)     #可视化代码
    # 进行 Transformer-based 增强和交互
        x1t, x2t = self.Trans(x1_FCN, x2_FCN)
        x1 = x1_FCN + x1t
        x2 = x2_FCN + x2t
    # RGB-D 融合
        # x1_fuse = self.RGBD_multi(x1,x3)
        # x2_fuse = self.RGBD_multi(x2,x4)
        x1_fuse = self.RGBD(x1,x3)
        x2_fuse = self.RGBD(x2,x4)
        # x1_fuse = self.AdaptiveDFormer(x1, x3)
        # x2_fuse = self.AdaptiveDFormer(x2, x4)
        x1 = self.conv4(torch.cat([x1, x1_fuse], 1))
        x2 = self.conv4(torch.cat([x2, x2_fuse], 1))
        # x1 = self.id_x1_after_trans(x1)   #可视化代码
        # x2 = self.id_x2_after_trans(x2)   #可视化代码
    # 对 low 特征也做Transformer
        # x1_low, x2_low = self.Trans_low(x1_low, x2_low)
        # x1_low = self.low_conv(x1_low)
        # x2_low = self.low_conv(x2_low)
####
        xc = self.change_specific_transfer(x1, x2)  #融合x1,x2特征 --> xc(8 128 64 64)
        xc = self.SBRM_128(xc)
        
        x1 = self.Dec1(x1, x1_low)  # x(8 128 64 64) x_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合
        x2 = self.Dec2(x2, x2_low)
        
        # xc_low = torch.abs(x1 - x2) # (B, 64, 128, 128) 融合以后的特征在128*128尺度 直接做差取绝对值
        xc_low = torch.cat([x1, x2], dim=1)  # (B, 128, 128, 128)
        xc_low = self.conv_diff(xc_low)  # (B, 64, 128, 128)
        xc_low = self.SBRM_64(xc_low)
        
        xc = self.DecCD(xc, xc_low)  # xc(8 128 64 64) xc_low(8 64 128 128) --> (8 64 128 128) 其实就是一层多尺度融合   !!!又用了一次多尺度融合!!!
        # xc1 = self.xc_hook(xc1)   #    xc_fuse可视化代码
    # Classifier
        init_BCD_change = self.classifierCD(xc)  #(8 2 128 128)  先初步预测变化图 用于后续的任务交互模块 即BCD任务
        # BCD_change = self.BCD_1_hook(BCD_change)   #可视化代码
        # new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, init_BCD_change)  #new_xc(8 64 128 128)
        new_xc, pixel_sim_loss = self.depth_change_interaction_module(x1, x2, xc, init_BCD_change)  #new_xc(8 64 128 128)
        
        
        final_BCD_change = self.classifierCD(new_xc)  #new_change(8 2 128 128)  最终变化图预测 交互后，又输出了一次BCD任务的结果？？？？
        # new_change = self.BCD_2_hook(new_change)   #可视化代码

        out1 = self.classifierSem1(x1)  #out1(8 7 128 128)
        out2 = self.classifierSem2(x2)
        
        out1 = F.upsample(out1, x_size[2:], mode='bilinear')    #(8 7 128 128) --> (8 7 512 512)
        # # # out1 = self.id_out1(out1)
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        # # # out2 = self.id_out2(out2)
        change_out = F.upsample(final_BCD_change, x_size[2:], mode='bilinear')
        # out1 = self.Dysample_out(out1)    #(8 7 128 128) --> (8 7 512 512)
        # out2 = self.Dysample_out(out2)
        # change_out = self.Dysample_chage_out(final_BCD_change)

        # boundary_x1 = self.boundary_decoder(x1, x_size[2:])
        # boundary_x2 = self.boundary_decoder(x2, x_size[2:])
        # # boundary_x1 = self.Dysample_BD(x1)
        # # boundary_x2 = self.Dysample_BD(x2)
        
        # boundary_change = self.boundary_decoder(new_xc, x_size[2:])
        # # boundary_change = self.Dysample_BD(new_xc)

        # boundary_sem = self.eca(boundary_x1 + boundary_x2)
        # boundary_sem = self.boundary_classifier(boundary_sem)
        # boundary_change = self.boundary_classifier(boundary_change)

        # return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change
        return out1, out2, change_out, pixel_sim_loss

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # 最大池化 & 平均池化
        max_out = self.max_pool(x).view(b, c)  # 变为 (b, c)
        avg_out = self.avg_pool(x).view(b, c)  # 变为 (b, c)
        # 经过 MLP 并求和
        max_out = self.mlp(max_out)
        avg_out = self.mlp(avg_out)
        out = max_out + avg_out
        # 通过 Sigmoid
        out = self.sigmoid(out).view(b, c, 1, 1)
        return out    

class SpatialAttention(nn.Module): #空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class SBRM(nn.Module):
    '''Dual Attention Fusion Module'''
    def __init__(self, channel, reduction=16, kernel_size=7,input_size=(128,128)):
        super(SBRM, self).__init__()
        self.channelattention = ChannelAttention(channel, reduction=reduction)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.Conv3_1 = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.Conv3_2 = nn.Sequential(
                    nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        self.Conv3_3 = nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel),
                    nn.ReLU())
        # self.dilated_conv_1 = nn.Sequential(
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        # self.dilated_conv_2 = nn.Sequential(
        #             nn.Conv2d(channel * 2, channel * 2,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel * 2, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        # self.dilated_conv_3 = nn.Sequential(
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=2, dilation=2),
        #             nn.Conv2d(channel, channel,kernel_size=3, padding=4, dilation=4),
        #             nn.BatchNorm2d(num_features=channel),
        #             nn.ReLU())
        self.Conv1 = nn.Conv2d(channel * 2,channel * 2,kernel_size=1)
        H,W = input_size
        # self.FAM = FAM_Module(channel,channel,shapes=H)
    def forward(self,x):
        
        x1 = x
        fc = x1 * self.channelattention(x1)     #Wc = self.channelattention(x1)
        fc = self.Conv3_1(fc)              #   dilated_conv    Conv3_1     FAM
        fs = fc * self.spatialattention(fc)     #Ws = self.spatialattention(fc)
        cat = torch.concat([x1,fs],dim=1)
        fd = cat * self.Conv1(cat)
        fd = self.Conv3_2(fd)
        # f_out_res = self.Conv3_3(f_out)
        f_out_res = self.Conv3_3(fd)
        f_out = fd + f_out_res
        x  = x + f_out
        return x


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



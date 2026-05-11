import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.DT_SCD.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from einops import rearrange

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
    
class Trans(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=64, dropout=0.):
        super(Trans, self).__init__()
        self.dim = dim
        self.token_len = 4
        self.conv_a = nn.Conv2d(self.dim, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.with_pos = 'learned'
        if self.with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len*3, self.dim))
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
    
    def forward(self, x1, x2, x3):
        # extract tokens
        token1 = self._forward_semantic_tokens(x1)  #token1(8 4 128)    (1 4 64)
        token2 = self._forward_semantic_tokens(x2)
        token3 = self._forward_semantic_tokens(x3)
        # token1_dep = self._forward_semantic_tokens(x1_dep)
        # token2_dep = self._forward_semantic_tokens(x2_dep)
        tokens_ = torch.cat([token1, token2, token3], dim=1)    # cat --> tokens --> transformer encoder
        # tokens_dep_ = torch.cat([token1_dep, token2_dep], dim=1)    # cat --> tokens --> transformer encoder
        # forward transformer Encoder
        tokens = self._forward_transformer(tokens_)
        # tokens_dep = self._forward_transformer(tokens_dep_)
        token1, token2, token3 = tokens.chunk(3, dim=1)    #split into three parts
        # token1_dep, token2_dep = tokens_dep.chunk(2, dim=1)
        # forward transformer Decoder
        x1 = self._forward_transformer_decoder(x1, token1)  #x1(8 32 64 64)     Q:x1 K,V: token1(new)
        x2 = self._forward_transformer_decoder(x2, token2)  #x2(8 32 64 64)     Q:x2 K,V: token2(new)
        x3 = self._forward_transformer_decoder(x3, token3)  #x3(8 32 64 64)     Q:x3 K,V: token3(new)
        return x1, x2, x3
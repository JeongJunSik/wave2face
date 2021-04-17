"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from core.wing import FAN



class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.num_features = num_features
        self.style_dim = style_dim
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # print("@@ adain num_features", self.num_features)
        # print("@@ adain style_dim: ", self.style_dim)
        # print("@@ adain x.shape: ", x.shape)
        # print("@@ adain s.shape: ", s.shape)
        h = self.fc(s)
        #print("@@ adin h.shape: ", h.shape)
        h = h.view(h.size(0), h.size(1), 1, 1)
        #print("@@ adin view h.shape: ", h.shape)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        # x.shape: (batch*sync_t, d, h, w)
        # s.shape: (batch*sync, d)
        x = self.norm1(x, s)
        #print("adainresbik norm1 x.shape: ", x.shape)
        x = self.actv(x)
        #print("adainresbik actv x.shape: ", x.shape)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        #print("adainresbik upsample x.shape: ", x.shape)
        x = self.conv1(x)
        #print("adainresbik conv1 x.shape: ", x.shape)
        x = self.norm2(x, s)
        #print("adainresbik norm2 x.shape: ", x.shape)
        x = self.actv(x)
        
        x = self.conv2(x)
        #print("adainresbik conv2 x.shape: ", x.shape)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=512, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(6, dim_in, 3, 1, 1)# landmark guided, input channel: 3->6
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0),
            nn.Tanh())

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        # if w_hpf > 0:
        #     device = torch.device(
        #         'cuda' if torch.cuda.is_available() else 'cpu')
        #     self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, gt, masks=None):
        # x.shape: (batch*sync_t, c*2, h, w) , gt_mask
        # s.shape: (batch*sync_t, 512)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        zero_tensor = torch.cuda.FloatTensor( [0.] ).to(device)
        
        residual = x[:, 3:].clone().detach()
        gt = gt.detach()
        x = x.detach()
        
        x = self.from_rgb(x)
        cache = {}

        """
        @@ generator: encode block0, x.shape:torch.Size([5, 128, 128, 128]) 
        @@ generator: encode block1, x.shape:torch.Size([5, 256, 64, 64]) 
        @@ generator: encode block2, x.shape:torch.Size([5, 512, 32, 32]) 
        @@ generator: encode block3, x.shape:torch.Size([5, 512, 16, 16]) 
        @@ generator: encode block4, x.shape:torch.Size([5, 512, 16, 16]) 
        @@ generator: encode block5, x.shape:torch.Size([5, 512, 16, 16]) 
        """

        """
        @@ generator: decode block0, x.shape:torch.Size([5, 512, 16, 16]) 
        @@ generator: decode block1, x.shape:torch.Size([5, 512, 16, 16]) 
        @@ generator: decode block2, x.shape:torch.Size([5, 512, 32, 32]) 
        @@ generator: decode block3, x.shape:torch.Size([5, 256, 64, 64]) 
        @@ generator: decode block4, x.shape:torch.Size([5, 128, 128, 128]) 
        @@ generator: decode block5, x.shape:torch.Size([5, 64, 256, 256]) 
        """

        for i, block in enumerate(self.encode):
            x = block(x)
        for j, block in enumerate(self.decode):
            x = block(x, s)
        
        x = self.to_rgb(x)

        zero_tensor = torch.cuda.FloatTensor( [-1.] ).to(device)
        fake_reverse_masked = torch.where( gt==residual, zero_tensor, x )

        fake_reverse_masked_denorm = (fake_reverse_masked+1)/2
        residual_denorm = (residual+1)/2
        result = fake_reverse_masked_denorm + residual_denorm
        result_norm = 2*result-1

        return result_norm


class StyleEncoder(nn.Module):
    """
    out = torch.cat((x,y),dim = -3) #out 6*224*224
    out = self.pad(out) #out 6*256*256
    out = self.resDown1(out) #out 64*128*128
    out = self.resDown2(out) #out 128*64*64
    out = self.resDown3(out) #out 256*32*32
    
    out = self.self_att(out) #out 256*32*32
    
    out = self.resDown4(out) #out 512*16*16
    out = self.resDown5(out) #out 512*8*8
    out = self.resDown6(out) #out 512*4*4
    
    out = self.sum_pooling(out) #out 512*1*1
    out = self.relu(out) #out 512*1*1
    out = out.view(-1,512,1) #out B*512*1
    """
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)] # landmark guided, input channel: 3->6

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

    def forward(self, x):
        # x.shape: (batch*sync_t, c*2, h,w)
        h = self.shared(x) #(batch*sync_t, 512, 1, 1)
        h = h.view(h.size(0), -1) #(batch*sync_t, 512)
        s = h 
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        #blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)] # landmark guided, input channel: 6

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1 , 1, 1, 0)] # num_domain = 1(dim=1)
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        # x.shape: (batch * sync_t * 2, c, h, w)
        out = self.main(x) # (batch*sync_t, 1, 1, 1)
        out = out.view(out.size(0), -1)  # (batch*sync_t, 1)
        # idx = torch.LongTensor(range(y.size(0))).to(y.device)
        # out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    #mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim)
    discriminator = Discriminator(args.img_size) #, args.num_domains)
    
    """ not use moving average """
    # generator_ema = copy.deepcopy(generator)
    # mapping_network_ema = copy.deepcopy(mapping_network)
    # style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 #mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    # nets_ema = Munch(generator=generator_ema,
    #                  mapping_network=mapping_network_ema,
    #                  style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = FAN(fname_pretrained=args.wing_path).eval()
    #     nets.fan = fan
    #     # nets_ema.fan = fan

    return nets
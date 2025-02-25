# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 
import copy
import math
import sys
'''
Computational complexity:       26.69 GMac
Number of parameters:           7.57 M 
'''
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out

class TriPixelMambaLayer(nn.Module):
    '''
    1. use mamba v3 (3 direction)
    2. divide --> mamba --> construct

    p: 將各軸切成p等分 (不是window size)
    '''
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2,num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.triMamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )

     
        # self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)
      
        # adjust the window size here to fit the feature map
        self.p = p
        self.p1 = p
        self.p2 = p
        self.p3 = p
        # self.mamba_forw.out_proj = nn.Identity()
        # self.mamba_backw.out_proj = nn.Identity()
       


    def forward(self, x):
        
        B, C = x.shape[:2]
        assert C == self.dim
        img_dims = x.shape[2:]
         
        Z,H,W = x.shape[2:]

        # divide
        if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
            x_div = x.reshape(B, C, Z//self.p1, self.p1, H//self.p2, self.p2, W//self.p3, self.p3)
            x_div = x_div.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous().view(B*self.p1*self.p2*self.p3, C, Z//self.p1, H//self.p2, W//self.p3)
        else:
            x_div = x
        # prepare and go through mamba
        NB = x_div.shape[0]
        NZ,NH,NW = x_div.shape[2:]
        
        n_tokens = x_div.shape[2:].numel()
        

        x_flat = x_div.reshape(NB, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        
        x_out = self.triMamba(x_norm)

        # x_out = self.out_proj(x_mamba)
        
        # construct
        if Z%self.p1==0 and H%self.p2==0 and W%self.p3==0:
            x_out = x_out.transpose(-1, -2).reshape(B, self.p1, self.p2, self.p3, C, NZ, NH, NW).permute(0, 4, 5, 1, 6, 2, 7, 3).contiguous().reshape(B, C, *img_dims)
        else:
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)

        out = x_out + x

        return out



class TriWindowMambaLayer(nn.Module):
    def __init__(self, dim, p, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.p = p
        self.triMamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )

     
        # self.out_proj = copy.deepcopy(self.mamba_forw.out_proj)

      

       
        # self.mamba_forw.out_proj = nn.Identity()
        # self.mamba_backw.out_proj = nn.Identity()
       

    def forward(self, x):
        
        ll = len(x.shape)
        B, C = x.shape[:2]
        assert C == self.dim
        img_dims = x.shape[2:]


        #------------------
        #!!!!!!!!!!!!!!
        #self.p = self.p*4

        # Pooling
        Z,H,W = x.shape[2:]

        if Z%self.p==0 and H%self.p==0 and W%self.p==0:
            pool_layer = nn.AvgPool3d(self.p, stride=self.p)
            x_div = pool_layer(x)
        else:
            x_div = x
        

        # TriMamba
        NZ,NH,NW = x_div.shape[2:]
        n_tokens = x_div.shape[2:].numel()
        x_flat = x_div.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_out = self.triMamba(x_norm)

        #x_out = self.out_proj(x_mamba)
        # Unpooling
        if Z%self.p==0 and H%self.p==0 and W%self.p==0:
            unpool_layer = nn.Upsample(scale_factor=self.p, mode='trilinear')
            x_out = x_out.transpose(-1, -2).reshape(B, C, NZ, NH, NW)
            x_out = unpool_layer(x_out)
        else:
            x_out = x_out.transpose(-1, -2).reshape(B, C, *img_dims)

                
        out = x_out + x

        return out

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.dims = dims
        self.depths = depths
        self.out_indices = out_indices

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        num_slices_list = [48, 24, 12, 6]
        window_size_list = [8, 8, 4, 3]
        pool_list = [2, 2, 2, 2]
        for i in range(len(depths)):
            stage= []
            for j in range(depths[i]):
                pim = TriPixelMambaLayer(
                    dim=dims[i],
                    p=num_slices_list[i] // window_size_list[i],
                    num_slices=window_size_list[i]
                )
                pam = TriWindowMambaLayer(
                    dim=dims[i],
                    p=pool_list[i],
                    num_slices=num_slices_list[i] // pool_list[i]
                )
 
                LMblock = nn.Sequential(pim, pam)
                stage.append(LMblock)

            self.stages.append(nn.Sequential(*stage))


        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(len(self.depths)):

            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.vit(x_in)
        # outs.shape = [[48*3,48c],[24*3,96c],[12*3,192c],[6*3,384c]])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)           
        return dec2

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    device = torch.device("cuda:0")
    net  = SegMamba(in_chans=1,
                        out_chans=4,
                        depths=[2,2,2,2],
                        feat_size=[64, 96, 128, 160]).to(device)
    macs, params = get_model_complexity_info(net, (1, 96, 96, 96), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


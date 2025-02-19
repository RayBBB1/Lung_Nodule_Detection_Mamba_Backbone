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

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

'''
Computational complexity (with ECA):   63.74 GMac
Number of parameters (with ECA):   9.85 M  
'''

class LayerNorm(nn.Module):
    r"""
    LayerNorm 層，支援兩種資料格式：channels_last (預設) 或 channels_first。
    ... (省略 LayerNorm 註解，與之前版本相同)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        # ... (省略 LayerNorm 初始化，與之前版本相同)
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 可學習的增益參數
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # 可學習的偏差參數
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Data format '{self.data_format}' not implemented, should be 'channels_last' or 'channels_first'.")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ... (省略 LayerNorm 前向傳播，與之前版本相同)
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ECAModule(nn.Module):
    """
    ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://doi.org/10.48550/arXiv.1910.03151
    https://github.com/BangguWu/ECANet
    """

    def __init__(self, n_channels, k_size=3, gamma=2, b=1):
        super(ECAModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
	
	# https://github.com/BangguWu/ECANet/issues/243 
	# dynamically computing the k_size 
        t = int(abs((math.log(n_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
	
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # feature descriptor on the global spatial information
        y = self.global_avg_pool(x)

        # y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1) # b, c, z, h, w = x.size()
        y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-2, -1)).transpose(-2, -1).unsqueeze(-2).unsqueeze(-1) # b, c, w, h, z = x.size()

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MambaLayer(nn.Module):
    """
    Mamba 層，包含 LayerNorm、Mamba SSM 模組和 ECA 模組 (可選)。
    ... (省略 MambaLayer 註解，與之前版本類似，但需更新以包含 ECA)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None, use_eca=True): # 添加 use_eca 參數，預設為 True
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.eca = ECAModule(dim) if use_eca else None # 根據 use_eca 參數決定是否使用 ECA 模組

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數。
        ... (省略 MambaLayer 前向傳播註解，與之前版本類似，但需更新以包含 ECA)
        """
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim, f"Input channel {C} not match configured dimension {self.dim}"
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip

        if self.eca is not None: # 如果啟用了 ECA 模組，則應用 ECA
            out = self.eca(out)

        return out


class MlpChannel(nn.Module):
    """
    通道 MLP (多層感知器) 模組。
    ... (省略 MlpChannel 註解，與之前版本相同)
    """
    def __init__(self, hidden_size, mlp_dim):
        # ... (省略 MlpChannel 初始化，與之前版本相同)
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數。
        ... (省略 MlpChannel 前向傳播，與之前版本相同)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class GSC(nn.Module):
    """
    GSC (Gated Spatial Convolution) 模組。
    ... (省略 GSC 註解，與之前版本相同)
    """
    def __init__(self, in_channles: int) -> None:
        # ... (省略 GSC 初始化，與之前版本相同)
        super().__init__()
        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數。
        ... (省略 GSC 前向傳播，與之前版本相同)
        """
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual


class MambaEncoder(nn.Module):
    """
    Mamba 編碼器。
    ... (省略 MambaEncoder 註解，與之前版本類似，但需更新以包含 ECA 配置)
    """
    def __init__(
        self,
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        use_eca=True # 添加 use_eca 參數，預設為 True，控制是否在 MambaLayer 中使用 ECA
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [48, 24, 12, 6]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i], use_eca=use_eca) for j in range(depths[i])] # 將 use_eca 傳遞給 MambaLayer
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        提取編碼器特徵。
        ... (省略 MambaEncoder 前向特徵提取註解，與之前版本相同)
        """
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        前向傳播函數。
        ... (省略 MambaEncoder 前向傳播註解，與之前版本相同)
        """
        x = self.forward_features(x)
        return x


class SegMamba(nn.Module):
    """
    SegMamba 模型。
    ... (省略 SegMamba 註解，與之前版本類似，但需更新以包含 ECA 配置)
    """
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name="instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        use_eca=True # 添加 use_eca 參數，預設為 True，控制是否在 MambaLayer 中使用 ECA
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
        self.use_eca = use_eca # 保存 use_eca 參數

        self.vit = MambaEncoder(
            in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            use_eca=use_eca # 將 use_eca 傳遞給 MambaEncoder
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

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=out_chans,
        )

    def proj_feat(self, x): # 這個方法在 forward 中未使用，可以考慮移除
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函數。
        ... (省略 SegMamba 前向傳播註解，與之前版本相同)
        """
        outs = self.vit(x_in)
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)

        segmentation_output = self.out(dec2)
        return segmentation_output


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    device = torch.device("cuda:0")
    net = SegMamba(in_chans=1, out_chans=4, depths=[2, 2, 2, 2], feat_size=[64, 96, 128, 160], use_eca=True).to(device) # 預設使用 ECA
    macs, params = get_model_complexity_info(
        net, (1, 96, 96, 96), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity (with ECA): ", macs)) # 更新提示資訊
    print("{:<30}  {:<8}".format("Number of parameters (with ECA): ", params)) # 更新提示資訊

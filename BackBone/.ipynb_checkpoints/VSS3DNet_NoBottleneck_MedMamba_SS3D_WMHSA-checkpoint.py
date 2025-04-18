# from fvcore.nn import FlopCountAnalysis
import ipdb
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    # from flopth import flopth
    from fvcore.nn import FlopCountAnalysis
except:
    pass
try:
    from VSS3D import VSSLayer3D
    from Transformer import TransformerBottleneck
except:
    from models.SS3D import SS3D, SS3D_v5, SS3D_v6
    from models.VSS3D import VSSLayer3D
    from models.Transformer import TransformerBottleneck
"""
Computational complexity:       95.35 GMac
Number of parameters:           5.5 M  
"""
class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class ResConvBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, id = True, preact = True): #id was False
        super().__init__()
        if preact:
            self.conv = nn.Sequential(
                nn.GroupNorm(8, ch_in),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
                nn.GroupNorm(8, ch_out),
                nn.ReLU(inplace = True),
                )
        id = (ch_in == ch_out) and id
        self.identity = (nn.Identity()) if id else nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, inp):
        x = self.conv(inp)
        residual = self.identity(inp)
        return residual + x
# to_3tuple, window_partition_3d, window_reverse_3d (來自上次回應)
def to_3tuple(x):
    """將輸入轉換為 3 維元組。"""
    if isinstance(x, tuple):
        assert len(x) == 3, "窗口大小的元組長度必須為 3"
        return x
    return (x, x, x)

def window_partition_3d(x: torch.Tensor, window_size: tuple[int, int, int]):
    """將張量分割成 3D 窗口。"""
    B, C, D, H, W = x.shape
    wD, wH, wW = to_3tuple(window_size)
    assert D % wD == 0 and H % wH == 0 and W % wW == 0, \
        f"特徵圖維度 ({D},{H},{W}) 必須能被窗口大小 ({wD},{wH},{wW}) 整除"
    x = x.view(B, C, D // wD, wD, H // wH, wH, W // wW, wW)
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = windows.view(-1, wD * wH * wW, C)
    return windows

def window_reverse_3d(windows: torch.Tensor, window_size:tuple[int, int, int], D: int, H: int, W: int):
    """將窗口還原為 3D 特徵圖。"""
    wD, wH, wW = to_3tuple(window_size)
    num_windows_D = D // wD
    num_windows_H = H // wH
    num_windows_W = W // wW
    B = int(windows.shape[0] / (num_windows_D * num_windows_H * num_windows_W))
    C = windows.shape[2]
    x = windows.view(B, num_windows_D, num_windows_H, num_windows_W, wD, wH, wW, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(B, C, D, H, W)
    return x

# Attention3D (來自上次回應)
class Attention3D(nn.Module):
    """ 從 MambaVision 改編的 3D 注意力模塊。 """
    def __init__(
            self,
            dim, num_heads=8, qkv_bias=False, qk_norm=False,
            attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = hasattr(F, 'scaled_dot_product_attention')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_win_vol, N_win, C = x.shape
        qkv = self.qkv(x).reshape(B_win_vol, N_win, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B_win_vol, N_win, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
def channel_shuffle_3d(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, depth, height, width = x.size()
    assert num_channels % groups == 0, f"num_channels ({num_channels}) must be divisible by groups ({groups})"
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, depth, height, width] -> [batch_size, groups, channels_per_group, depth, height, width]
    x = x.view(batch_size, groups, channels_per_group, depth, height, width)

    # transpose groups and channels_per_group
    x = torch.transpose(x, 1, 2).contiguous()  # 交換第1維(groups)和第2維(channels_per_group)

    # flatten
    x = x.view(batch_size, num_channels, depth, height, width)

    return x   
class SS_Conv_SSM_3D(nn.Module): # 維持原始類名
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer= partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0, # *** 假設這是給 SS3D 的 dropout ***
        d_state: int = 16,
        expansion_factor = 1,
        mlp_drop_rate=0., # 未使用
        orientation = 0,
        scan_type = 'scan',
        size = 12, # 未使用 (除非 SS3D 需要)
        # --- 注意力參數 ---
        window_size = (6,6,6), # 默認值來自您的簽名
        num_heads = 4,         # 默認值來自您的簽名
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.,          # *** 假設這是給 Attention3D 的 attention dropout ***
        proj_drop=0.,          # *** 假設這是給 Attention3D 的 projection dropout ***
        **kwargs,
    ):
        super().__init__()
        if hidden_dim == 0:
             raise ValueError("hidden_dim 不能為 0")
        assert hidden_dim % 2 == 0, "hidden_dim 必須是偶數"
        dim_half = hidden_dim // 2
        self.window_size = to_3tuple(window_size)

        # --- 左路徑：卷積 (來自原始 SS_Conv_SSM_3D) ---
        self.conv_path = nn.Sequential(
            nn.BatchNorm3d(dim_half), # 注意：原始碼這裡使用 BatchNorm，與右路徑的 LayerNorm 不同
            nn.Conv3d(in_channels=dim_half, out_channels=dim_half, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_half),
            nn.ReLU(),
            nn.Conv3d(in_channels=dim_half, out_channels=dim_half, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim_half),
            nn.ReLU(),
            nn.Conv3d(in_channels=dim_half, out_channels=dim_half, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # --- 右路徑：注意力 -> SSM ---
        # 注意力部分
        self.norm_attn = norm_layer(dim_half) # 注意力前的歸一化
        self.attn = Attention3D(
            dim=dim_half,
            num_heads=num_heads,         # 使用傳入的 num_heads
            qkv_bias=qkv_bias,         # 使用傳入的 qkv_bias
            qk_norm=qk_norm,           # 使用傳入的 qk_norm
            attn_drop=attn_drop,       # 使用傳入的 attn_drop
            proj_drop=proj_drop,       # 使用傳入的 proj_drop
            norm_layer=norm_layer,     # 使用傳入的 norm_layer
        )
        # SSM 部分
        self.norm_ssm = norm_layer(dim_half) # SSM 前的歸一化
        print(f"SSM Orientation: {orientation}", end='') # 顯示 SSM 方向
        self.ssm = SS3D(
            d_model=dim_half,
            dropout=attn_drop_rate,    # 使用 attn_drop_rate 作為 SS3D dropout
            d_state=d_state,           # 使用傳入的 d_state
            expand=expansion_factor,   # 使用傳入的 expansion_factor
            # 傳遞 orientation, scan_type 等給 SS3D (如果它接受這些參數)
            # 例如：orientation=orientation, scan_type=scan_type
            **kwargs                   # 傳遞其他參數給 SS3D
        )
        # DropPath 應用於 SSM 輸出之後
        self.drop_path_ssm = DropPath(drop_path) if drop_path > 0. else nn.Identity() # 使用傳入的 drop_path

    def forward(self, x: torch.Tensor):
        # 輸入 x: (B, C, D, H, W) 其中 C == hidden_dim
        B, C, D, H, W = x.shape
        wD, wH, wW = self.window_size
        dim_half = C // 2

        # 1. 通道分割
        input_left, input_right = x.chunk(2, dim=1) # 每個 (B, C/2, D, H, W)

        # -------------------------------------
        # 2. 左路徑 (卷積)
        # -------------------------------------
        processed_left = self.conv_path(input_left) # (B, C/2, D, H, W)

        # -------------------------------------
        # 3. 右路徑 (注意力 -> SSM)
        # -------------------------------------
        right_branch = input_right # (B, C/2, D, H, W)

        # == 注意力部分 ==
        #   a. 填充 (確保維度可被窗口大小整除)
        pad_d = (wD - D % wD) % wD
        pad_h = (wH - H % wH) % wH
        pad_w = (wW - W % wW) % wW
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            right_branch_padded = F.pad(right_branch, (0, pad_w, 0, pad_h, 0, pad_d))
        else:
            right_branch_padded = right_branch
        # 獲取填充後的維度
        Bp, Cp_half, Dp, Hp, Wp = right_branch_padded.shape

        #   b. 注意力前的歸一化 (通道移到最後)
        right_permuted_attn = right_branch_padded.permute(0, 2, 3, 4, 1).contiguous() # B, Dp, Hp, Wp, C/2
        right_norm_attn = self.norm_attn(right_permuted_attn)

        #   c. 窗口分割 (通道移回標準位置)
        right_windows = window_partition_3d(right_norm_attn.permute(0, 4, 1, 2, 3).contiguous(), self.window_size) # B*nW, win_vol, C/2

        #   d. 應用注意力
        attn_windows = self.attn(right_windows) # B*nW, win_vol, C/2

        #   e. 窗口還原 (需要傳遞填充後的維度 Dp, Hp, Wp)
        attn_out_padded = window_reverse_3d(attn_windows, self.window_size, Dp, Hp, Wp) # B, C/2, Dp, Hp, Wp

        #   f. 移除填充
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            attn_out = attn_out_padded[:, :, :D, :H, :W].contiguous()
        else:
            attn_out = attn_out_padded
        # attn_out: (B, C/2, D, H, W) - 作為 SSM 的輸入

        # == SSM 部分 ==
        #   g. SSM 前的歸一化 (通道移到最後)
        ssm_input_permuted = attn_out.permute(0, 2, 3, 4, 1).contiguous() # B, D, H, W, C/2
        ssm_input_norm = self.norm_ssm(ssm_input_permuted)

        #   h. 應用 SSM 和 DropPath
        #      確保 SS3D 的輸入形狀是它所期望的 (可能是 B, D, H, W, C/2)
        ssm_out_permuted = self.ssm(ssm_input_norm) # 假設 SS3D 輸出 (B, D, H, W, C/2)
                                                    # **再次確認 SS3D 的輸出形狀**
        ssm_out_dropped = self.drop_path_ssm(ssm_out_permuted)

        #   i. 將維度排列回標準順序
        processed_right = ssm_out_dropped.permute(0, 4, 1, 2, 3).contiguous() # B, C/2, D, H, W

        # -------------------------------------
        # 4. 合併輸出
        # -------------------------------------
        #   a. 沿通道維度合併兩條路徑
        output = torch.cat((processed_left, processed_right), dim=1) # B, C, D, H, W

        #   b. 通道重排
        output_shuffled = channel_shuffle_3d(output, groups=2)

        #   c. 最終殘差連接 (與原始輸入 x 相加)
        final_output = output_shuffled + x

        return final_output

class upconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3, upsample = True, preact = True):
        super().__init__()
        if upsample:
            upconv = nn.Sequential(nn.Upsample(scale_factor = 2),
                                    nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            )
        else:
            upconv = nn.ConvTranspose3d(
                                        in_channels=ch_in,
                                        out_channels=ch_out,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=kernel_size//2 - (1 - kernel_size % 2),
                                        output_padding=kernel_size % 2,
                                        )
        if preact:
            act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True))# if act else (lambda x: x)
            self.up = nn.Sequential(act, upconv)
        else:
            act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True))# if act else (lambda x: x)
            self.up = nn.Sequential(upconv, act)
            
            
    def forward(self,x):
        x = self.up(x)
        return x


class down_block(nn.Module):
    def __init__(self, ch_in, ch_out, maxpool=False, id=True, preact=True, kernel_size=3, size = None):
        super().__init__()
        
        self.preact = preact
        self.act = nn.Sequential(nn.GroupNorm(8, ch_out), nn.ReLU(inplace=True))
        
        #-----------------------
        # downsample approach  
        #-----------------------
        if maxpool:
            downsample = nn.MaxPool3d(kernel_size=2, stride=2)
            resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        else:
            self.down = nn.Conv3d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size//2 - (1 - kernel_size % 2),
            )
            
        #-----------------------
        # encode block using SS_Conv_SSM_3D
        #-----------------------
        self.encode_block = SS_Conv_SSM_3D(
            hidden_dim=ch_out,
            drop_path=0.3,  # 可以根據需要調整
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.0,  # 可以根據需要調整
            d_state=16,  # 可以根據需要調整
            expansion_factor=1,
            mlp_drop_rate=0.0,  # 可以根據需要調整
            orientation=0,  # 可以根據需要調整
            scan_type='scan',  # 可以根據需要調整
            size=size  # 根據你的輸入尺寸調整
        )

    def forward(self, x):
        x = self.down(x)
        x = self.act(x)
        x = self.encode_block(x)
        return x

class MedSegMamba(nn.Module):
    def __init__(
        self,
        img_dim = 96,
        patch_dim = 8, # 2^(number of downsample/upsample stages in the encoder/decoder)
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,256], #last element is the size which the bottleneck processes
        mamba_d_state = 64, # for vss block
        num_layers = 9,
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        drop_path_rate=0.3,
        ssm_expansion_factor=1,
        scan_type = 'scan',
        id = False, #whether to have conv1x1 in the residual path when input and output channel sizes of residual block is the same
        preact = False, #whether to have preactivation residual blocks (norm>act>conv vs conv>norm>act)
        maxpool = True, #maxpool vs strided conv
        upsample = True, #upsample vs transposed conv
        full_final_block = True, # found to do better when the last resblock is large on TABSurfer, but a lot more memory
        ):
        super().__init__()
        self.hidden_dim = int((img_dim // patch_dim) ** 3)
        
        self.preconv = nn.Conv3d(img_ch, channel_sizes[1], kernel_size=3,stride=1,padding=1) if preact else (nn.Identity())
        if preact:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[1],ch_out=channel_sizes[1], id=id, preact=preact) # 96
        else:
            self.Conv1 = ResConvBlock3D(ch_in=channel_sizes[0],ch_out=channel_sizes[1], id=id, preact=preact) # 96
            
        self.downconv1 = down_block(channel_sizes[1], channel_sizes[2], maxpool, id, preact, size = 48)

        self.downconv2 = down_block(channel_sizes[2], channel_sizes[3], maxpool, id, preact, size = 24)

        self.downconv3 = down_block(channel_sizes[3], channel_sizes[4], maxpool, id, preact, size = 12)

        self.gn = nn.GroupNorm(8, channel_sizes[4])
        self.relu = nn.ReLU(inplace=True)



        self.Up4 = upconv(ch_in=channel_sizes[4],ch_out=channel_sizes[3], upsample=upsample, preact=preact)
        self.Up_conv4 = ResConvBlock3D(ch_in=channel_sizes[3]*2, ch_out=channel_sizes[3], id=id, preact=preact)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch

        self.num_layers = num_layers

    def forward(self,input):
        # encoding path
        input = self.preconv(input)
        
        x1 = self.Conv1(input)

        x2 = self.downconv1(x1)

        x3 = self.downconv2(x2)

        x = self.downconv3(x3)

        #decoding
        d4 = self.Up4(x)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        return d4

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    device = torch.device("cuda:0")
    net  = MedSegMamba(img_dim = 96,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 32,
        channel_sizes = [1,32,64,128,192], 
        mamba_d_state = 64, # for vss block
        num_layers = 6, #6 or 9
        vss_version = 'v5', # None for vanilla
        mlp_dropout_rate = 0, #0.1
        attn_dropout_rate = 0, #0.1
        drop_path_rate=0, #0
        ssm_expansion_factor=1,
        id = False, #TABSurfer default
        preact = False, #TABSurfer default
        maxpool = True, #TABSurfer default
        upsample = True, #TABSurfer default
        full_final_block=True,
        scan_type='scan'
        ).to(device)
    
    macs, params = get_model_complexity_info(net, (1, 96, 96, 96), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

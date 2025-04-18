# from fvcore.nn import FlopCountAnalysis
import ipdb
import torch
import torch.nn as nn
try:
    # from flopth import flopth
    from fvcore.nn import FlopCountAnalysis
except:
    pass
try:
    from VSS3D import VSSLayer3D
    from Transformer import TransformerBottleneck
except:
    from models.VSS3D import VSSLayer3D
    from models.Transformer import TransformerBottleneck
"""
Computational complexity:       84.51 GMac
Number of parameters:           4.58 M   
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

class downconv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size = 3, maxpool = False, act = True, preact = True):
        super().__init__()
        if maxpool:
            downconv = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                  nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True))
        else:
            downconv = nn.Conv3d(
                                in_channels=ch_in,
                                out_channels=ch_out,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size//2 - (1 - kernel_size % 2),
                                )
        if preact:
            act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True)) if act else (nn.Identity())
            self.down = nn.Sequential(act, downconv)
        else:
            act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True)) if act else (nn.Identity())
            self.down = nn.Sequential(downconv, act)
    def forward(self,x):
        x = self.down(x)
        return x

class down_block(nn.Module):
    def __init__(self, ch_in, ch_out, maxpool = False, id=True, preact = True, kernel_size = 3):
        super().__init__()
        if maxpool:
            downsample = nn.MaxPool3d(kernel_size=2,stride=2)
            resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=preact)
            self.down = nn.Sequential(downsample, resblock)
        else:
            downconv = nn.Conv3d(
                                in_channels=ch_in,
                                out_channels=ch_out,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=kernel_size//2 - (1 - kernel_size % 2),
                                )
            if preact:
                act = nn.Sequential(nn.GroupNorm(8, ch_in),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_out, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(act, downconv, resblock)
            else:
                act = nn.Sequential(nn.GroupNorm(8, ch_out),nn.ReLU(inplace = True))# if act else (lambda x: x)
                resblock = ResConvBlock3D(ch_in=ch_in, ch_out=ch_out, id=id, preact=True)
                self.down = nn.Sequential(downconv, act, resblock)

    def forward(self,x):
        x = self.down(x)
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
            
        self.downconv1 = down_block(channel_sizes[1], channel_sizes[2], maxpool, id, preact)

        self.downconv2 = down_block(channel_sizes[2], channel_sizes[3], maxpool, id, preact)

        self.downconv3 = down_block(channel_sizes[3], channel_sizes[4], maxpool, id, preact)

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

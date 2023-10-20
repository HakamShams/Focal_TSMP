# ------------------------------------------------------------
# 2D Focal Modulation Network Encoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# The original code is available at https://github.com/microsoft/FocalNet/tree/main
# and is licensed under The MIT License. Copyright (c) 2022 Microsoft
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from models.channel_attention import ChannelAttention_2D

# ------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggreation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)

        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

class FocalNetBlock(nn.Module):
    """ Focal Modulation Network Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        focal_level (int): Number of focal levels.
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether to use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layernorm after modulation
    """

    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, focal_level=1, focal_window=3,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False, use_postln_in_modulation=False,
                 normalize_modulator=False
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln

        self.norm1 = nn.LayerNorm(dim)

        self.modulation = FocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level,
            use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = None
        self.W = None

    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H * W, C)
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x


class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether to use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether to use layernorm after modulation
    """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 downsample=None, use_checkpoint=False,
                 focal_level=1, focal_window=1,
                 use_layerscale=False, layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=2,
                in_chans=out_dim,
                embed_dim=dim,
                norm_layer=None,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W

        for blk in self.blocks:
            blk.H, blk.W = Ho, Wo
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x, Ho, Wo


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple = (224, 224), patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):

        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W


class FocalNet_2D(nn.Module):
    """ 2D Focal Modulation Networks (FocalNets) Encoder """

    def __init__(self, config):
        super().__init__()

        """
        Parameters
        ----------
        config : argparse
           configuration file from config.py
        """

        self.n_lat = config.n_lat - (config.cut_lateral_boundary * 2)
        self.n_lon = config.n_lon - (config.cut_lateral_boundary * 2)

        self.out_channels = config.out_channels
        self.n_layers = len(config.out_channels)
        self.depths = config.en_depths
        self.patch_size = config.en_patch_size
        self.in_channels = config.out_channels
        self.in_chans = config.in_channels
        self.embed_dim = config.out_channels[0]
        self.mlp_ratio = config.en_mlp_ratio
        self.ape = config.en_ape
        self.drop_rate = config.en_drop_rate
        self.drop_path_rate = config.en_drop_path_rate
        self.patch_norm = config.en_patch_norm
        self.use_checkpoint = config.en_use_checkpoint

        self.pretrained = config.en_pretrained if config.phase == 'train' else None

        self.focal_levels = config.en_focal_levels
        self.focal_windows = config.en_focal_windows
        self.use_layerscale = config.en_use_layerscale
        self.layerscale_value = config.en_layerscale_value
        self.use_postln = config.en_use_postln
        self.use_postln_in_modulation = config.en_use_postln_in_modulation
        self.normalize_modulator = config.en_normalize_modulator

        # split image into patches using either non-overlapped embedding or overlapped embedding
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=(self.n_lat, self.n_lon), patch_size=self.patch_size,
                                      in_chans=self.in_chans, embed_dim=self.embed_dim,
                                      norm_layer=nn.LayerNorm if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.input_resolution = []
        for i in range(self.n_layers):
            self.input_resolution.append((patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            layer = BasicLayer(dim=self.in_channels[i_layer],
                               out_dim=self.in_channels[i_layer - 1] if (i_layer < self.n_layers) else None,
                               input_resolution=self.input_resolution[i_layer],
                               depth=self.depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=self.drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               downsample=PatchEmbed if (i_layer != 0) else None,
                               focal_level=self.focal_levels[i_layer],
                               focal_window=self.focal_windows[i_layer],
                               use_checkpoint=self.use_checkpoint,
                               use_layerscale=self.use_layerscale,
                               layerscale_value=self.layerscale_value,
                               use_postln=self.use_postln,
                               use_postln_in_modulation=self.use_postln_in_modulation,
                               normalize_modulator=self.normalize_modulator,
                               )

            self.layers.append(layer)

        self.en_channel_att = config.en_channel_att

        # create channel attention layers
        if self.en_channel_att:
            self.en_channel_att_r = config.en_channel_att_r
            self.en_channel_att_p = config.en_channel_att_p
            self.channel_attn = ChannelAttention_2D(channels_in=self.in_chans,
                                                    reduction_ratio=self.en_channel_att_r,
                                                    pooling_type=self.en_channel_att_p)

        self._init_weights()

    def _init_weights(self):
        def init_func(m):

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

        if self.pretrained:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model']

            # delete head and last norm layer
            del state_dict['head.weight'], state_dict['head.bias']
            if not self.patch_norm:
                del state_dict['norm.weight'], state_dict['norm.bias']

            # prepare patch embed weights
            m1 = state_dict['patch_embed.proj.weight'].mean()
            s1 = state_dict['patch_embed.proj.weight'].std()
            state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].mean(axis=(1, 2, 3)).\
            unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.in_chans, self.patch_size, self.patch_size)
            m2 = state_dict['patch_embed.proj.weight'].mean()
            s2 = state_dict['patch_embed.proj.weight'].std()
            state_dict['patch_embed.proj.weight'] = m1 + (state_dict['patch_embed.proj.weight'] - m2) * s1/s2

            # change downsample name since we changed their order in our implementation
            downsample_index_keys = sorted([k for k in state_dict.keys() if "downsample" in k])
            downsample_index_keys_ori = sorted([k for k in self.state_dict().keys() if "downsample" in k])
            downsample_tmp = {}

            if not self.patch_norm:
                downsample_index_keys = sorted([k for k in downsample_index_keys if "norm" not in k])

            for k, j in zip(downsample_index_keys, downsample_index_keys_ori):
                downsample_tmp[j] = state_dict[k]

            for k in downsample_tmp.keys():
                state_dict[k] = downsample_tmp[k]

            self.load_state_dict(state_dict, strict=False)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def forward(self, x):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """
        x_out = []

        N, C, H, W = x.shape

        if self.en_channel_att:
            x_scaled = self.channel_attn(x[:, :, :, :])
            x[:, :, :, :] = x[:, :, :, :] * x_scaled[:, :, None, None]

        x, H, W = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            h, w, c = self.input_resolution[i][0], self.input_resolution[i][1], self.out_channels[i]

            x_out.append(x.view(N, h, w, c).permute(0, 3, 1, 2))

        return x_out


if __name__ == '__main__':

    import config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_x = torch.randn((2, 65 + 6, 397, 409), device=device)

    model = FocalNet_2D(config_file).to(device)
    print(model)

    test = model(test_x)
    for test_x in test:
        print(test_x.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")


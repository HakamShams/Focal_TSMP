# ------------------------------------------------------------------
# 2D CNN Decoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from models.regression_head import regression_head

# ------------------------------------------------------------------

class conv_block(nn.Module):
    """ Residual CNN Block """
    def __init__(self, in_channels=96, out_channels=96):
        super(conv_block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 96)
            number of input channels
        out_channels : int (default 96)
            number of output channels
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_layer1 = nn.LayerNorm(self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_layer2 = nn.LayerNorm(self.out_channels)

        self.conv_shortcut = nn.Conv2d(self.in_channels, self.out_channels,
                                       kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_shortcut = nn.LayerNorm(self.out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """
        x_short = self.conv_shortcut(x)

        N, C, W, H = x_short.shape
        x_short = self.norm_shortcut(x_short.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x_short = x_short.view(N, H, W, C).permute(0, 3, 2, 1)
        x_short = self.act(x_short)

        x = self.conv1(x)
        N, C, W, H = x.shape
        x = self.norm_layer1(x.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x = x.view(N, H, W, C).permute(0, 3, 1, 2)
        x = self.act(x)

        x = self.conv2(x)
        N, C, W, H = x.shape
        x = self.norm_layer2(x.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x = x.view(N, H, W, C).permute(0, 3, 1, 2)
        x = self.act(x)

        return x + x_short



class CNN_2D(nn.Module):
    """ 2D CNN Decoder """
    def __init__(self, config):
        super(CNN_2D, self).__init__()

        """
        Parameters
        ----------
        config : argparse
          configuration file from config.py
        """

        # read configuration from file
        self.n_layers = len(config.out_channels)

        self.out_channels = []
        self.in_channels = [config.out_channels[-1]]

        for i in reversed(range(self.n_layers - 1)):
            if i == (self.n_layers - 2):
                self.out_channels.append(config.out_channels[i + 1] // 2)
            else:
                self.out_channels.append(config.out_channels[i + 1])
            self.in_channels.append(config.out_channels[i] + self.out_channels[(self.n_layers - 2) - i])

        self.out_channels.append(config.out_channels[0] * 3)

        self.up_sampling = config.de_up_sampling

        # create conv layers with two blocks
        self.conv_blocks = nn.ModuleList([])

        for layer in range(self.n_layers):
            self.conv_blocks.append(conv_block(self.in_channels[layer], self.out_channels[layer]))

        # create up sampling layers
        self.up_sampling_block = nn.ModuleList([])

        for layer in range(self.n_layers - 1):
            if self.up_sampling == 'conv':
                block = [nn.ConvTranspose2d(self.out_channels[layer], self.out_channels[layer],
                                            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)]
            elif self.up_sampling in ('bilinear', 'nearest', 'bicubic'):
                block = [nn.Upsample(scale_factor=2, mode=self.up_sampling)]
            else:
                raise ValueError('%s is not a recognized down_sampling type.\
                 Supported are conv, bilinear, nearest or bicubic' % self.up_sampling)
            self.up_sampling_block.append(nn.Sequential(*block))

        # create regression head
        self.BT_block = regression_head(self.out_channels[-1])
        self.NDVI_block = regression_head(self.out_channels[-1])

        self._init_weights()

    def _init_weights(self):

        def init_func(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

        for m in self.BT_block.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()
            else:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)

        for m in self.NDVI_block.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()
            else:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)

        for m in self.conv_blocks.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()

    def forward(self, x):
        """
        Args:
            x : list of input torch.tensor [N, C, H, W]
        """
        for i, x_i in enumerate(reversed(x)):
            if i != 0:
                if x_i.size()[-2:] != x_out.size()[-2:]:
                    x_out = F.interpolate(x_out, size=x_i.size()[-2:], mode='bilinear', align_corners=False)
                x_i = torch.cat((x_i, x_out), dim=1)

            x_out = self.conv_blocks[i](x_i)
            if i != self.n_layers - 1:
                x_out = self.up_sampling_block[i](x_out)

        BT_out = self.BT_block(x_out)[:, 0, :, :]
        NDVI_out = self.NDVI_block(x_out)[:, 0, :, :]

        return BT_out, NDVI_out


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

    test_x = [torch.randn((1, 96, 397, 409)).to(device), torch.randn((1, 192, 198, 204)).to(device),
              torch.randn((1, 384, 99, 102)).to(device)]

    model = CNN_2D(config_file).to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    BT, NDVI = model(test_x)

    print(BT.shape)
    print(NDVI.shape)


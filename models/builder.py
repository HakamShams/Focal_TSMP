# ------------------------------------------------------------
# Encoder Decoder U-Net
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------

import torch
import torch.nn as nn
import importlib

# ------------------------------------------------------------

def import_class(en_de, name):
    module = importlib.import_module("models." + en_de + '.' + name)
    return getattr(module, name)


class encoder_decoder(nn.Module):
    """ Encoder-Decoder U-Net """
    def __init__(self, config):
        super(encoder_decoder, self).__init__()

        """
        Parameters
        ----------
        config : argparse
            configuration file from config.py
        """

        self.encoder = import_class('encoder', config.encoder)(config)
        self.decoder = import_class('decoder', config.decoder)(config)

        self._init_weights()

    def _init_weights(self, init_type='xavier', gain=1):  # gain=nn.init.calculate_gain('leaky_relu', 0.2)

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm3d') != -1 or classname.find('LayerNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    #torch.nn.init.normal_(m.weight.data, 1.0, gain)
                    torch.nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # initialization code is based on https://github.com/nvlabs/spade/
        for m in self.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()

    def forward(self, x):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """

        x = self.encoder(x)
        x_BT, x_NDVI = self.decoder(x)

        return x_BT, x_NDVI


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

    model = encoder_decoder(config_file).to(device)
    print(model)

    BT, NDVI = model(test_x)

    print(BT.shape)
    print(NDVI.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")


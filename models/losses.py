# ------------------------------------------------------------------
# Loss functions
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision

# ------------------------------------------------------------------
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights='VGG19_Weights.DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()

        # VGG architecter, used for the perceptual loss using a pretrained VGG network from spade paper
        # https://github.com/NVlabs/SPADE

        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [8., 1., 1., 1., 1.]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class G_loss(nn.Module):
    def __init__(self, config):
        super(G_loss, self).__init__()

        self.lambda_NDVI = config.lambda_NDVI
        self.lambda_BT = config.lambda_BT
        self.lambda_vgg = config.lambda_vgg

        if config.loss_NDVI == 'L1':
            self.loss_NDVI = nn.L1Loss(reduction='none')
        elif config.loss_NDVI == 'L2':
            self.loss_NDVI = nn.MSELoss(reduction='none')
        elif config.loss_NDVI == 'Huber':
            self.loss_NDVI = nn.HuberLoss(reduction='none', delta=0.10)
        else:
            raise ValueError('Unexpected loss_NDVI {}, supported are L1, L2 and Huber'.format(config.loss_BT))

        if config.loss_BT == 'L1':
            self.loss_BT = nn.L1Loss(reduction='none')
        elif config.loss_BT == 'L2':
            self.loss_BT = nn.MSELoss(reduction='none')
        elif config.loss_BT == 'Huber':
            self.loss_BT = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError('Unexpected loss_BT {}, supported are L1, L2 and Huber'.format(config.loss_BT))

        self.reduction = config.loss_reduction

        if self.reduction not in ['sum', 'mean']:
            raise ValueError('Unexpected loss_reduction {}, supported are mean and sum reductions'.format(config.loss_reduction))

        if config.gpu_id != -1:
            device = 'cuda'
        else:
            device = 'cpu'

        self.VGG_loss = VGGLoss(device=device)

    def forward(self, NDVI_p, BT_p, NDVI, BT, weights_ndvi, weights_bt):

        mask_valid_NDVI = ~torch.isnan(NDVI)
        mask_valid_BT = ~torch.isnan(BT)

        NDVI = (NDVI - 0.20188083) / 0.14982702
        BT = (BT - 289.30905) / 17.59524

        NDVI[~mask_valid_NDVI] = NDVI_p[~mask_valid_NDVI].clone().detach()
        BT[~mask_valid_BT] = BT_p[~mask_valid_BT].clone().detach()
        NDVI = NDVI.requires_grad_(False)
        BT = BT.requires_grad_(False)

        loss_NDVI = self.loss_NDVI(NDVI_p, NDVI) * weights_ndvi
        loss_BT = self.loss_BT(BT_p, BT) * weights_bt

        BT_p = BT_p.unsqueeze(1).repeat(1, 3, 1, 1)
        BT = BT.unsqueeze(1).repeat(1, 3, 1, 1)

        NDVI_p = NDVI_p.unsqueeze(1).repeat(1, 3, 1, 1)
        NDVI = NDVI.unsqueeze(1).repeat(1, 3, 1, 1)

        loss_NDVI_vgg = self.VGG_loss(NDVI_p, NDVI)
        loss_BT_vgg = self.VGG_loss(BT_p, BT)

        if self.reduction == 'sum':
            loss_NDVI = torch.nansum(loss_NDVI) + self.lambda_vgg * loss_NDVI_vgg
            loss_BT = torch.nansum(loss_BT) + self.lambda_vgg * loss_BT_vgg
        else:
            loss_NDVI = (torch.sum(loss_NDVI) / torch.sum(weights_ndvi)) + self.lambda_vgg * loss_NDVI_vgg
            loss_BT = (torch.sum(loss_BT) / torch.sum(weights_bt)) + self.lambda_vgg * loss_BT_vgg

        BT_p = BT_p[:, 0, :, :]
        NDVI_p = NDVI_p[:, 0, :, :]
        BT = BT[:, 0, :, :]
        NDVI = NDVI[:, 0, :, :]

        loss = self.lambda_BT * loss_BT + self.lambda_NDVI * loss_NDVI

        return loss, loss_BT, loss_NDVI


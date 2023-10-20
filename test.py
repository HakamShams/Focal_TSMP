# ------------------------------------------------------------------
# Script for testing on TerrSysMP_NET dataset
# Contact person: "Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>"
# ------------------------------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
from models.losses import G_loss
import utils.train_utils as utils
from models.builder import encoder_decoder
import time
import os

from TerrSysMP_NET_dataset import TerrSysMP_NET
import config as config_file

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
np.seterr(divide='ignore', invalid='ignore')

# ------------------------------------------------------------------

def test(config_file):

    # read config arguments
    config = config_file.read_arguments(train=False)

    # get logger
    logger = utils.get_logger(config)

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading testing dataset ...")
    test_dataset = TerrSysMP_NET(root=config.root_data,
                                 nan_fill=config.nan_fill,
                                 is_aug=False,
                                 is_shuffle=False,
                                 variables=config.variables,
                                 variables_static=config.variables_static,
                                 years=config.years_test,
                                 n_lat=config.n_lat,
                                 n_lon=config.n_lon,
                                 cut_lateral_boundary=config.cut_lateral_boundary
                                 )

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=config.batch_size,
                                                  drop_last=False,
                                                  shuffle=False,
                                                  pin_memory=config.pin_memory,
                                                  num_workers=config.n_workers)

    utils.log_string(logger, "# testing samples: %d" % len(test_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = encoder_decoder(config).to(device)

    if config.en_de_pretrained:
        utils.log_string(logger, "initialize weights from pretrained model {} ...".format(config.en_pretrained))
        checkpoint = torch.load(config.en_de_pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        raise ValueError('expected a pretrained model for testing')

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.decoder))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # testing loop
    utils.log_string(logger, 'testing on TerrSysMP_NET dataset ...\n')

    eval_test = utils.evaluator(logger, 'Testing', config)

    time.sleep(1)

    # get losses
    utils.log_string(logger, "get criterion ...")
    criterion = G_loss(config).to(device)

    # initialize the best values
    best_loss_val = np.inf
    best_loss_val_BT = np.inf
    best_loss_val_NDVI = np.inf

    # testing
    with torch.no_grad():

        model = model.eval()
        loss_val, loss_val_BT, loss_val_NDVI = 0, 0, 0

        time.sleep(1)

        for i, (data_d, data_smn, data_smt, data_climatology,
                data_cold_surface, data_sea, data_no_vegetation) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader),
                                                                         smoothing=0.9,
                                                                         postfix="  validation"):

            data_smn, data_smt, data_climatology, data_cold_surface, data_sea, data_no_vegetation = \
                data_smn.to(device), data_smt.to(device), data_climatology.to(device), data_cold_surface.to(device), \
                data_sea.to(device), data_no_vegetation.to(device)

            smt_p, smn_p = model(data_d.to(device))

            smn_p = torch.clip(smn_p, -15, 15)
            smt_p = torch.clip(smt_p, -15, 15)

            mask_non_valid_NDVI = torch.logical_or(torch.isnan(data_smn), torch.isnan(smn_p))
            mask_non_valid_BT = torch.logical_or(torch.isnan(data_smt), torch.isnan(smt_p))

            weights_ndvi, weights_bt = utils.calc_weights(data_smn, data_smt, data_sea)

            loss, loss_BT, loss_NDVI = criterion(smn_p, smt_p, data_smn, data_smt, weights_ndvi, weights_bt)

            loss_val += loss.item()
            loss_val_BT += loss_BT.item()
            loss_val_NDVI += loss_NDVI.item()

            smn_p = (smn_p * 0.14982702) + 0.20198083
            smt_p = (smt_p * 17.59524) + 289.30905

            data_smn[mask_non_valid_NDVI] = torch.nan
            data_smt[mask_non_valid_BT] = torch.nan
            smt_p[mask_non_valid_BT] = torch.nan
            smn_p[mask_non_valid_NDVI] = torch.nan

            VCI_p, TCI_p = utils.calc_VCI_TCI(smn_p, smt_p, data_climatology)
            VCI, TCI = utils.calc_VCI_TCI(data_smn, data_smt, data_climatology)

            VHI_p = utils.calc_VHI(VCI_p, TCI_p, 0.5)
            VHI = utils.calc_VHI(VCI, TCI, 0.5)

            VHI_p_c = utils.calc_VHI_classes(VHI_p.detach().clone(), data_cold_surface, data_sea, data_no_vegetation)
            VHI_c = utils.calc_VHI_classes(VHI.detach().clone(), data_cold_surface, data_sea, data_no_vegetation)

            eval_test(VHI_p_c.detach().clone().cpu().numpy(), VHI_c.clone().cpu().numpy(),
                      VHI_p.detach().clone().cpu().numpy(), VHI.clone().cpu().numpy(),
                      VCI_p.detach().clone().cpu().numpy(), VCI.clone().cpu().numpy(),
                      TCI_p.detach().clone().cpu().numpy(), TCI.clone().cpu().numpy(),
                      smn_p.detach().clone().cpu().numpy(), data_smn.clone().cpu().numpy(),
                      smt_p.detach().clone().cpu().numpy(), data_smt.clone().cpu().numpy(),
                      data_climatology[:, 2, :, :].clone().cpu().numpy(),
                      data_climatology[:, 7, :, :].clone().cpu().numpy())

        mean_loss_val = loss_val / float(len(test_dataloader))
        mean_loss_val_BT = loss_val_BT / float(len(test_dataloader))
        mean_loss_val_NDVI = loss_val_NDVI / float(len(test_dataloader))

        eval_test.get_results(mean_loss_val, best_loss_val,
                             mean_loss_val_BT, best_loss_val_BT,
                             mean_loss_val_NDVI, best_loss_val_NDVI)


if __name__ == '__main__':

    test(config_file)


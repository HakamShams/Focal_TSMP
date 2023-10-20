# ------------------------------------------------------------------
# Script for training and validating on TerrSysMP_NET dataset
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

def train(config_file):

    # read config arguments
    config = config_file.read_arguments(train=True)

    # get logger
    logger = utils.get_logger(config)

    # fix random seed
    utils.fix_seed(config.seed)

    # dataloader
    utils.log_string(logger, "loading training dataset ...")

    train_dataset = TerrSysMP_NET(root=config.root_data,
                                  nan_fill=config.nan_fill,
                                  is_aug=config.is_aug,
                                  is_shuffle=False,
                                  variables=config.variables,
                                  variables_static=config.variables_static,
                                  years=config.years_train,
                                  n_lat=config.n_lat,
                                  n_lon=config.n_lon,
                                  cut_lateral_boundary=config.cut_lateral_boundary
                                  )

    utils.log_string(logger, "loading validation dataset ...")
    val_dataset = TerrSysMP_NET(root=config.root_data,
                                nan_fill=config.nan_fill,
                                is_aug=False,
                                is_shuffle=False,
                                variables=config.variables,
                                variables_static=config.variables_static,
                                years=config.years_val,
                                n_lat=config.n_lat,
                                n_lon=config.n_lon,
                                cut_lateral_boundary=config.cut_lateral_boundary
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=config.batch_size,
                                                  drop_last=True,
                                                  shuffle=True,
                                                  pin_memory=config.pin_memory,
                                                  num_workers=config.n_workers)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=config.batch_size,
                                                 drop_last=False,
                                                 shuffle=True,
                                                 pin_memory=config.pin_memory,
                                                 num_workers=config.n_workers)

    utils.log_string(logger, "# training samples: %d" % len(train_dataset))
    utils.log_string(logger, "# evaluation samples: %d" % len(val_dataset))

    # get models
    utils.log_string(logger, "\nloading the model ...")

    if config.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    model = encoder_decoder(config).to(device)

    utils.log_string(logger, "model parameters ...")
    utils.log_string(logger, "encoder parameters: %d" % utils.count_parameters(model.encoder))
    utils.log_string(logger, "decoder parameters: %d" % utils.count_parameters(model.decoder))
    utils.log_string(logger, "all parameters: %d\n" % utils.count_parameters(model))

    # get losses
    utils.log_string(logger, "get criterion ...")
    criterion = G_loss(config).to(device)

    # get optimizer
    utils.log_string(logger, "get optimizer and learning rate scheduler ...")

    optim_groups = utils.get_optimizer_groups(model, config)
    optimizer = utils.get_optimizer(optim_groups, config)
    lr_scheduler = utils.get_learning_scheduler(optimizer, config)

    # training loop
    utils.log_string(logger, 'training on TerrSysMP_NET dataset ...\n')

    eval_train = utils.evaluator(logger, 'Training', config)
    eval_val = utils.evaluator(logger, 'Validation', config)

    time.sleep(1)

    # initialize the best values
    best_loss_train = np.inf
    best_loss_train_BT = np.inf
    best_loss_train_NDVI = np.inf

    best_loss_val = np.inf
    best_loss_val_BT = np.inf
    best_loss_val_NDVI = np.inf
    #best_F1_drought = 0

    for epoch in range(config.n_epochs):
        utils.log_string(logger, '################# Epoch (%s/%s) #################' % (epoch + 1, config.n_epochs))

        # train
        model = model.train()
        loss_train, loss_train_BT, loss_train_NDVI = 0, 0, 0

        time.sleep(1)

        for i, (data_d, data_smn, data_smt, data_climatology, data_cold_surface,
                data_sea, data_no_vegetation) in tqdm(enumerate(train_dataloader),
                                                      total=len(train_dataloader),
                                                      smoothing=0.9,
                                                      postfix="  training"):

            optimizer.zero_grad(set_to_none=True)

            data_smn, data_smt, data_climatology, data_cold_surface, data_sea, data_no_vegetation = \
            data_smn.to(device).requires_grad_(False), data_smt.to(device).requires_grad_(False), \
            data_climatology.to(device).requires_grad_(False), data_cold_surface.to(device).requires_grad_(False), \
            data_sea.to(device).requires_grad_(False), data_no_vegetation.to(device).requires_grad_(False)

            smt_p, smn_p = model(data_d.to(device))

            smn_p = torch.clip(smn_p, -15, 15)
            smt_p = torch.clip(smt_p, -15, 15)

            mask_non_valid_NDVI = torch.isnan(data_smn)
            mask_non_valid_BT = torch.isnan(data_smt)

            weights_ndvi, weights_bt = utils.calc_weights(data_smn, data_smt, data_sea)

            loss, loss_BT, loss_NDVI = criterion(smn_p, smt_p, data_smn, data_smt, weights_ndvi, weights_bt)

            loss_train += loss.item()
            loss_train_BT += loss_BT.item()
            loss_train_NDVI += loss_NDVI.item()

            loss.backward()
            optimizer.step()

            smn_p = (smn_p * 0.14982702) + 0.20188083
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

            eval_train(VHI_p_c.detach().cpu().numpy(), VHI_c.cpu().numpy(),
                       VHI_p.detach().cpu().numpy(), VHI.cpu().numpy(),
                       VCI_p.detach().cpu().numpy(), VCI.cpu().numpy(),
                       TCI_p.detach().cpu().numpy(), TCI.cpu().numpy(),
                       smn_p.detach().clone().cpu().numpy(), data_smn.clone().cpu().numpy(),
                       smt_p.detach().clone().cpu().numpy(), data_smt.clone().cpu().numpy(),
                       data_climatology[:, 2, :, :].clone().cpu().numpy(),
                       data_climatology[:, 7, :, :].clone().cpu().numpy())

        mean_loss_train = loss_train / float(len(train_dataloader))
        mean_loss_train_BT = loss_train_BT / float(len(train_dataloader))
        mean_loss_train_NDVI = loss_train_NDVI / float(len(train_dataloader))

        eval_train.get_results(mean_loss_train, best_loss_train,
                               mean_loss_train_BT, best_loss_train_BT,
                               mean_loss_train_NDVI, best_loss_train_NDVI)

        if mean_loss_train <= best_loss_train:
            best_loss_train = mean_loss_train
        if mean_loss_train_BT <= best_loss_train_BT:
            best_loss_train_BT = mean_loss_train_BT
        if mean_loss_train_NDVI <= best_loss_train_NDVI:
            best_loss_train_NDVI = mean_loss_train_NDVI

        # validation
        with torch.no_grad():

            model = model.eval()
            loss_val, loss_val_BT, loss_val_NDVI = 0, 0, 0

            time.sleep(1)

            for i, (data_d, data_smn, data_smt, data_climatology,
                    data_cold_surface, data_sea, data_no_vegetation) in tqdm(enumerate(val_dataloader),
                                                                             total=len(val_dataloader),
                                                                             smoothing=0.9,
                                                                             postfix="  validation"):

                data_smn, data_smt, data_climatology, data_cold_surface, data_sea, data_no_vegetation = \
                data_smn.to(device), data_smt.to(device), data_climatology.to(device), data_cold_surface.to(device), \
                data_sea.to(device), data_no_vegetation.to(device)

                smt_p, smn_p = model(data_d.to(device))

                smn_p = torch.clip(smn_p, -15, 15)
                smt_p = torch.clip(smt_p, -15, 15)

                mask_non_valid_NDVI = torch.isnan(data_smn)
                mask_non_valid_BT = torch.isnan(data_smt)

                weights_ndvi, weights_bt = utils.calc_weights(data_smn, data_smt, data_sea)

                loss, loss_BT, loss_NDVI = criterion(smn_p, smt_p, data_smn, data_smt, weights_ndvi, weights_bt)

                loss_val += loss.item()
                loss_val_BT += loss_BT.item()
                loss_val_NDVI += loss_NDVI.item()

                smn_p = (smn_p * 0.14982702) + 0.20188083
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

                eval_val(VHI_p_c.detach().clone().cpu().numpy(), VHI_c.clone().cpu().numpy(),
                         VHI_p.detach().clone().cpu().numpy(), VHI.clone().cpu().numpy(),
                         VCI_p.detach().clone().cpu().numpy(), VCI.clone().cpu().numpy(),
                         TCI_p.detach().clone().cpu().numpy(), TCI.clone().cpu().numpy(),
                         smn_p.detach().clone().cpu().numpy(), data_smn.clone().cpu().numpy(),
                         smt_p.detach().clone().cpu().numpy(), data_smt.clone().cpu().numpy(),
                         data_climatology[:, 2, :, :].clone().cpu().numpy(),
                         data_climatology[:, 7, :, :].clone().cpu().numpy())

            mean_loss_val = loss_val / float(len(val_dataloader))
            mean_loss_val_BT = loss_val_BT / float(len(val_dataloader))
            mean_loss_val_NDVI = loss_val_NDVI / float(len(val_dataloader))

            eval_val.get_results(mean_loss_val, best_loss_val,
                                 mean_loss_val_BT, best_loss_val_BT,
                                 mean_loss_val_NDVI, best_loss_val_NDVI)

            if mean_loss_val <= best_loss_val:
                best_loss_val = mean_loss_val
                utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'loss')
            if mean_loss_val_BT <= best_loss_val_BT:
                best_loss_val_BT = mean_loss_val_BT
            if mean_loss_val_NDVI <= best_loss_val_NDVI:
                best_loss_val_NDVI = mean_loss_val_NDVI

           # mean_F1_drought = eval_val.F1_drought
           # if mean_F1_drought >= best_F1_drought:
           #     best_F1_drought = mean_F1_drought
           #     utils.save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, 'F1')

        eval_train.reset()
        eval_val.reset()

        lr_scheduler.step()


if __name__ == '__main__':

    train(config_file)


# ------------------------------------------------------------------
# Utility functions
# Contact person: "Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>"
# ------------------------------------------------------------------

import torch
import numpy as np
import random
import os
import datetime
import logging
from functools import reduce
from scipy.stats import spearmanr, pearsonr

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

# ------------------------------------------------------------------

def log_string(logger, str):
    logger.info(str)
    print(str)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)


def get_logger(config):
    # Set Logger and create Directories

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    make_dir(dir_log)

    if config.phase == 'train':
        checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
        make_dir(checkpoints_dir)

    logger = logging.getLogger("Trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log_file.txt' % dir_log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_optimizer_groups(model, config):
    # Based on https://github.com/karpathy/minGPT/commit/bbbdac74fa9b2e55574d70056163ffbae42310c1

    # separate out all parameters to those that will and won't experience regularizing weight decay
    encoder_decay = set()
    encoder_no_decay = set()
    decoder_decay = set()
    decoder_no_decay = set()
    rest_decay = set()
    rest_no_decay = set()

    blacklist_weight_modules = (torch.nn.LayerNorm,
                                torch.nn.Embedding,
                                torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d,
                                torch.nn.BatchNorm1d,
                                torch.nn.GroupNorm,
                                torch.nn.InstanceNorm1d,
                                torch.nn.InstanceNorm2d,
                                torch.nn.InstanceNorm3d)
    u = 0
    for mn, m in model.encoder.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name##
            mm = reduce(getattr, fpn.split(sep='.')[:-1], model.encoder)
            u += 1
            if pn.endswith('bias') or pn.endswith('relative_position_bias_table'):
                # all biases will not be decayed
                encoder_no_decay.add('encoder.' + fpn)
            elif pn.endswith('weight') and isinstance(mm, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                encoder_no_decay.add('encoder.' + fpn)
            else:
                encoder_decay.add('encoder.' + fpn)

    for mn, m in model.decoder.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name##
            mm = reduce(getattr, fpn.split(sep='.')[:-1], model.decoder)
            if pn.endswith('bias') or pn.endswith('relative_position_bias_table'):
                # all biases will not be decayed
                decoder_no_decay.add('decoder.' + fpn)
            elif pn.endswith('weight') and isinstance(mm, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                decoder_no_decay.add('decoder.' + fpn)
            else:
                decoder_decay.add('decoder.' + fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = encoder_decay & encoder_no_decay & decoder_decay & decoder_no_decay & rest_decay & rest_no_decay
    union_params = encoder_decay | encoder_no_decay | decoder_decay | decoder_no_decay | rest_decay | rest_no_decay

    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [

        {"params": [param_dict[pn] for pn in sorted(list(encoder_decay))],
         'lr': config.en_lr, "weight_decay": config.en_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(encoder_no_decay))],
         'lr': config.en_lr, "weight_decay": 0.0},

        {"params": [param_dict[pn] for pn in sorted(list(decoder_decay))],
         'lr': config.de_lr, "weight_decay": config.de_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(decoder_no_decay))],
         'lr': config.de_lr, "weight_decay": 0.0},

        {"params": [param_dict[pn] for pn in sorted(list(rest_decay))],
         'lr': config.en_lr, "weight_decay": config.en_weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(rest_no_decay))],
         'lr': config.en_lr, "weight_decay": 0.0},

    ]

    return optim_groups


def get_optimizer(optim_groups, config):

    optim = config.optimizer

    if optim == 'Adam':
        optimizer = torch.optim.Adam(optim_groups, betas=(config.beta1, config.beta2))
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups, betas=(config.beta1, config.beta2))
    elif optim == 'SGD':
        optimizer = torch.optim.SGD(optim_groups, config.en_lr)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(optim_groups, lr=config.en_lr)
    else:
        raise ValueError('Unexpected optimizer {} supported optimizers are Adam, AdamW, SGD, and RMSprop'.format(config.optimizer))

    return optimizer


def get_learning_scheduler(optimizer, config):
    # it is recommended to have a better scheduler with warmup
    lr_scheduler = config.lr_scheduler

    if lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_decay)
    else:
        raise ValueError('Unexpected optimizer {}, supported scheduler is StepLR'.format(config.optimizer))

    return lr_scheduler


class evaluator():
    def __init__(self, logger, mode, config):

        # this function needs an optimization

        self.classes = ['exceptional', 'extreme', 'severe', 'moderate', 'dry condition', 'normal']
        self.n_classes = len(self.classes)

        self.mode = mode
        self.config = config
        self.logger = logger

        self.correct_all = 0
        self.seen_all = 0
        self.seen_iter = 0

        self.NDVI_MAE = 0
        self.BT_MAE = 0
        self.NDVI_RMSE = 0
        self.BT_RMSE = 0
        self.NDVI_R2 = 0
        self.BT_R2 = 0
        self.NDVI_Rs = 0
        self.BT_Rs = 0
        self.NDVI_Rp = 0
        self.BT_Rp = 0

        self.NDVI_a_MAE = 0
        self.BT_a_MAE = 0
        self.NDVI_a_RMSE = 0
        self.BT_a_RMSE = 0
        self.NDVI_a_R2 = 0
        self.BT_a_R2 = 0
        self.NDVI_a_Rs = 0
        self.BT_a_Rs = 0
        self.NDVI_a_Rp = 0
        self.BT_a_Rp = 0

        self.VHI_MAE = 0
        self.VCI_MAE = 0
        self.TCI_MAE = 0

        self.VHI_RMSE = 0
        self.VCI_RMSE = 0
        self.TCI_RMSE = 0

        self.VHI_R2 = 0
        self.VCI_R2 = 0
        self.TCI_R2 = 0

        self.VHI_Rs = 0
        self.VCI_Rs = 0
        self.TCI_Rs = 0

        self.VHI_Rp = 0
        self.VCI_Rp = 0
        self.TCI_Rp = 0

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.weights_drought = 0
        self.seen_drought = 0
        self.correct_drought = 0
        self.iou_de_drought = 0
        self.predicted_drought = 0

    def get_results(self, mean_loss, best_loss, mean_loss_BT, best_loss_BT, mean_loss_NDVI, best_loss_NDVI):

        weights_label = self.weights_label.astype(np.float32) / np.sum(self.weights_label.astype(np.float32))
        weights_drought = self.seen_drought / self.seen_all
        self.accuracy_all = self.correct_all / float(self.seen_all)

        self.NDVI_MAE = self.NDVI_MAE / float(self.seen_iter)
        self.BT_MAE = self.BT_MAE / float(self.seen_iter)
        self.NDVI_RMSE = self.NDVI_RMSE / float(self.seen_iter)
        self.BT_RMSE = self.BT_RMSE / float(self.seen_iter)
        self.NDVI_R2 = self.NDVI_R2 / float(self.seen_iter)
        self.BT_R2 = self.BT_R2 / float(self.seen_iter)
        self.NDVI_Rs = self.NDVI_Rs / float(self.seen_iter)
        self.BT_Rs = self.BT_Rs / float(self.seen_iter)
        self.NDVI_Rp = self.NDVI_Rp / float(self.seen_iter)
        self.BT_Rp = self.BT_Rp / float(self.seen_iter)

        self.NDVI_a_MAE = self.NDVI_a_MAE / float(self.seen_iter)
        self.BT_a_MAE = self.BT_a_MAE / float(self.seen_iter)
        self.NDVI_a_RMSE = self.NDVI_a_RMSE / float(self.seen_iter)
        self.BT_a_RMSE = self.BT_a_RMSE / float(self.seen_iter)
        self.NDVI_a_R2 = self.NDVI_a_R2 / float(self.seen_iter)
        self.BT_a_R2 = self.BT_a_R2 / float(self.seen_iter)
        self.NDVI_a_Rs = self.NDVI_a_Rs / float(self.seen_iter)
        self.BT_a_Rs = self.BT_a_Rs / float(self.seen_iter)
        self.NDVI_a_Rp = self.NDVI_a_Rp / float(self.seen_iter)
        self.BT_a_Rp = self.BT_a_Rp / float(self.seen_iter)

        self.VHI_MAE = self.VHI_MAE / float(self.seen_iter)
        self.VCI_MAE = self.VCI_MAE / float(self.seen_iter)
        self.TCI_MAE = self.TCI_MAE / float(self.seen_iter)

        self.VHI_RMSE = self.VHI_RMSE / float(self.seen_iter)
        self.VCI_RMSE = self.VCI_RMSE / float(self.seen_iter)
        self.TCI_RMSE = self.TCI_RMSE / float(self.seen_iter)

        self.VHI_R2 = self.VHI_R2 / float(self.seen_iter)
        self.VCI_R2 = self.VCI_R2 / float(self.seen_iter)
        self.TCI_R2 = self.TCI_R2 / float(self.seen_iter)

        self.VHI_Rs = self.VHI_Rs / float(self.seen_iter)
        self.VCI_Rs = self.VCI_Rs / float(self.seen_iter)
        self.TCI_Rs = self.TCI_Rs / float(self.seen_iter)

        self.VHI_Rp = self.VHI_Rp / float(self.seen_iter)
        self.VCI_Rp = self.VCI_Rp / float(self.seen_iter)
        self.TCI_Rp = self.TCI_Rp / float(self.seen_iter)

        message = '-----------------   %s   -----------------\n' % self.mode

        accuracy = [0 for _ in range(self.n_classes)]
        precision = [0 for _ in range(self.n_classes)]
        F1 = [0 for _ in range(self.n_classes)]
        iou = [0 for _ in range(self.n_classes)]

        for label in range(self.n_classes):
            precision[label] = self.correct_label_all[label] / float(self.predicted_label_all[label])
            accuracy[label] = self.correct_label_all[label] / (np.array(self.seen_label_all[label], dtype=float) + 1e-6)
            F1[label] = 2 * precision[label] * accuracy[label] / (accuracy[label] + precision[label])
            iou[label] = self.correct_label_all[label] / float(self.iou_de_label_all[label])

        precision_drought = self.correct_drought / float(self.predicted_drought)
        accuracy_drought = self.correct_drought / (np.array(self.seen_drought, dtype=float) + 1e-6)
        self.F1_drought = 2 * precision_drought * accuracy_drought / (accuracy_drought + precision_drought)
        iou_drought = self.correct_drought / float(self.iou_de_drought)

        for label in range(self.n_classes):
            message += 'class %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
                self.classes[label] + ' '*(14 - len(self.classes[label])), weights_label[label],
                precision[label],
                accuracy[label],
                F1[label],
                iou[label])

        message += '\nclass %s weight: %.4f, precision: %.4f, accuracy: %.4f, F1: %.4f IoU: %.4f \n' % (
            'drought' + ' ' * 7, weights_drought, precision_drought, accuracy_drought, self.F1_drought, iou_drought)

        message += '\n%s accuracy       : %.4f' % (self.mode, self.accuracy_all)
        message += '\n%s mean accuracy  : %.4f' % (self.mode, np.mean(accuracy))
        message += '\n%s mean IoU       : %.4f' % (self.mode, np.mean(iou))
        message += '\n%s mean F1        : %.4f' % (self.mode, np.mean(F1))

        message += '\n%s mean loss      : %.4f    best mean loss      : %.4f' % (self.mode, mean_loss, best_loss)
        message += '\n%s mean BT loss   : %.4f    best mean BT loss   : %.4f' % (self.mode, mean_loss_BT, best_loss_BT)
        message += '\n%s mean NDVI loss : %.4f    best mean NDVI loss : %.4f' % (self.mode, mean_loss_NDVI, best_loss_NDVI)

        message += '\n%s NDVI R2        : %.4f   BT R2  : %.4f' % (self.mode, self.NDVI_R2, self.BT_R2)
        message += '\n%s NDVI Rp        : %.4f   BT Rp  : %.4f' % (self.mode, self.NDVI_Rp, self.BT_Rp)
        message += '\n%s NDVI Rs        : %.4f   BT Rs  : %.4f' % (self.mode, self.NDVI_Rs, self.BT_Rs)
        message += '\n%s NDVI MAE       : %.4f   BT MAE : %.4f' % (self.mode, self.NDVI_MAE, self.BT_MAE)
        message += '\n%s NDVI RMSE      : %.4f   BT RMSE: %.4f' % (self.mode, self.NDVI_RMSE, self.BT_RMSE)

        message += '\n%s NDVI anomaly R2        : %.4f   BT anomaly R2  : %.4f' % (self.mode, self.NDVI_a_R2, self.BT_a_R2)
        message += '\n%s NDVI anomaly Rp        : %.4f   BT anomaly Rp  : %.4f' % (self.mode, self.NDVI_a_Rp, self.BT_a_Rp)
        message += '\n%s NDVI anomaly Rs        : %.4f   BT anomaly Rs  : %.4f' % (self.mode, self.NDVI_a_Rs, self.BT_a_Rs)
        message += '\n%s NDVI anomaly MAE       : %.4f   BT anomaly MAE : %.4f' % (self.mode, self.NDVI_a_MAE, self.BT_a_MAE)
        message += '\n%s NDVI anomaly RMSE      : %.4f   BT anomaly RMSE: %.4f' % (self.mode, self.NDVI_a_RMSE, self.BT_a_RMSE)

        message += '\n%s VCI R2        : %.4f   TCI R2  : %.4f    VHI R2  : %.4f' % (self.mode, self.VCI_R2, self.TCI_R2, self.VHI_R2)
        message += '\n%s VCI Rp        : %.4f   TCI Rp  : %.4f    VHI Rp  : %.4f' % (self.mode, self.VCI_Rp, self.TCI_Rp, self.VHI_Rp)
        message += '\n%s VCI Rs        : %.4f   TCI Rs  : %.4f    VHI Rs  : %.4f' % (self.mode, self.VCI_Rs, self.TCI_Rs, self.VHI_Rs)
        message += '\n%s VCI MAE       : %.4f   TCI MAE : %.4f    VHI MAE : %.4f' % (self.mode, self.VCI_MAE, self.TCI_MAE, self.VHI_MAE)
        message += '\n%s VCI RMSE      : %.4f   TCI RMSE: %.4f    VHI RMSE: %.4f' % (self.mode, self.VCI_RMSE, self.TCI_RMSE, self.VHI_RMSE)

        log_string(self.logger, message)


    def reset(self):

        self.correct_all = 0
        self.seen_all = 0

        self.seen_iter = 0

        self.NDVI_MAE = 0
        self.BT_MAE = 0
        self.NDVI_RMSE = 0
        self.BT_RMSE = 0
        self.NDVI_R2 = 0
        self.BT_R2 = 0
        self.NDVI_Rs = 0
        self.BT_Rs = 0
        self.NDVI_Rp = 0
        self.BT_Rp = 0

        self.NDVI_a_MAE = 0
        self.BT_a_MAE = 0
        self.NDVI_a_RMSE = 0
        self.BT_a_RMSE = 0
        self.NDVI_a_R2 = 0
        self.BT_a_R2 = 0
        self.NDVI_a_Rs = 0
        self.BT_a_Rs = 0
        self.NDVI_a_Rp = 0
        self.BT_a_Rp = 0

        self.VHI_MAE = 0
        self.VCI_MAE = 0
        self.TCI_MAE = 0
        self.VHI_RMSE = 0
        self.VCI_RMSE = 0
        self.TCI_RMSE = 0
        self.VHI_R2 = 0
        self.VCI_R2 = 0
        self.TCI_R2 = 0
        self.VHI_Rs = 0
        self.VCI_Rs = 0
        self.TCI_Rs = 0
        self.VHI_Rp = 0
        self.VCI_Rp = 0
        self.TCI_Rp = 0

        self.weights_label = np.zeros(self.n_classes)
        self.seen_label_all = [0 for _ in range(self.n_classes)]
        self.correct_label_all = [0 for _ in range(self.n_classes)]
        self.iou_de_label_all = [0 for _ in range(self.n_classes)]
        self.predicted_label_all = [0 for _ in range(self.n_classes)]

        self.weights_drought = 0
        self.seen_drought = 0
        self.correct_drought = 0
        self.iou_de_drought = 0
        self.predicted_drought = 0

    def __call__(self, VHI_p_c, VHI_c, VHI_p, VHI, VCI_p, VCI, TCI_p, TCI, NDVI_p, NDVI, BT_p, BT, NDVI_mean, BT_mean):

        VHI_c = VHI_c.flatten()
        VHI_p_c = VHI_p_c.flatten()

        VCI = VCI.flatten()
        VCI_p = VCI_p.flatten()
        TCI = TCI.flatten()
        TCI_p = TCI_p.flatten()
        VHI = VHI.flatten()
        VHI_p = VHI_p.flatten()

        index = np.logical_and(~np.isnan(VHI_c), VHI_c != 6)
        VHI_p_c = VHI_p_c[index]

        VCI = VCI[index]
        VCI_p = VCI_p[index]
        TCI = TCI[index]
        TCI_p = TCI_p[index]
        VHI = VHI[index]
        VHI_p = VHI_p[index]

        # ANOMALY
        NDVI_a = NDVI.flatten() - NDVI_mean.flatten()
        NDVI_a_p = NDVI_p.flatten() - NDVI_mean.flatten()

        index_ndvi = np.logical_and(~np.logical_or(np.isnan(NDVI_a), np.isnan(NDVI_a_p)), ~np.isnan(VHI_c))

        NDVI_a = NDVI_a[index_ndvi]
        NDVI_a_p = NDVI_a_p[index_ndvi]

        BT_a = BT.flatten() - BT_mean.flatten()
        BT_a_p = BT_p.flatten() - BT_mean.flatten()

        index_bt = np.logical_and(~np.logical_or(np.isnan(BT_a), np.isnan(BT_a_p)), ~np.isnan(VHI_c))
        VHI_c = VHI_c[index]

        BT_a = BT_a[index_bt]
        BT_a_p = BT_a_p[index_bt]

        # RAW
        NDVI = NDVI.flatten()
        NDVI_p = NDVI_p.flatten()

        NDVI = NDVI[index_ndvi]
        NDVI_p = NDVI_p[index_ndvi]

        BT = BT.flatten()
        BT_p = BT_p.flatten()

        BT = BT[index_bt]
        BT_p = BT_p[index_bt]

        correct = np.sum(VHI_p_c == VHI_c)
        self.correct_all += correct
        self.seen_all += len(VHI_c)

        self.seen_iter += 1

        self.NDVI_MAE += np.mean(np.abs(NDVI_p - NDVI))
        self.BT_MAE += np.mean(np.abs(BT_p - BT))

        self.NDVI_RMSE += np.sqrt(np.mean((NDVI_p - NDVI) ** 2))
        self.BT_RMSE += np.sqrt(np.mean((BT_p - BT) ** 2))

        self.NDVI_R2 += calc_R2(NDVI_p, NDVI)
        self.BT_R2 += calc_R2(BT_p, BT)

        self.NDVI_a_MAE += np.mean(np.abs(NDVI_a_p - NDVI_a))
        self.BT_a_MAE += np.mean(np.abs(BT_a_p - BT_a))

        self.NDVI_a_RMSE += np.sqrt(np.mean((NDVI_a_p - NDVI_a) ** 2))
        self.BT_a_RMSE += np.sqrt(np.mean((BT_a_p - BT_a) ** 2))

        self.NDVI_a_R2 += calc_R2(NDVI_a_p, NDVI_a)
        self.BT_a_R2 += calc_R2(BT_a_p, BT_a)

        self.VHI_MAE += np.mean(np.abs(VHI_p - VHI))
        self.VCI_MAE += np.mean(np.abs(VCI_p - VCI))
        self.TCI_MAE += np.mean(np.abs(TCI_p - TCI))

        self.VHI_RMSE += np.sqrt(np.mean((VHI_p - VHI)**2))
        self.VCI_RMSE += np.sqrt(np.mean((VCI_p - VCI)**2))
        self.TCI_RMSE += np.sqrt(np.mean((TCI_p - TCI)**2))

        self.VHI_R2 += calc_R2(VHI_p, VHI)
        self.VCI_R2 += calc_R2(VCI_p, VCI)
        self.TCI_R2 += calc_R2(TCI_p, TCI)

        coef_VHI_Rs, _ = spearmanr(VHI_p, VHI)
        coef_VCI_Rs, _ = spearmanr(VCI_p, VCI)
        coef_TCI_Rs, _ = spearmanr(TCI_p, TCI)

        coef_VHI_Rp, _ = pearsonr(VHI_p, VHI)
        coef_VCI_Rp, _ = pearsonr(VCI_p, VCI)
        coef_TCI_Rp, _ = pearsonr(TCI_p, TCI)

        coef_NDVI_Rs, _ = spearmanr(NDVI_p, NDVI)
        coef_BT_Rs, _ = spearmanr(BT_p, BT)

        coef_NDVI_Rp, _ = pearsonr(NDVI_p, NDVI)
        coef_BT_Rp, _ = pearsonr(BT_p, BT)

        coef_NDVI_a_Rs, _ = spearmanr(NDVI_a_p, NDVI_a)
        coef_BT_a_Rs, _ = spearmanr(BT_a_p, BT_a)

        coef_NDVI_a_Rp, _ = pearsonr(NDVI_a_p, NDVI_a)
        coef_BT_a_Rp, _ = pearsonr(BT_a_p, BT_a)

        self.VHI_Rs += coef_VHI_Rs
        self.VCI_Rs += coef_VCI_Rs
        self.TCI_Rs += coef_TCI_Rs

        self.VHI_Rp += coef_VHI_Rp
        self.VCI_Rp += coef_VCI_Rp
        self.TCI_Rp += coef_TCI_Rp

        self.NDVI_Rs += coef_NDVI_Rs
        self.BT_Rs += coef_BT_Rs

        self.NDVI_Rp += coef_NDVI_Rp
        self.BT_Rp += coef_BT_Rp

        self.NDVI_a_Rs += coef_NDVI_a_Rs
        self.BT_a_Rs += coef_BT_a_Rs

        self.NDVI_a_Rp += coef_NDVI_a_Rp
        self.BT_a_Rp += coef_BT_a_Rp

        weights, _ = np.histogram(VHI_c, range(self.n_classes + 1))
        self.weights_label += weights

        for label in range(self.n_classes):
            self.correct_label_all[label] += np.sum((VHI_p_c == label) & (VHI_c == label))
            self.seen_label_all[label] += np.sum((VHI_c == label))
            self.iou_de_label_all[label] += np.sum(((VHI_p_c == label) | (VHI_c == label)))
            self.predicted_label_all[label] += np.sum(VHI_p_c == label)

        VHI_c[VHI_c <= 2] = 0
        VHI_c[VHI_c > 2] = 1
        VHI_p_c[VHI_p_c <= 2] = 0
        VHI_p_c[VHI_p_c > 2] = 1

        self.correct_drought += np.sum((VHI_p_c == 0) & (VHI_c == 0))
        self.seen_drought += np.sum((VHI_c == 0))
        self.iou_de_drought += np.sum(((VHI_p_c == 0) | (VHI_c == 0)))
        self.predicted_drought += np.sum(VHI_p_c == 0)

def min_max_scale(array, min_array, max_array, min_new=-1., max_new=1.):
    array = ((max_new - min_new) * (array - min_array) / (max_array - min_array)) + min_new
    return array


def calc_R2(pred, target):
    target_hat = np.mean(target)
    residuals_sum = np.sum((target - pred) ** 2)
    total_sum = np.sum((target - target_hat) ** 2)
    R2 = 1 - (residuals_sum / total_sum)
    return R2

def calc_VHI(VCI, TCI, alpha=0.5):
    VHI = alpha * VCI + (1 - alpha) * TCI
    return VHI


def calc_weights(data_smn, data_smt, data_sea):

    weights_ndvi = torch.ones_like(data_smn)
    weights_bt = torch.ones_like(data_smt)

    weights_ndvi[torch.isnan(weights_ndvi)] = 0  # mask
    weights_ndvi[torch.isinf(weights_ndvi)] = 0  # mask
    weights_bt[torch.isnan(weights_bt)] = 0  # mask
    weights_bt[torch.isinf(weights_bt)] = 0  # mask

    weights_ndvi[data_sea == 1] = 0  # mask
    weights_bt[data_sea == 1] = 0  # mask

    return weights_ndvi.requires_grad_(False), weights_bt.requires_grad_(False)


def calc_VHI_classes(VHI, mask_cold_surface, mask_sea, mask_no_vegetation):

    VHI[mask_sea == 1] = torch.nan  # mask
    VHI[mask_no_vegetation == 1] = torch.nan  # no vegetation/desert

    VHI[torch.logical_and(0 <= VHI, VHI < 6)] = 0  # exceptional
    VHI[torch.logical_and(6 <= VHI, VHI < 16)] = 1  # extreme
    VHI[torch.logical_and(16 <= VHI, VHI < 26)] = 2  # severe
    VHI[torch.logical_and(26 <= VHI, VHI < 36)] = 3  # moderate
    VHI[torch.logical_and(36 <= VHI, VHI < 40)] = 4  # dry condition
    VHI[40 <= VHI] = 5  # normal
    VHI[mask_cold_surface == 1] = 6  # very cold surface/ice/snow

    return VHI

def calc_VCI_TCI(NDVI, BT, climatology):

    VCI = 100 * (NDVI - climatology[:, 0, :, :]) / (climatology[:, 1, :, :] - climatology[:, 0, :, :])
    TCI = 100 * (climatology[:, 6, :, :] - BT) / (climatology[:, 6, :, :] - climatology[:, 5, :, :])

    VCI = torch.clip(VCI, min=0, max=100)
    TCI = torch.clip(TCI, min=0, max=100)

    return VCI, TCI


def save_model(model, optimizer, epoch, mean_loss_train, mean_loss_val, logger, config, metric):

    dir_log = os.path.join(config.dir_log, config.name)
    checkpoints_dir = os.path.join(dir_log, 'model_checkpoints/')
    if metric == 'loss':
        path = os.path.join(checkpoints_dir, 'best_loss_model.pth')
    elif metric == 'F1':
        path = os.path.join(checkpoints_dir, 'best_F1_model.pth')

    log_string(logger, 'saving model to %s' % path)

    state = {'model_state_dict': model.state_dict()}

    torch.save(state, path)


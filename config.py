# ------------------------------------------------------------------
# Main config file
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# Based on https://github.com/sj-li/DP_GAN/blob/main/config.py
# ------------------------------------------------------------------

import numpy as np
import argparse
import pickle
import os
import datetime

# ------------------------------------------------------------------

def add_all_arguments(parser):

    # --- general options ---
    parser.add_argument('--seed', type=int, default=43, help='random seed')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers for multiprocessing')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='allocate the loaded samples in GPU memory. Use it with training on GPU')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--name', type=str, default='', help='name of the experiment')
    parser.add_argument('--dir_log', type=str, default=r'./log', help='log folder')
    parser.add_argument('--root_data', type=str, default=r'./data/TerrSysMP_NET',
                        help='root of the TerrSysMP_NET dataset')

    parser.add_argument('--encoder', type=str, default='FocalNet_2D', help='name of the encoder model')
    parser.add_argument('--decoder', type=str, default='FocalNet_2D', help='name of the decoder model')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: i.e. 0  (0,1,2, use -1 for CPU)')

    parser.add_argument('--n_lat', type=int, default=411, help='latitudinal extension [image height]')
    parser.add_argument('--n_lon', type=int, default=423, help='longitudinal extension [image width]')
    parser.add_argument('--cut_lateral_boundary', type=int, default=7,
                        help='number of pixels to be excluded from the lateral boundary relaxation zone')
    parser.add_argument('--nan_fill', type=float, default=0.0, help='a value to fill missing data')

    # --- encoder ---
    parser.add_argument('--in_channels', type=int, default=65+6, help='number of input variables')
    parser.add_argument('--out_channels', type=int, default=[96, 192, 384], help='hidden dimensions in the model')
    parser.add_argument('--en_down_sampling', type=str, default='conv', help='type of the down-sampling (conv or max)')
    parser.add_argument('--en_depths', type=int, default=[2, 2, 2], help='number transformer blocks inside each layer')
    parser.add_argument('--en_patch_size', type=int, default=1, help='keep it 1 for regression tasks')
    parser.add_argument('--en_channel_att', type=bool, default=False, help='if True, add channel attention')
    parser.add_argument('--en_channel_att_r', type=int, default=5, help='reduction rate for channel attention')
    parser.add_argument('--en_channel_att_p', type=str, default=['mean', 'std'],
                        help='pooling type for channel attention')
    # encoder Swin v1 + Swin v2 + Focal + Wave_MLP
    parser.add_argument('--en_n_heads', type=int, default=[3, 6, 12], help='number of heads for self-attention')
    parser.add_argument('--en_window_size', type=int, default=8, help='window size for self-attention')
    parser.add_argument('--en_mlp_ratio', type=float, default=4., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--en_drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--en_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--en_drop_path_rate', type=float, default=0.3, help='stochastic depth rate')
    parser.add_argument('--en_qkv_bias', type=bool, default=True,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--en_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--en_ape', type=bool, default=False,
                        help='if True, add absolute position embedding to the patch embedding')
    parser.add_argument('--en_patch_norm', type=bool, default=False,
                        help='if True, add normalization after patch embedding')
    parser.add_argument('--en_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--en_fused_window_process', type=bool, default=False,
                        help='if True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part')
    # encoder Swin v2
    parser.add_argument('--en_pretrained_window_sizes', type=int, default=[12, 12, 12],
                        help='the height and width of the window in pre-training')
    # encoder Focal
    parser.add_argument('--en_focal_levels', type=int, default=[3, 3, 3],
                        help='how many focal levels at all stages. Note that this excludes the finest-grain level')
    parser.add_argument('--en_focal_windows', type=int, default=[3, 3, 3], help='the focal window size at all stages')
    parser.add_argument('--en_use_layerscale', type=bool, default=True,
                        help='whether to use layerscale proposed in CaiT')
    parser.add_argument('--en_layerscale_value', type=float, default=1/np.sqrt(2), help='value for layer scale')
    parser.add_argument('--en_use_postln', type=bool, default=False,
                        help='whether to use layernorm after modulation. It helps stablize training of large models')
    parser.add_argument('--en_use_postln_in_modulation', type=bool, default=False,
                        help='whether to use layernorm after modulation')
    parser.add_argument('--en_normalize_modulator', type=bool, default=False,
                        help='whether to normaize the context in the modulation')
    # pretrained encoder model
    parser.add_argument('--en_pretrained', type=str, default=None,
                        help='pretrained model i.e. focalnet_tiny_lrf.pth or trained model with best loss')

    # --- decoder ---
    parser.add_argument('--de_up_sampling', type=str, default='bilinear', help='type of the up sampling conv or bilinear')
    parser.add_argument('--de_depths', type=int, default=[2, 2, 2], help='number transformer blocks inside each layer')
    # decoder Swin v1 + Swin v2 + Focal + Wave_MLP
    parser.add_argument('--de_n_heads', type=int, default=[3, 6, 12], help='number of heads for self-attention')
    parser.add_argument('--de_window_size', type=int, default=8, help='window size for self-attention')
    parser.add_argument('--de_mlp_ratio', type=float, default=2., help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--de_drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--de_attn_drop_rate', type=float, default=0.0, help='attention dropout rate')
    parser.add_argument('--de_drop_path_rate', type=float, default=0.3, help='stochastic depth rate')
    parser.add_argument('--de_qkv_bias', type=bool, default=True,
                        help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--de_qk_scale', type=float, default=None,
                        help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--de_use_checkpoint', type=bool, default=False,
                        help='whether to use checkpointing to save memory')
    parser.add_argument('--de_fused_window_process', type=bool, default=False,
                        help='if True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part')
    # decoder Swin v2
    parser.add_argument('--de_pretrained_window_sizes', type=int, default=[12, 12, 12],
                        help='the height and width of the window in pre-training')
    # decoder Focal
    parser.add_argument('--de_focal_levels', type=int, default=[3, 3, 3],
                        help='how many focal levels at all stages. Note that this excludes the finest-grain level')
    parser.add_argument('--de_focal_windows', type=int, default=[3, 3, 3], help='the focal window size at all stages')
    parser.add_argument('--de_use_layerscale', type=bool, default=True,
                        help='whether to use layerscale proposed in CaiT')
    parser.add_argument('--de_layerscale_value', type=float, default=1/np.sqrt(2), help='value for layer scale')
    parser.add_argument('--de_use_postln', type=bool, default=False,
                        help='whether to use layernorm after modulation. It helps stabilize training of large models')
    parser.add_argument('--de_use_postln_in_modulation', type=bool, default=False,
                        help='whether to use layernorm after modulation')
    parser.add_argument('--de_normalize_modulator', type=bool, default=False,
                        help='whether to normalize the context in the modulation')

    parser.add_argument('--en_de_pretrained', type=str, default=None,
                        help='pretrained model for testing i.e., a trained model called FocalNet_2D_96.pth')

    parser.add_argument('--years_train', type=str,
                        default=[str(year) for year in range(1989, 2010)] +
                                [str(year) for year in range(2013, 2017)], help='years for training')
    parser.add_argument('--years_val', type=str, default=['2010', '2011', '2017'], help='years for validation')
    parser.add_argument('--years_test', type=str, default=['2012', '2018', '2019'], help='years for testing')

    parser.add_argument('--is_aug', type=bool, default=True, help='if True, apply data augmentation')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training')
    parser.add_argument('--en_lr', type=float, default=0.0003, help='encoder learning rate')
    parser.add_argument('--de_lr', type=float, default=0.0003, help='decoder learning rate')
    parser.add_argument('--en_weight_decay', type=float, default=0.05, help='encoder weight decay')
    parser.add_argument('--de_weight_decay', type=float, default=0.05, help='decoder weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 momentum term for Adam/AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 momentum term for Adam/AdamW')

    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=16, help='learning rate step decay')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')

    parser.add_argument('--loss_NDVI', type=str, default='L1', help='type of Loss function for NDVI (L1, L2 or Huber)')
    parser.add_argument('--loss_BT', type=str, default='L1', help='type of Loss function for BT (L1, L2 or Huber)')
    parser.add_argument('--loss_reduction', type=str, default='mean', help='reduction type for loss functions')
    parser.add_argument('--lambda_NDVI', type=float, default=1., help='weight for NDVI Loss')
    parser.add_argument('--lambda_BT', type=float, default=1., help='weight for BT Loss')
    parser.add_argument('--lambda_vgg', type=float, default=0.1, help='weight for VGG loss')

    # input variables
    parser.add_argument('--variables', type=str,
                        default=['awt',
                                 'capec',
                                 'capeml',
                                 'ceiling',
                                 'cli',
                                 'clt',
                                 'clw',
                                 'evspsbl',
                               #  'gh',
                                 'hfls',
                                 'hfss',
                                 'hudiv',
                                 'hur2',
                                 'hur200',
                                 'hur500',
                                 'hur850',
                                 'hus2',
                                 'hus200',
                                 'hus500',
                                 'hus850',
                                 'incml',
                                # 'lf',#
                                # 'pgw',
                                 'pr',
                                 'prc',
                                 'prg',
                                 'prsn',
                                 'prso',
                                 'prt',
                                 'ps',
                                 'psl',
                                 'rlds',
                               #  'rs',
                                 'sgw',
                                 'snt',
                               #  'sr',
                                 'ta200',
                                 'ta500',
                                 'ta850',
                                 'tas',
                                 'tch',
                                 'td2',
                                 'trspsbl',
                                 'ua200',
                                 'ua500',
                                 'ua850',
                                 'uas',
                                 'va200',
                                 'va500',
                                 'va850',
                                 'vas',
                                 'wtd',
                                 'zg200',
                                 'zg500',
                                 'zg850',
                                 'zmla'
                                 ]
                        , help='input variables')

    parser.add_argument('--variables_static', type=str,
                        default=['FR_LAND',
                                 'HSURF',
                                 'ZBOT',
                                 'DIS_SEA',
                                 'ROUGHNESS',
                                 'SLOPE'],
                        help='input variables')
    return parser


def read_arguments(train=True, print=True, save=True):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser)
    parser.add_argument('--phase', type=str, default='train')
    config = parser.parse_args()
    config.phase = 'train' if train else 'test'
    if print:
        print_options(config, parser)
    if save:
        save_options(config, parser)

    return config


def save_options(config, parser):

    if config.name is None or len(config.name) == 0:
        config.name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    dir_log = os.path.join(config.dir_log, config.name)
    os.makedirs(dir_log, exist_ok=True)

    with open(dir_log + '/config.txt', 'wt') as config_file:
        message = ''
        message += '----------------- Options ---------------       -------------------\n\n'
        for k, v in sorted(vars(config).items()):
            if k in ['variables', 'years_train', 'years_val', 'years_test', 'dir_log', 'root_data']:
                continue
            # comment = ''
            default = parser.get_default(k)
            # if v != default:
            comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

        comment = '\t[default: %s]' % str(parser.get_default('root_data'))
        message += '\n{:>25}: {:<20}{}\n'.format('root_data', vars(config)['root_data'], comment)

        comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
        message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

        message += '\n----------------- Input Variables -------      -------------------'
        message += '\n\n{}\n'.format(str(config.variables))

        message += '\n----------------- Years -----------------      -------------------'
        if config.phase == 'train':
            message += '\n\nTraining: {}'.format(str(config.years_train))
            message += '\nValidation: {}\n'.format(str(config.years_val))
        else:
            message += '\n\nTesting: {}\n'.format(str(config.years_test))

        message += '\n----------------- End -------------------      -------------------'
        config_file.write(message)

    with open(dir_log + '/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)


def print_options(config, parser):
    message = ''
    message += '----------------- Options ---------------       -------------------\n\n'
    for k, v in sorted(vars(config).items()):
        if k in ['variables', 'years_train', 'years_val', 'years_test', 'dir_log', 'root_data']:
            continue
        # comment = ''
        default = parser.get_default(k)
        # if v != default:
        comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<20}{}\n'.format(str(k), str(v), comment)

    comment = '\t[default: %s]' % str(parser.get_default('root_data'))
    message += '\n{:>25}: {:<20}{}\n'.format('root_data', vars(config)['root_data'], comment)

    comment = '\t[default: %s]' % str(parser.get_default('dir_log'))
    message += '{:>25}: {:<20}{}\n'.format('dir_log', vars(config)['dir_log'], comment)

    message += '\n----------------- Input Variables -------      -------------------'
    message += '\n\n{}\n'.format(str(config.variables))

    message += '\n----------------- Years -----------------      -------------------'
    if config.phase == 'train':
        message += '\n\nTraining: {}'.format(str(config.years_train))
        message += '\nValidation: {}\n'.format(str(config.years_val))
    else:
        message += '\n\nTesting: {}\n'.format(str(config.years_test))

    message += '\n----------------- End -------------------      -------------------'
    print(message)


if __name__ == '__main__':

    config = read_arguments(train=True, print=True, save=False)



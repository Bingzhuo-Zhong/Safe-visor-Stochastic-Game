"""
This script is the main entrance for training and testing the agent.
"""


import argparse
import os
import tensorflow as tf

from rein.system.drn_ddpg import DrnDDPG, DrnDDPGParams
from utils import *


def main_train(p):
    drn = DrnDDPG(p)
    drn.train()


def main_test(p):
    p.stats_params.visualize_eval = True
    drn = DrnDDPG(p)
    drn.test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Activate usage of GPU')
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='./config/default_ddpg_adv_ips.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='If set overrides the logfile name and the save name')
    parser.add_argument('--force', action='store_true', help='Override log file without asking')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights')
    parser.add_argument('--params', nargs='*', default=None)
    parser.add_argument('--mode', default='train', help='Choose the mode, train or test')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(DrnDDPGParams(), "config/default_ddpg_drn.json")
        exit("ddpg_drn_config file generated")

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            exit("GPU allocated failed")

    if args.config is None:
        exit("config file needed")

    params = read_config(args.config)

    if args.params is not None:
        params = override_params(params, args.params)

    if args.id is not None:
        params.stats_params.model_name = args.id
        params.stats_params.log_file_name = args.id

    if args.force:
        params.stats_params.force_override = True

    if args.weights is not None:
        params.stats_params.weights_path = args.weights

    params.stats_params.running_mode = args.mode

    if args.mode == 'train':
        main_train(params)
    elif args.mode == 'test':
        main_test(params)
    else:
        assert NameError('No such mode. train or test?')

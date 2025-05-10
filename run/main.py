import os
import sys
import copy
import torch
import random
import logging
import argparse
import numpy as np
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import config
from run import Run


# Get Parser from config
def get_parser():
    '''Parse the config file.'''
    print('ok')
    parser = argparse.ArgumentParser(description='PV Modules Failures Detection with ResNet.')
    parser.add_argument('--config', type=str,
                        default='config/config.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/config.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)

    return cfg

def get_logger(args):
    '''Define logger.'''

    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    file_handler = logging.FileHandler(f'logs/tmp.log')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(file_handler)
    
    return logger_in

def main():

    args = get_parser()

    # ID run configuration: Date and Time Format
    args.id_run = datetime.now().strftime("%d%m%Y_%H%M")
    id_run = copy.deepcopy(args.id_run)

    # WandB experiment name configuration
    args.experiment_name = args.id_run 

    # Device configuration
    DEFAULT_DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        print("Using GPU as default device!")
        DEFAULT_DEVICE = torch.device("cuda")
        torch.backends.cudnn.enabled = True # Accelerate cuda
    else:
        print("Using CPU as default device!\nThe performances will decrease significantly!")

    args['device'] = DEFAULT_DEVICE

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    global logger
    logger = get_logger(args)

    logger.info(args)
    args.logger = logger
    r = Run(args)
    if args.tag == 'train':
        r.train()
    elif args.tag == 'eval': 
        r.eval()
    else: 
        raise Exception(f'Variable tag equal to {args.tag} not implemented!')
    return

if __name__ == '__main__':
    main()
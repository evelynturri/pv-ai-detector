import os
import yaml
import argparse

from utils import config
from openai import OpenAI

def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='PV Modules Failures Detection with GPT-4o.')
    parser.add_argument('--config', type=str,
                        default='config\config.yaml',
                        help='config file')
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
    return cfg


def call_key(key:None):

    OPENAI_API_KEY=key
    client = OpenAI()
    # defaults to getting the key using os.environ.get("OPENAI_API_KEY")
    # if you saved the key under a different environment variable name, you can do something like:
    # client = OpenAI(
    #   api_key=os.environ.get("CUSTOM_ENV_NAME"),
    # )
    return client
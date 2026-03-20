import argparse
from types import SimpleNamespace

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_dict_to_namespace(v) for v in data]
    return data


def dict_to_namespace(data):
    return _dict_to_namespace(data)


def parse_yaml_opt(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-opt", type=str, required=True, help="Path to option yaml.")
    args = parser.parse_args()
    opt = load_yaml(args.opt)
    return opt, args.opt

def print_opt(opt):
    print("Option:")
    for k, v in opt.items():
        print(f"  {k}: {v}")
"""
Author: Zhou Chen
Date: 2020/6/9
Desc: read yaml file
"""
import os
import yaml


def read_config(config_file="../config/default.yml"):
    assert os.path.isfile(config_file), "not a config file"
    with open(config_file, 'r', encoding="utf8") as f:
        cfg = yaml.safe_load(f.read())
    return cfg

"""
Author: Zhou Chen
Date: 2020/6/9
Desc: read yaml file
"""
import os
import yaml
import time

current_path = os.path.dirname(__file__)


def read_config(config_file="../../config/default.yaml"):
    config_file = os.path.join(current_path, config_file)
    assert os.path.isfile(config_file), "not a config file"
    print("load config file from", config_file)
    with open(config_file, 'r', encoding="utf8") as f:
        cfg = yaml.safe_load(f.read())
    return cfg


class Config(object):
    def __init__(self):
        config = read_config()
        ds = config['dataset']['dataset_name']
        dataset_info = config['dataset'][ds]
        self.dataset_name = ds
        self.feat_dir = dataset_info['feat_dir']
        self.video_info_file = dataset_info['video_info_file']
        self.test_info_file = dataset_info['test_info_file']
        self.test_dir = dataset_info['test_dir']
        self.video_filter = dataset_info['video_filter']
        self.tscale = dataset_info['tscale']
        self.data_aug = dataset_info['data_aug']
        self.feature_dim = dataset_info['feature_dim']

        """ Set model and results paths """
        saver_info = config['saver']
        root_dir = saver_info['root_dir']
        timestamp = time.strftime('%m%d-%H%M', time.localtime())
        checkpoint_dir = '{}-{}'.format(saver_info['model_name'], timestamp)
        self.checkpoint_dir = os.path.join(root_dir, checkpoint_dir ) #TODO: comment out the next line
        # self.checkpoint_dir = os.path.join(root_dir, saver_info['checkpoint_dir'])
        self.result_dir = saver_info['result_dir']
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        """ Set training information """
        training_info = config['training']
        learning_rate = training_info['learning_rate']
        lr_epochs = training_info['lr_epochs']
        assert len(learning_rate) == len(lr_epochs)
        self.learning_rate = []
        for lr, n in zip(learning_rate, lr_epochs):
            self.learning_rate.extend([float(lr)] * n)
        self.epoch_num = len(self.learning_rate)
        self.epoch_num = training_info['epoch_num']
        self.batch_size = training_info['batch_size']

        """ Set testing information """
        testing_info = config['testing']
        self.test_mode = testing_info['mode']
        self.test_batch_size = testing_info['batch_size']

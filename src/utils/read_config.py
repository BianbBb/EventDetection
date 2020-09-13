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

        """ Globel Setting """
        config = read_config()
        self.config = config
        timestamp = time.strftime('%m%d-%H%M', time.localtime())


        """ Dataset Setting """
        self.dataset_name = config['dataset']['dataset_name']
        dataset_info = config['dataset'][self.dataset_name]
        self.num_classes = dataset_info['num_classes']
        self.feat_dir = dataset_info['feat_dir']
        self.video_info_file = dataset_info['video_info_file']
        self.test_info_file = dataset_info['test_info_file']
        self.index_file = dataset_info['index_file']
        self.test_dir = dataset_info['test_dir']
        self.video_filter = dataset_info['video_filter']
        self.tscale = dataset_info['tscale']
        self.data_aug = dataset_info['data_aug']
        self.feature_dim = dataset_info['feature_dim']

        """ Save Path Setting """
        saver_info = config['saver']
        self.exp_dir = saver_info['exp_dir']
        self.results_dir = saver_info['results_dir']
        self.log_root = saver_info['log_root']
        self.log_dir = os.path.join(self.log_root, '{}-{}'.format(timestamp, self.dataset_name))

        """model setting"""
        model_info = config['model']
        self.model_name = model_info['model_name']

        # transformer configure
        transformer_info = model_info['transformer']
        self.enc_layers = int(transformer_info['enc_layers'])
        self.dec_layers = int(transformer_info['dec_layers'])
        self.dim_feedforward = int(transformer_info['dim_feedforward'])
        self.hidden_dim = int(transformer_info['hidden_dim'])
        self.dropout = float(transformer_info['dropout'])
        self.nheads = int(transformer_info['nheads'])
        self.num_queries = int(transformer_info['num_queries'])
        self.pre_norm = bool(transformer_info['pre_norm'])

        # loss configure
        loss_info = model_info['loss']
        self.eos_coef = float(loss_info['eos_coef'])
        self.aux_loss = False

        loss_weight = loss_info['loss_weight']
        self.set_cost_classes = float(loss_weight['set_cost_classes'])
        self.set_cost_segments = float(loss_weight['set_cost_segments'])
        self.set_cost_diou = float(loss_weight['set_cost_diou'])

        matcher = loss_info['matcher']
        self.matcher_cost_classes = float(matcher['matcher_cost_classes'])
        self.matcher_cost_segments = float(matcher['matcher_cost_segments'])
        self.matcher_cost_diou = float(matcher['matcher_cost_diou'])


        """ Training Setting """
        training_info = config['training']
        self.gpu_id = training_info['gpu_id']
        self.resume = bool(training_info['resume'])
        self.train_pth_name = training_info['pth_name']
        self.train_pth_file = training_info['pth_file']
        self.train_pth_load_dir = os.path.join(self.log_root, self.train_pth_name, self.train_pth_file)
        self.learning_rate = float(training_info['learning_rate'])
        self.lr_schduler = bool(training_info['lr_scheduler'])
        self.epoch_num = training_info['epoch_num']
        self.batch_size = training_info['batch_size']

        train_pth_save_dir = '{}-{}-{}'.format(self.model_name, timestamp, self.dataset_name)
        self.train_pth_save_dir = os.path.join(self.exp_dir, train_pth_save_dir)

        """ Testing Setting """
        testing_info = config['testing']
        self.mode = testing_info['mode']
        self.test_batch_size = testing_info['test_batch_size']
        self.test_pth_name = testing_info['pth_name']
        self.test_pth_file = testing_info["pth_file"]
        self.test_pth_load_dir = os.path.join(self.log_root, self.test_pth_name,self.test_pth_file)
        self.test_csv_save_dir = os.path.join(self.results_dir, '{}-{}'.format(self.test_pth_name, self.mode))

        # Post Processing
        self.post_csv_load_dir = self.test_csv_save_dir
        self.post_json_save_path = os.path.join(self.results_dir, '{}-{}.json'.format(self.test_pth_name, self.mode))

        # eval 测评结果图表, 只能用于val dataset
        self.eval_json_load_path = os.path.join(self.results_dir, '{}-validation.json'.format(self.test_pth_name))

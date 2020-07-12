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


# TODO：修改 self.XXX 与 default.yaml 的名称相同
# train，test，post_processing, eval 需要的路径与设置
class Config(object):
    def __init__(self):

        """ Globel Setting """
        config = read_config()
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

        """ Training Setting """
        training_info = config['training']
        self.gpu_id = training_info['gpu_id']
        self.resume = bool(training_info['resume'])
        self.learning_rate = float(training_info['learning_rate'])
        self.epoch_num = training_info['epoch_num']
        self.batch_size = training_info['batch_size']
        self.aux_loss = False

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
        # matcher configure
        matcher_info = model_info['matcher']
        self.set_cost_classes = float(matcher_info['set_cost_classes'])
        self.set_cost_segments = float(matcher_info['set_cost_segments'])
        self.set_cost_diou = float(matcher_info['set_cost_diou'])

        """ Testing Setting """
        testing_info = config['testing']
        self.mode = testing_info['mode']
        self.test_batch_size = testing_info['test_batch_size']
        self.pth_name = testing_info['pth_name']

        """ Save Path Setting """
        saver_info = config['saver']
        self.exp_dir = saver_info['exp_dir']
        self.results_dir = saver_info['results_dir']
        self.log_dir = os.path.join(saver_info['log_dir'], 'logs{}'.format(timestamp))

        # Train

        train_pth_save_dir = '{}-{}-{}'.format(self.model_name, timestamp, self.dataset_name)
        self.train_pth_save_dir = os.path.join(self.exp_dir, train_pth_save_dir)

        # Test
        self.test_pth_load_dir = os.path.join(self.exp_dir, self.pth_name)
        self.test_csv_save_dir = os.path.join(self.results_dir, '{}-{}'.format(self.pth_name, self.mode))

        # Post Processing
        self.post_csv_load_dir = self.test_csv_save_dir
        self.post_json_save_path = os.path.join(self.results_dir, '{}-{}.json'.format(self.pth_name, self.mode))

        # eval 测评结果图表, 只能用于val dataset
        self.eval_json_load_path = os.path.join(self.results_dir, '{}-validation.json'.format(self.pth_name))

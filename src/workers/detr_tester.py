  # load net
    # load weight
# load feature
# tester.run()
# tester get output
# tester change the outputs to result
import json
import os
import time
import numpy as np
from tqdm import tqdm
import torch

from utils.proposal_ops import cl2xy


class ADTR_tester():
    def __init__(self, config, net, test_loader):
        self.device = "cuda:{}".format(config.gpu_id) if torch.cuda.is_available() else "cpu"
        self.batch_size = config.batch_size
        self.pth_path = config.test_pth_load_dir

        self.net = net
        self.net.to(self.device)
        self.net.eval()

        self.load_weight(self.pth_path)
        self.test_loader = test_loader

        with open(config.index_file, 'r') as f:
            self.classes_index = json.load(f)

        self.num_classes = config.num_classes
        self.num_queries = config.num_queries

    def load_weight(self,pth_path):
        try:
            self.net.load_state_dict(torch.load(pth_path)["state_dict"])
            print('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            print('Can not find feature.pkl !')

    def forward(self): # network 输出
        outputs = {}
        for n_iter, (samples, _, infos) in tqdm(enumerate(self.test_loader)):
            samples = torch.cat([i.unsqueeze(0) for i in samples], dim=0)
            samples = samples.to(self.device)
            batch_infos = list(infos)
            # forward
            batch_outputs = self.net(samples)
            # 转为标准格式
            result = self.batch_process(batch_outputs,batch_infos)
            outputs.update(result)
        return outputs

    def batch_process(self,batch_outputs,batch_infos): # 对一个batch的处理
        batch_result = {}
        batch_classes = batch_outputs["classes"].softmax(-1).cpu() # b,q,num+1

        batch_segmentes = batch_outputs["segments"]                # b,q,2
        # print(batch_segmentes)
        for v in range(len(batch_infos)):
            video_info = batch_infos[v]
            video_name = video_info["video_name"]
            video_duration = video_info["video_duration"] # 1
            video_class = batch_classes[v]                # q,num+1
            #print(video_class)

            video_segment = batch_segmentes[v]            # q,2
            video_segment = cl2xy(video_segment).clamp(min=0,max=video_duration)          # 转换坐标表示

            # 100个query对应的 classes的每一个的最大值位置和最大值 转为numpy
            query_max = torch.max(video_class,dim=-1)
            query_confidence = query_max[0].data # 每个query的类别置信度 # q
            query_class = query_max[1].data.numpy() # 每个query的类别索引        # q
            #  print(query_class)

            query_segment = video_segment * video_duration                 # q,2
            batch_result.update(self.video_process(video_name,query_confidence,query_class,query_segment))
        return batch_result

    def video_process(self,video_name,query_confidence,query_class,query_segment): #对batch中一个video的处理,过滤掉预测值为空的query
        # TODO:后处理（NMS）
        values = []
        assert self.num_queries == len(query_confidence)
        assert self.num_queries == len(query_class)
        assert self.num_queries == len(query_segment)

        for i in range(self.num_queries):
            query_result = {}
            if query_class[i] == self.num_classes:
                continue
            # if query_confidence[i].tolist() < 0.02: # 设定阈值
            #     continue
            query_result["score"] = query_confidence[i].tolist()
            query_result["label"] = list(self.classes_index.keys())[query_class[i]]

            query_result["segment"] = query_segment[i].tolist()
            values.append(query_result)
        return {video_name:values}

    def normal_result(self,result): # 生成标准化结果文件
        pass

    @torch.no_grad()
    def run(self):
        test_time = time.time()
        output = self.forward()
        # result = self.normal_result(output)
        print('Test time: {}'.format(time.time()-test_time))
        return output

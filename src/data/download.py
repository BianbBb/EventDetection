# this script surport to get i3d features of the train and test dataset
# read train_annotations.json    # train and cal dataset
# read val_video_ids.txt   # test dataset 4835

import os
import json
import urllib.request as request
import time


def down_train_feature(type='video', train_annotations_file="../../data/train_annotations.json",
                       saved_path="../../data/dataset/train"):
    if type == "video":
        root_url = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798/train/video/{}.mp4"
        saved_path = os.path.join(saved_path, "video")
        suffix = ".mp4"
    else:
        root_url = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798/train/i3d_feature/{}.npy"
        saved_path = os.path.join(saved_path, "i3d")
        suffix = ".npy"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    f = open(train_annotations_file, 'r')
    content = f.readline()  # this file only one line
    anno = json.loads(content)
    total_num = len(anno)
    num = 0
    for video_id in anno.keys():
        num += 1
        file_url = root_url.format(video_id)
        print("file {}/{} starting download from {}".format(num, total_num, file_url))
        download_from_url(file_url, os.path.join(saved_path, video_id+suffix))


def down_test_feature(info_path, folder_dir, type='i3d',
                      url_path='http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798/'):
    print(folder_dir)
    time.sleep(3)
    if not os.path.exists(folder_dir):
        print("Selected folder not exist, try to create it.")
        os.makedirs(folder_dir)

    with open(info_path, 'r') as f:
        lines = f.readlines()
        num = 0
        for line in lines:
            num = num + 1
            feature_url = url_path + type + '_feature/' + line[:-1] + '.npy'
            save_path = os.path.join(folder_dir, line[:-1] + '.npy')
            download_from_url(feature_url, save_path)
            if num % 100 == 0:
                print('{} files have been doenloaded '.format(num))

        print('Download Finished ! Total : {}'.format(num))


def download_from_url(url, filepath):
    if os.path.exists(filepath):
        print("File have already exist.")
    else:
        try:
            opener = request.build_opener()
            opener.addheaders = [('User-Agent',
                                  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            request.install_opener(opener)
            request.urlretrieve(url, filename=filepath)
        except Exception as e:
            print("Error occurred when downloading file, error message:")
            print(e)


if __name__ == '__main__':
    down_train_feature(type="i3d")

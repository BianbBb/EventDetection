# this script surport to get i3d features of the train and test dataset
# read train_annotations.json    # train and cal dataset
# read val_video_ids.txt   # test dataset 4835

import os
import json
#from urllib.request import urlretrieve
import urllib.request as request
import time

def down_trian_feature(type='i3d', ):
    pass
    # if type == 'i3d':



def down_test_feature(info_path, folder_dir, type='i3d',url_path='http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798/'):
    print(folder_dir)
    time.sleep(3)
    if not os.path.exists(folder_dir):
        print("Selected folder not exist, try to create it.")
        os.makedirs(folder_dir)

    with open(info_path,'r') as f:
        lines = f.readlines()
        num = 0
        for line in lines:
            num = num + 1
            feature_url = url_path + type + '_feature/' + line[:-1] + '.npy'
            save_path = os.path.join(folder_dir, line[:-1]+'.npy')
            download_from_url(feature_url, save_path)
            if num % 100 == 0:
                print('{} files have been doenloaded '.format(num))

        print('Download Finished ! Total : {}'.format(num))


def download_from_url(url, filepath):
    if os.path.exists(filepath):
            print("File have already exist.")
    else:
        try:
            opener =request.build_opener()
            opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            request.install_opener(opener)
            request.urlretrieve(url, filename=filepath)
        except Exception as e:
            print("Error occurred when downloading file, error message:")
            print(e)




if __name__ == '__main__':
    url_path = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531798/'
    download_dir = 'G:/BBBLib/TianChi/data/'
    train_info_path = 'G:/BBBLib/TianChi/data/train_annotations.json'
    test_info_path = 'G:/BBBLib/TianChi/data/val_video_ids.txt'
    down_test_feature(info_path=test_info_path, folder_dir=os.path.join(download_dir, 'test/'),type='i3d')




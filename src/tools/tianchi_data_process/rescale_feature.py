"""
1024*N -> 1024*400
extract 400 columns from source feature sequence

DBG 的csv数据：T:100,D:400
Tianchi 的npy数据：T:N，D:1024 - > 100,1024
"""

import numpy as np
import os
from tqdm import tqdm
import scipy
import scipy.interpolate


def load_npy(npy_file):
    return np.load(npy_file)


def resize_feature(inputData, newSize=100):
    # refer to DBG code
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = scipy.interpolate.interp1d(x, inputData, axis=0)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new


def sample(feature, new_n=100):
    old_n, dim = np.shape(feature)
    index = np.floor(np.linspace(0, old_n - 1, new_n)).reshape(-1)
    new_feature = feature[index.astype("int")]
    return new_feature


if __name__ == '__main__':
    old_dir = '/data/byh/EventDetection/test/i3d/'
    new_dir = '/data/byh/EventDetection/test/i3d_400/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    count = 0
    for filename in tqdm(os.listdir(old_dir)):  # 44611
        count = count + 1
        fea = load_npy(os.path.join(old_dir, filename))
        new_fea = sample(fea)
        np.save(os.path.join(new_dir, filename), new_fea)

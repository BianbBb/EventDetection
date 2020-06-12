
def draw_loss():
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)####
    # Grab some test data.

    x = y = np.arange(0.0,1.0, 0.05)
    X, Y = np.meshgrid(x, y)

    gamma = 5
    Z = (1-X)**gamma * Y**gamma  + X**gamma  * (1-Y)**gamma
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

#
# from torchvision import transforms
# from torch.utils.data import dataset, dataloader
# from torchvision.datasets.folder import default_loader
# from utils.RandomErasing import RandomErasing
# from utils.RandomSampler import RandomSampler
# from PIL import Image
# import os
# import re
#
#
# class Data():
#     def __init__(self):
#         train_transform = transforms.Compose([
#             transforms.Resize((384, 128), interpolation=3),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
#         ])
#
#         test_transform = transforms.Compose([
#             transforms.Resize((384, 128), interpolation=3),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         self.trainset = Cartoon(train_transform, 'train', '\iCartoon')  # path可以设到超参里
#         self.testset = Cartoon(test_transform, 'test', '\iCartoon')
#         self.queryset = Cartoon(test_transform, 'query', '\iCartoon')
#
#         self.train_loader = dataloader.DataLoader(self.trainset,
#                                                   sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
#                                                                         batch_image=opt.batchimage),
#                                                   batch_size=opt.batchid * opt.batchimage, num_workers=8,
#                                                   pin_memory=True)
#         self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=8, pin_memory=True)
#         self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchtest, num_workers=8,
#                                                   pin_memory=True)
#
# class Cartoon(dataset.Dataset):
#     def __init__(self, transform, dtype, data_path):
#         self.transform = transform
#         if dtype == 'train':
#             data_path += '/....'
#             fh = open(data_path+'/....txt', 'r')
#         elif dtype == 'test':
#             data_path += '/....'
#             fh = open(data_path + '/....txt', 'r')
#         else:
#             data_path += '/....'
#             fh = open(data_path + '/...txt', 'r')
#
#         imgs = []
#         for line in fh:
#             words = line.split() #以空格将每行数据分成多列
#             imgs.append(data_path + '/' + words[0], words[0].split(_)[3])  # 图片路径
#         self.imgs = imgs
#
#     def __getitem__(self, index):
#         path,label = self.imgs[index]
#         img = Image.open(path).covert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)
#
#
#


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred+0.0000001) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def similarity(a, b):
    A = np.array(a)
    B = np.array(b)
    dot_product = float(np.dot(A, B))
    magnitude_product = np.linalg.norm(A) * np.linalg.norm(B)
    cos = dot_product / magnitude_product
    return cos

if __name__ == '__main__':
    import torch
    # from torch.nn import functional as F
    # n=5
    # x=torch.randn(n).unsqueeze(-1)
    # print(x)
    # y = torch.eye(n)*0.5+ torch.randn(n,n)*0.5*(1/n)
    # n1= torch.mm(y,x)
    # n2 = torch.mm(y, n1)
    # n3 = torch.mm(y, n2)
    # print(n3)
    # print(n3-1.5)
    # print(torch.sigmoid(n3-1.5))
    # #print(np.mean((a-b)**2))
    mask = torch.ones((128,128),dtype=torch.ByteTensor)
    print(mask)






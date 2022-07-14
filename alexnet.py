# import time
# # import numpy as np
# import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# # from torchvision import datasets
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from torchvision import models
# from PIL import Image
# import yaml
# import os
# from utils import *
#
# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True
#
# # path = 'E:/Code/Python/vgg/'
#
# # Device
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device:', DEVICE)
#
# # hyperparameter
# data_cfg = open('data/data.yaml', 'r', encoding="utf-8")
# # 使用文件对象作为参数
# data = yaml.load(data_cfg, Loader=yaml.SafeLoader)
# num_classes = data['nc']
# train_dir = data['train']
# test_dir = data['test']
#
# random_seed = 1
# learning_rate = 0.001
# num_epochs = 150
# batch_size = 8
#
# # class number
# IMG_SIZE = (224, 224)  # resize image
# # IMG_MEAN = [0.485, 0.456, 0.406]
# # IMG_STD = [0.229, 0.224, 0.225]
#
# transforms = transforms.Compose([
#     transforms.Resize(IMG_SIZE),
#     transforms.ToTensor()
#     #     transforms.Normalize(IMG_MEAN, IMG_STD)
# ])
#
#
# class MyDataset(Dataset):
#     def __init__(self, root, datatxt, transform=None, target_transform=None):
#         super(MyDataset, self).__init__()
#         with open(root + datatxt, 'r') as f:
#             imgs = []
#             for line in f:
#                 line = line.rstrip()
#                 words = line.split()
#                 imgs.append((words[0], int(words[1])))
#             self.root = root
#             self.imgs = imgs
#             self.transform = transform
#             self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         f, label = self.imgs[index]
#         img = Image.open(self.root + f).convert('RGB')
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)
#
#
# train_data = MyDataset(train_dir, 'train_origin.txt', transform=transforms)
# test_data = MyDataset(test_dir, 'test_origin.txt', transform=transforms)
#
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size)
#
# """
# initial model
# """
# model = models.alexnet(pretrained=True)  # 选择是否预训练
#
# # for param in model.parameters():
# #     param.requires_grad = False
# # model = models.vgg16(pretrained=False)
#
# model.classifier[6] = nn.Linear(4096, num_classes)  # 按类分类
# model = model.to(DEVICE)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
# # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # save results
# save_path=make_save_folder()
# result_file = open(save_path + "/results.txt", 'w')
# cfg_file = open(save_path + "/cfg.txt", 'w')
# cfg_file.write('alexnet_origin\n'+'learning_rate: {:.4f}\nnum_epochs: {:d}%\npretrained=1\naug=0'.format(learning_rate, num_epochs))
# cfg_file.close()
# best_prec = 0
#
# """
# train
# """
# for epoch in range(num_epochs):
#     start = time.perf_counter()
#     model.train()
#     running_loss = 0.0
#     correct_pred = 0
#     if epoch==50:
#         learning_rate = 0.0005
#     if epoch==100:
#         learning_rate = 0.0001
#     for index, data in enumerate(train_loader):
#         image, label = data
#         image = image.to(DEVICE)
#         label = label.to(DEVICE)
#         y_pred = model(image)
#
#         _, pred = torch.max(y_pred, 1)
#         correct_pred += (pred == label).sum()
#
#         loss = criterion(y_pred, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += float(loss.item())
#     end = time.perf_counter()
#
#     now_loss = running_loss / (index + 1)
#     accuracy = correct_pred.item() / (batch_size * (index + 1)) * 100
#     print('best accu is {:.2f}%, now accu is {:.2f}%'.format(best_prec, accuracy))
#     print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.
#           format(epoch + 1, num_epochs, running_loss / (index + 1),
#                  correct_pred.item() / (batch_size * (index + 1)) * 100))
#     print('Time: {:.2f}s'.format(end - start))
#     result_file.write('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%\n'.
#                       format(epoch + 1, num_epochs, running_loss / (index + 1), accuracy))
#     if best_prec < accuracy:
#         best_prec = accuracy
#         torch.save(model, os.path.join(save_path, str(num_classes) + 'c5best.pth'))
#         print("model saved\n")
#     else:
#         torch.save(model, os.path.join(save_path, str(num_classes) + 'c5last.pth'))
# print('Finished training!')
#
# """
# test
# """
# test_loss = 0.0
# correct_pred = 0
# model.eval()
# for _, data in enumerate(test_loader):
#     image, label = data
#     image = image.to(DEVICE)
#     lable = label.to(DEVICE)
#     y_pred = model(image).to(torch.device("cpu"))
#
#     _, pred = torch.max(y_pred, 1)
#     correct_pred += (pred == label).sum()
#
#     loss = criterion(y_pred, label)
#     test_loss += float(loss.item())
# print('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / 12, correct_pred.item() / 120 * 100))
# test_result_path = make_test_folder()
# test_result_file = open(test_result_path + "/results.txt", 'w')
# test_result_file.write('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / 316, correct_pred.item() / 31600))
# cfg_file = open(test_result_path + "/alexnet_origin.txt", 'w')
# cfg_file.close()

import time
# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import yaml
import os
from utils import *
import argparse

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

IMG_SIZE=(224, 224)
transforms = transforms.Compose({
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
})

class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        with open(root + datatxt, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.root = root
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        f, label = self.imgs[index]
        img = Image.open(self.root + f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

"""
train
"""
def train(save_path,opt):
    # data load
    data_cfg = open(opt.data, 'r', encoding="utf-8")
    data = yaml.load(data_cfg, Loader=yaml.SafeLoader)

    num_classes = data['nc']
    train_dir = data['train']
    test_dir = data['test']
    batch_size = opt.batch_size
    train_data = MyDataset(train_dir, 'train.txt', transform=transforms)
    test_data = MyDataset(test_dir, 'test.txt', transform=transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    #IMG_SIZE = opt.img_size  # resize image

    # save results

    result_file = open(save_path + "/results.txt", 'w')
    test_file = open(save_path + "/test_results.txt", 'a')
    with open(test_dir + 'test.txt') as f:
        test_len = len(f.readlines())

    best_prec = 0

    #init model
    pretrained=opt.pretrained
    model = models.alexnet(pretrained=pretrained)  # 选择是否预训练
    print("pretrained=",pretrained)

    model.classifier[6] = nn.Linear(4096, num_classes)  # 按类分类
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=opt.learning_rate)
    num_epochs=opt.epochs
    for epoch in range(num_epochs):
        if epoch==10:
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=opt.learning_rate/2)
        if epoch==30:
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=opt.learning_rate/10)
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct_pred = 0
        for index, data in enumerate(train_loader):
            image, label = data
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            y_pred = model(image)

            _, pred = torch.max(y_pred, 1)
            correct_pred += (pred == label).sum()

            loss = criterion(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
        end = time.perf_counter()

        #now_loss = running_loss / (index + 1)
        accuracy = correct_pred.item() / (batch_size * (index + 1)) * 100
        print('best accu is {:.2f}%, now accu is {:.2f}%'.format(best_prec, accuracy))
        print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.
              format(epoch + 1, num_epochs, running_loss / (index + 1), correct_pred.item() / (batch_size * (index + 1)) * 100))
        print('Time: {:.2f}s'.format(end - start))
        result_file.write('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%\n'.
                          format(epoch + 1, num_epochs, running_loss / (index + 1), accuracy))
        if best_prec < accuracy:
            best_prec = accuracy
            torch.save(model, os.path.join(save_path, str(num_classes) + 'c5best.pth'))
            print("model saved\n")
        else:
            pass
        torch.save(model, os.path.join(save_path, str(num_classes) + 'c5last.pth'))

        # test part
        test_loss = 0.0
        correct_pred = 0
        model.eval()
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for _, data in enumerate(test_loader):
            image, label = data
            image = image.to(torch.device("cuda:0"))
            lable = label.to(torch.device("cuda:0"))
            y_pred = model(image).to(torch.device("cpu"))

            _, pred = torch.max(y_pred, 1)
            correct_pred += (pred == label).sum()

            loss = criterion(y_pred, label)
            test_loss += float(loss.item())

        with open(test_dir + 'test.txt') as f:
            test_len = len(f.readlines())
        print('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / test_len,
                                                                 correct_pred.item() / test_len * 100))
        test_file.write('Test loss: {:.4f}\tTest accuracy: {:.2f}%\n'.format(test_loss / test_len,
                                                                 correct_pred.item() / test_len * 100))

    print('Finished training!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='runs/train/exp3/best.pt',help='initial weights path')  # runs/train/exp10/weights/best.pt
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[224, 224], help='[train, test] image sizes')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)

    opt = parser.parse_args()
    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', DEVICE)

    # hyperparameter
    data_cfg = open('data/data.yaml', 'r', encoding="utf-8")
    # 使用文件对象作为参数
    data = yaml.load(data_cfg, Loader=yaml.SafeLoader)
    num_classes = data['nc']
    train_dir = data['train']
    test_dir = data['test']
    random_seed = opt.random_seed
    batch_size = opt.batch_size
    train_data = MyDataset(train_dir, 'train.txt', transform=transforms)
    test_data = MyDataset(test_dir, 'test.txt', transform=transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    IMG_SIZE = opt.img_size  # resize image

    save_path= make_save_folder()
    cfg_file = open(save_path + "/cfg.txt", 'w')
    cfg_file.write(
        'alex_origin\n' + 'learning_rate: {:.4f}\nnum_epochs: {:d}%\npretrained=0\naug=2'.format(opt.learning_rate,                                                                                              opt.epochs))
    cfg_file.close()
    print('start training for {:d} epochs: '.format(opt.epochs))
    train(save_path,opt)

import os
from PIL import Image
import glob
import numpy as np
import cv2
import random
from PIL import ImageEnhance

def mkdir(path):#查看是否有路径，若无则创建路径
    # 判断文件夹有无后再进行创建文件夹和子文件内容
    isExists = os.path.exists(path)
    if not isExists:  # 如果不存在相同的，则创建文件夹
        os.makedirs(path)
        print(path + "文件夹创建成功")
    else:
        print("文件已存在")
    return

def make_5class_aug_subset5(class_num,prob=0.8):
    '''
    在亮度上做增强，但不在偏移量上做增强
    相比于上次的方法这次先切分再扩增
    :param class_num:
    :param prob:
    :return:
    '''
    bright1 = 0.8
    bright2 = 1.2
    bright3 = 1.4
    index=[0.45,0.45,0.45,0.45,0.45,0.5,0.3,0.32,0.35,0.2]
    in_dir = r'images'#images文件夹是存放原图的位置，images文件夹中有10个子文件夹，具体见实际文件

    train_dir = r'\train'
    test_dir = r'\test'

    mkdir(train_dir)
    mkdir(test_dir)

    outfile_train = open(os.path.join(train_dir, "train.txt"), 'w')
    outfile_train_rotate = open(os.path.join(train_dir, "train_rotate.txt"), 'w')
    outfile_train_light = open(os.path.join(train_dir, "train_light.txt"), 'w')
    outfile_train_sync = open(os.path.join(train_dir, "train_sync.txt"), 'w')
    outfile_test = open(os.path.join(test_dir, "test.txt"), 'w')

    j = 0
    k = 1
    count=0
    for i in class_num:
        crop_size = index[i]
        print('crop_size: ',crop_size)
        srcPath=os.path.join(in_dir,str(i)) #取到每个每个分组下面
        for filename in os.listdir(srcPath):
            if i==6:
                count+=1
                if count==65:
                    crop_size=0.4
            srcFile = os.path.join(srcPath, filename)   #取到具体原图地址
            #srcFile=r'F:\ai_learining\yolov5\img_press\images\images\9\451 (1).jpg'
            sImg = Image.open(srcFile)
            filename = str(k) + '.jpg'
            k+=1
            half_the_width = sImg.size[0] / 2
            half_the_height = sImg.size[1] / 2
            #第1大类是旋转+原图
            out1 = sImg.crop(
                (
                    half_the_width - crop_size * sImg.size[0],
                    half_the_height - crop_size * sImg.size[0],
                    half_the_width + crop_size * sImg.size[0],
                    half_the_height + crop_size * sImg.size[0]
                ))

            out11 = out1.rotate(90)
            out12 = out1.rotate(180)
            out13 = out1.rotate(270)
            # 3 rotate images

            brightEnhancer = ImageEnhance.Brightness(out1)
            out21 = brightEnhancer.enhance(bright1)

            brightEnhancer = ImageEnhance.Brightness(out1)
            out22 = brightEnhancer.enhance(bright2)

            brightEnhancer = ImageEnhance.Brightness(out1)
            out23 = brightEnhancer.enhance(bright3)
            # 3 bright images

            out31=out21.rotate(90)
            out32=out22.rotate(180)
            out33=out23.rotate(270)
            # 3 sync images


            rand = random.random()
            classi=j//2
            if rand < prob:
                out1.save(os.path.join(train_dir, filename))
                outfile_train.write(filename + ' ' + str(classi) + '\n')

                outfile_train_rotate.write(filename + ' ' + str(classi) + '\n')
                out11.save(os.path.join(train_dir, (filename[0:-4] + '-11.jpg')))
                outfile_train_rotate.write((filename[0:-4] + '-11.jpg') + ' ' + str(classi) + '\n')
                out12.save(os.path.join(train_dir, (filename[0:-4] + '-12.jpg')))
                outfile_train_rotate.write((filename[0:-4] + '-12.jpg') + ' ' + str(classi) + '\n')
                out13.save(os.path.join(train_dir, (filename[0:-4] + '-13.jpg')))
                outfile_train_rotate.write((filename[0:-4] + '-13.jpg') + ' ' + str(classi) + '\n')

                outfile_train_light.write(filename + ' ' + str(classi) + '\n')
                out21.save(os.path.join(train_dir, (filename[0:-4] + '-21.jpg')))
                outfile_train_light.write((filename[0:-4] + '-21.jpg') + ' ' + str(classi) + '\n')
                out22.save(os.path.join(train_dir, (filename[0:-4] + '-22.jpg')))
                outfile_train_light.write((filename[0:-4] + '-22.jpg') + ' ' + str(classi) + '\n')
                out23.save(os.path.join(train_dir, (filename[0:-4] + '-23.jpg')))
                outfile_train_light.write((filename[0:-4] + '-23.jpg') + ' ' + str(classi) + '\n')


                outfile_train_sync.write(filename + ' ' + str(classi) + '\n')
                out31.save(os.path.join(train_dir, (filename[0:-4] + '-31.jpg')))
                outfile_train_sync.write((filename[0:-4] + '-31.jpg') + ' ' + str(classi) + '\n')
                out32.save(os.path.join(train_dir, (filename[0:-4] + '-32.jpg')))
                outfile_train_sync.write((filename[0:-4] + '-32.jpg') + ' ' + str(classi) + '\n')
                out33.save(os.path.join(train_dir, (filename[0:-4] + '-33.jpg')))
                outfile_train_sync.write((filename[0:-4] + '-33.jpg') + ' ' + str(classi) + '\n')


            else:
                out1.save(os.path.join(test_dir, filename))
                outfile_test.write(filename + ' ' + str(classi) + '\n')
        j+=1

if __name__ == '__main__':
    class_num = range(0,10)
    make_5class_aug_subset5(class_num)

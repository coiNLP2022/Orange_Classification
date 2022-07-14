import os
import matplotlib.pyplot as plt
import math
from numpy import linspace
from pathlib import Path
# x='/mnt/YOLOv4/data/coco/images/train2017'
# val = Path(x)#.resolve()
#
# if val.exists():
#     print("exist")
#
#
# from pathlib import Path
# dir1=Path('runs')
# results_file = dir1/ 'results.txt'
# print(results_file)
# #print(os.environ['WORLD_SIZE'])#['RANK']

# accu,loss plot
f = open("results_600.txt",encoding = "utf-8")
#输出读取到的数据
#print(f.read())
lines=f.readlines()
f.close()
#print(lines)
loss=[]
mAP=[]
i=0
for line in lines:
    # if i==150:
    #     break
    # if i<150:
    #     i+=1
    # else:
    #     data=line.split()
    #     #loss.append(float(data[4]))
    #     mAP.append(float(data[7][0:5]))
    #     # print(loss,mAP)
    data = line.split()
    loss.append(float(data[5])*100)
    mAP.append(float(data[10])*100)
    # print(loss,mAP)

# print(mAP)
# print(len(mAP))
for i in range(2):
    xx=range(599)
    plt.figure()
    plt.xlabel('epochs')
    xt = range(0, 601, 50)
    yt = range(0, 101,10)
    plt.yticks(yt)
    plt.xticks(xt)
    if i==0:
        plt.ylabel('mAP@.5')
        plt.plot(xx, mAP)
        plt.title("mAP@.5")
        plt.savefig('mAP.png')
        #plt.show()
    else:
        plt.ylabel('loss')
        plt.plot(xx, loss)
        plt.title("loss")
        plt.savefig('loss.png')


#关闭文件

'''
#mish,LeakyReLu plot
x=linspace(-4,4)
mish=[]
leaky=[]
for xx in x:
    mish.append(xx * (math.tanh(math.log(1+math.exp(xx)))))
    if xx<0:
        leaky.append(xx*0.2)
    else:
        leaky.append(xx)
plt.plot(x,mish)
plt.plot(x,leaky)
plt.legend(labels=["Mish","LeakyReLu"],loc="upper left",fontsize=16)
plt.yticks([-1,0,1,2,3,4])
ax = plt.gca()
# 删除右边框
ax.spines['right'].set_color('none')
# 删除上边框
ax.spines['top'].set_color('none')
# 删除右边框
#ax.spines['left'].set_color('none')
# 删除上边框
#ax.spines['bottom'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
# 将横坐标上 0 当做 纵坐标的原点（即横坐标左移一个单位,将 0 处当做横坐标原点）
ax.spines['left'].set_position(('data', 0))
plt.savefig('activation.png')
plt.show()
'''


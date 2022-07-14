import glob
import os

def make_save_folder():
    # 第一步创建文件夹
    j = 1
    if not os.path.exists('runs/train'):
        os.makedirs('runs/train')
    else:
        all_folder = glob.glob('runs/train/exp*')
        if len(all_folder) > 0:
            for i in range(len(all_folder)):
                #all_folder[i] = all_folder[i].split('/')[-1]
                all_folder[i] = all_folder[i].split('\\')[-1]
                #print(all_folder)
                all_folder[i] =int(all_folder[i][3:])
            j = max(all_folder) + 1
    os.makedirs(r'runs/train/exp' + str(j))
    return r'runs/train/exp' + str(j)

#def save_results():

def make_test_folder():
    # 第一步创建文件夹
    j = 1
    if not os.path.exists('runs/test'):
        os.makedirs('runs/test')
    else:
        all_folder = glob.glob('runs/test/exp*')
        if len(all_folder) > 0:
            for i in range(len(all_folder)):
                all_folder[i] = all_folder[i].split('/')[-1]
                all_folder[i] =int(all_folder[i][3:])
            j = max(all_folder) + 1
    os.makedirs('runs/test/exp' + str(j))
    return 'runs/test/exp' + str(j)

if __name__ == '__main__':
    pass
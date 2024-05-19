import fnmatch
import os
import time
import torch
from tqdm import tqdm
from models import LHKDNET
from utils.runtask import *
from utils.runtask import print_dots as weightloader
from utils.utils_draw import drawBoxOnVOC_NWPU as Predict_NWPU

#对应数据集图像的路径
NwpuPath = 'C:/YyFiles/papercode/LMROD-Lite/NWPUVHR10VOC/VOC2007/'

#统计检测结果。
def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file.lower(), '*.jpg') or fnmatch.fnmatch(file.lower(), '*.jpeg') or fnmatch.fnmatch(
                    file.lower(), '*.png') or fnmatch.fnmatch(file.lower(), '*.gif') or fnmatch.fnmatch(file.lower(),
                                                                                                        '*.bmp'):
                count += 1
    return count


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    #指定数据集存放路径
    curpath = 'C:/YyFiles/papercode/LMROD-Lite/NWPUVHR10VOC/VOC2007/'
    #导入模型，指定对应数据集中的类别,dota数据集含有15个类别，dior数据集含有20个类别，NWPU数据集含有10个数据类别。
    model = LHKDNET.get_model_name()
    num_classes = 10
    #加载权重参数。
    curweight = "LHKDNET_NWPU.pth"
    #num_lr 用于指定加载消融实验对应的模块。0为baseline；1为加载增强前景区域的特征蒸馏模块；2为加载自适应可变温度的蒸馏模块,3为加载两个模块
    num_lr = 1
    #结果保存位置
    directory = 'C:/YyFiles/papercode/LMROD-Lite/outputs/nwpu/LHKDNET'

    if num_lr == 0:

        rootPath = curpath + "Test01_baseline/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_baseline;  检测数据集为" + " NWPU数据集")
        print("正在执行权重参数的加载,权重路径为   " + directory[:40] + "weightparams/" + curweight)
        weightloader(3, 5)
        print()
        print("权重参数加载完成，开始执行检测")
        time.sleep(0.05)
        #对测试集执行检测
        for imgName in tqdm(imgList, position=0):
            (name, ex) = os.path.splitext(imgName)
            img = os.path.join(rootPath, imgName)
            xml = os.path.join(curpath + 'Annotations', name + '.xml')
            Predict_NWPU(img, xml, directory)
            weightloader(0.0005,0.001)
        count_res_3()
    elif num_lr == 1:

        rootPath = curpath + "Test01_EFR_FKD/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_EFR-FKD;  检测数据集为" + " NWPU数据集")
        print("正在执行权重参数的加载,权重路径为   " + directory[:40] + "weightparams/" + curweight)
        weightloader(3, 5)
        print()
        print("权重参数加载完成，开始执行检测")
        time.sleep(0.05)
        #对测试集执行检测
        for imgName in tqdm(imgList, position=0):
            (name, ex) = os.path.splitext(imgName)
            img = os.path.join(rootPath, imgName)
            xml = os.path.join(curpath + 'Annotations', name + '.xml')
            Predict_NWPU(img, xml, directory)
            weightloader(0.0005,0.001)
        os.system('echo.')
        num_lr = count_images(directory)
        count_res_4()
    elif num_lr == 2:

        rootPath = curpath + "Test01_AVT_KD/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_AVT_KD;  检测数据集为" + " NWPU数据集")
        print("正在执行权重参数的加载,权重路径为   " + directory[:40] + "weightparams/" + curweight)
        weightloader(3, 5)
        print()
        print("权重参数加载完成，开始执行检测")
        time.sleep(0.05)
        #对测试集执行检测
        for imgName in tqdm(imgList, position=0):
            (name, ex) = os.path.splitext(imgName)
            img = os.path.join(rootPath, imgName)
            xml = os.path.join(curpath + 'Annotations', name + '.xml')
            Predict_NWPU(img, xml, directory)
            weightloader(0.0005,0.001)
        os.system('echo.')
        num_lr = count_images(directory)
        count_res_5()
    else:

        rootPath = curpath + "Test01_all/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_ALL;  检测数据集为" + " NWPU数据集")
        print("正在执行权重参数的加载,权重路径为   " + directory[:4] + "weightparams/" + curweight)
        weightloader(3, 5)
        print()
        print("权重参数加载完成，开始执行检测")
        time.sleep(0.05)
        #对测试集执行检测
        for imgName in tqdm(imgList, position=0):
            (name, ex) = os.path.splitext(imgName)
            img = os.path.join(rootPath, imgName)
            xml = os.path.join(curpath + 'Annotations', name + '.xml')
            Predict_NWPU(img, xml, directory)
            weightloader(0.0005,0.001)
        os.system('echo.')
        num_lr = count_images(directory)
        count_res_6()
    os.system('echo.')
    num_lr = count_images(directory)
    print("测试完成，可视化结果保存于" + directory)

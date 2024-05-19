import fnmatch
import os
import time
import torch
from tqdm import tqdm
from utils.runtask import *
from utils.runtask import print_dots as weightloader
from utils.utils_draw import drawBoxOnVOC_NWPU as Predict_NWPU
from models.LMROD import *
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
    model = get_model_name()
    num_classes = 10
    #加载权重参数。
    curweight = "LMROD_NWPU.pth"
    #num_lr 用于指定加载消融实验对应的模块。0为baseline；1为加载上下文信息增强模块；2为加载级联三重注意力模块,3为加载两个模块
    num_lr = 0
    #结果保存位置
    directory = 'C:/YyFiles/papercode/LMROD-Lite/outputs/nwpu/LMROD'
    if num_lr == 0:

        rootPath = curpath + "Test02_baseline/"
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
        count_res_13()
    elif num_lr == 1:

        rootPath = curpath + "Test02_CIEM/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_CIEM;  检测数据集为" + " NWPU数据集")
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
        count_res_14()
    elif num_lr == 2:

        rootPath = curpath + "Test02_CTAM/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_CTAM;  检测数据集为" + " NWPU数据集")
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
        count_res_15()
    else:

        rootPath = curpath + "Test02_all/"
        imgList = os.listdir(rootPath)

        print("目前测试的模型为 " + model + "_ALL;  检测数据集为" + " NWPU数据集")
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
        count_res_16()
    os.system('echo.')
    num_lr = count_images(directory)
    print("测试完成，可视化结果保存于" + directory)
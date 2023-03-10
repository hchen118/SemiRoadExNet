from os import listdir
from os.path import join
import random

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class DatasetFromFolder_l(data.Dataset):
    def __init__(self, image_dir,work_mode=0,stuff=True,double_channel = False,zero_channel = False):
        super(DatasetFromFolder_l, self).__init__()
        ##########图像路径及文件名##########
        self.a_path = join(image_dir, "a")  # 有监督遥感图像路径
        self.b_path = join(image_dir, "b")  # 有监督道路图像路径

        self.work_mode = work_mode

        self.label_image_filenames = [[x,float(1)] for x in listdir(self.a_path)] # 有监督图像列表(元素类似于["xmp_b0005.png",1])
        self.label_image_num = len(self.label_image_filenames)  # 有监督图像数量
        self.img_filenames = self.label_image_filenames  # 图像列表
        self.double_channel  = double_channel  # 是否是双通道道路图像
        self.zero_channel = zero_channel  # 是否输出无通道
        self.stuff = stuff  # 是否随机打乱
        if self.stuff:
            random.shuffle(self.img_filenames)  # 随机打乱
        ##########.##########

        ##########数据转换##########
        if self.work_mode == 0:  # 如果不是训练阶段，即验证阶段
            self.transform_sat = transforms.Compose([# 有监督遥感图像数据转换
                transforms.CenterCrop((512,512)),
                transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化，[0,1]->[-1,1]
            self.transform_road = transforms.Compose([  # 有监督道路图像数据转换
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor()])# 转换为张量，[0,255]->[0,1]
        else:
            rot_degree = random.choice([0, 90, 180, 270])  # 随机旋转角度
            sat_enhance_list = [
                transforms.RandomCrop((512,512)),
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree)),  # 随机旋转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# 随机水平、垂直平移-0.1~0.1，随机缩放到0.9~1.1
                transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，[0,1]->[-1,1]
            ]  # 遥感图像有而道路图像没有的数据增强
            road_enhance_list = [
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree)),  # 随机旋转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移-0.1~0.1距离，随机缩放到0.9~1.1
                transforms.ToTensor()  # 转换为张量，[0,255]->[0,1]
            ]  # 遥感图像和道路图像的数据增强
            self.transform_sat = transforms.Compose(sat_enhance_list)
            self.transform_road = transforms.Compose(road_enhance_list)
        ##########.##########

    def __getitem__(self, index):
        ##########读取图像##########
        # 读取第index个图像数据，shape为(W,H)，模式为RGB或L
        sat_data = Image.open(join(self.a_path, self.img_filenames[index][0])).convert('RGB')
        road_data = Image.open(join(self.b_path, self.img_filenames[index][0])).convert('L')
        if self.work_mode:  # 如果是训练阶段
            sat_data=transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.1)(sat_data)  # 随机改变亮度、对比度、饱和度、色相

        seed = np.random.randint(2147483647)  # 随机数种子

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sat_data = self.transform_sat(sat_data)  # 遥感图像数据增强

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        road_data = self.transform_road(road_data)  # 道路图像数据增强
        if self.double_channel:
            return sat_data, torch.cat((1-road_data,road_data),-3)
        if self.zero_channel:
            return sat_data, road_data.squeeze(0)
        return sat_data, road_data
        ##########.##########

    def __len__(self):
        return len(self.img_filenames)

class DatasetFromFolder_u(data.Dataset):
    def __init__(self, image_dir,work_mode=0,stuff=True,double_channel = False,zero_channel = False):
        super(DatasetFromFolder_u, self).__init__()
        ##########图像路径及文件名##########
        self.c_path = join(image_dir, "c")  # 无监督遥感图像路径
        self.d_path = join(image_dir, "d")  # 无监督遥感图像路径

        self.work_mode = work_mode

        self.unlabel_image_filenames = [[x,float(1)] for x in listdir(self.c_path)] # 无监督图像列表(元素类似于["xmp_b0005.png",1])
        self.unlabel_image_num = len(self.unlabel_image_filenames)  # 无监督图像数量
        self.img_filenames = self.unlabel_image_filenames  # 图像列表
        self.double_channel  = double_channel  # 是否是双通道道路图像
        self.zero_channel = zero_channel  # 是否输出无通道
        self.stuff = stuff  # 是否随机打乱
        if self.stuff:
            random.shuffle(self.img_filenames)  # 随机打乱
        ##########.##########

        ##########数据转换##########
        if self.work_mode == 0:  # 如果不是训练阶段，即验证阶段
            self.transform_sat = transforms.Compose([# 有监督遥感图像数据转换
                transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 归一化，[0,1]->[-1,1]
        else:
            rot_degree = random.choice([0, 90, 180, 270])  # 随机旋转角度
            sat_enhance_list = [
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree)),  # 随机旋转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# 随机水平、垂直平移-0.1~0.1，随机缩放到0.9~1.1
                transforms.ToTensor(),  # 转换为张量，[0,255]->[0,1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，[0,1]->[-1,1]
            ]  # 遥感图像有而道路图像没有的数据增强
            road_enhance_list = [
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(0.5),  # 0.5的概率随机左右翻转
                transforms.RandomVerticalFlip(0.5),  # 0.5的概率随机上下翻转
                transforms.RandomRotation((rot_degree, rot_degree)),  # 随机旋转
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),# 随机平移-0.1~0.1距离，随机缩放到0.9~1.1
                transforms.ToTensor()  # 转换为张量，[0,255]->[0,1]
            ]  # 遥感图像和道路图像的数据增强
            self.transform_sat = transforms.Compose(sat_enhance_list)
            self.transform_road = transforms.Compose(road_enhance_list)
        ##########.##########

    def __getitem__(self, index):
        ##########读取图像##########
        # 读取第index个图像数据，shape为(W,H)，模式为RGB或L
        sat_data = Image.open(join(self.c_path, self.img_filenames[index][0])).convert('RGB')
        road_data = Image.open(join(self.d_path, self.img_filenames[index][0])).convert('L')
        if self.work_mode:  # 如果是训练阶段
            sat_data=transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.1)(sat_data)  # 随机改变亮度、对比度、饱和度、色相

        seed = np.random.randint(2147483647)  # 随机数种子

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        sat_data = self.transform_sat(sat_data)  # 遥感图像数据增强

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        road_data = self.transform_road(road_data)  # 道路图像数据增强

        if self.double_channel:
            return sat_data, torch.cat((1-road_data,road_data),-3)
        if self.zero_channel:
            return sat_data, road_data.squeeze(0)
        return sat_data,road_data
        ##########.##########

    def __len__(self):
        return len(self.img_filenames)
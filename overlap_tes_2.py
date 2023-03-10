import torch
import torchvision.transforms as transforms

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from other.cal_img_entropy import cal_entropy
from other_information import FILE_NAME,TEST_EPOCH_NUM,SET_NAME,LABEL_RATE,EXTRA_WORD

def tes_result():
    ####################定义参数####################
    ##########路径参数##########
    file_name = FILE_NAME  # 训练的代码文件名字
    epoch_num = TEST_EPOCH_NUM  # 训练代数
    test_set_name = SET_NAME  # 数据集名字
    label_rate = LABEL_RATE  # 有标签数据所占百分比
    extra_word = EXTRA_WORD # 附加标志信息

    test_root = 'dataset/{}_{}/test'.format(test_set_name,label_rate)  # 测试集根目录
    file_a_path = test_root+'/a'  # 有监督遥感图像路径
    stride_step = 256  # overlap的间隔
    net_input_size = 512  # 网络训练时的输入形状，512*512
    road_threshold=0.5  # 道路阈值0.5
    ##########.##########

    ##########图像列表##########
    #file_a_list = [[x,1] for x in os.listdir(file_a_path)]  # 有监督遥感图像名字列表
    file_a_list = [['113050.jpg',1],['133797.jpg',1],['120056.jpg',1],['121763.jpg',1]]
    img_num = len(file_a_list)  # 测试图像总数
    ##########.##########

    ##########保存路径##########
    save_root = 'result/{}_{}'.format(test_set_name,label_rate)  # 测试结果根目录
    #save_b_path = save_root+'/b'  # 有监督遥感图像测试结果路径
    save_b_path = 'result/entropy'  # 有监督遥感图像测试结果路径
    if not os.path.exists(save_root):  # 测试结果根目录
        os.mkdir(save_root)
    if not os.path.exists(save_b_path):  # 有监督遥感图像测试结果路径
        os.mkdir(save_b_path)
    ##########.##########

    ##########转换方式##########
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)  # 数据转换方式
    ##########.##########

    ##########读取模型##########
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'checkpoint/{}/{}/net_g_{}_{}.pth'.format(file_name,test_set_name,epoch_num,extra_word)  # 模型路径
    net_g = torch.load(model_path)  # 道路提取网络
    net_g.to(device)
    net_g.eval()
    ##########.##########
    ####################.####################

    ####################迭代测试####################
    bar=tqdm(range(img_num))
    bar.write('\ntest:')
    for i in bar:  # 遍历测试图像
        img_name, img_type = file_a_list[i]  # 图像名字，图像类型
        ##########读取路径和保存路径##########
        read_path = file_a_path  # 文件读取路径
        save_path = save_b_path  # 文件保存路径
        ##########.##########

        ##########输入图像及滑动位置##########
        img = Image.open(read_path+'/'+img_name).convert('RGB')  # 读取遥感图像
        img = transform(img)  # 转换为tensor,3*h*w

        _,h_size,_ = img.shape  # 输入图像的通道数，高，宽
        stride_num = (h_size - net_input_size) // stride_step + 1  # 完整滑动窗口的次数
        stride_x_place_list = [stride_step * i for i in range(stride_num)]  # x方向滑动位置列表
        rest_size = (h_size - net_input_size) % stride_step  # 没有输出部分的长度
        if rest_size!=0:
            stride_x_place_list.append(h_size - net_input_size)  # 没有输出部分对应的x方向滑动位置
        stride_xy_place_list = [(j,k) for j in stride_x_place_list for k in stride_x_place_list]  # 滑动的xy坐标
        cover_num_tensor = torch.zeros((h_size,h_size)).to(device)  # 原图像各像素滑动覆盖次数
        out_img_all = torch.zeros((h_size,h_size)).to(device)  # 完整的整幅输出图像
        ##########.##########

        ##########滑动输出##########
        with torch.no_grad():
            for x_place, y_place in stride_xy_place_list:  # 当前滑动的x、y坐标
                in_img_part = img[:,x_place:x_place + net_input_size, y_place:y_place + net_input_size].unsqueeze(0).to(device)  # 输入窗口内图像，1*c*h*w
                out_img_part = net_g(in_img_part).squeeze(0)  # 网络预测输出,c*h*w
                # out_img_part=nn.MaxPool2d(kernel_size=8)(out_img_part).unsqueeze(0)
                # out_img_part=nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)(out_img_part).squeeze(0)
                out_img_part = out_img_part.squeeze(0)  # 1*h*w->h*w
                out_img_all[x_place:x_place + net_input_size, y_place:y_place + net_input_size]+=out_img_part  # 输出部分加到幅输出图像
                cover_num_tensor[x_place:x_place + net_input_size, y_place:y_place + net_input_size]+=1  # 更新对应像素的滑动覆盖次数
            out_img_all = out_img_all/cover_num_tensor  # 各像素的平均输出,h*w
            # test1=torch.min(out_img_all)
            # test2=torch.max(out_img_all)

            # 保存熵图
            entropy_img=torch.clone(out_img_all.detach())
            entropy_img=cal_entropy(entropy_img)
            entropy_np = entropy_img.cpu().float().numpy()  # 转换为numpy
            entropy_img_np = np.array([entropy_np, entropy_np, entropy_np])  # 1*H*W->3*H*W
            entropy_img_np = (np.transpose(entropy_img_np, (1, 2, 0))) * 255  # 3*H*W->H*W*3,[0,1]->[0,255]
            entropy_img_np = entropy_img_np.astype(np.uint8)  # 转换为整数
            entropy_hot_img_cv = cv2.applyColorMap(entropy_img_np, cv2.COLORMAP_HOT)  # 熵图由灰度图转热度图
            #entropy_hot_img_cv = cv2.applyColorMap(entropy_img_np, cv2.COLORMAP_JET)  # 熵图由灰度图转热度图
            #entropy_hot_img_cv=cv2.cvtColor(entropy_hot_img_cv, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path + '/' + img_name.split('.')[0]+'_2.jpg',entropy_hot_img_cv)  # 保存熵图的热度图
            entropy_img_pil = Image.fromarray(entropy_img_np, mode='RGB')  # numpy转换为Image
            entropy_img_pil.save(save_path + '/' + img_name.split('.')[0]+'_1.jpg')  # 保存网络输出结果

            # 保存区域图
            area_img=torch.clone(out_img_all.detach())
            area_np=area_img.cpu().float().numpy()  # 转换为numpy,1024*1024
            threshold=0.03
            road_area=area_np>1-threshold
            uncertain_area=(1-threshold>=area_np)*(area_np>threshold)
            non_road_area = area_np <= threshold
            #area_img_np=np.array([road_area*255, uncertain_area*255, np.zeros((1024,1024))])  # 1*H*W->3*H*W (3*1024*1024)
            area_img_np=np.array([road_area*255+uncertain_area*50+non_road_area*0,
                                  road_area*50+uncertain_area*255+non_road_area*0,
                                  road_area*0+uncertain_area*255+non_road_area*0])  # 1*H*W->3*H*W (3*1024*1024)
            # area_img_np=np.array([road_area*255, np.zeros((1024,1024)), uncertain_area*255])  # 1*H*W->3*H*W (3*1024*1024)
            # area_img_np[:,area_np>0.97]=np.array([255,0,0])
            # area_img_np[:,0.97>=area_np>0.03]=np.array([0,0,255])
            # area_img_np[:,0.03>=area_np]=np.array([0,0,0])

            out_img_all[out_img_all>road_threshold]=255  # 道路部分
            out_img_all[out_img_all<=road_threshold]=0  # 背景部分
            out_img_all=out_img_all.expand((3,-1,-1)).permute(1,2,0)  # h*w->3*h*w->h*w*3
        ##########.##########

        ##########保存图像##########
        # out_np = out_img_all.float().numpy()  # 转换为numpy,h*w
        # out_np[out_np>road_threshold] = 1  # 道路部分
        # out_np[out_np<=road_threshold] = 0  # 背景部分

        # out_img_np = np.array([out_np, out_np, out_np])  # H*W->3*H*W
        # out_img_np = (np.transpose(out_img_np, (1, 2, 0)))*255  # 3HW->HW3,[0,1]->[0,255]
        # out_img_np = out_img_np.astype(np.uint8)  # 转换为整数

        # 保存区域图
        area_img_np=area_img_np.transpose((1,2,0)).astype(np.uint8)
        area_img_pil = Image.fromarray(area_img_np,mode='RGB')  # numpy转换为Image
        area_img_pil.save(save_path + '/' + img_name.split('.')[0]+'_area.jpg')  # 保存网络输出结果

        out_img_np=out_img_all.cpu().numpy().astype(np.uint8)
        out_img_pil = Image.fromarray(out_img_np,mode='RGB')  # numpy转换为Image
        out_img_pil.save(save_path + '/' + img_name.split('.')[0]+'.jpg')  # 保存网络输出结果

        bar.set_description("[{:4d}/{:4d}]Image saved as {}".format(i+1,img_num,img_name))
        ##########.##########
    ####################.####################


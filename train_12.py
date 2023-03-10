import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import random
import shutil
import warnings
import numpy as np
from time import time
from tqdm import tqdm

from data.data_set_3 import DatasetFromFolder
from data.data_set_6 import DatasetFromFolder_l
from network.dinknet5 import DinkNet34
# from network.DSnet import DSNet32,DSNet64
# from network.my_d import my_netd
from network.unet import Unet
from other.cal_img_entropy import cal_entropy
from other_information import DEVICE,SET_NAME,LABEL_RATE,EXTRA_WORD,sever_root,BATCH_SIZE,PRE_EPOCH,START_EPOCH,END_EPOCH,NUM_WORKER


device = DEVICE
warnings.filterwarnings("ignore")
####################定义函数####################
def find_not_zero_index(data1):
    out_list = []  # 初始化结果列表
    place_num = len(data1)  # 向量长度
    for i in range(place_num):
        if data1[i]:
            out_list.append(i)
    return out_list

def find_not_one_index(data1):
    out_list = []  # 初始化结果列表
    place_num = len(data1)  # 向量长度
    for i in range(place_num):
        if data1[i] != 1:
            out_list.append(i)
    return out_list

def dice_loss(y_pre, y_lab):
    smooth = 0.01
    pre_num = torch.sum(y_pre, (1, 2, 3))  # 预测结果相加和
    lab_num = torch.sum(y_lab, (1, 2, 3))  # 实际结果相加和
    pre_inter_lab = torch.sum(y_pre * y_lab, (1, 2, 3))  # 预测结果和实际结果相交部分相加和
    dice_loss_val = 1.0-((2 * pre_inter_lab+smooth) / (pre_num+lab_num+smooth))
    return dice_loss_val.mean()

def cal_iou_loss(y_pre_, y_lab_):
    smooth = 0.1
    y_pre = y_pre_.clone().detach()
    y_lab = y_lab_.clone().detach()
    y_pre[y_pre <= 0.5] = 0.0
    y_pre[y_pre > 0.5] = 1.0
    y_lab[y_lab <= 0.5] = 0.0
    y_lab[y_lab > 0.5] = 1.0

    # pre_num = torch.sum(y_pre, (1, 2, 3))
    # lab_num = torch.sum(y_lab, (1, 2, 3))
    # pre_inter_lab = torch.sum(y_pre * y_lab, (1, 2, 3))
    pre_num = torch.sum(y_pre).item()
    lab_num = torch.sum(y_lab).item()
    pre_inter_lab = torch.sum(y_pre * y_lab).item()

    #iou_loss = np.array((1-((pre_inter_lab+smooth) / (pre_num+lab_num-pre_inter_lab+smooth))).cpu())
    iou_loss = (1-((pre_inter_lab+smooth) / (pre_num+lab_num-pre_inter_lab+smooth)))
    #return np.nanmean(iou_loss)
    return iou_loss

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.],
                         [torch.sin(theta), torch.cos(theta), 0.]])

def rot_img(x, theta):
    rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).to(device)
    x = F.grid_sample(x, grid)
    return x

def type2mask(data_type_, h_size_=512):
    batch_size_ = len(data_type_)
    return data_type_.expand((h_size_, h_size_, 1, batch_size_)).permute((3, 2, 1, 0))

def threshold_label_loss(y_pre, label_threshold=0.95):
    y_lab = y_pre.detach().clone()
    y_compare = (y_pre < 1-label_threshold)+(y_pre >= label_threshold)
    if torch.sum(y_compare) == 0:
        return torch.tensor(0.0, requires_grad=True)
    y_lab[y_lab < 1-label_threshold] = 0.0
    y_lab[y_lab >= label_threshold] = 1.0

    bce_label_loss = nn.BCELoss()(y_pre[y_compare], y_lab[y_compare])

    smooth = 0.01
    y_tp = torch.sum(y_pre[y_compare] * y_lab[y_compare])
    dice_label_loss = 1.0-(2 * y_tp+smooth) / (torch.sum(y_pre[y_compare])+torch.sum(y_lab[y_compare])+smooth)
    #del y_lab,y_compare
    return bce_label_loss+dice_label_loss

####################.####################

def main():
    ####################定义路径和损失####################
    ##########定义训练信息##########
    start_epoch = START_EPOCH  # 开始时的已训练epoch数
    end_epoch = END_EPOCH  # 结束时的已训练epoch数
    pre_epoch = PRE_EPOCH  # 训练开始时的预训练epoch数

    gap_epoch = 10  # 保存的epoch间隔
    batch_size = BATCH_SIZE  # batch数
    num_workers = NUM_WORKER  # 子进程数

    # lambda4 = 1  # loss5系数

    train_set_name = SET_NAME  # 数据集名字
    label_rate = LABEL_RATE  # 有标签数据所占百分比
    file_name = 'train_12'  # 本代码文件名字
    extra_word = EXTRA_WORD  # 附加标志信息

    train_root = sever_root+'dataset/{}_{}/train'.format(train_set_name, label_rate)  # 训练集根目录
    val_root = sever_root+'dataset/{}_{}/val'.format(train_set_name, label_rate)  # 验证集根目录
    #pre_root = sever_root+'/checkpoint/pre_{}'.format(label_rate)  # 预训练保存位置
    base_degree = np.pi / 2  # 基本旋转角度，即90度
    degree_list = [base_degree, base_degree * 2, base_degree * 3]  # loss4选择的旋转角度
    ##########.##########

    ##########生成器和判别器##########
    if start_epoch == 0:  # 如果是从头开始训练
        write_mode = 'w'  # 日志模式为重新写
        #net_g = DinkNet34(pool_old=1,d_size=64)  # 定义生成网络
        if pre_epoch==0:
            net_g = DinkNet34(pool_old=1)  # 定义生成网络
            #net_g = DSNet32()  # 定义生成网络
            #net_g = DSNet64()  # 定义生成网络
        else:
            net_g = torch.load('pre/{}/{}.pth'.format(train_set_name, label_rate), map_location=device)  # 加载生成网络
            print('load pre_train net_{}'.format(pre_epoch))
        # net_d1 = my_netd(4)  # 定义判别网络1
        # net_d2 = my_netd(4)  # 定义判别网络2
        net_d1 = Unet(in_channel=4, out_channel=1)  # 定义判别网络1
        net_d2 = Unet(in_channel=4, out_channel=1)  # 定义判别网络2
    else:  # 如果是接着训练
        write_mode = 'a'  # 日志模式为添加
        net_g = torch.load(sever_root+'checkpoint/{}/{}_{}/net_g_{}.pth'.format(file_name, train_set_name, extra_word,
                                                                                 start_epoch))  # 加载生成网络
        # net_d1 = my_netd(4)  # 定义判别网络1
        # net_d2 = my_netd(4)  # 定义判别网络2
        net_d1 = Unet(in_channel=4, out_channel=1)  # 定义判别网络1
        net_d2 = Unet(in_channel=4, out_channel=1)  # 定义判别网络2
        net_d1.load_state_dict(torch.load(
            sever_root+'checkpoint/{}/{}_{}/net_d1_{}.pth'.format(file_name, train_set_name, extra_word,
                                                                   start_epoch)))  # 加载判别网络1
        net_d2.load_state_dict(torch.load(
            sever_root+'checkpoint/{}/{}_{}/net_d2_{}.pth'.format(file_name, train_set_name, extra_word,
                                                                   start_epoch)))  # 加载判别网络2
    net_g.to(device)
    net_d1.to(device)
    net_d2.to(device)
    net_g.train()
    net_d1.train()
    net_d2.train()
    ##########.##########

    ##########数据集、日志和模型保存位置##########
    data_set = DatasetFromFolder(train_root, work_mode=1)  # 数据集
    val_data_set = DatasetFromFolder_l(val_root, work_mode=0)  # 验证集数据集
    label_num = data_set.label_image_num  # 有监督图像数量
    unlabel_num = data_set.unlabel_image_num  # 无监督图像数量
    label_proportion = label_num / (label_num+unlabel_num)  # 有标签占比
    unlabel_proportion = unlabel_num / (label_num+unlabel_num)  # 无标签占比
    data_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data_loader = data.DataLoader(val_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    batch_num_per_epoch = len(data_loader)  # 一个epoch的batch数

    if not os.path.exists(sever_root+'log/{}'.format(file_name)):
        os.mkdir(sever_root+'log/{}'.format(file_name))
    if not os.path.exists(sever_root+'log/{}/{}_{}'.format(file_name, train_set_name, extra_word)):
        os.mkdir(sever_root+'log/{}/{}_{}'.format(file_name, train_set_name, extra_word))
    epoch_log = open(sever_root+'log/{}/{}_{}/{}_epoch.txt'.format(file_name, train_set_name, extra_word, label_rate),
                     write_mode)  # epoch日志文件句柄
    val_log = open(sever_root+'log/{}/{}_{}/{}_val.txt'.format(file_name, train_set_name, extra_word, label_rate),
                   write_mode)  # 验证信息日志文件句柄

    other_log = open(sever_root+'log/{}/{}_{}_{}_other.txt'.format(file_name, train_set_name, extra_word, label_rate),
                     'w')  # 参数信息日志文件句柄
    other_words = 'dataset:[{}_{}] total_epoch[{}] batch_size[{:d}] num_workers[{:d}]pre_train[{}] extra_word[{}]'.format(
        train_set_name, label_rate, end_epoch, batch_size, num_workers,
        pre_epoch, extra_word
    )  # 参数信息
    print(other_words)
    other_log.write(other_words+'\n')
    other_log.flush()

    if not os.path.exists(sever_root+'checkpoint/{}'.format(file_name)):  # 本代码神经网络保存位置
        os.mkdir(sever_root+'checkpoint/{}'.format(file_name))
    if not os.path.exists(
            sever_root+'checkpoint/{}/{}'.format(file_name, train_set_name)):  # 本代码对应数据集的训练神经网络保存位置
        os.mkdir(sever_root+'checkpoint/{}/{}'.format(file_name, train_set_name))
    ##########.##########

    ##########损失函数和优化器##########
    loss_val_best = 10000  # 初始化生成器的最佳验证损失
    loss_val_up_num = 0  # 初始化生成器验证损失连续超过最佳验证损失的代数

    loss_bce = nn.BCELoss().to(device)  # BCE损失
    learn_rate_g = 0.0004  # 生成器初始学习率
    learn_rate_d1 = 0.0004  # 判别器1初始学习率
    learn_rate_d2 = 0.0004  # 判别器2初始学习率
    weight_decay=1e-5
    optimizer_g = optim.Adam(net_g.parameters(), lr=learn_rate_g,weight_decay=weight_decay,betas=(0.9,0.99))  # 生成网络优化器
    optimizer_d1 = optim.Adam(net_d1.parameters(), lr=learn_rate_d1,weight_decay=weight_decay,betas=(0.9,0.99))  # 判别网络1优化器
    optimizer_d2 = optim.Adam(net_d2.parameters(), lr=learn_rate_d2,weight_decay=weight_decay,betas=(0.9,0.99))  # 判别网络2优化器

    loss1_g_val, loss1_d_val, loss2_g_val, loss2_d_val, loss3_bce_val, loss3_dice_val, loss3_dice_unweight_val, loss4_val, loss5_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 初始化loss1、2、3、4、5输出值
    ##########.##########
    ####################.####################

    ####################迭代训练####################
    print('*' * 100)
    # time_start = time()  # 开始训练时间
    bar=tqdm(range(start_epoch+1, end_epoch+1))
    for epoch in bar:  # 迭代训练
        # time_epoch_start = time()  # 当前epoch开始训练时间
        loss1_g_epoch = 0.0  # 初始化生成器当前epoch的loss1损失
        loss1_d_epoch = 0.0  # 初始化判别器1当前epoch的loss1损失
        loss2_g_epoch = 0.0  # 初始化生成器当前epoch的loss2损失
        loss2_d_epoch = 0.0  # 初始化判别器2当前epoch的loss2损失
        loss3_bce_epoch = 0.0  # 初始化loss3当前epoch的bce损失
        loss3_dice_unweight_epoch_list = []  # 初始化当前epoch的平均不加权dice损失
        loss3_dice_epoch = 0.0  # 初始化loss3当前epoch的dice损失
        loss4_epoch = 0.0  # 初始化生成器当前epoch的loss4损失
        loss5_epoch = 0.0  # 初始化生成器当前epoch的loss5损失

        d1_weight = 1  # 初始化鉴别器权重
        d2_weight = 1  # 初始化鉴别器权重
        act_words=''
        for batch, batch_data in enumerate(data_loader, 1):  # batch从1开始增加
            ##########图像数据##########
            sat_img = batch_data[0].to(device)  # 遥感图像
            road_img = batch_data[1].to(device)  # 道路图像
            data_type = batch_data[2].float().to(device)  # 数据类型列表，1表示有监督，0表示无监督
            h_size = sat_img.shape[-1]  # 图像尺寸
            # data_weight = data_type / label_proportion+(1-data_type) / unlabel_proportion  # 有监督数据和无监督数据的损失权重
            data_mask = type2mask(data_type, h_size_=h_size).to(device)
            # data_mask = data_type
            data_weight = data_mask / label_proportion+(1-data_mask) / unlabel_proportion  # 有监督数据和无监督数据的损失权重

            loss_BCE_weight = nn.BCELoss(weight=data_weight).to(device)  # BCE损失进行加权
            ##########.##########

            ##########loss1##########
            out_img = net_g(sat_img).to(device)  # 道路预测图像
            #####D1#####
            optimizer_d1.zero_grad()

            # 判别网络1的预测结果形状是1*1024*1024，每个像素值表示是有监督图像的概率
            rate1_img = net_d1(torch.cat((sat_img, out_img.detach()), 1).detach().to(device))  # 预测是否有监督结果
            loss1_d = loss_BCE_weight(rate1_img, data_mask)  # 判别网络预测结果与真实数据类型之间的损失

            (d1_weight*loss1_d).backward()
            optimizer_d1.step()

            loss1_d_val = loss1_d.item()  # 当前batch的loss1_d损失
            loss1_d_epoch += loss1_d_val  # 当前batch的loss1_d损失加入判别器1的epoch损失
            #####.#####

            #####G1#####
            optimizer_g.zero_grad()

            rate1_img = net_d1(torch.cat((sat_img, out_img), 1).to(device))  # 预测是否有监督结果
            loss1_g = loss_BCE_weight(rate1_img, (1-data_mask))  # 判别网络预测结果与虚假数据类型之间的损失

            loss1_g.backward()
            optimizer_g.step()

            loss1_g_val = loss1_g.item()
            loss1_g_epoch += loss1_g_val
            #####.#####
            if loss1_d_val>0.1*loss1_g_val:  # 如果判别器损失太小则更改鉴别器权重
                d1_weight=0.1
            else:
                d1_weight=1
            # del rate1_img
            ##########.##########

            ##########loss2##########
            out_img = net_g(sat_img).to(device)  # 道路预测图像
            entropy_img = cal_entropy(out_img).to(device)  # 生成图像的熵图
            #####D2#####
            optimizer_d2.zero_grad()

            rate2_img = net_d2(torch.cat((sat_img, entropy_img), 1).detach().to(device))  # 预测是否有监督结果
            loss2_d = loss_BCE_weight(rate2_img, data_mask)  # 判别网络预测结果与真实数据类型之间的损失

            (d2_weight*loss2_d).backward()
            optimizer_d2.step()

            loss2_d_val = loss2_d.item()
            loss2_d_epoch += loss2_d_val
            #####.#####

            #####G2#####
            optimizer_g.zero_grad()

            rate2_img = net_d2(torch.cat((sat_img, entropy_img), 1).to(device))  # 预测是否有监督结果
            loss2_g = loss_BCE_weight(rate2_img, (1-data_mask))  # 判别网络预测结果与虚假数据类型之间的损失

            loss2_g.backward()
            optimizer_g.step()

            loss2_g_val = loss2_g.item()
            loss2_g_epoch += loss2_g_val
            #####.#####
            if loss2_d_val>0.1*loss2_g_val:  # 如果判别器损失太小则更改鉴别器权重
                d2_weight=0.1
            else:
                d2_weight=1
            ##########.##########

            ##########loss3##########
            optimizer_g.zero_grad()

            if torch.sum(data_type) != 0:  # 如果非空，即当前batch数据不全为无监督数据
                full_sup_data_index = find_not_zero_index(data_type)  # 全监督对应索引
                if torch.sum(road_img[full_sup_data_index].to(device)) != 0:
                    optimizer_g.zero_grad()
                    out_img = net_g(sat_img).to(device)  # 道路预测图像
                    loss3_bce = loss_bce(out_img[full_sup_data_index],
                                                   road_img[full_sup_data_index]) / label_proportion
                    loss3_dice_unweight = dice_loss(out_img[full_sup_data_index],
                                                    road_img[full_sup_data_index])
                    loss3_dice = loss3_dice_unweight / label_proportion
                    loss3 = loss3_bce+loss3_dice
                    loss3.backward()
                    optimizer_g.step()

                    loss3_bce_val = loss3_bce.item()
                    loss3_bce_epoch += loss3_bce_val
                    loss3_dice_unweight_val = loss3_dice_unweight.item()
                    loss3_dice_val = loss3_dice.item()
                    loss3_dice_unweight_epoch_list.append(loss3_dice_unweight_val)
                    loss3_dice_epoch += loss3_dice_val
            ##########.##########

            ##########loss4##########
            optimizer_g.zero_grad()

            rot_degree = random.choice(degree_list)  # 随机选取旋转角度
            # rot_trans=transforms.RandomRotation((rot_degree,rot_degree))

            out_img = net_g(sat_img).to(device)  # 道路预测图像
            out_img_r = net_g(rot_img(sat_img, rot_degree).detach()).to(device)  # 旋转后遥感图像的道路预测图像

            loss4 = loss_bce(out_img_r, rot_img(out_img.detach(), rot_degree).detach().to(device))
            loss4_val = loss4.item()
            loss4_epoch += loss4_val

            loss4.backward()
            optimizer_g.step()
            ##########.##########

            ##########loss5##########
            optimizer_g.zero_grad()

            no_sup_data_index = find_not_one_index(data_type)  # 无监督对应索引
            if len(no_sup_data_index) != 0:  # 当前batch数据不全为有监督数据
                optimizer_g.zero_grad()
                out_img = net_g(sat_img).to(device)  # 道路预测图像

                loss5 = threshold_label_loss(out_img[no_sup_data_index]) / unlabel_proportion  # 伪标签损失
                loss5_val = loss5.item()
                loss5_epoch += loss5_val

                if loss5_val != 0:  # 如果有伪标签
                    loss5.backward()
                    optimizer_g.step()
            ##########.##########

            ##########batch输出信息##########
            batch_words = 'batch[{:4d}/{:4d}] loss[1:(g:{:.2e},d:{:.2e}),2:(g:{:.2e},d:{:.2e}),3:(bce:{:.2e},dice:{:.2e}),4:{:.2e},5:{:.2e}]'.format(
                batch, batch_num_per_epoch, loss1_g_val, loss1_d_val, loss2_g_val, loss2_d_val,
                loss3_bce_val, loss3_dice_val, loss4_val, loss5_val)
            bar.set_description(batch_words)
            ##########.##########

        ##########epoch输出信息##########
        # time_epoch_end = time()  # 当前epoch结束训练时间
        # time_epoch_used = (time_epoch_end-time_epoch_start) / 60  # 当前epoch用时（单位min）
        # loss_g_epoch = loss1_g_epoch+loss2_g_epoch+loss3_bce_epoch+loss3_dice_epoch+loss4_epoch+loss5_epoch  # 当前epoch的生成器损失
        # loss3_dice_unweight_mean = np.mean(loss3_dice_unweight_epoch_list)

        epoch_words = "epoch[{:3d}] learn_rate[g:{:.2e}] loss[1:(g:{:.2e},d:{:.2e}),2:(g:{:.2e},d:{:.2e}),3:(bce:{:.2e},dice:{:.2e},4:{:.2e},5:{:.2e}]".format(
            epoch, learn_rate_g, loss1_g_epoch, loss1_d_epoch, loss2_g_epoch, loss2_d_epoch, loss3_bce_epoch,
            loss3_dice_epoch, loss4_epoch, loss5_epoch)
        # bar.write(epoch_words)
        epoch_log.write('\n'+epoch_words)
        epoch_log.flush()
        ##########.##########

        ##########学习率更新##########
        val_loss_epoch_list = []  # 初始化验证损失
        with torch.no_grad():
            for val_batch, val_batch_data in enumerate(val_data_loader, 1):
                sat_img = val_batch_data[0].to(device)  # 遥感图像
                road_img = val_batch_data[1].to(device)  # 道路图像
                out_img = net_g(sat_img)  # 预测图像
                val_loss_batch = cal_iou_loss(out_img, road_img)  # IOU loss=1-IOU越小越好
                #val_loss_epoch_list.append(val_loss_batch.item())
                val_loss_epoch_list.append(val_loss_batch)
        val_loss_epoch_val = np.nanmean(np.array(val_loss_epoch_list))
        val_words = ' iou[{:.2f}]'.format((1-val_loss_epoch_val)*100)
        # print(val_loss_epoch_val_list)

        val_log.write('\n'+val_words)
        val_log.flush()

        if val_loss_epoch_val > loss_val_best:  # 当前的生成器损失在上升
            loss_val_up_num += 1
        else:  # 当前的生成器损失在下降
            loss_val_up_num = 0
            loss_val_best = val_loss_epoch_val
            torch.save(net_g,
                       sever_root+'checkpoint/{}/{}/net_g_{}_{}.pth'.format(file_name, train_set_name,
                                                                             'best', extra_word))  # 保存损失最小的生成网络
            act_words=' 1->saved'
        if loss_val_up_num > 7:  # 如果连续很多次验证损失上升则提前终止
            bar.write('early stop at {:3d}th epoch'.format(epoch))
            epoch_log.write('\n'+'early stop at {:3d}th epoch'.format(epoch))
            epoch_log.flush()
            break

        if loss_val_up_num > 6:  # 如果连续多次验证损失上升则调整学习率
            if learn_rate_g < 5e-7:
                pass
            else:
                # net_g = torch.load(
                #     sever_root+'/checkpoint/{}/{}_{}/net_g_{}.pth'.format(file_name, train_set_name, extra_word,
                #                                                           'best'))
                loss_val_up_num = 0
                learn_rate_g /= 5
                act_words='2->new rate:{:.2e}'.format(learn_rate_g)
                # bar.write('2->new rate:{:.2e}'.format(learn_rate_g))
                for param_group in optimizer_g.param_groups:  # 学习率更新
                    param_group['lr'] = learn_rate_g
                for param_group in optimizer_d1.param_groups:  # 学习率更新
                    param_group['lr'] = learn_rate_g
                for param_group in optimizer_d2.param_groups:  # 学习率更新
                    param_group['lr'] = learn_rate_g
        bar.write('\n' + epoch_words + val_words+act_words)
        ##########.##########

        ##########保存模型##########
        if epoch % gap_epoch == 0:  # 如果达到保存条件
            torch.save(net_g,
                       sever_root+'checkpoint/{}/{}/net_g_{}_{}.pth'.format(file_name, train_set_name,
                                                                             epoch, extra_word))  # 保存生成网络
            torch.save(net_d1.state_dict(),
                       sever_root+'checkpoint/{}/{}/net_d1_{}_{}.pth'.format(file_name, train_set_name,
                                                                              epoch, extra_word))  # 保存判别网络1
            torch.save(net_d2.state_dict(),
                       sever_root+'checkpoint/{}/{}/net_d2_{}_{}.pth'.format(file_name, train_set_name,
                                                                              epoch, extra_word))  # 保存判别网络2
        ##########.##########
    # time_end = time()
    # bar.write('epoch:{:3d}~{:3d},time used:{:2.1f}h'.format(start_epoch, end_epoch, (time_end-time_start) / 3600))
    bar.close()
    ####################.####################

if __name__ == '__main__':
    main()



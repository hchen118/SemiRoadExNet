from PIL.Image import Image
import torchvision.transforms as transforms

import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from other_information import DEVICE,SET_NAME,LABEL_RATE,EXTRA_WORD ,sever_root

def compare_result():
    ####################定义参数####################
    ##########道路标签和测试结果路径##########
    label_rate = LABEL_RATE  # 有标签数据所占百分比
    test_set_name = SET_NAME  # 数据集名字

    cut_bool = False  # 是否将输入图像cut成固定尺寸(512)

    test_root = sever_root+'dataset/{}_{}/test'.format(test_set_name,label_rate)  # 测试集根目录
    file_b_path = test_root+'/b'  # 有监督道路图像路径

    save_root = sever_root+'result/{}_{}'.format(test_set_name,label_rate)  # 测试结果根目录
    save_b_path = save_root+'/b'  # 有监督遥感图像测试结果路径
    ##########.##########

    ##########图像列表##########
    file_b_list = [[x,1] for x in os.listdir(file_b_path)]  # 有监督道路图像名字列表
    img_num = len(file_b_list)  # 比较图像总数
    ##########.##########

    ##########比较结果路径##########
    compare_root = save_root  # 比较结果根目录
    compare_b_path = compare_root + '/compare_img'  # 有监督比较结果路径
    compare_csv_path = compare_root + '/csv'  # 比较结果的csv文件路径
    # if not os.path.exists(compare_root):  # 比较结果根目录
    #     os.mkdir(compare_root)
    if not os.path.exists(compare_b_path):  # 有监督比较结果路径
        os.mkdir(compare_b_path)
    if not os.path.exists(compare_csv_path):  # 比较结果的csv文件路径
        os.mkdir(compare_csv_path)

    csv_header = ('tp_num', 'tn_num', 'fp_num', 'fn_num', 'all_num', 'rate_precision', 'rate_recall',
                      'rete_f1', 'rate_oa', 'rate_iou_road', 'rate_iou_background', 'rete_miou','kappa')  # csv文件的第一行头标题
    #result_header = ('type','precision','recall','f1','oa','iou_road','kappa')
    result_header = ('type','f1','iou_road','kappa')
    compare_b_csv = pd.DataFrame(columns=csv_header)  # 有监督比较结果的csv数据
    compare_result_csv = pd.DataFrame(columns=result_header)  # 总览结果的csv数据
    ##########.##########
    ####################.####################

    ####################遍历比较####################
    cbar=tqdm()
    cbar.write('\ncompare:')
    for i in range(img_num):  # 遍历测试图像
        img_name, img_type = file_b_list[i]  # 图像名字，图像类型
        ##########文件路径##########
        read_real_path = file_b_path  # 道路标签路径
        read_test_path = save_b_path  # 测试结果路径
        save_path = compare_b_path  # 图像保存路径
        compare_csv = compare_b_csv  # csv数据
        ##########.##########

        ##########读取数据##########
        real_img = Image.open(read_real_path+'/'+img_name).convert('L')  # 道路标签图像
        test_img = Image.open(read_test_path+'/'+img_name.split('.')[0]+'.jpg').convert('L')  # 测试结果图像
        if cut_bool:  # 如果中心裁剪
            real_img = transforms.CenterCrop((512,512))(real_img)
            test_img = transforms.CenterCrop((512,512))(test_img)

        real_np = np.asarray(real_img)/255  # 道路标签数据
        test_np = np.asarray(test_img)/255  # 测试结果数据

        real_np[real_np<=0.5]=0
        real_np[real_np>0.5]=1
        test_np[test_np <= 0.5] = 0  # 由于保存数据是有损的jpg，所以要把它转为jpg免得数值不止0和1
        test_np[test_np > 0.5] = 1
        ##########.##########

        ##########数据处理##########
        result_1_np = test_np + real_np  # 0:TN,1:FN+FP,2:TP
        result_2_np = test_np-real_np  # 0:TP+TN,1:FP,-1:FN

        tp_np = result_1_np==2  # TP布尔数组
        tn_np = result_1_np==0  # TN布尔数组
        fp_np = result_2_np==1  # FP布尔数组
        fn_np = result_2_np==-1  # FN布尔数组

        tp_num = np.sum(tp_np)  # TP像素数
        tn_num = np.sum(tn_np)  # TN像素数
        fp_num = np.sum(fp_np)  # FP像素数
        fn_num = np.sum(fn_np)  # FN像素数
        all_num = result_1_np.size  # 总像素数

        # 如果检查有问题则报错
        smooth = 0
        if all_num != tp_num+tn_num+fp_num+fn_num:  # 相加不等于像素总数
            print('wrong!img_name:[{}]img_type:[{}]num_all:[{}]add_result:[{}]'.format(
                img_name,img_type,all_num, tp_num + tn_num + fp_num + fn_num))
        # if tp_num + fp_num == 0 or tp_num + fn_num==0:  # 分母为0
        #     print(i+1,'分母为0',tp_num,tn_num,fp_num,fn_num)

        rate_precision = tp_num / (tp_num + fp_num+smooth)  # 查准率
        rate_recall = tp_num / (tp_num + fn_num+smooth)  # 查全率
        rete_f1 = 2*rate_precision*rate_recall/(rate_precision+rate_recall+smooth)  # F1分数
        rate_oa = (tp_num+tn_num) / (tp_num+tn_num+fp_num+fn_num+smooth)  # PA，即像素准确度
        rate_iou_road = tp_num / (tp_num + fp_num + fn_num+smooth)  # 道路交并比
        rate_iou_background = tn_num / (tn_num + fn_num + fp_num+smooth)  # 背景交并比
        rate_miou = (rate_iou_road+rate_iou_background) / 2  # 平均交并比

        rate_pe=((tp_num+fp_num)/all_num*(tp_num+fn_num)+(tn_num+fp_num)/all_num*(tn_num+fn_num))/all_num
        rate_kappa=(rate_oa-rate_pe)/(1-rate_pe)  # kappa系数
        ##########.##########

        ##########保存比较结果图片##########
        result_shape = np.array(real_np.shape)  # 保存比较数据的形状,[1024,1024]
        result_np = np.zeros(np.insert(result_shape,2,3))  # 初始化比较结果图像  # 1024*1024*3
        color_list = np.array([[255,255,255],[0,0,0],[0,0,255],[0,255,0]])  # TP、TN、FP、FN的BGR颜色列表（白、黑、红、绿）

        result_np[tp_np] = color_list[0]  # 绘制TP
        result_np[tn_np] = color_list[1]  # 绘制TN
        result_np[fp_np] = color_list[2]  # 绘制FP
        result_np[fn_np] = color_list[3]  # 绘制FN

        cv2.imwrite(save_path+'/'+img_name.split('.')[0]+'.jpg',result_np)  # 保存图片
        ##########.##########

        ##########写入比较结果csv##########
        result_list = [tp_num, tn_num, fp_num, fn_num, all_num, rate_precision, rate_recall,
                       rete_f1, rate_oa, rate_iou_road, rate_iou_background, rate_miou,rate_kappa]  # 结果指标列表
        compare_csv.loc[i + 1]=result_list  # 写入一行数据
        cbar.set_postfix_str("[{:4d}/{:4d}][{}] {}".format(i+1, img_num, img_type, img_name))
        ##########.##########
    cbar.close()
    ##########写入总览csv##########
    compare_b_np = compare_b_csv.values

    #compare_b_result = np.nanmean(compare_b_np[::,[5,6,7,8,9,12]],axis=0)  # 有监督数据的平均查准率，平均查全率，平均F1，平均精度，平均IOU,kappa
    compare_b_result = np.nanmean(compare_b_np[::,[7,9,12]],axis=0)  # 有监督数据的平均F1，平均IOU,kappa
    print(compare_b_result)
    compare_result_csv.loc[1]=['supervised']+list(compare_b_result)
    ##########.##########

    ##########保存结果csv##########
    compare_result_csv.to_csv(compare_csv_path+'/overview.csv', mode='w', header=True)  # 保存结果数据
    compare_b_csv.to_csv(compare_csv_path+'/b.csv', mode='w', header=True)  # 保存有监督数据
    ##########.##########
    ####################.####################




'''
在123批次数据上训练，4批次数据上测试
'''
import os
import torch
import sys
import copy
import ttach as tta
from Dataloader.dataloader import \
    TestRegionLevel_Dloader
from models.FCN_resnet18 import FCN_res18
from utils.utils import \
    load_weight, \
    PR_score, \
    Logger
import numpy as np

### 模型训练过程中的一些基本参数的设置 ###
class Opt:
    def __init__(self):
        self.gpu = 0  # 指定gpu
        self.cls = 1  # 预测的类别
        self.size = 416  # 图像块的裁切大小
        '''
        label
        =0采样Dimple图片,
        =1采样压痕不良图片,
        =2采样脏污等其他非Dimple图片,
        =3采样非Dimple图片
        '''

        self.root = '../Dimple-dataset/'  # 数据集根目录
        self.weight_pth = list()
        self.transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),     
        ]
)


### 模型训练, 验证 + 保存 ###
def main(opt):
    #########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### init ###
    size = opt.size
    root = opt.root
    weight_pth = opt.weight_pth
    transforms =  opt.transforms
    ############

    model = FCN_res18(num_classes=1, cls_classes=1)
    model = model.cuda()
    model_ensemble = []
    for i in range(len(weight_pth)):
        model, _ = load_weight(model, weight_pth[i], device, is_backbone=False)
        model_ensemble.append(copy.deepcopy(model))
        # tta_model = tta.ClassificationTTAWrapper(copy.deepcopy(model), transforms=transforms)
        # model_ensemble.append(tta_model)


    val_data = TestRegionLevel_Dloader(
                                    # root=root,
                                    root=root + 'old_data/',
                                    #    bbox_json='Dataloader/bboxes.json',
                                       bbox_json='Dataloader/old_bboxes.json',
                                       size=size)


    probs = list()
    gts = list()
    model_ensemble[0].eval()
    model_ensemble[1].eval()
    model_ensemble[2].eval()
    model_ensemble[3].eval()

    for img_pth, img, label in val_data():
        vote = 0
        img = img.to(device)
        with torch.no_grad():
            # prob = model_ensemble[0](img).sigmoid().min()
            prob_0 = model_ensemble[0](img)[0].sigmoid().min()
            prob_1 = model_ensemble[1](img)[0].sigmoid().min()
            prob_2 = model_ensemble[2](img)[0].sigmoid().min()
            # prob_3 = model_ensemble[3](img)[0].sigmoid().min()
            prob_3 = model_ensemble[3](img)[0].sigmoid().min()


            if prob_0 >= 0.5:
                vote += 1
            if prob_1 >= 0.5:
                vote += 1
            if prob_2 >= 0.5:
                vote += 1
            if prob_3 >= 0.5:
                vote += 1
            
            if (vote >= 2):
                prob = np.array(1)

            else:
                prob = np.array(0)

            # prob = min(prob_0, prob_1, prob_2, prob_3)
            # prob = (prob_0 + prob_1 + prob_2 + prob_3) / 4
            # prob = max(prob_0, prob_1, prob_2, prob_3)
            # prob = prob.data.cpu().tolist()

        probs.append(prob)
        gts.append(label)

        # 打印残差>0.5的所有样本
        if abs(label - prob) >= 0.5:
            print("prob_0=", prob_0, "  prob_1=", prob_1,"  prob_2=", prob_2,"  prob_3=", prob_3)
            print('{}: pred prob={}, label={}'.format(
                img_pth,
                # round(prob, 2),
                prob,
                label)
            )

    acc, recall = PR_score(gts, probs, threshold=0.5)
    print('acc={:.4f}, recall={:.4f}'.format(acc, recall))



if __name__ == '__main__':
    opt = Opt()
    opt.gpu = 0
    # sys.stdout = Logger("./logs/sample_data_ensemble_models/" + "test_imbalance124_tta.txt")
    # sys.stdout = Logger("./logs/train124test3/sample_data_ensemble_models/" + "test_ensemble_reclean_data_imbalance_rate_124.txt")

    # data_ensemble_train
    # weight_pth_0 = 'model_0_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train12_test3batch/1500.pth'
    # weight_pth_1 = 'model_1_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train14_test3batch/500.pth'
    # weight_pth_2 = 'model_2_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train24_test3batch/1000.pth'
    # weight_pth_3 = 'ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train124_test3batch/1000.pth'

    # imbalance_ensemble_train
    weight_pth_0 = 'model_0_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train124_test3batch/1500.pth'
    weight_pth_1 = 'model_1_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train124_test3batch/2000.pth'
    weight_pth_2 = 'model_2_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train124_test3batch/1000.pth'
    weight_pth_3 = 'model_3_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_FCN_resnet18_train124_test3batch/500.pth'


    opt.weight_pth.append(weight_pth_0)
    opt.weight_pth.append(weight_pth_1)
    opt.weight_pth.append(weight_pth_2)
    opt.weight_pth.append(weight_pth_3)
    main(opt)

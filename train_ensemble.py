'''
在123批次数据上训练，4批次数据上测试
'''
import os
import torch
import sys
import copy
from torch.utils.data import DataLoader
from Dataloader.dataloader import \
    Train_Dloader, \
    SamplerByLabel, \
    TestRegionLevel_Dloader
from models.FCN_resnet18 import FCN_res18
from utils.utils import \
    save_weight, \
    load_weight, \
    PR_score, \
    BCE_criterion, \
    BCE_neg_pixellevel_criterion, \
    DICE_criterion, Logger

from utils.losses import TripletLoss

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
        self.labels = 8 * [0] + 8 * [1] + 8 * [2] + 16 * [3]
        self.lr = 1e-3  # 学习率
        self.iterations = 2000  # 迭代次数
        self.root = '../Dimple-dataset/'  # 数据集根目录
        self.weight_pth = ''
         # 预训练模型路径
        self.pre_model_path = ''


### 模型训练, 验证 + 保存 ###
def main(opt):
    #########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### init ###
    size = opt.size
    labels = opt.labels
    lr = opt.lr
    iterations = opt.iterations
    root = opt.root
    weight_pth = opt.weight_pth
    pre_model_path = opt.pre_model_path
    ############

    model = FCN_res18(cls_classes=1)
    # model = model.cuda()
    model_backbone, model, _ = load_weight(model, pre_model_path, device, is_backbone=True)
    step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-5,
                                 amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    backone_optimizer = torch.optim.SGD(model_backbone.parameters(),
                          lr=0.0001,
                          momentum=0.9,
                          weight_decay=1e-5)


    Train = Train_Dloader(
                        root=root,
                        bbox_json='Dataloader/bboxes.json',
                        # root=root + 'old_data/',
                        # bbox_json='Dataloader/old_bboxes.json',
                        seg_root='./Dataloader/seg_set/',
                        size=size,
                        supcon=False,
                        suptrans=False)
    train_data = DataLoader(Train,
                            num_workers=8,
                            batch_size=len(labels),
                            collate_fn=Train.collate_fn,
                            sampler=SamplerByLabel(labels))
    val_data = TestRegionLevel_Dloader(
                                    root=root,
                                     bbox_json='Dataloader/bboxes.json',
                                    # root=root + 'old_data/',
                                    # bbox_json='Dataloader/old_bboxes.json',
                                    size=size)

    model.train()
    for flag, img, seg in train_data:

        for g in optimizer.param_groups:
            g['lr'] = lr * (iterations - step) / iterations

        for g in backone_optimizer.param_groups:
            g['lr'] = 0.0001 * (iterations - step) / iterations

        flag = flag.to(device)
        img = img.to(device)
        seg = seg.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels).to(device)

        optimizer.zero_grad()
        backone_optimizer.zero_grad()
        logit, seg_logit, cls_features = model(img)

        '''
        每个mini batch(batch size=40)中
        对于Dimple图片，16张图片忽略1张
        对于难例图片，默认为负例，但是忽略1/4的图片（2张）
        合计忽略3张图片
        '''
        loss_cls = BCE_criterion(logit[labels == 0], label=1, ignore_k=1) + \
                   BCE_criterion(logit[labels == 1], label=0, ignore_k=0) + \
                   BCE_criterion(logit[labels == 2], label=0, ignore_k=0) + \
                   BCE_criterion(logit[labels == 3], label=0, ignore_k=2)

        triplet_labels = copy.deepcopy(labels)
        for i in range(len(triplet_labels)):
            if triplet_labels[i] == 0:
                triplet_labels[i] = 1
            else:
                triplet_labels[i] = 0

        loss_metric = TripletLoss()(cls_features, triplet_labels)
        loss_cls = loss_cls / (len(labels) - 3)
        loss_seg = 0
        if flag.sum() != 0:
            loss_seg = DICE_criterion(seg_logit[flag == 1], seg)
        loss_seg_neg = BCE_neg_pixellevel_criterion(seg_logit[(labels == 1)+(labels == 2)])
        loss = loss_cls + loss_seg + 0.05 * loss_seg_neg + loss_metric

        loss.backward()
        optimizer.step()
        backone_optimizer.step()

        # 打印
        step += 1
        if step % 10 == 0:
           
            print('loss_cls={:.4f}, loss_seg={:.4f}, loss_metric={:.4f}, step={}, iterations={}'
                  .format(loss_cls, loss_seg, loss_metric,step, iterations))

        # 验证+保存
        if step % 500 == 0:
            probs = list()
            gts = list()
            model.eval()
            for img_pth, img, label in val_data():
                img = img.to(device)
                with torch.no_grad():
                    prob = model(img)[0].sigmoid().min()
                    prob = prob.data.cpu().tolist()
                probs.append(prob)
                gts.append(label)
                # 打印残差>0.5的所有样本
                if abs(label - prob) >= 0.5:
                    print('{}: pred prob={}, label={}'.format(
                        img_pth,
                        round(prob, 2),
                        label)
                    )
            model.train()
            acc, recall = PR_score(gts, probs)
            print('acc={:.4f}, recall={:.4f}'.format(acc, recall))
            save_weight(model, step, weight_pth.format(step))
        if step >= iterations:
            break
 

if __name__ == '__main__':
    opt = Opt()
    opt.gpu = 0
    # sys.stdout = Logger("ensemble_FPN_model_2_ignore_k_2_backbone_lr_0.0001_SGD_lr_0.001_Adam_triplet_supcon_cls_seg_train124_test3batch.txt")
    sys.stdout = Logger("06_13_17_15_FCN_model_3_supcon_fcn_train123test4.txt")

    opt.weight_pth = '06_13_17_15_supcon_FCN_model_3_train123test4/{}.pth'
    opt.pre_model_path = 'supcon_models_02-18-19-22/last.pth'
    main(opt)

'''
修改fcn的结构，采用孔洞卷积以保持模型的感受野与原backbone一致。
即layer3, layer4中非下采样全采用了孔洞卷积。

相比于train2， train3加入了对非Dimple样本的分割loss约束。以避免错误的预测区域
'''
import os
import torch
from torch.utils.data import DataLoader
from utils.losses import SupConLoss

from Dataloader.dataloader import Train_Dloader
from utils.utils import save_weight, load_weight, warmup_learning_rate, adjust_learning_rate, AverageMeter, Logger, save_weight

from models.FCN_resnet18 import Resnet18_Backbone
import math

import time
import sys

### 模型训练过程中的一些基本参数的设置 ###
class Opt:
    def __init__(self):
        self.gpu = 0  # 指定gpu
        self.size = 416  # 图像块的裁切大小
        self.bs = 32  # 训练的batch size
        self.learning_rate = 0.5  # 学习率
        self.root = '../Dimple-dataset/'  # 数据集根目录
  
        self.epochs = 1000
        self.temp = 0.1
        self.lr_decay_epochs = '700,800,900'
        self.lr_decay_rate = 0.1
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.cosine = True
        self.syncBN = False
        self.warm = True
        self.print_freq = 10
        self.save_freq = 50
        self.record_time = None



### 模型训练, 验证 + 保存 ###
def main(opt):
    #########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### init ###
    size = opt.size
    bs = opt.bs
    root = opt.root

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if bs > 256:
        opt.warm = True
    if opt.warm:
        # opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate



    ############
    model = Resnet18_Backbone()
    criterion = SupConLoss(temperature=opt.temp)
    model, step = load_weight(model, 'none.pth', device)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    Train = Train_Dloader(root=root,
                          bbox_json='./Dataloader/bboxes.json',
                          seg_root='./Dataloader/seg_set/',
                          size=size,
                          supcon=True,
                          suptrans=True)

    train_data = DataLoader(Train,
                            num_workers=16,
                            batch_size=bs,
                            shuffle=True,
                            drop_last=True)



    #training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        idx = 0
        model.train()

        end = time.time()
        for _, (img_1, img_2, label) in enumerate(train_data):
            # measure data loading time
            data_time.update(time.time() - end)

            idx += 1
            step += 1
            images = torch.cat([img_1, img_2], dim=0)
            images = images.to(device)
            bsz = label.shape[0]

            label = label.to(device)
         
            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_data), optimizer)

            # compute loss
            _, x, _ = model(images)
            f1, f2 = torch.split(x, [bsz, bsz], dim=0)
            x = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(x, label)

            # update metric
            loss = loss.float()
            losses.update(loss.item(), bsz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                    epoch, idx + 1, len(train_data), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                'supcon_models_{}/'.format((time.strftime('%m-%d-%H-%M', opt.record_time)))
                 + 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_weight(model, step, save_file)

    # save the last model
    save_file = os.path.join(
       './supcon_models_{}/'.format((time.strftime('%m-%d-%H-%M', opt.record_time))) + 'last.pth')
    save_weight(model, step, save_file)

if __name__ == '__main__':
    opt = Opt()
    opt.record_time = time.localtime(time.time())
    sys.stdout = Logger("cls_supCon_train_info_debug_{}.txt".format(time.strftime('%m-%d-%H-%M', opt.record_time)))
    main(opt)

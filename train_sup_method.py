'''
在123批次数据上训练，4批次数据上测试
'''
import os
import sys
import torch
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
from timm.utils import ModelEma

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
        self.labels = 16 * [0] + 8 * [1] + 8 * [2] + 8 * [3]
        self.lr = 1e-3  # 学习率
        self.iterations = 2000  # 迭代次数
        self.root = '../Dimple-dataset/'  # 数据集根目录
        self.weight_pth = 'sup_FCN_resnet18/{}.pth'


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
    ############

    model = FCN_res18(cls_classes=1)
    model, step = load_weight(model, 'none.pth', device)

    # model_ema = ModelEma(
    #     model,
    #     decay=0.99992,
    #     device='',
    #     resume='')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    Train = Train_Dloader(root=root,
                          bbox_json='Dataloader/bboxes.json',
                          seg_root='Dataloader/seg_set/',
                          size=size)
    train_data = DataLoader(Train,
                            num_workers=8,
                            batch_size=len(labels),
                            collate_fn=Train.collate_fn,
                            sampler=SamplerByLabel(labels))
    val_data = TestRegionLevel_Dloader(root=root,
                                       bbox_json='Dataloader/bboxes.json',
                                       size=size)

    model.train()
    for flag, img, seg in train_data:
        flag = flag.to(device)
        img = img.to(device)
        seg = seg.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels).to(device)

        for g in optimizer.param_groups:
                g['lr'] = lr * (opt.iterations - step) / opt.iterations

        optimizer.zero_grad()
        logit, seg_logit,_ = model(img)

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
        loss_cls = loss_cls / (len(labels) - 3)
        loss_seg = 0
        if flag.sum() != 0:
            loss_seg = DICE_criterion(seg_logit[flag == 1], seg)
        loss_seg_neg = BCE_neg_pixellevel_criterion(seg_logit[(labels == 1)+(labels == 2)])
        loss = loss_cls + loss_seg + 0.05 * loss_seg_neg

        loss.backward()
        optimizer.step()

        # 打印
        step += 1
        if step % 10 == 0:
            print('loss_cls = {:.4f}, loss_seg = {:.4f}, step = {}, iterations={}'
                  .format(loss_cls, loss_seg, step, iterations))

        # 每200轮ema一次
        # if step % 200 == 0:
        #     model_ema.update(model)


        # 验证+保存
        # new_ema_model = model_ema.ema
        if step % 500 == 0:
            probs = list()
            gts = list()
            # new_ema_model.eval()
            model.eval()

            for img_pth, img, label in val_data():
                img = img.to(device)
                with torch.no_grad():
                    # prob = new_ema_model(img)[0].sigmoid().min()
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
            save_weight(model, step, weight_pth.format(step))
            # new_ema_model.train()
            model.train()
            acc, recall = PR_score(gts, probs)
            print('acc={:.4f}, recall={:.4f}'.format(acc, recall))
        if step >= iterations:
            break


if __name__ == '__main__':
    opt = Opt()
    opt.gpu = 0
    sys.stdout = Logger("0519_sup_fcn_train123test4.txt")

    opt.weight_pth = 'sup_FCN_resnet18_train123test4/{}.pth'
    main(opt)

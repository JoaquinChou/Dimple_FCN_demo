import os
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def load_weight(net, net_pth, device, is_backbone=False):
    root = './save_net/'
    step = 0
    if not os.path.exists(root):
        os.makedirs(root)
    if os.path.isfile(os.path.join(root, net_pth)):
        save_dic = torch.load(os.path.join(root, net_pth), map_location=lambda storage, loc: storage)
        if is_backbone:
            # net.encoder.load_state_dict(save_dic['state_dict'])
            net.backbone.load_state_dict(save_dic['state_dict'])
        else:
            net.load_state_dict(save_dic['state_dict'])
        step = save_dic['step']
        print('weight loaded successfully')
    else:
        print('no weight file')
    net = net.to(device)
    if is_backbone:
        # return net.encoder, net, step
        return net.backbone, net, step
    else:
        return net, step


def save_weight(net, step, net_pth):
    root = './save_net/'
    if not os.path.exists(os.path.join(root, net_pth.split('/')[0])):
        os.makedirs(os.path.join(root, net_pth.split('/')[0]))
    save_dic = {}
    save_dic['state_dict'] = net.state_dict()
    save_dic['step'] = step
    torch.save(save_dic, os.path.join(root, net_pth))
    return


def PR_score(y_true, y_pred, threshold=0.5):
    '''
    :param y_true: list, 0或1
    :param y_pred: list, 01之间的浮点数
    :param threshold: y_pred threshold, 大于threshold认为为正样本
    :return: accuracy, recall
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred != 1] = 0
    inter = (y_pred * y_true).sum()
    acc = inter / y_pred.sum()
    recall = inter / y_true.sum()
    return acc, recall


def save_PRcurve(y_true, y_pred, pic_pth):
    root = './save_PRcurve/'
    if not os.path.exists(os.path.join(root, pic_pth.split('/')[0])):
        os.makedirs(os.path.join(root, pic_pth.split('/')[0]))
    # matplotlib draw PR
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision)
    plt.savefig(os.path.join(root, pic_pth))
    plt.close()
    return


### loss ###
def BCE_criterion(logits, label, ignore_k=1, reduction='sum'):
    '''
    logit: shape = NC(C=1),
    label: 0 或 1
    ignore_k: 假设一个batch有n张Dimple图像块， 对Dimple图像中拟合最差的k张图片不监督。(相当于忽略极难例，可能是噪声标签)
    reduction: 'mean' 或则是 'sum'
    '''
    loss = F.binary_cross_entropy_with_logits(
        logits,
        torch.full_like(logits, fill_value=label, dtype=torch.float32).to(logits.device),
        reduction='none').squeeze()
    n = loss.size(0)
    if n > ignore_k:
        n -= ignore_k
        loss, _ = torch.topk(loss, k=n, largest=False)
        loss = loss.sum()
    else:
        n = 0
        loss = 0
    if reduction == 'mean':
        return loss / n
    elif reduction == 'sum':
        return loss


def CE_criterion(logits, label, ignore_k=1, reduction='sum'):
    labels = torch.full((logits.shape[0], 1), fill_value=label, dtype=torch.long).to(logits.device)
    loss = F.cross_entropy(logits, labels.squeeze(), reduction='none')

    n = loss.size(0)
    if n > ignore_k:
        n -= ignore_k
        loss, _ = torch.topk(loss, k=n, largest=False)
        loss = loss.sum()
    else:
        n = 0
        loss = 0
    if reduction == 'mean':
        return loss / n
    elif reduction == 'sum':
        return loss 


def DICE_criterion(logits, segs):
    '''
    把医学分割常用的Dice loss拿过来用了。相比交叉熵，Dice loss损失大小受填充面积控制，
    在类别不均衡，标注不精细时可能会带来一些好处。
    '''
    preds = torch.sigmoid(logits)
    BS = segs.size(0)
    preds = preds.view(BS, -1)
    segs = segs.view(BS, -1)
    smooth = 1
    dice = (2 * torch.sum(preds * segs, dim=-1) + smooth) / (preds.pow(2).sum(-1) + segs.pow(2).sum(-1) + smooth)
    return 1 - dice.mean()


def BCE_neg_pixellevel_criterion(logits):
    '''
    对于模型的分割预测，对于所有的非Dimple，我们使用Pixel级别的二值交叉熵进行约束
    '''
    return F.binary_cross_entropy_with_logits(logits,
                                              torch.zeros_like(logits, dtype=torch.float32).to(logits.device),
                                              reduction='mean')


# 返回对比学习的两种增扩
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(image=x)["image"], self.transform(image=x)["image"]]



def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#log the terminal message in the txt
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w+", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
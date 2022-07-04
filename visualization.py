'''
修改fcn的结构，采用孔洞卷积以保持模型的感受野与原backbone一致。
即layer3, layer4中非下采样全采用了孔洞卷积。
'''
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet18 import Resnet18
from Dataloader.dataloader import TestRegionLevel_Dloader
from utils import save_weight, load_weight, PR_score, save_PRcurve

from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation._utils import _SimpleSegmentationModel


### 模型结构，基于resnet18, 加入了scSE block ###
'''
模型性能探索:
加入scSE block, 略微提升，主要是模型的预测概率变大了：
举个例子:
加入前: 预测概率0.2, label1.0
加入后: 预测概率0.5, label1.0
大概是这种级别的提升
'''
class cSE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(cSE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class sSE_Module(nn.Module):
    def __init__(self, channel):
        super(sSE_Module, self).__init__()
        self.spatial_excitation = nn.Sequential(
                nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class scSE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(scSE_Module, self).__init__()
        self.cSE = cSE_Module(channel, ratio)
        self.sSE = sSE_Module(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class Resnet18(Resnet18):
    def __init__(self, *args, **kwargs):
        super(Resnet18, self).__init__(*args, **kwargs)
        self.scSE1 = scSE_Module(channel=64, ratio=16)
        self.scSE2 = scSE_Module(channel=128, ratio=16)
        self.scSE3 = scSE_Module(channel=256, ratio=32)
        self.scSE4 = scSE_Module(channel=512, ratio=32)

        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fc = nn.Linear(21 * 256, 1)

        for m in self.resnet.layer3.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                if isinstance(m.stride, int):
                    m.stride = (m.stride, m.stride)
                if m.stride[0] == 2:
                    m.stride = (1, 1)
                elif isinstance(m, nn.Conv2d):
                    m.dilation = 2
                    m.padding = m.dilation
        for m in self.resnet.layer4.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                if isinstance(m.stride, int):
                    m.stride = (m.stride, m.stride)
                if m.stride[0] == 2:
                    m.stride = (1, 1)
                elif isinstance(m, nn.Conv2d):
                    m.dilation = 2
                    m.padding = m.dilation

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # N1HW -> N3HW
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.scSE1(x)
        x = self.resnet.layer2(x)
        x = self.scSE2(x)
        x = self.resnet.layer3(x)
        x = self.scSE3(x)
        x = self.resnet.layer4(x)
        x = self.scSE4(x)
        features = x
        x = self.conv1x1(x)

        # 以下为目标检测所用的SPP与其实现
        x = torch.cat([
            F.adaptive_avg_pool2d(x, (4, 4)).flatten(1),
            F.adaptive_avg_pool2d(x, (2, 2)).flatten(1),
            F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)],
            dim=-1)
        x = self.fc(x)
        ############################
        return features, x


class FCN_res18(_SimpleSegmentationModel):
    def __init__(self, num_classes=1):
        super(FCN_res18, self).__init__(
            backbone=Resnet18(num_classes=num_classes),
            classifier=FCNHead(in_channels=512, channels=num_classes),
            aux_classifier=None,
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features, cls_logits = self.backbone(x)
        seg_logits = self.classifier(features)
        seg_logits = F.interpolate(seg_logits, size=input_shape, mode='bilinear', align_corners=False)
        return cls_logits, seg_logits


### dataloader ###
class Test_Dloader(TestRegionLevel_Dloader):
    def __init__(self, *args, **kwargs):
        super(Test_Dloader, self).__init__(*args, **kwargs)

    def __call__(self):
        for pth, bboxes in self.dimple_dic.items():
            pth = os.path.join(self.root, pth)
            img_cv2 = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), 0)
            yield pth.replace(self.root, ''), \
                  cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR), \
                  torch.from_numpy(self.norm(img_cv2)[np.newaxis, np.newaxis]).float()

        for pth, bboxes in self.neg_dic.items():
            pth = os.path.join(self.root, pth)
            img_cv2 = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), 0)
            yield pth.replace(self.root, ''), \
                  cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR), \
                  torch.from_numpy(self.norm(img_cv2)[np.newaxis, np.newaxis]).float()


### 模型训练过程中的一些基本参数的设置 ###
class Opt:
    def __init__(self):
        self.gpu = 1  # 指定gpu
        self.cls = 1  # 预测的类别
        self.step = 3000  # 选用第几次迭代
        self.root = 'F:\Dimple\Dimple-dataset\\'  # 数据集根目录
        self.weight_pth = 'resnet18—dilation/{}.pth'


### 模型训练, 验证 + 保存 ###
def main(opt):
    #########################################
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    #########################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### init ###
    cls = opt.cls
    step = opt.step
    root = opt.root
    weight_pth = opt.weight_pth
    ############

    model = FCN_res18(num_classes=cls)
    model, _ = load_weight(model, weight_pth.format(step), device)

    data = Test_Dloader(root=root,
                        bbox_json='Dataloader/bboxes.json',
                        size=320,
                        )()


    model.eval()
    for img_pth, img_cv2, img_torch in data:
        img_torch = img_torch.to(device)
        with torch.no_grad():
            pred = model(img_torch)[-1].sigmoid()
        pred = pred[0].cpu().numpy()
        pred[pred <= 0.5] = 0

        # visual
        pred = (pred * 255).astype('uint8').transpose(1, 2, 0).repeat(3, -1)
        pred[:, :, :2] = 0
        cv2.addWeighted(pred, 0.3, img_cv2, 0.7, 0, img_cv2)
        if not os.path.exists('./visualization_val/' + '/'.join(img_pth.split('/')[:-1])):
            os.makedirs('./visualization_val/' + '/'.join(img_pth.split('/')[:-1]))
        save_pth = './visualization_val/' + img_pth
        cv2.imwrite(save_pth, img_cv2)


if __name__ == '__main__':
    opt = Opt()
    opt.gpu = 1
    opt.step = 2000
    opt.weight_pth = 'resnet18—dilation-addneg/{}.pth'
    main(opt)

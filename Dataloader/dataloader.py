from cProfile import label
import torch
import torch.utils.data as data
import os
import cv2
import json
import random
import numpy as np
import albumentations as A
from torch.utils.data import Sampler
from utils.utils import TwoCropTransform


class SamplerByLabel(Sampler):
    '''
    用标签类型去数据集内索引图片，
    label=0随机采样Dimple图片,=1随机采样压痕不良图片,=2随机采样脏污等图片,=3随机采样非Dimple图片
    '''
    def __init__(self, labels):
        '''
        将标签作为输入, 按类别随机从图像中抽取样本
        '''
        if isinstance(labels, list) or isinstance(labels, tuple):
            self.labels = torch.LongTensor(labels)
        else:
            self.labels = labels

    def __iter__(self):
        for _ in range(int(1e+7)):
            for label in self.labels:
                yield label

    def __len__(self):
        return int(1e+7)


def dic_filter(bbox_json, keywords, supcon=False, supcon_label=None):
    '''
    只保留字典中，key有关键字的部分
    dimple keywords:
    ['1-batch/点状Dimple',
    '1-batch/条状Dimple',
    '2-batch/点状Dimple',
    '2-batch/条状Dimple',
    '3-batch/点状Dimple',
    '3-batch/条状Dimple',
    '4-batch/点状Dimple',
    '4-batch/条状Dimple',]

    非Dimple keywords(未考虑保存异常以及过亮):
    ['1-batch/非Dimple/1.',
    '1-batch/非Dimple/4.',
    '1-batch/非Dimple/破片',
    '2-batch/压痕不良',
    '2-batch/脏污异物',
    '3-batch/压痕不良',
    '3-batch/脏污',
    '3-batch/非Dimple',
    '4-batch/压痕不良',
    '4-batch/脏污',
    '4-batch/非Dimple']

    难例（可能dimple可能脏污）keywords:
    ['1-batch/非Dimple/5.',
     '2-batch/难例',
     '3-batch/可能',
     '4-batch/可能']
    '''
    with open(bbox_json, 'r') as f:
        dic = json.load(f)
    keys = list(dic.keys())
    for k in keys:
        bboxes = dic.pop(k)
        if bboxes == list():
            continue
        if supcon:
            for w in keywords:
                if w in k:
                    dic[k] = []
                    dic[k].append(bboxes)
                    dic[k].append(supcon_label)
                    break

        else:
            for w in keywords:
                if w in k:
                    dic[k] = bboxes
                    break
    return dic



class Train_Dloader(data.Dataset):
    def __init__(self,
                 root,
                 bbox_json='bboxes.json',
                 seg_root='./seg_set/',
                 size=320,
                 supcon=False,
                 suptrans=False,
                 ):
        '''
        :param root: 数据集根目录
        :param bbox_json: 图片中所有的红框坐标(y1x1y2x2)
        :param seg_root: 分割标签存储位置
        :param size:裁切图像块大小
        '''
        self.supcon_data = None
        self.supcon = supcon
        self.suptrans = suptrans
        
        self.root = root
        self.size = size
        # 整理所有的Dimple图片
        dimple_dic = dic_filter(bbox_json,
                                keywords=
                                [
                                 '1-batch/点状Dimple',
                                 '1-batch/条状Dimple',
                                 '2-batch/点状Dimple',
                                 '2-batch/条状Dimple',
                                 '3-batch/点状Dimple',
                                 '3-batch/条状Dimple',
                                #  '4-batch/点状Dimple',
                                #  '4-batch/条状Dimple',
                                 '5-batch/点状Dimple',
                                 '5-batch/条状Dimple',
                                 ],
                                 supcon=self.supcon,
                                 supcon_label=1)
                                #  supcon_label=0)

        # 整理所有的非Dimple图片(压痕不良)
        neg_yahen_dic = dic_filter(bbox_json,
                                   keywords=
                                   [
                                    '1-batch/非Dimple/1.',
                                    '2-batch/压痕不良',
                                    '3-batch/压痕不良',
                                    # '4-batch/压痕不良',
                                    '5-batch/非dimple/Dimple与其他不良的混合',
                                    ],
                                    supcon=self.supcon,
                                    supcon_label=0)
                                    # supcon_label=1)

        # 整理所有的非Dimple图片(脏污破片等other)
        neg_zhang_dic = dic_filter(bbox_json,
                                   keywords=
                                   [
                                    '1-batch/非Dimple/4.',
                                    '1-batch/非Dimple/破片',
                                    '2-batch/脏污异物',
                                    '3-batch/脏污',
                                    '3-batch/非Dimple',
                                    # '4-batch/脏污',
                                    # '4-batch/非Dimple',
                                    '5-batch/非dimple/Dimple与其他不良的混合',
                                    ],
                                    supcon=self.supcon,
                                    supcon_label=0)
                                    # supcon_label=2)

        # 整理所有的难例图片(可能是Dimple也可能是非Dimple)
        hard_dic = dic_filter(bbox_json,
                              keywords=
                              [
                               '1-batch/非Dimple/5.',
                               '2-batch/难例',
                               '3-batch/可能',
                            #    '4-batch/可能'
                                '5-batch/非dimple/Dimple与其他不良的混合',
                               ],
                                supcon=self.supcon,
                                supcon_label=0)
                                # supcon_label=3)

        self.dimple_dic = dimple_dic
        self.neg_yahen_dic = neg_yahen_dic
        self.neg_zhang_dic = neg_zhang_dic
        self.hard_dic = hard_dic
        print(
            '''
            加载的数据中：
            Dimple图片:{}张
            压痕不良图片:{}张
            脏污异物等图片:{}张
            难以区分图片:{}张
            '''.format(len(dimple_dic),
                       len(neg_yahen_dic),
                       len(neg_zhang_dic),
                       len(hard_dic)),
        )
        self.seg_root = seg_root
        self.seg_names = list(os.listdir(seg_root))

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625,
                                   scale_limit=0.1,
                                   rotate_limit=1,
                                   interpolation=cv2.INTER_LINEAR,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   value=-1,
                                   mask_value=0,
                                   p=0.5),
           ],
        )

        if self.supcon:
            self.supcon_data = list( self.dimple_dic.items()) + list(self.neg_yahen_dic.items()) + list(self.neg_zhang_dic.items()) + list(self.hard_dic.items())
            random.shuffle(self.supcon_data)

    def norm(self, img):
        '''
        instance级归一化，未来要加速的话可以固定均值和方差，归一化更快
        '''
        img = img.astype('float32')
        return (img - img.mean()) / (img.std() + 1e-5)
        # return (img - 56.23) / (39.28)

    def __len__(self):
        if self.supcon:
            return len(self.supcon_data)
        else:
            return int(1e+7)

    def __getitem__(self, label):
        if self.supcon:
            img_pth, [bboxes, supcon_label] = self.supcon_data[label]

        else:
            '''
            label=0随机采样Dimple图片,=1随机采样压痕不良图片,=2随机采样脏污等图片,=3随机采样非Dimple图片
            '''
            if label == 0:
                dic = self.dimple_dic
            elif label == 1:
                dic = self.neg_yahen_dic
            elif label == 2:
                dic = self.neg_zhang_dic
            elif label == 3:
                dic = self.hard_dic

            img_pth, bboxes = random.choice(list(dic.items()))

        img_pth = os.path.join(self.root, img_pth)
        img = cv2.imdecode(np.fromfile(img_pth, dtype=np.uint8), 0)
        H, W = img.shape
        bbox = random.choice(bboxes)
        center = [np.clip(random.randint(bbox[0], bbox[2]), self.size // 2, H - self.size + self.size // 2),
                np.clip(random.randint(bbox[1], bbox[3]), self.size // 2, W - self.size + self.size // 2)]
        seg_name = img_pth.split('/')[-1].split('\\')[-1].replace('.jpg', '.png')

        # 图片归一化
        img = self.norm(img)
        # 倘若采样中心位于图片边缘，则需对输入的图片进行pad, 然后再进行裁切
        img = np.pad(img,
                     ((self.size//2, self.size - self.size//2),
                      (self.size//2, self.size - self.size//2)),
                     'constant',
                     constant_values=0,
                     )
        img = img[center[0]: center[0] + self.size, center[1]: center[1] + self.size]
        # 如果有分割标签，就加载分割标签

        if seg_name in self.seg_names:
            seg = cv2.imdecode(np.fromfile(self.seg_root + seg_name, dtype=np.uint8), 0)
            seg[seg != 0] = 1
            seg = np.pad(seg,
                         ((self.size // 2, self.size - self.size // 2),
                          (self.size // 2, self.size - self.size // 2)),
                         'constant',
                         constant_values=0,
                         )
            seg = seg[center[0]: center[0] + self.size, center[1]: center[1] + self.size]
        else:
            seg = 0

        if self.supcon:
            if  self.suptrans:
                twins_images = TwoCropTransform(self.transform)
                img = twins_images(img)
                return torch.from_numpy(img[0][np.newaxis]).float(), torch.from_numpy(img[1][np.newaxis]).float(), torch.FloatTensor([supcon_label])
            else:
                return img[np.newaxis], np.array(supcon_label), img_pth
            
        # 图像增扩
        elif not isinstance(seg, int):
            transformed = self.transform(image=img, mask=seg)
            img = transformed["image"]
            seg = transformed["mask"]
        else:
            transformed = self.transform(image=img)
            img = transformed["image"]

        return torch.from_numpy(img[np.newaxis]).float(), \
               torch.from_numpy(seg[np.newaxis]).float() if not isinstance(seg, int) else 0


    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        segs = [x[1] for x in batch]
        flags = torch.LongTensor([1 if not isinstance(s, int) else 0 for s in segs])
        imgs = torch.stack(imgs, dim=0)
        if flags.sum() != 0:
            segs = torch.stack([s for s in segs if not isinstance(s, int)], dim=0)
        else:
            segs = torch.FloatTensor([])
        return flags, imgs, segs


class TestRegionLevel_Dloader():
    def __init__(self,
                 root,
                 bbox_json='bboxes.json',
                 size=320,
                 ):
        '''
        :param root: 数据集根目录
        :param bbox_json: 图片中所有的红框坐标(y1x1y2x2)
        :param size:裁切图像块大小
        '''
        self.root = root
        self.size = size
        # 整理所有的Dimple图片
        dimple_dic = dic_filter(bbox_json,
                                keywords=
                                [
                                #  '1-batch/点状Dimple',
                                #  '1-batch/条状Dimple',
                                #  '2-batch/点状Dimple',
                                #  '2-batch/条状Dimple',
                                #  '3-batch/点状Dimple',
                                #  '3-batch/条状Dimple',
                                '4-batch/点状Dimple',
                                '4-batch/条状Dimple',
                                '5-batch/点状Dimple',
                                '5-batch/条状Dimple',
                                ])
        # 整理所有的非Dimple图片
        neg_dic = dic_filter(bbox_json,
                             keywords=
                             [
                            #   '1-batch/非Dimple/1.',
                            #   '1-batch/非Dimple/4.'
                            #   '1-batch/非Dimple/破片',
                            #   '2-batch/压痕不良',
                            #   '2-batch/脏污异物',
                            #   '3-batch/压痕不良',
                            #   '3-batch/脏污',
                            #   '3-batch/非Dimple',
                            '4-batch/压痕不良',
                            '4-batch/脏污',
                            '4-batch/非Dimple',
                            '5-batch/非dimple/Dimple与其他不良的混合',
                             ])
        self.dimple_dic = dimple_dic
        self.neg_dic = neg_dic

    def norm(self, img):
        '''
        instance级归一化，未来要加速的话可以固定均值和方差，归一化更快
        '''
        img = img.astype('float32')

        return (img - img.mean()) / (img.std() + 1e-5)

        # return (img - 56.23) / (39.28)

    def __call__(self):
        '''
        返回路径，图像块，以及是否是Dimple图像(1是Dimple，0是非Dimple)
        '''
        for pth, bboxes in self.dimple_dic.items():
            pth = os.path.join(self.root, pth)
            img = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), 0)
            H, W = img.shape
            # 图片归一化
            img = self.norm(img)
            # 倘若采样中心位于图片边缘，则需对输入的图片进行pad
            img = np.pad(img,
                         ((self.size // 2, self.size - self.size // 2),
                          (self.size // 2, self.size - self.size // 2)),
                         'constant',
                         constant_values=0,
                         )
            img_patch = list()
            for bbox in bboxes:
                center = [np.clip((bbox[0] + bbox[2]) // 2, self.size // 2, H - self.size + self.size // 2),
                          np.clip((bbox[1] + bbox[3]) // 2, self.size // 2, W - self.size + self.size // 2)]
                img_patch.append(img[center[0]: center[0] + self.size, center[1]: center[1] + self.size])
            if img_patch is not None:
                img_patch = np.stack(img_patch, axis=0)[:, np.newaxis]
                yield pth.replace(self.root, ''), \
                      torch.from_numpy(img_patch).float(), \
                      1

        for pth, bboxes in self.neg_dic.items():
            pth = os.path.join(self.root, pth)
            img = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), 0)
            H, W = img.shape
            # 图片归一化
            img = self.norm(img)
            # 倘若采样中心位于图片边缘，则需对输入的图片进行pad
            img = np.pad(img,
                         ((self.size // 2, self.size - self.size // 2),
                          (self.size // 2, self.size - self.size // 2)),
                         'constant',
                         constant_values=0,
                         )
            img_patch = list()
            for bbox in bboxes:
                center = [np.clip((bbox[0] + bbox[2]) // 2, self.size // 2, H - self.size + self.size // 2),
                          np.clip((bbox[1] + bbox[3]) // 2, self.size // 2, W - self.size + self.size // 2)]
                img_patch.append(img[center[0]: center[0] + self.size, center[1]: center[1] + self.size])
            if img_patch != list():
                img_patch = np.stack(img_patch, axis=0)[:, np.newaxis]
                yield pth.replace(self.root, ''), \
                      torch.from_numpy(img_patch).float(), \
                      0


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 训练数据dataloader
    labels = [0, 1, 2, 3]
    loader = Train_Dloader('F:\Dimple\Dimple-dataset\\', size=320)
    train_data = DataLoader(loader,
                            num_workers=1,
                            batch_size=len(labels),
                            collate_fn=loader.collate_fn,
                            sampler=SamplerByLabel(labels))
    for flags, imgs, segs in train_data:
        print(flags)
        print(imgs.size())
        print(segs.size())
        for img in imgs:
            img = img.numpy()[0].astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(0)
        # if flags.sum() != 0:
        #     imgs = imgs[flags == 1][0].numpy()[0].astype('uint8')
        #     segs = segs[0, 0].numpy().astype('uint8') * 255
        #     print(segs.shape, imgs.shape)
        #     cv2.imshow('imgs', imgs)
        #     cv2.imshow('segs', segs)
        #     cv2.waitKey(0)
        print(20 * '=')

    # 测试数据dataloader
    loader = TestRegionLevel_Dloader('F:\Dimple\Dimple-dataset\\', size=320)
    for n, a, b in loader():
        print(n, a.shape, b)
        a = a[0, 0].numpy().astype('uint8')
        cv2.imshow('a', a)
        cv2.waitKey(0)

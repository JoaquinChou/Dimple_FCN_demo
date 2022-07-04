import torch
import os
import argparse
from models.FCN_resnet18 import Resnet18_Backbone
from torch.utils.data import DataLoader
from Dataloader.dataloader import Train_Dloader
from utils.utils import load_weight, AverageMeter

if __name__ == '__main__':
        
    size = 416  # 图像块的裁切大小
    bs = 32  # 训练的batch size
    root = '../Dimple-dataset/'  # 数据集根目录
    device = 0

    parser = argparse.ArgumentParser(description='SigSiam Evaluating!')
    parser.add_argument('--model_path', default=None, type=str, help='model_path')
    # parser.add_argument('--train_folder', default='D:/Ftp_Server/zgx/data/CWRU_data/class_balance/train/', type=str, help='train_folder')
    parser.add_argument('--results_txt', default=None, type=str, help='results_txt')
    parser.add_argument('--isPrintImgListName', default=False, type=str, help='whether print img list name or not')
    parser.add_argument('--img_list_name_txt', default=None, type=str, help='img_list_name_txt')

    args = parser.parse_args()        

    model_path = args.model_path
    # train_folder = args.train_folder

    model = Resnet18_Backbone()
    model, _ = load_weight(model, model_path, device)
    # state_dict = ckpt['model']
    # model.cuda()
    # model.load_state_dict(state_dict)



    Train = Train_Dloader(
                        # root=root,
                        root=root + 'old_data/',
                        #  bbox_json='Dataloader/bboxes.json',
                        bbox_json='Dataloader/old_bboxes.json',
                        seg_root='Dataloader/seg_set/',
                        size=size,
                        supcon=True,
                        suptrans=False)

    train_data = DataLoader(Train,
                            num_workers=8,
                            batch_size=bs,
                            shuffle=True,
                            drop_last=True)


    model.eval()
    top1 = AverageMeter()
    # acc1 = 0
    label_list, feature_list = [], []
    img_path_list = []
    with torch.no_grad():
        for batch_idx, (img, target, img_path) in enumerate(train_data):
            for i in range(bs):
                # print(img_path[0])
                img_path_list.append(img_path[i])

            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            bsz = bs
            features = model(img)[0]
            print(features.shape)
            feature_list += features.tolist()
            label_list += target.tolist()
            # if batch_idx == 1005:
            #     break

    if not os.path.exists('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/'):
        os.makedirs('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/')

    with open('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/' + args.results_txt.split('_')[-1] +'.txt', 'w+') as f:
        for i in range(len(label_list)):
            data = str(label_list[i]) + " " + " ".join('%s' % num
                                                    for num in feature_list[i])
            f.write(data)
            f.write('\n')

    if args.isPrintImgListName:    
        with open('./tsne/results_txt/'+ args.results_txt.split('_')[0] + '/' + args.img_list_name_txt + '.txt', 'w+', encoding='utf-8') as f:
            for i in range(len(img_path_list)):
                path = img_path_list[i]
                f.write(path)
                f.write('\n')
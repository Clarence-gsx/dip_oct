#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author:ShawnWang
##### System library #####
import os
import os.path as osp
from os.path import exists
import argparse
import json
import logging
import time
import numpy as np
##### pytorch library #####
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
##### My own library #####
import data.seg_transforms as st
from data.Seg_dataset import SegList
from utils.logger import Logger
from models.net_builder import net_builder
from utils.utils import AverageMeter, aic_fundus_lesion_segmentation, compute_segment_score, \
    compute_single_segment_score, target_seg2target_cls, aic_fundus_lesion_classification
from utils.vis import vis_multi_class

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)


###### eval ########

def test(args, test_data_loader, model, result_path, logger):
    model.eval()
    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    Dice_4 = AverageMeter()
    Dice_5 = AverageMeter()
    Dice_6 = AverageMeter()
    Dice_7 = AverageMeter()
    Dice_8 = AverageMeter()
    Dice_9 = AverageMeter()
    ret_segmentation = []

    for iter, (image, label, imt) in enumerate(test_data_loader):
        # if iter > 1:
        #     break
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        #target_cls = target_seg2target_cls(target_seg)

        with torch.no_grad():
            # batch test for memory reduce
            batch = 1
            pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
            #pred_cls = torch.zeros(image.shape[0], 10)
            for i in range(0, image.shape[0], batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]
                image_batch = image[start_id:end_id, :, :, :]
                image_var = Variable(image_batch).cuda()
                # wangshen model forward
                output_seg = model(image_var)
                # print(output_seg.shape)
                # print(torch.max(output_seg, 1))
                _, pred_batch = torch.max(output_seg, 1)

                pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data

                # pred_cls[start_id:end_id,:] = output_cls.cpu().data

            pred_seg = pred_seg.numpy().astype('uint8')

            if args.vis:
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                model_name = args.seg_path.split('/')[-3]
                save_dir = osp.join(result_path, 'vis', '%04d' % iter)
                if not exists(save_dir): os.makedirs(save_dir)
                vis_multi_class(imt, ant, pred_seg, save_dir)
                print('save vis, finished!')

            batch_time.update(time.time() - end)
            label_seg = label.numpy().astype('uint8')

            ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])
            Dice_4.update(ret[4])
            Dice_5.update(ret[5])
            Dice_6.update(ret[6])
            Dice_7.update(ret[7])
            Dice_8.update(ret[8])
            Dice_9.update(ret[9])

            # ground_truth = target_cls.numpy().astype('float32')
            # prediction = pred_cls.numpy().astype('float32') # predict label

            # if iter == 0:
            #    detection_ref_all = ground_truth
            #    detection_pre_all = prediction
            # else:
            #    detection_ref_all = np.concatenate((detection_ref_all, ground_truth), axis=0)
            #    detection_pre_all = np.concatenate((detection_pre_all, prediction), axis=0)

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                        'Dice {dice.val:.3f} ({dice.avg:.3f})\t'
                        'Dice_1 {dice_1.val:.6f} ({dice_1.avg:.4f})\t'
                        'Dice_2 {dice_2.val:.6f} ({dice_2.avg:.4f})\t'
                        'Dice_3 {dice_3.val:.6f} ({dice_3.avg:.4f})\t'
                        'Dice_4 {dice_4.val:.6f} ({dice_4.avg:.4f})\t'
                        'Dice_5 {dice_5.val:.6f} ({dice_5.avg:.4f})\t'
                        'Dice_6 {dice_6.val:.6f} ({dice_6.avg:.4f})\t'
                        'Dice_7 {dice_7.val:.6f} ({dice_7.avg:.4f})\t'
                        'Dice_8 {dice_8.val:.6f} ({dice_8.avg:.4f})\t'
                        'Dice_9 {dice_9.val:.6f} ({dice_9.avg:.4f})\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(iter, len(test_data_loader), dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3,
                                dice_4=Dice_4, dice_5=Dice_5, dice_6=Dice_6, dice_7=Dice_7, dice_8=Dice_8,
                                dice_9=Dice_9, batch_time=batch_time))

    final_seg, seg_1, seg_2, seg_3, seg_4, seg_5, seg_6, seg_7, seg_8, seg_9 = compute_segment_score(
        ret_segmentation)
    print('### Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))
    print('Final Seg_4 Score:{}'.format(seg_4))
    print('Final Seg_5 Score:{}'.format(seg_5))
    print('Final Seg_6 Score:{}'.format(seg_6))
    print('Final Seg_7 Score:{}'.format(seg_7))
    print('Final Seg_8 Score:{}'.format(seg_8))
    print('Final Seg_9 Score:{}'.format(seg_9))

    # ret_detection = aic_fundus_lesion_classification( detection_ref_all, detection_pre_all, num_samples=len(eval_data_loader)*42)   #######val number
    # auc = np.array(ret_detection).mean()
    # print('AUC :',auc)
    # auc_1 = ret_detection[0]
    # auc_2 = ret_detection[1]
    # auc_3 = ret_detection[2]
    # auc_4 = ret_detection[3]
    # auc_5 = ret_detection[4]
    # auc_6 = ret_detection[5]
    # auc_7 = ret_detection[6]
    # auc_8 = ret_detection[7]
    # auc_9 = ret_detection[8]
    epoch = 0
    logger.append([epoch, final_seg, seg_1, seg_2, seg_3, seg_4, seg_5, seg_6, seg_7, seg_8, seg_9])


def eval_fusion(args, eval_data_loader, model_list, result_path, logger):
    for model in model_list:
        model.eval()

    batch_time = AverageMeter()
    dice = AverageMeter()
    end = time.time()
    dice_list = []
    Dice_1 = AverageMeter()
    Dice_2 = AverageMeter()
    Dice_3 = AverageMeter()
    ret_segmentation = []

    for iter, (image, label, imt) in enumerate(eval_data_loader):
        # batchsize = 1 ,so squeeze dim 1
        image = image.squeeze(dim=0)
        label = label.squeeze(dim=0)

        target_seg = label.numpy()
        target_cls = target_seg2target_cls(target_seg)

        with torch.no_grad():
            # batch test for memory reduce
            batch = 8
            pred_seg = torch.zeros(image.shape[0], image.shape[2], image.shape[3])
            pred_cls = torch.zeros(image.shape[0], 3)
            for i in range(0, image.shape[0], batch):
                start_id = i
                end_id = i + batch
                if end_id > image.shape[0]:
                    end_id = image.shape[0]

                image_batch = image[start_id:end_id, :, :, :]
                image_var = Variable(image_batch).cuda()

                Output_Seg = Variable(torch.zeros(batch, 4, image.shape[2], image.shape[3])).cuda()
                Output_Cls = Variable(torch.zeros(batch, 3)).cuda()
                # wangshen model forward
                weight = torch.tensor([0.5, 0.5]).cuda()
                for j, model in enumerate(model_list):
                    output_seg, output_cls = model(image_var)
                    Output_Seg += weight[j] * torch.exp(output_seg)
                    Output_Cls += weight[j] * output_cls

                _, pred_batch = torch.max(Output_Seg, 1)
                pred_seg[start_id:end_id, :, :] = pred_batch.cpu().data
                pred_cls[start_id:end_id, :] = Output_Cls.cpu().data

            pred_seg = pred_seg.numpy().astype('uint8')  # predict label

            if args.vis:
                imt = (imt.squeeze().numpy()).astype('uint8')
                ant = label.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis', '%04d' % iter)
                if not exists(save_dir):
                    os.makedirs(save_dir)
                vis_multi_class(imt, ant, pred_seg, save_dir)
                print('save vis, finished!')

            batch_time.update(time.time() - end)
            # metrice dice for seg
            label_seg = label.numpy().astype('uint8')
            ret = aic_fundus_lesion_segmentation(label_seg, pred_seg)
            ret_segmentation.append(ret)
            dice_score = compute_single_segment_score(ret)
            dice_list.append(dice_score)
            # update dice
            dice.update(dice_score)
            Dice_1.update(ret[1])
            Dice_2.update(ret[2])
            Dice_3.update(ret[3])
            # metrice auc for cls
            ground_truth = target_cls.numpy().astype('float32')
            prediction = pred_cls.numpy().astype('float32')  # predict label

            if iter == 0:
                detection_ref_all = ground_truth
                detection_pre_all = prediction
            else:
                detection_ref_all = np.concatenate((detection_ref_all, ground_truth), axis=0)
                detection_pre_all = np.concatenate((detection_pre_all, prediction), axis=0)

        end = time.time()
        logger_vis.info('Eval: [{0}/{1}]\t'
                        'Dice {dice.val:.3f} ({dice.avg:.3f})\t'
                        'Dice_1 {dice_1.val:.3f} ({dice_1.avg:.3f})\t'
                        'Dice_2 {dice_2.val:.3f} ({dice_2.avg:.3f})\t'
                        'Dice_3 {dice_3.val:.3f} ({dice_3.avg:.3f})\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(iter, len(eval_data_loader), dice=dice, dice_1=Dice_1, dice_2=Dice_2, dice_3=Dice_3,
                                batch_time=batch_time))

    final_seg, seg_1, seg_2, seg_3 = compute_segment_score(ret_segmentation)
    print('### Seg ###')
    print('Final Seg Score:{}'.format(final_seg))
    print('Final Seg_1 Score:{}'.format(seg_1))
    print('Final Seg_2 Score:{}'.format(seg_2))
    print('Final Seg_3 Score:{}'.format(seg_3))

    ret_detection = aic_fundus_lesion_classification(detection_ref_all, detection_pre_all,
                                                     num_samples=len(eval_data_loader) * 128)
    auc = np.array(ret_detection).mean()
    print('AUC :', auc)
    auc_1 = ret_detection[0]
    auc_2 = ret_detection[1]
    auc_3 = ret_detection[2]

    epoch = 0
    logger.append([epoch, final_seg, seg_1, seg_2, seg_3, auc, auc_1, auc_2, auc_3])


def test_seg(args, result_path, logger):
    print('Loading eval model ...')
    if args.fusion:
        # 1
        net_1 = net_builder('unet_nested')
        net_1 = nn.DataParallel(net_1).cuda()
        checkpoint_1 = torch.load('result/ori_3D/train/unet_nested/checkpoint/model_best.pth.tar')
        net_1.load_state_dict(checkpoint_1['state_dict'])
        # 2
        net_2 = net_builder('unet')
        net_2 = nn.DataParallel(net_2).cuda()
        checkpoint_2 = torch.load('result/ori_3D/train/unet/checkpoint/model_best.pth.tar')
        net_2.load_state_dict(checkpoint_2['state_dict'])

        net = [net_1, net_2]
    else:
        net = net_builder(args.seg_name)
        net = nn.DataParallel(net).cuda()
        checkpoint = torch.load(args.seg_path)
        net.load_state_dict(checkpoint['state_dict'])

    print('model loaded!')
    info = json.load(open(osp.join(args.list_dir, 'info.json'), 'r'))
    normalize = st.Normalize(mean=info['mean'], std=info['std'])

    t = []
    if args.resize:
        t.append(st.Resize(args.resize))
    t.extend([st.Label_Transform(), st.ToTensor(), normalize])
    dataset = SegList(args.data_dir, 'test', st.Compose(t), list_dir=args.list_dir)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    cudnn.benchmark = True
    if args.fusion:
        eval_fusion(args, test_loader, net, result_path, logger)
    else:
        test(args, test_loader, net, result_path, logger)


def parse_args():
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--seg-name', dest='seg_name', help='seg model', default=None, type=str)
    parser.add_argument('--seg-path', help='pretrained model test', default='./', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--fusion', action='store_true')
    parser.add_argument('--resize', default=0, type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    task_name = args.list_dir.split('/')[-1]
    ##### logger setting #####
    model_name = args.seg_path.split('/')[-3] if len(args.seg_path) > 2 else 'fusion'
    result_path = osp.join('result', task_name, 'test', model_name)
    if not exists(result_path):
        os.makedirs(result_path)
    logger = Logger(osp.join(result_path, 'dice_epoch.txt'), title='dice', resume=False)
    # if not resume:
    logger.set_names(
        ['Epoch', 'Dice', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4', 'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8', 'Dice_9'])
    test_seg(args, result_path, logger)


if __name__ == '__main__':
    main()

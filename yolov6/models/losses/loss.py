#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner

# 损失计算类
class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,  
                 fpn_strides=[8, 16, 32],       # 传入参数7：步长
                 grid_cell_size=5.0,            
                 grid_cell_offset=0.5,
                 num_classes=80,                # 传入参数1：类别数
                 ori_img_size=640,              # 传入参数2：输入图片大小 
                 warmup_epoch=4,                # 传入参数3：预热期epochs数量
                 use_dfl=True,                  # 传入参数4：是否使用dfl
                 reg_max=16,                    # 传入参数5：dfl参数
                 iou_type='giou',               # 传入参数6：iou损失计算方式
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5},
                 ):

        self.fpn_strides = fpn_strides

        self.cached_feat_sizes = [torch.Size([0, 0]) for _ in fpn_strides]   # 创建和fpn_strides 数组中的元素值相同数量的 torch.Size([0, 0]) 的张量大小
                                                                             # [torch.Size([0, 0]), torch.Size([0, 0]), torch.Size([0, 0])] 
                                                                             # 计算的时候会变成[torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])]
        self.cached_anchors = None
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,        # 头网络输出的特征，包括[[neck的三层特征], cls_score_list, reg_distri_list]
        targets,        # 当前对应的labels
        epoch_num,      # 当前epoch数
        step_num,       # 当前step数
        batch_height,   # 当前batch的高度
        batch_width     # 当前batch的宽度
    ):

        feats, pred_scores, pred_distri = outputs
        if all(feat.shape[2:] == cfsize for feat, cfsize in zip(feats, self.cached_feat_sizes)):
            anchors, anchor_points, n_anchors_list, stride_tensor = self.cached_anchors
        else:
            self.cached_feat_sizes = [feat.shape[2:] for feat in feats]
            
            anchors, anchor_points, n_anchors_list, stride_tensor = \
                   generate_anchors(feats,                       # neck输出的三层特征 
                                    self.fpn_strides,            # 步长
                                    self.grid_cell_size,
                                    self.grid_cell_offset,
                                    device=feats[0].device)
            
            # 把生成的anchor [8400, 4], anchor_points [8400, 2], n_anchors_list [6400, 1600, 400], stride_tensor [8400, 1]缓存起来
            # 后面计算的时候直接使用缓存的结果  
            self.cached_anchors = anchors, anchor_points, n_anchors_list, stride_tensor

        assert pred_scores.type() == pred_distri.type()
        
        # [640, 640, 640, 640]
        gt_bboxes_scale = torch.tensor([batch_width, batch_height, batch_width, batch_height]).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # [b, n_gt, 5]
        targets =self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]           # [b, n_gt, 1]
        gt_bboxes = targets[:, :, 1:]           # [b, n_gt, 4]  xyxy格式的坐标
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()  # [b, n_gt, 1]  掩码表示哪些位置有gt

        # 得到尺寸合适的候选点[8400, 2]
        anchor_points_s = anchor_points / stride_tensor

        # 通过候选点[8400, 2] 和 预测坐标的分布[b,8400, 17*4] ---> 预测的坐标[b,8400,4]
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            else:
                # 得到目标分类，目标框坐标，分类分数，真实框掩码 。其中分类分数根据预测的类别和真实类别计算得到。
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach(),                               # 预测类别[b, 8400, nc]
                        pred_bboxes.detach() * stride_tensor,               # 预测坐标[b, 8400, 4]
                        anchor_points,                                      # 原始的候选点[8400, 2]     
                        gt_labels,                                         # 真实类别[b, n_gt, 1]                           
                        gt_bboxes,                                         # 真实坐标[b, n_gt, 4]
                        mask_gt)                                           # 掩码表示哪些位置有gt

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # 把目标框映射到对应特征图的大小 [b, 8400, 4]
        target_bboxes /= stride_tensor

        # target_labels [b, 8400] ---> one-hot形式的标签 [b, 8400, nc+1] ---> [b, 8400, nc]
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        # dfl回归损失，通过:预测的分布[b, 8400, 17*4] 预测的坐标[b, 8400, 4] 候选点[8400, 2] 目标框坐标[b, 8400, 4] 目标框掩码[b, 8400, 1] 
        # 和target_scores [b, 8400, 6] 总的target_scores_sum  和掩码得到DFL损失和iou的损失一起知道bbox的回归
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl

        # 得到总的loss和各个部分loss
        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        # 转换为[b, n_gt, 5]的格式的targets
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)


# 计算cls分类损失
class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


# 计算bbox回归损失
class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask，正样本的选择
        num_pos = fg_mask.sum()   # 正样本的个数
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])

            # 得到正样本坐标[m, 4]
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])   
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum > 1:
                loss_iou = loss_iou.sum() / target_scores_sum
            else:
                loss_iou = loss_iou.sum()

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum > 1:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                else:
                    loss_dfl = loss_dfl.sum()
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

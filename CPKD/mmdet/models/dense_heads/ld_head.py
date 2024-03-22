# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.core import (bbox2distance, bbox_overlaps, distance2bbox,
                        images_to_levels, anchor_inside_flags, unmap,
                        multi_apply, reduce_mean)

from mmdet.core import bbox_overlaps, multi_apply, reduce_mean
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from ..builder import HEADS, build_loss
from .gfl_head import GFLHead
import math
import os

def intersect(box_a, box_b):
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                   box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)


def jaccard(box_a, box_b, iscrowd: bool = False):
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
                (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(
                    inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
                (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(
                    inter)  # [A,B]
    union = area_a + area_b - inter
    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

@HEADS.register_module()
class LDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_ld=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.0,
                     T=10),
                 loss_ld_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.0,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.0,
                     T=2),
                 loss_im=dict(type='IMLoss', loss_weight=0.0),
                 imitation_method='decouple',
                 **kwargs):

        super(LDHead, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_im = build_loss(loss_im)
        self.iou_calculator = build_iou_calculator(
            dict(type='BboxOverlaps2D'), )
        self.imitation_method = 'finegrained'
        self.loss_ld_vlr = build_loss(loss_ld_vlr)
        self.loss_kd = build_loss(loss_kd)
        self.loss_ld = build_loss(loss_ld)
        # self.loss_ld = build_loss(loss_ld)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, soft_targets, soft_label, x,
                    teacher_x, vlr_region, im_region, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        '''
        print("anchors size is before")
        print(anchors.size())
        '''
        anchors = anchors.reshape(-1, 4)
        '''
        print("anchors size is")
        print(anchors.size())
        '''
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3,
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))
        # 加进来的
        soft_label = soft_label.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
        teacher_x = teacher_x.permute(0, 2, 3, 1).reshape(-1, 256)
        x = x.permute(0, 2, 3, 1).reshape(-1, 256)
        vlr_region = vlr_region.reshape(-1)
        im_region = im_region.reshape(-1)

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        remain_inds = (vlr_region > 0).nonzero().squeeze(1)

        if self.imitation_method == 'finegrained':
            '''
            fg_inds = (im_region == 2).nonzero().squeeze(1)
            near_bg_inds = (im_region == 1).nonzero().squeeze(1)
            bg_inds = (im_region == 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                #loss_im = self.loss_im(x[fg_inds], teacher_x[fg_inds])
                #loss_fg = self.loss_im(x[fg_inds], teacher_x[fg_inds])
                #loss_near_bg = self.loss_im(x[near_bg_inds], teacher_x[near_bg_inds])
                #loss_bg = self.loss_im(x[bg_inds], teacher_x[bg_inds])
                loss_im = self.loss_im(x[fg_inds], teacher_x[fg_inds]) + self.loss_im(x[near_bg_inds], teacher_x[near_bg_inds])+self.loss_im(x[bg_inds], teacher_x[bg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
                '''
            '''
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
                '''
            '''
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            ng_inds = (im_region == 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds],
                                       teacher_x[fg_inds]) + 2 * self.loss_im(
                                           x[ng_inds], teacher_x[ng_inds])
            else:
                loss_im = bbox_pred.sum() * 0
            '''
            reg_im_region = torch.zeros_like(im_region)
            reg_im_region=self.tfd_reg(soft_label, cls_score, anchors,
                                        bbox_pred, soft_targets, stride,bbox_targets,pos_inds, im_region)
            
            cls_im_region = torch.zeros_like(im_region)
            cls_im_region =self.tfd_cls(soft_label, im_region)
            '''
            fg_inds1 = (reg_im_region > 0).nonzero().squeeze(1)
            ng_inds1 = (reg_im_region == 0).nonzero().squeeze(1)
            fg_inds2 = (cls_im_region > 0).nonzero().squeeze(1)
            ng_inds2 = (cls_im_region == 0).nonzero().squeeze(1)
            '''
            fg_inds1 = (reg_im_region >= 0).nonzero().squeeze(1)
            fg_inds2 = (cls_im_region >= 0).nonzero().squeeze(1)
            fg_weight1 = torch.ones_like(reg_im_region, dtype=torch.float)
            fg_weight2 = torch.ones_like(cls_im_region, dtype=torch.float)
            loss_im = (self.loss_im(x[fg_inds1],teacher_x[fg_inds1])* fg_weight1[fg_inds1]).mean() +  (self.loss_im(x[fg_inds2], teacher_x[fg_inds2])*fg_weight2[fg_inds2]).mean()
            
            

        elif self.imitation_method == 'decouple':
                #max_value = torch.max(im_region)
            #print("max value  iou")
            #print(max_value)
            
            #计算giscore，取前十个地方的蒸馏值*2
            gi_idx = self.get_gi_region(soft_label, cls_score, anchors,
                                        bbox_pred, soft_targets, stride, im_region)
            overlap_idx = torch.where(gi_idx[:, None] == im_region)[1]   
            # 将重合部分对应的im_region乘2
            im_region[overlap_idx] *= 2
            #max_value = torch.max(im_region)
            #print("max value iou+gi")
            #print(max_value)
            
            #分类奖惩
            #top_100_indices = self.get_cls_region(soft_label, im_region)
            #im_region[top_100_indices] *= 1.5
            #mask = (im_region[top_100_indices] <= 0.1)
            #im_region[top_100_indices][mask] = 0.6            
                   
            #max_value = torch.max(im_region)
            
            #print("max value iou+gi+cls")
            #print(max_value)
            #im_region = im_region / max_value
            #渐进三元遮罩+iou定位奖惩
            '''
            fg_inds = (im_region >= 0.5).nonzero().squeeze(1)
            near_fg_inds = ((im_region > 0.2) & (im_region < 0.5)).nonzero().squeeze(1)
            ng_inds = (im_region <=0.2).nonzero().squeeze(1)
            '''
            #torch.set_printoptions(profile="full")
            #print("Size7 of im_regions is: ", im_region.size())
            '''
            #准备输出im_region
            print("im size is")
            print(im_region.size())
            # 计算第二个维度的大小
            x1 = im_region.shape[0] // 2
            # 使用reshape函数
            im_region2 = im_region.reshape(2, x1)
            print("im2 size is")
            print(im_region2.size())
            #im_region3 = im_region2.reshape(2, 136, 100)
            xf = im_region2.shape[1]
            xc=math.ceil(xf)
            print("xc  is")
            print(xc)
            yf = (1088*800) / xc
            y=math.ceil(yf)
            print("y is")
            print(y)
            y1f = math.sqrt(y)
            y1=math.ceil(y1f)
            print("y1就是步长 is")
            print(y1)
            if y1 == 8:
                im_region3 = im_region2.reshape(2, 136, 100)
                torch.set_printoptions(profile="full")
                with open('../gtflag.txt', 'a') as f:
                    f.write("\nim_region 136*100 :\n\n")
                    t1 = im_region3[0]
                    #f.write(str(im_region3))
                    f.write(str(t1))
                #with open('../gtflag.txt', 'a') as f:
                #f.write("\nim_region:\n")
                #f.write(str(im_region3))
                #print("over")
            elif y1 == 16:
                im_region3 = im_region2.reshape(2, 68, 50)
                torch.set_printoptions(profile="full")
                with open('../gtflag.txt', 'a') as f:
                    f.write("\nim_region 68*50 :\n\n")
                    t1 = im_region3[0]
                    #f.write(str(im_region3))
                    f.write(str(t1))
            elif y1 == 32:
                im_region3 = im_region2.reshape(2, 34, 25)
                torch.set_printoptions(profile="full")
                with open('../gtflag.txt', 'a') as f:
                    f.write("\nim_region 34*25 :\n\n")
                    t1 = im_region3[0]
                    #f.write(str(im_region3))
                    f.write(str(t1))
            elif y1 == 63:
                im_region3 = im_region2.reshape(2, 17, 13)
                torch.set_printoptions(profile="full")
                with open('../gtflag.txt', 'a') as f:
                    f.write("\nim_region 17*13 :\n\n")
                    t1 = im_region3[0]
                    #f.write(str(im_region3))
                    f.write(str(t1))
            elif y1 == 118:
                im_region3 = im_region2.reshape(2, 9, 7)
                with open('../gtflag.txt', 'a') as f:
                    f.write("\nim_region 9*7 :\n\n")
                    t1 = im_region3[0]
                    #f.write(str(im_region3))
                    f.write(str(t1))
                    
                print("over")
            else:
                print("未知的步长值")

            #print(f"当步长为{stride}时，特征图的大小为{feature_map_size}")
            # 使用reshape函数
            print("im3 size is")
            print(im_region3.size())
            '''
            fg_inds = (im_region >= 1).nonzero().squeeze(1)
            near_fg_inds = ((im_region > 0) & (im_region < 1)).nonzero().squeeze(1)
            ng_inds = (im_region == 0).nonzero().squeeze(1)
            fg_weight = torch.ones_like(im_region, dtype=torch.float)
            alls = (im_region).nonzero().squeeze(1)
            
            #print("Size of fg_inds is: ", len(fg_inds))
            #print("Size of near_fg_inds is: ", len(near_fg_inds))
            #print("Size of ng_inds is: ", len(ng_inds))
            #result1 = np.array(im_region)
            
            #with open('../gtflag.txt', 'a') as f:
            #    f.write("im_region:\n")
            #    f.write(str(im_region))
            
            #print("over")
            #print("fg_inds")
            #print(len(fg_inds))
            #print("near_fg_inds")
            #print(len(near_fg_inds))
            #print("ng_inds")
            #print(len(ng_inds))
            if len(fg_inds) > 0:
                if len(ng_inds)>0:
                    #loss_im = self.loss_im(x[fg_inds],teacher_x[fg_inds]) + 0.5*self.loss_im(x[near_fg_inds], teacher_x[near_fg_inds])+ 0.05* self.loss_im(x[ng_inds], teacher_x[ng_inds])
                    loss_im = (self.loss_im(x[fg_inds], teacher_x[fg_inds]) * fg_weight[fg_inds]).mean() + (self.loss_im(x[near_fg_inds], teacher_x[near_fg_inds]) * (fg_weight[near_fg_inds]).mean()*0.7) + (self.loss_im(x[ng_inds], teacher_x[ng_inds]) * ((fg_weight[ng_inds]).mean()+0.005))
                   # loss_im = self.loss_im(x[alls], teacher_x[alls]) * fg_weight[alls].mean()
                else:
                    loss_im = (self.loss_im(x[fg_inds], teacher_x[fg_inds]) * fg_weight[fg_inds]).mean() + \
                              (self.loss_im(x[near_fg_inds], teacher_x[near_fg_inds]) * fg_weight[near_fg_inds]).mean() *0.7
                    #loss_im = self.loss_im(x[alls], teacher_x[alls]) * fg_weight[alls].mean()
                    
            else:
                loss_im = bbox_pred.sum() * 0

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            #loss_im = self.loss_im(x[fg_inds],teacher_x[fg_inds]) + 1.5 * self.loss_im(x[ng_inds], teacher_x[ng_inds])+ 1.5 * self.loss_im(x[near_fg_inds], teacher_x[near_fg_inds])
            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)


            loss_ld = self.loss_ld(
                pred_corners,
                soft_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            loss_kd = self.loss_kd(
                cls_score[pos_inds],
                soft_label[pos_inds],
                weight=label_weights[pos_inds],
                avg_factor=pos_inds.shape[0])

        else:
            loss_ld = bbox_pred.sum() * 0
            loss_kd = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_im = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        if len(remain_inds) > 0:
            neg_pred_corners = bbox_pred[remain_inds].reshape(
                -1, self.reg_max + 1)
            neg_soft_corners = soft_targets[remain_inds].reshape(
                -1, self.reg_max + 1)
            remain_targets = vlr_region[remain_inds]

            loss_ld_vlr = self.loss_ld_vlr(
                neg_pred_corners,
                neg_soft_corners,
                weight=remain_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=16.0)
            #loss_kd_neg = 0 * self.loss_kd(
            #    cls_score[remain_inds],
            #    soft_label[remain_inds],
            #    weight=label_weights[remain_inds],
            #    avg_factor=remain_inds.shape[0])
        else:
            loss_ld_vlr = bbox_pred.sum() * 0
            #loss_kd_neg = bbox_pred.sum() * 0
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, loss_ld_vlr,loss_kd, loss_im, weight_targets.sum()

    def forward_train(self,
                      x,
                      out_teacher,
                      teacher_x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        # soft_target = out_teacher[1]
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, out_teacher, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, x,
                                  teacher_x, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             soft_teacher,
             x,
             teacher_x,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        soft_label, soft_target = soft_teacher
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, assigned_neg_list,
         im_region_list) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)
        ##wdawd
        losses_cls, losses_bbox, losses_dfl, losses_ld,losses_kd, losses_ld_vlr, losses_im, \
        avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            soft_target,
            soft_label,
            x,
            teacher_x,
            assigned_neg_list,
            im_region_list,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        
        #torch.set_printoptions(profile="full")
        #print("Size2 of im_region_list is: ", len(im_region_list))
        
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            loss_ld_vlr=losses_ld_vlr,
            loss_kd=losses_kd,
            loss_im=losses_im, )

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_vlr_region,
         all_im_region) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        
        #torch.set_printoptions(profile="full")
        #print("Size5 of all_im_region is: ", len(all_im_region))
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # num_total_remain_neg = sum([max(inds.numel(), 1) for inds in assigned_neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        vlr_regions_list = images_to_levels(all_vlr_region, num_level_anchors)
        im_regions_list = images_to_levels(all_im_region, num_level_anchors)
        # sampled anchors of all images
        #torch.set_printoptions(profile="full")
        #print("Size6 of im_regions_list is: ", len(im_regions_list))
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, vlr_regions_list, im_regions_list)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        vlr_region = self.assigner.get_vlr_region(anchors,
                                                  num_level_anchors_inside,
                                                  gt_bboxes, gt_bboxes_ignore,
                                                  gt_labels)

        im_region = self.get_im_region(
            anchors, gt_bboxes, mode='decouple')

        #torch.set_printoptions(profile="full")
        #print("Size3 of im_region_list is: ", len(im_region))
        
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        labels_neg = anchors.new_full((num_valid_anchors,),
                                      self.num_classes,
                                      dtype=torch.long)

        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            vlr_region = unmap(vlr_region, num_total_anchors, inside_flags)
            im_region = unmap(im_region, num_total_anchors, inside_flags)

            labels_neg = unmap(
                labels_neg,
                num_total_anchors,
                inside_flags,
                fill=self.num_classes)
        #torch.set_printoptions(profile="full")
        #print("Size4 of im_region_list is: ", len(im_region))
        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, vlr_region, im_region)
    

        
    
    def get_im_region(self, bboxes, gt_bboxes, mode='decouple'):
        assert mode in ['gibox', 'finegrained', 'fitnet', 'decouple']
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        bboxes = bboxes[:, :4]
        #print("gt_bboxes is*********************************")
        #print(len(gt_bboxes))
        gt_bbox_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        iou_dotd = self.iou_calculator(bboxes, gt_bboxes, mode='iou')
        #print("iou_dotd.max(0)[0]")
        #print(iou_dotd.max(0)[0])
        
        
        # compute iou between all bbox and gt
        gt_flag = torch.zeros(bboxes.shape[0], device=bboxes.device)
        anchor_center = self.anchor_center(bboxes)
        for gt_bbox in gt_bboxes:
            width = gt_bbox[2] - gt_bbox[0]
            height = gt_bbox[3] - gt_bbox[1]
            width_dist_threshold = width/4
            height_dist_threshold = height/4
            in_gt_flag = torch.nonzero(
                (anchor_center[:, 0] > gt_bbox[0])
                & (anchor_center[:, 0] < gt_bbox[2])
                & (anchor_center[:, 1] > gt_bbox[1])
                & (anchor_center[:, 1] < gt_bbox[3]),
                as_tuple=False)
            in_near_gt_flag = torch.nonzero(
                (anchor_center[:, 0] > gt_bbox[0] - width_dist_threshold)
                & (anchor_center[:, 0] < gt_bbox[2] + width_dist_threshold)
                & (anchor_center[:, 1] > gt_bbox[1] - height_dist_threshold)
                & (anchor_center[:, 1] < gt_bbox[3] + height_dist_threshold),
                as_tuple=False)
            in_near_gt_flag = set(in_near_gt_flag.flatten().tolist()) - set(in_gt_flag.flatten().tolist()) # 去掉与GT框重合的anchor
            in_near_gt_flag = torch.tensor(list(in_near_gt_flag), dtype=torch.long)

            dists = torch.sqrt(((anchor_center[in_near_gt_flag] - gt_bbox[:2]) ** 2).sum(dim=1))
            if len(dists) > 0:
                dists = (dists - dists.min()) / (dists.max() - dists.min())
            else:
                dists = torch.zeros_like(dists)

            #gt_flag[in_near_gt_flag] = 1 - dists
            #in_gt_flag = in_gt_flag.squeeze()
            gt_flag[in_gt_flag] = 1
            #print("+++++++++++++")
            #print(gt_flag[in_gt_flag])
            #size = gt_flag[in_gt_flag].size()
            #print("Size of gt_flag[in_gt_flag] is: ", size)
            gt_flag[in_near_gt_flag] = torch.where(gt_flag[in_near_gt_flag] < (1 - dists), 1 - dists, gt_flag[in_near_gt_flag])
        size1 = gt_flag.size()
        #torch.set_printoptions(profile="full")
        #print("Size of gt_flag is: ", size1)
        '''
        with open('../gtflag.txt', 'a') as f:
                f.write("im_region:\n")
                f.write(str(gt_flag))
        print("over")
        '''
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),0,dtype=torch.long)
        assigned_fg = (assigned_gt_inds + 0).float()
        fine_grained = torch.nonzero((iou_dotd >=0.7 * iou_dotd.max(0)[0]))
        near_bg_fine_grained=torch.nonzero((iou_dotd>=0.5 * iou_dotd.max(0)[0]) & (iou_dotd<0.7 * iou_dotd.max(0)[0]))
        bg_fine_grained=torch.nonzero((iou_dotd <0.5 * iou_dotd.max(0)[0]) & (iou_dotd >= 0 * iou_dotd.max(0)[0]))
        #bg_fine_grained=torch.nonzero((iou_dotd <0.5 * iou_dotd.max(0)[0]))
        assigned_fg[bg_fine_grained[:, 0]] = 0.4
        assigned_fg[near_bg_fine_grained[:, 0]] = 1.5
        assigned_fg[fine_grained[:, 0]] = 2
        #注销下面一行即解除了重点区域
        #gt_flag=gt_flag * assigned_fg
        return gt_flag  
    
    
    def get_gi_region(self, soft_label, cls_score, anchors, bbox_pred,
                      soft_targets, stride, gt_flag):

        teacher_score = soft_label.detach().sigmoid()

        student_score = cls_score.detach().sigmoid()  # [num,80]
        #student_score = student_score *gt_flag
        #iou = self.ious(bbox_pred, anchors)
        #teacher_score = teacher_score * gt_flag
        
        
        anchor_centers = self.anchor_center(anchors) / stride[0]
        sdistribution = self.integral(bbox_pred)
        tdistribution = self.integral(soft_targets)
        sbox = distance2bbox(anchor_centers, sdistribution)  # [num,4]
        tbox = distance2bbox(anchor_centers, tdistribution)
        #iou = self.iou_calculator(bbox_pred, anchors, mode='iou')
        z = teacher_score - student_score  # difference between teacher score and student score on the whole locations.
        '''
        print("anchors is")
        print(len(anchors))
        print(anchors.size())
        print(anchors)
        print("z is")
        print(len(z))
        print(z.size())
        print(z)
        
        print("gt is")
        print(len(gt_flag))
        print(gt_flag.size())
        print(gt_flag)
        
        max_value = torch.max(gt_flag)
        print("gtmax is")
        print(max_value)
        '''
        giscore, index = torch.abs(z).max(dim=1)  # GI scores
        giscore = giscore * gt_flag
        #print("giscore is")
        #print(giscore.size())
        #print(giscore)
        k = z >= 0  # who is bigger
        j = torch.take(
            k, index + self.cls_out_channels *
               (torch.arange(student_score.size(0)).cuda()))
        h = j == 0
        gibox = sbox.new_zeros(sbox.shape)
        gibox[j] = tbox[j] + 0
        gibox[h] = sbox[h] + 0  # GI boxes
    
        idx_out = torch.ops.torchvision.nms(gibox, giscore, 0.3)[:10]
        #print("idx_out is")
        #print(idx_out.size())
        #print(idx_out)
        return idx_out
    
    def get_cls_region(self, soft_label, im_region):
        teacher_score = soft_label.detach().sigmoid()
        giscores, index = torch.abs(teacher_score).max(dim=1)  # GI scores
        #print("giscore size is")
        #print(giscore.size())
        #print("im_region size is")
        #print(im_region.size())
        act = int(giscores.size()[0]*0.025)
        if act < 10:
            top_100_values, top_100_indices=torch.topk(giscores, 5)
            #print("act<10,re is")
            #print(im_region[top_100_indices])
        else:
            #top_100_values, top_100_indices=torch.topk(giscore, k=int(giscore.size()[0]*0.025))
            top_100_values, top_100_indices=torch.topk(giscores, 10)
            #print("act>10,ress is")
            #print(im_region[top_100_indices])
        #print("the k is")
        #print(int(giscore.size()[0]*0.1))
        return top_100_indices
    
    def tfd_reg(self, soft_label, cls_score, anchors, bbox_pred,
                      soft_targets, stride,bbox_targets,pos_inds , gt_flag):
        
        anchor_centers = self.anchor_center(anchors) / stride[0]
        sdistribution = self.integral(bbox_pred)
        tdistribution = self.integral(soft_targets)
        sbox = distance2bbox(anchor_centers, sdistribution)  # [num,4]
        tbox = distance2bbox(anchor_centers, tdistribution)
        bbox_targets = bbox_targets/stride[0]  
        
        unique_elements, _ = torch.unique(bbox_targets, dim=0, return_inverse=True)
        iou_dotd_t = self.iou_calculator(tbox, unique_elements, mode='iou')
        
        #z = teacher_score - student_score  # difference between teacher score and student score on the whole locations.
        giscore, index = torch.abs(iou_dotd_t).max(dim=1)  # GI scores
        return giscore
    
    def tfd_cls(self, soft_label, im_region):
        teacher_score = soft_label.detach().sigmoid()
        giscores, index = torch.abs(teacher_score).max(dim=1)  # GI scores
        return giscores
import fnmatch
import math
import os
import sys
import time
from operator import itemgetter
import torch.nn.functional as F # Make sure F is imported at the top of load_data.py
import gc
import numpy as np
from numpy.random import randn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import camo_utils
import numpy as np
from arch.yolov3_models import YOLOv3Darknet



class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.loss_target = lambda obj, cls: obj
    
    def forward(self, output, gt, loss_type, iou_thresh):
        det_loss = []
        max_probs = []
        num = 0
        for i, boxes in enumerate(output):
            ious = torchvision.ops.box_iou(boxes['boxes'], gt[i].unsqueeze(0)).squeeze(1)
            mask = ious.ge(iou_thresh)
            if True:
                mask = mask.logical_and(boxes['labels'] == 1)
            ious = ious[mask]
            scores = boxes['scores'][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)


                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        if num < 1:
            raise RuntimeError()
        return det_loss, max_probs
    

class YOLOv3MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, model,figsize):
        super(YOLOv3MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize
        self.model = model

    # for v3 training output
    
    def forward(self, YOLOoutputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        box_all = utils.get_region_boxes_general(YOLOoutputs, self.model, conf_thresh=0.2, name="yolov3")
        for i in range(len(box_all)):
            boxes = box_all[i]
            assert boxes.shape[1] == 7
            # boxes = boxes.view(-1,7) # [x,y,w,h,obj_conf,cls_score,class_idx]
            w1 = boxes[...,0] - boxes[..., 2]/2
            h1 = boxes[...,1] - boxes[..., 3]/2
            w2 = boxes[...,0] + boxes[..., 2]/2
            h2 = boxes[...,1] + boxes[..., 3]/2
            bbox = torch.stack([w1,h1,w2,h2],dim=-1)
            ious = torchvision.ops.box_iou(bbox.view(-1,4).detach()*self.figsize,gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            if True:
                mask = mask.logical_and(boxes[...,6]==0)
            ious = ious[mask]
            scores = boxes[...,4][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)


                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1

                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        if num < 1:
            raise RuntimeError()
        return det_loss, max_probs

class DetrMaxProbExtractor(nn.Module):
    
    def __init__(self, cls_id, num_cls, figsize):
        super(DetrMaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize

    def forward(self, outputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        
        prob, labels = outputs['pred_logits'].softmax(dim=-1)[..., :-1].max(-1)
        batch = prob.shape[0]
        for i in range(batch):
            bbox = outputs['pred_boxes'][i]
            w1 = bbox[...,0] - bbox[..., 2]/2
            h1 = bbox[...,1] - bbox[..., 3]/2
            w2 = bbox[...,0] + bbox[..., 2]/2
            h2 = bbox[...,1] + bbox[..., 3]/2
            bboxes = torch.stack([w1,h1,w2,h2], dim=-1)
            ious = torchvision.ops.box_iou(bboxes.view(-1,4).detach()*self.figsize,gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            mask = mask.logical_and(labels[i] == 1)
            ious = ious[mask]
            scores = prob[i][mask]
            # logit = logits[i][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_logit':
                    _, ids = torch.max(ious,dim=0)
                    det_loss.append(logit[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)


                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1

                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        if num < 1:
            raise RuntimeError()
        return det_loss, max_probs

class DeformableDetrProbExtractor(nn.Module):

    def __init__(self, cls_id, num_cls, figsize):
        super(DeformableDetrProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.figsize = figsize

    def forward(self, outputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss = []
        num = 0
        logits = outputs['logits'][...,1]
        prob = F.softmax(outputs['logits'],dim=-1)[...,1]
        labels = torch.argmax(outputs['logits'],dim=-1)
        batch = prob.shape[0]
        for i in range(batch):
            bbox = outputs['pred_boxes'][i]
            w1 = bbox[...,0] - bbox[..., 2]/2
            h1 = bbox[...,1] - bbox[..., 3]/2
            w2 = bbox[...,0] + bbox[..., 2]/2
            h2 = bbox[...,1] + bbox[..., 3]/2
            bboxes = torch.stack([w1,h1,w2,h2],dim=-1)
            ious = torchvision.ops.box_iou(bboxes.view(-1,4).detach()*self.figsize,gt[i].unsqueeze(0)).squeeze(-1)
            mask = ious.ge(iou_thresh)
            mask = mask.logical_and(labels[i] == 1)
            ious = ious[mask]
            scores = prob[i][mask]
            logit = logits[i][mask]
            if len(ious) > 0:
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_iou_mtconf':
                    _, ids = torch.max(ious*scores,dim=0)
                    det_loss.append(scores[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_logit':
                    _, ids = torch.max(ious,dim=0)
                    det_loss.append(logit[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf':
                    det_loss.append(scores.max())
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_max':
                    max_conf = - torch.log(1.0 / scores.max() - 1.0)
                    max_conf = F.softplus(max_conf)
                    det_loss.append(max_conf)
                    max_probs.append(scores.max())
                    num += 1
                elif loss_type == 'softplus_sum':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious.detach()).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'max_iou_mtiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] * ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_mtiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) * ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_mtiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) * ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)


                elif loss_type == 'max_iou_adiou':
                    _, ids = torch.max(ious, dim=0) # get the bbox w/ biggest iou compared to gt
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'max_conf_adiou':
                    _, ids = torch.max(scores, dim=0)
                    det_loss.append(scores[ids] + ious[ids])
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_max_adiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + ious[ids]
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1
                elif loss_type == 'softplus_sum_adiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + ious).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                elif loss_type == 'softplus_max_adspiou':
                    _, ids = torch.max(scores, dim=0)
                    max_conf = - torch.log(1.0 / scores[ids] - 1.0)
                    max_conf = F.softplus(max_conf) + F.softplus(- torch.log(1.0 / ious[ids] - 1.0))
                    det_loss.append(max_conf)
                    max_probs.append(scores[ids])
                    num += 1

                elif loss_type == 'softplus_sum_adspiou':
                    max_conf = (F.softplus(- torch.log(1.0 / scores - 1.0)) + F.softplus(- torch.log(1.0 / ious - 1.0))).sum()
                    det_loss.append(max_conf)
                    max_probs.append(scores.mean())
                    num += len(scores)

                else:
                    raise ValueError
            else:
                det_loss.append(ious.new([0.0])[0])
                max_probs.append(ious.new([0.0])[0])
        det_loss = torch.stack(det_loss).mean()
        max_probs = torch.stack(max_probs)
        if num < 1:
            raise RuntimeError()
        return det_loss, max_probs

    
class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.02
        self.min_scale = -0.28  # log 0.75
        self.max_scale = 0.47  # log 1.60
        self.translation_x = 0.8
        self.translation_y = 1.0

    def forward(self, img_batch, adv_patch):
        # import matplotlib.pyplot as plt
        B, _, Ht, Wt = img_batch.shape
        _, _, Ho, Wo = adv_patch.shape
        adv_patch = adv_patch[:B]

        mask = (adv_patch[:, -1:, ...] > 0).to(adv_patch)
        adv_patch = adv_patch[:, :-1, ...]

        contrast = adv_patch.new(size=[B]).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        brightness = adv_patch.new(size=[B]).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        noise = adv_patch.new(adv_patch.shape).uniform_(-1, 1) * self.noise_factor

        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch = adv_patch.clamp(0, 1)
        adv_patch = torch.cat([adv_patch, mask], dim=1)

        scale = adv_patch.new(size=[B]).uniform_(self.min_scale, self.max_scale).exp()
        mesh_bord = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        mesh_bord = mesh_bord / mesh_bord.new([Ho, Wo, Ho, Wo]) * 2 - 1
        #         mesh_bord = mesh_bord / scale
        pos_param = mesh_bord + mesh_bord.new([1, 1, -1, -1]) * scale.unsqueeze(-1)
        tymin, txmin, tymax, txmax = pos_param.unbind(-1)

        xdiff = (-txmax + txmin).clamp(min=0)
        xmiddle = (txmax + txmin) / 2
        ydiff = (-tymax + tymin).clamp(min=0)
        ymiddle = (tymax + tymin) / 2

        tx = txmin.new(txmin.shape).uniform_(-0.5, 0.5) * xdiff * self.translation_x + xmiddle
        ty = tymin.new(tymin.shape).uniform_(-0.5, 0.5) * ydiff * self.translation_y + ymiddle

        theta = adv_patch.new_zeros(B, 2, 3)
        theta[:, 0, 0] = scale
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, img_batch.shape)
        adv_batch = F.grid_sample(adv_patch, grid, padding_mode='zeros')
        mask = adv_batch[:, -1:]
        adv_batch = adv_batch[:, :-1] * mask + img_batch * (1 - mask)

        gt = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
        gt = gt[:, [1, 0, 3, 2]].unbind(0)
        return adv_batch, gt

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, imgsize, shuffle=True, if_square=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        # n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        # assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        # self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        self.if_square = if_square
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        # lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')


        image = self.pad_and_scale(image)
        transform = transforms.ToTensor()
        image = transform(image)
        # label = self.pad_lab(label)
        return image

    def pad_and_scale(self, img):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w==h:
            padded_img = img
        elif self.if_square:
            a = min(w, h)
            ww = (w - a) // 2
            hh = (h - a) // 2
            padded_img = img.crop([ww, hh, ww+a, hh+a])
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                # lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                # lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                # lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                # lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize, self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab
    
class YOLOv3InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab
        #self.batch_count = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        YOLOv2label = self.pad_lab(label.clone())
        bb_targets = torch.zeros((len(label), 6))
        bb_targets[:, 1:] = label
        return image, YOLOv2label, bb_targets

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab
    
    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab
    
    def collate_fn(self, batch):
        #self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, YOLOv2label, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        #if self.multiscale and self.batch_count % 10 == 0:
        #    self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([img for img in imgs])
        YOLOv2label = torch.stack([lab for lab in YOLOv2label])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, YOLOv2label, bb_targets



class Tshirt_dataset(Dataset):
    def __init__(self, img_dir="data/Tshirts_fashion_clean/", shuffle=True, size = 256):
        super(Tshirt_dataset,self).__init__()
        self.shuffle = shuffle
        self.size = size
        self.path = []
        names = os.listdir(img_dir)
        self.len = len(names)
        for i in names:
            self.path.append(img_dir+i)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_dir = self.path[idx]
        img = Image.open(img_dir).convert('RGB')
        transform = transforms.ToTensor()
        img = transform(img)
        # print(img.type)
        resize = transforms.Resize((self.size,self.size))
        img = resize(img)
        return img

class landscape_dataset(Dataset):
    def __init__(self, shuffle=True):
        super(landscape_dataset,self).__init__()
        self.len = 2192
        self.shuffle = shuffle
        self.path = []
        for i in range(self.len):
            self.path.append("data/landscape/"+str(i)+".jpg")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_dir = self.path[idx]
        img = Image.open(img_dir).convert('RGB')
        transform = transforms.ToTensor()
        img = transform(img)
        return img

# In load_data.py, add this new class. The rest of the file can remain the same.

# In load_data.py, find the old YOLOv5MaxProbExtractor class and REPLACE it with this one.


# ... (other classes like MaxProbExtractor, YOLOv3MaxProbExtractor, etc.) ...

class YOLOv5MaxProbExtractor(nn.Module):
    """
    Extracts max class probability for a class from YOLOv5 output.
    This class is designed to work with the raw output from the YOLOv5 model's detection head,
    which is required for backpropagation during the adversarial attack.
    """
# In train_yolov5.py, inside the YOLOv5MaxProbExtractor class definition

    def __init__(self, cls_id, num_cls, img_size, model):
        super(YOLOv5MaxProbExtractor, self).__init__()
        print("\n--- SUCCESS: Initializing LOCALLY DEFINED YOLOv5MaxProbExtractor (v6) ---\n")
        self.cls_id, self.num_cls, self.img_size = cls_id, num_cls, img_size
        
        # --- ROBUST METHOD TO FIND THE 'Detect' LAYER ---
        # The 'model' passed in is the AutoShape wrapper.
        # Its '.model' attribute is the DetectionModel.
        detection_model = model.model
        
        # The last module of the DetectionModel is the Detect layer.
        # We can access it by getting the last module from its children.
        # This is more robust than hardcoding indices.
        detect_layer = list(detection_model.modules())[-1]

        # A quick check to make sure we found the right thing
        from yolov5.models.yolo import Detect
        if not isinstance(detect_layer, Detect):
             raise TypeError(f"Last layer is not a Detect module, but {type(detect_layer)}. Model structure may have changed.")
        # ----------------------------------------------------

        self.strides = detect_layer.stride.to(model.device)
        self.anchors = detect_layer.anchors.clone().detach().to(model.device)
        self.num_anchors, self.num_layers = self.anchors.shape[1], self.anchors.shape[0]

    def _make_grid(self, nx=20, ny=20, i=0):
        # Creates a grid for decoding predictions
        d = self.anchors.device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, self.num_anchors, ny, nx, 2).float()
        anchor_grid = (self.anchors[i].clone() * self.strides[i]) \
            .view(1, self.num_anchors, 1, 1, 2).expand(1, self.num_anchors, ny, nx, 2).float()
        return grid, anchor_grid

    def forward(self, yolo_outputs, gt, loss_type, iou_thresh):
        max_probs = []
        det_loss_batch = []
        num_detections = 0
        
        all_boxes_batch = [[] for _ in range(gt.shape[0])]

        # yolo_outputs is a list of 3 tensors from the detection head
        for i, p in enumerate(yolo_outputs):
            bs, na, ny, nx, _ = p.shape # batch_size, anchors, grid, grid, 5+num_classes
            p = p.sigmoid()
            
            grid, anchor_grid = self._make_grid(nx, ny, i)
            
            # Decode boxes
            xy = (p[..., 0:2] * 2. - 0.5 + grid) * self.strides[i]
            wh = (p[..., 2:4] * 2) ** 2 * anchor_grid
            
            x1y1 = xy - wh / 2
            x2y2 = xy + wh / 2
            bbox = torch.cat((x1y1, x2y2), -1)

            # Use objectness confidence for the attack
            conf = p[..., 4]
            # Get the confidence for the target class (person)
            class_probs = p[..., 5:]
            class_conf = class_probs[..., self.cls_id]

            # The score for the attack is obj_conf * class_conf
            scores = conf * class_conf
            
            # Reshape for batch processing
            bbox = bbox.view(bs, -1, 4)
            scores = scores.view(bs, -1)

            for batch_idx in range(bs):
                # We don't need to filter by class anymore, since 'scores' is already class-specific
                batch_boxes = bbox[batch_idx]
                batch_scores = scores[batch_idx]
                
                if batch_boxes.shape[0] > 0:
                    all_boxes_batch[batch_idx].append(torch.cat([batch_boxes, batch_scores.unsqueeze(-1)], dim=-1))

        # Process each image in the batch
        for i in range(gt.shape[0]):
            if not all_boxes_batch[i]:
                det_loss_batch.append(gt.new_zeros(1)[0])
                max_probs.append(gt.new_zeros(1)[0])
                continue

            all_boxes_img = torch.cat(all_boxes_batch[i], dim=0)
            bboxes_img = all_boxes_img[:, :4]
            scores_img = all_boxes_img[:, 4]
            
            ious = torchvision.ops.box_iou(bboxes_img.detach(), gt[i].unsqueeze(0)).squeeze(1)
            mask = ious.ge(iou_thresh)
            
            ious_filtered = ious[mask]
            scores_filtered = scores_img[mask]
            
            if len(ious_filtered) > 0:
                num_detections += len(scores_filtered)
                if loss_type == 'max_iou':
                    _, ids = torch.max(ious_filtered, dim=0)
                    det_loss = scores_filtered[ids]
                    max_prob = scores_filtered[ids]
                elif loss_type == 'max_conf':
                    det_loss = scores_filtered.max()
                    max_prob = scores_filtered.max()
                elif loss_type == 'softplus_max':
                    max_conf_logit = -torch.log(1.0 / (scores_filtered.max() + 1e-9) - 1.0)
                    det_loss = F.softplus(max_conf_logit)
                    max_prob = scores_filtered.max()
                elif loss_type == 'softplus_sum':
                    logits = -torch.log(1.0 / (scores_filtered + 1e-9) - 1.0)
                    det_loss = (F.softplus(logits) * ious_filtered.detach()).sum()
                    max_prob = scores_filtered.mean()
                else:
                    raise ValueError(f"Unsupported loss_type for YOLOv5: {loss_type}")
                
                det_loss_batch.append(det_loss)
                max_probs.append(max_prob)
            else:
                det_loss_batch.append(gt.new_zeros(1)[0])
                max_probs.append(gt.new_zeros(1)[0])
                
        if num_detections < 1:
            raise RuntimeError("No detections found for any image in the batch meeting the IoU threshold.")
            
        final_det_loss = torch.stack(det_loss_batch).mean()
        final_max_probs = torch.stack(max_probs)
        
        return final_det_loss, final_max_probs
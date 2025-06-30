"""
Training code for Adversarial Patch Training on YOLOv5.
This version uses an explicitly targeted loss function necessary for the YOLOv5 architecture.
"""
# --- FIX: Add parent directory to Python path ---
# This allows the script to find and import from the 'yolov5' directory.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---------------------------------------------

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import PIL
import time
import subprocess
import warnings
warnings.simplefilter("ignore", FutureWarning)


# --- IMPORTS FROM YOUR PROJECT ---
from load_data import InriaDataset, PatchTransformer, PatchApplier, NPSCalculator, TotalVariation
import patch_config

# --- UTILITY FOR IoU CALCULATION from the yolov5 repository ---
# This is essential for our targeted loss function.


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

class PatchTrainer(object):
    def __init__(self, mode_config):
        self.config = mode_config

        print("Setting up YOLOv5 model...")
        # --- FIX: Use the correct path to the local repository ---
        self.yolo_model = torch.hub.load('../yolov5', 'custom', path=self.config.weightfile, source='local')
        self.yolo_model = self.yolo_model.cuda().eval()
        
        for param in self.yolo_model.parameters():
            param.requires_grad = False
        
        print("Model loaded successfully.")

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.writer = self.init_tensorboard(self.config.patch_name)

    def init_tensorboard(self, name=None):
        try:
            subprocess.Popen(['tensorboard', '--logdir=runs'])
        except FileNotFoundError:
            print("Tensorboard not found. Continuing without it.")
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def calculate_v5_loss(self, pred_raw, gt_labels):
        """
        Calculates a targeted detection loss by only considering predictions
        that overlap with the ground-truth locations of people.
        """
        if pred_raw.dim() == 2:
            pred_raw = pred_raw.unsqueeze(0)
        
        pred_nms = non_max_suppression(pred_raw, conf_thres=0.01, iou_thres=0.5)

        det_loss = torch.tensor(0.0).cuda()
        for i in range(pred_raw.size(0)):
            pred_img_nms = pred_nms[i]
            gt_boxes_img = gt_labels[i][gt_labels[i, :, 0] == 0]

            if gt_boxes_img.size(0) == 0 or pred_img_nms is None or pred_img_nms.size(0) == 0:
                continue

            gt_boxes_xyxy = xywh2xyxy(gt_boxes_img[:, 1:5])
            pred_boxes_xyxy = pred_img_nms[:, :4]
            
            iou = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
            max_iou, _ = torch.max(iou, dim=1)
            
            overlapping_preds_mask = max_iou > 0.05
            
            if overlapping_preds_mask.sum() == 0:
                continue

            conf_scores_of_interest = pred_img_nms[overlapping_preds_mask, 4]
            max_conf_for_img = torch.max(conf_scores_of_interest)
            det_loss += max_conf_for_img

        return det_loss / pred_raw.size(0)

    def train(self):
        img_size = (self.config.image_size, self.config.image_size)
        print(f"Using image size: {img_size}")
        
        n_epochs = 1000
        time_str = time.strftime("%Y%m%d-%H%M%S")

        adv_patch_cpu = self.generate_patch("random")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, 14, img_size[0], shuffle=True),
            batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.epoch_length = len(train_loader)
        print(f'One epoch is {self.epoch_length} batches')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        for epoch in range(n_epochs):
            ep_det_loss, ep_nps_loss, ep_tv_loss, ep_loss = 0, 0, 0, 0
            
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                img_batch, lab_batch = img_batch.cuda(), lab_batch.cuda()
                
                adv_patch = adv_patch_cpu.cuda()
                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size[0], do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                
                output = self.yolo_model(p_img_batch, augment=False)
                raw_pred = output[0] if isinstance(output, tuple) else output
                
                det_loss = self.calculate_v5_loss(raw_pred, lab_batch)
                nps = self.nps_calculator(adv_patch)
                tv = self.total_variation(adv_patch)

                nps_loss = nps * self.config.nps_weight
                tv_loss = tv * self.config.tv_weight
                loss = det_loss + nps_loss + tv_loss

                ep_det_loss += det_loss.item()
                ep_nps_loss += nps_loss.item()
                ep_tv_loss += tv_loss.item()
                ep_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                adv_patch_cpu.data.clamp_(0, 1)

                if i_batch % 10 == 0 and self.writer is not None:
                    iteration = self.epoch_length * epoch + i_batch
                    self.writer.add_scalar('total_loss', loss.item(), iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
                    self.writer.add_scalar('loss/nps_loss', nps_loss.item(), iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss.item(), iteration)
                    self.writer.add_image('patch', adv_patch_cpu, iteration)

            ep_loss /= self.epoch_length
            ep_det_loss /= self.epoch_length
            
            scheduler.step(ep_loss)
            
            print(f'\n--- EPOCH {epoch} SUMMARY ---')
            print(f'  EPOCH LOSS: {ep_loss:.4f} (Det: {ep_det_loss:.4f}, NPS: {ep_nps_loss/self.epoch_length:.4f}, TV: {ep_tv_loss/self.epoch_length:.4f})')
            
            patch_filename = f"saved_patches/{time_str}_{self.config.patch_name}_{epoch}.png"
            transforms.ToPILImage()(adv_patch_cpu).save(patch_filename)
            print(f"  Patch saved to {patch_filename}")

    def generate_patch(self, type):
        if type == 'gray':
            return torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            return torch.rand((3, self.config.patch_size, self.config.patch_size))

def main():
    if len(sys.argv) != 2:
        print('Usage: python train_patch_v5.py [config_mode]')
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode not in patch_config.patch_configs:
        print(f"Error: Mode '{mode}' not found in patch_config.py")
        sys.exit(1)
        
    trainer = PatchTrainer(patch_config.patch_configs[mode]())
    trainer.train()

if __name__ == '__main__':
    main()


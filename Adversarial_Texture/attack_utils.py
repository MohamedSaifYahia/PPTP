import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from yolo2 import utils

def truths_length(truths):
    for i in range(len(truths)):
        if truths[i][1] == -1:
            return i
    return len(truths)

def get_det_loss(darknet_model, p_img, lab_batch, args, kwargs):
    valid_num = 0
    det_loss = p_img.new_zeros([])
    output = darknet_model(p_img)
    if kwargs['name'] == 'yolov2':
        all_boxes_t = [utils.get_region_boxes_general(output, darknet_model, conf_thresh=args.conf_thresh, name=kwargs['name'])]
    else:
        raise ValueError

    for all_boxes in all_boxes_t:
        for ii in range(p_img.shape[0]):
            if all_boxes[ii].shape[0] > 0:
                iou_mat = utils.bbox_iou_mat(all_boxes[ii][..., :4], lab_batch[ii][:truths_length(lab_batch[ii]), 1:], False)
                iou_max = iou_mat.max(1)[0]
                idxs = iou_max > args.iou_thresh
                det_confs = all_boxes[ii][idxs][:, 4]
                if det_confs.shape[0] > 0:
                    max_prob = det_confs.max()
                    det_loss = det_loss + max_prob
                    valid_num += 1

    return det_loss, valid_num

def gauss_kernel(ksize=5, sigma=None, conv=False, dtype=np.float32):
    half = (ksize - 1) * 0.5
    if sigma is None:
        sigma = 0.3 * (half - 1) + 0.8
    x = np.arange(-half, half + 1)
    x = np.exp(- np.square(x / sigma) / 2)
    x = np.outer(x, x)
    x = x / x.sum()
    if conv:
        kernel = np.zeros((3, 3, ksize, ksize))
        for i in range(3):
            kernel[i, i] = x
    else:
        kernel = x
    return kernel.astype(dtype)


def pad_and_scale(img, lab=None, size=(416, 416), color=(127, 127, 127)):
    """

    Args:
        img:

    Returns:

    """
    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=color)
            padded_img.paste(img, (int(padding), 0))
            if lab is not None:
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=color)
            padded_img.paste(img, (0, int(padding)))
            if lab is not None:
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
    padded_img = padded_img.resize((size[0], size[1]))
    if lab is None:
        return padded_img
    else:
        return padded_img, lab


def random_crop(cloth, crop_size, pos=None, crop_type=None, fill=0):
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size is 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]

        p1 = max(0, 0 - r_h)
        p2 = max(0, r_h + crop_size[1] - h)
        p3 = max(0, 0 - r_w)
        p4 = max(0, r_w + crop_size[1] - w)
        cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
        patch = cloth_pad[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h


def random_stick(inputs, patch, stick_size=None, mode='replace', pos=None):
    if stick_size is None:
        stick_size = patch.shape[2:4]
    w = inputs.shape[2]
    h = inputs.shape[3]
    d_w = w - stick_size[0]
    d_h = h - stick_size[1]
    if pos is None:
        r_w = np.random.randint(d_w + 1)
        r_h = np.random.randint(d_h + 1)
    elif pos == 'center':
        r_w, r_h = (np.array(inputs.shape[2:]) - np.array(stick_size)) // 2
    else:
        r_w = pos[0]
        r_h = pos[1]

    patch_stick = inputs.new_zeros(inputs.shape)
    patch_resized = patch
    patch_stick[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = patch_resized

    assert mode in ['add', 'replace']

    if mode == 'add':
        inputs_stick = (inputs + patch_stick).clamp(0, 1)
    #         return inputs_stick

    elif mode == 'replace':
        mask = inputs.new_zeros(inputs.shape)
        mask[:, :, r_w:r_w + stick_size[0], r_h:r_h + stick_size[1]] = 1
        inputs_stick = mask * patch_stick + (1 - mask) * inputs
        inputs_stick = inputs_stick.clamp(0, 1)
    else:
        inputs_stick = None

    return inputs_stick, r_w, r_h


def TVLoss(patch):

    t1 = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().sum()
    t2 = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().sum()

    tv = t1 + t2

    return tv / patch.numel()

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
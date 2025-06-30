import torch
import torch.nn.functional as F

class MaxProbExtractor:
    """
    Extracts the maximum probability of a given class from the YOLOv5 output.
    The attack goal is to minimize this probability.
    """
    def __init__(self, target_cls_id, num_classes, model):
        self.target_cls_id = target_cls_id
        self.num_classes = num_classes
        self.loss_fn = model.loss if hasattr(model, 'loss') else None

    def __call__(self, yolov5_output):
        # yolov5_output is the raw output tensor: (batch_size, num_anchors, 5+num_classes)
        # Columns: [x, y, w, h, objectness, class_0, class_1, ...]
        
        # Ensure output is on the correct device
        output = yolov5_output[0] if isinstance(yolov5_output, tuple) else yolov5_output
        
        # Get objectness and class probabilities
        objectness = output[..., 4]
        cls_probs = output[..., 5:]
        
        # Get the probability of the target class
        target_cls_prob = cls_probs[..., self.target_cls_id]
        
        # The score to minimize is objectness * target_class_probability
        # This is what the detector uses to decide if an object is present
        detection_score = objectness * target_cls_prob
        
        # We want to minimize the maximum detection score
        max_score = torch.max(detection_score)
        
        return max_score

def get_det_loss(model, patched_images, labels, prob_extractor):
    """
    Calculates the detection loss for the adversarial attack.
    For YOLOv5, this is the maximum probability of detecting the target class.
    
    Args:
        model: The YOLOv5 model.
        patched_images: The batch of images with the adversarial patch applied.
        labels: Ground truth labels (not directly used for this loss formulation, but kept for API consistency).
        prob_extractor: The MaxProbExtractor instance.

    Returns:
        det_loss: The scalar loss value to be minimized.
        valid_batches: Number of batches used (always 1 here).
    """
    # Get raw model output (before NMS)
    # We pass the images through the model's 'model' attribute to get raw predictions
    predictions = model.model(patched_images)[0] # The first element of the tuple is the prediction tensor
    
    # Use the probability extractor to get the loss
    det_loss = prob_extractor(predictions)
    
    # We count every batch as valid for this simplified loss
    valid_batches = patched_images.size(0)

    return det_loss, valid_batches

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
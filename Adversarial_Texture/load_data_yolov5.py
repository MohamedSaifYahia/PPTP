"""
Modified data loading module for YOLOv5 compatibility
Adapts the original load_data module to work with YOLOv5's preprocessing requirements
"""

import torch
import torch.nn as nn
from yolo2.load_data import *  # Import all original functionality

# Override or add any YOLOv5-specific modifications here
# Most of the original load_data functionality should work as-is

class MaxProbExtractorYOLOv5(nn.Module):
    """MaxProbExtractor modified for YOLOv5 output format"""
    def __init__(self, cls_id, num_cls, target_func, model_name='yolov5'):
        super(MaxProbExtractorYOLOv5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.target_func = target_func
        self.model_name = model_name

    def forward(self, output):
        # For YOLOv5, we need to handle the output differently
        # YOLOv5 outputs are already processed through the Detect layer
        # We'll extract the person class probabilities
        
        if isinstance(output, list):
            output = output[0]  # Take main output
        
        # Extract objectness and class scores
        if len(output.shape) == 3:  # [batch, num_detections, 85]
            obj_conf = output[..., 4]
            cls_conf = output[..., 5 + self.cls_id]  # Person class
            combined_conf = obj_conf * cls_conf
        else:
            # Handle raw model outputs if needed
            combined_conf = output
        
        # Get max confidence per image
        batch_size = combined_conf.shape[0]
        max_probs = []
        
        for i in range(batch_size):
            if len(combined_conf.shape) > 2:
                conf = combined_conf[i].reshape(-1)
            else:
                conf = combined_conf[i]
            
            if len(conf) > 0:
                max_probs.append(conf.max())
            else:
                max_probs.append(torch.tensor(0.0, device=combined_conf.device))
        
        return torch.stack(max_probs)


# You can add more YOLOv5-specific modifications here if needed
# The existing PatchApplier, PatchTransformer, etc. should work fine with YOLOv5
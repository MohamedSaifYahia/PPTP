import sys
import time
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", FutureWarning)


# Import the necessary components from our project
from load_data import PatchTransformer, PatchApplier

# --- HELPER FUNCTION FOR YOLOv5 DETECTION ---
def detect_yolov5(model, img, img_size):
    """
    Runs detection on a single image using a YOLOv5 model.

    Args:
        model: The loaded YOLOv5 model.
        img: A PIL Image.
        img_size: The size to which the image will be resized for detection.

    Returns:
        A list of detection boxes in the format:
        [x_center, y_center, width, height, confidence, class_confidence, class_id]
        Coordinates are normalized to the range [0, 1].
    """
    # The model expects a list of images, so we wrap our single image
    results = model([img], size=img_size)

    # Detections for the first (and only) image
    # .xyxy[0] gives detections in [xmin, ymin, xmax, ymax, conf, class] format
    detections = results.xyxy[0].cpu().numpy()

    boxes = []
    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det

            # Convert from [x1, y1, x2, y2] to [x_center, y_center, width, height]
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2

            # Normalize coordinates to [0, 1]
            x_center /= img_size
            y_center /= img_size
            width /= img_size
            height /= img_size
            
            # The original format had two confidence scores. YOLOv5 provides one. We'll use it for both.
            class_confidence = conf

            boxes.append([x_center, y_center, width, height, conf, class_confidence, cls_id])
    
    return boxes


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_patch_v5.py <path_to_patch_file>")
        sys.exit(1)

    print("Setting everything up...")
    
    # --- Configuration ---
    imgdir = "inria/Test/pos"
    weightfile = "../yolov5s.pt"  # Path to your YOLOv5 weights
    patchfile = sys.argv[1]       # Get patch file from command line
    savedir = "testing_v5"        # Use a new directory for YOLOv5 results
    img_size = 640                # Standard YOLOv5 input size
    patch_size = 300              # Should match the size of the patch you trained
    conf_threshold = 0.4          # Confidence threshold for detections

    # --- Create Output Directories ---
    os.makedirs(os.path.join(savedir, 'clean', 'yolo-labels'), exist_ok=True)
    os.makedirs(os.path.join(savedir, 'proper_patched', 'yolo-labels'), exist_ok=True)
    os.makedirs(os.path.join(savedir, 'random_patched', 'yolo-labels'), exist_ok=True)

    # --- Load YOLOv5 Model ---
    print("Loading YOLOv5 model...")
    model = torch.hub.load('../yolov5', 'custom', path=weightfile, source='local')
    model = model.cuda().eval()
    print("Model loaded.")

    # --- Load Helpers and Patch ---
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size, patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch = tf(patch_img).cuda()

    clean_results = []
    noise_results = []
    patch_results = []
    
    print("Setup complete. Starting evaluation...")
    
    image_files = [f for f in os.listdir(imgdir) if f.endswith('.jpg') or f.endswith('.png')]
    for imgfile in tqdm(image_files, desc="Evaluating Images"):
        name = os.path.splitext(imgfile)[0]
        
        # --- 1. Process Clean Image ---
        img_path = os.path.join(imgdir, imgfile)
        img = Image.open(img_path).convert('RGB')

        # Pad and resize image
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) // 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (padding, 0))
            else:
                padding = (w - h) // 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, padding))
        
        padded_img = padded_img.resize((img_size, img_size))
        
        # Save clean processed image
        cleanname = name + ".png"
        padded_img.save(os.path.join(savedir, 'clean', cleanname))
        
        # Detect on clean image to get ground truth locations for patch
        boxes = detect_yolov5(model, padded_img, img_size)
        
        # Write clean detections to label file and results list
        txtpath = os.path.join(savedir, 'clean', 'yolo-labels', name + '.txt')
        with open(txtpath, 'w+') as textfile:
            for box in boxes:
                if box[4] > conf_threshold and int(box[6]) == 0: # Check confidence and if it's a person (class 0)
                    x_c, y_c, w_b, h_b = box[0], box[1], box[2], box[3]
                    textfile.write(f'{int(box[6])} {x_c} {y_c} {w_b} {h_b}\n')
                    # --- FIX: Convert numpy float32 to standard Python float for JSON ---
                    clean_results.append({
                        'image_id': name,
                        'bbox': [float(x_c - w_b / 2), float(y_c - h_b / 2), float(w_b), float(h_b)],
                        'score': float(box[4]),
                        'category_id': 1
                    })

        # --- 2. Process Adversarial Patched Image ---
        # Load the saved labels to know where to apply the patch
        if os.path.getsize(txtpath):
            label = np.loadtxt(txtpath)
        else:
            continue # Skip if no person was detected in the clean image

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        # Apply patch
        img_tensor = transforms.ToTensor()(padded_img).cuda()
        img_fake_batch = img_tensor.unsqueeze(0)
        lab_fake_batch = label.unsqueeze(0).cuda()
        adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=False, rand_loc=False)
        p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        p_img_pil = transforms.ToPILImage('RGB')(p_img_batch.squeeze(0).cpu())
        
        # Save patched image
        patched_name = name + "_p.png"
        p_img_pil.save(os.path.join(savedir, 'proper_patched', patched_name))

        # Detect on patched image
        patched_boxes = detect_yolov5(model, p_img_pil, img_size)
        
        # Write patched detections
        txtpath_p = os.path.join(savedir, 'proper_patched', 'yolo-labels', name + '_p.txt')
        with open(txtpath_p, 'w+') as textfile:
            for box in patched_boxes:
                if box[4] > conf_threshold and int(box[6]) == 0:
                    x_c, y_c, w_b, h_b = box[0], box[1], box[2], box[3]
                    textfile.write(f'{int(box[6])} {x_c} {y_c} {w_b} {h_b}\n')
                    # --- FIX: Convert numpy float32 to standard Python float for JSON ---
                    patch_results.append({
                        'image_id': name,
                        'bbox': [float(x_c - w_b / 2), float(y_c - h_b / 2), float(w_b), float(h_b)],
                        'score': float(box[4]),
                        'category_id': 1
                    })

        # --- 3. Process Random Patched Image ---
        random_patch = torch.rand(adv_patch.size()).cuda()
        adv_batch_t_random = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=False, rand_loc=False)
        p_img_batch_random = patch_applier(img_fake_batch, adv_batch_t_random)
        p_img_pil_random = transforms.ToPILImage('RGB')(p_img_batch_random.squeeze(0).cpu())

        # Save random patched image
        random_name = name + "_rdp.png"
        p_img_pil_random.save(os.path.join(savedir, 'random_patched', random_name))

        # Detect on random patched image
        random_boxes = detect_yolov5(model, p_img_pil_random, img_size)

        # Write random patched detections
        txtpath_r = os.path.join(savedir, 'random_patched', 'yolo-labels', name + '_rdp.txt')
        with open(txtpath_r, 'w+') as textfile:
            for box in random_boxes:
                if box[4] > conf_threshold and int(box[6]) == 0:
                    x_c, y_c, w_b, h_b = box[0], box[1], box[2], box[3]
                    textfile.write(f'{int(box[6])} {x_c} {y_c} {w_b} {h_b}\n')
                    # --- FIX: Convert numpy float32 to standard Python float for JSON ---
                    noise_results.append({
                        'image_id': name,
                        'bbox': [float(x_c - w_b / 2), float(y_c - h_b / 2), float(w_b), float(h_b)],
                        'score': float(box[4]),
                        'category_id': 1
                    })
    
    # --- Save Final Results ---
    print("Saving final results to JSON files...")
    with open(os.path.join(savedir, 'clean_results.json'), 'w') as fp:
        json.dump(clean_results, fp, indent=4)
    with open(os.path.join(savedir, 'noise_results.json'), 'w') as fp:
        json.dump(noise_results, fp, indent=4)
    with open(os.path.join(savedir, 'patch_results.json'), 'w') as fp:
        json.dump(patch_results, fp, indent=4)
    
    print("Evaluation finished.")
    print(f"Clean detections: {len(clean_results)}")
    print(f"Patched detections: {len(patch_results)}")
    print(f"Random patch detections: {len(noise_results)}")
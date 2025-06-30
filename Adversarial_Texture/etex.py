import os
import torch
import itertools
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnmatch
import re
import numpy as np

# Assuming original utils are in the same directory or accessible via PYTHONPATH
from yolo2 import load_data # Keep using the original data loader and patch applier
from attack_utils import random_crop, non_max_suppression
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis

# Import new YOLOv5-specific utilities
from yolov5_utils import bbox_iou


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='YOLOv5 Adversarial Texture Evaluation')
parser.add_argument('--net', default='yolov5s', help='target net name (e.g., yolov5s, yolov5m)')
parser.add_argument('--method', default='TCEGA', help='method name for evaluation')
parser.add_argument('--suffix', default=None, help='suffix for saving results')
parser.add_argument('--device', default='cuda:0', help='device to use (e.g., cuda:0, cpu)')
parser.add_argument('--prepare_data', action='store_true', help='Generate test labels using the clean YOLOv5 model')
parser.add_argument('--load_path', default=None, help='Path to the trained patch (.npy) or generator (.pkl)')
parser.add_argument('--load_path_z', default=None, help='Path to the trained latent vector z (.npy)')
parser.add_argument('--npz_dir', default=None, help='Directory with .npz results to plot a combined PR curve')
pargs = parser.parse_args()


# --- Configuration and Setup ---
args, kwargs = get_cfgs(pargs.net, pargs.method, 'test')
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device(pargs.device)
unloader = transforms.ToPILImage()
PERSON_CLASS_ID = 0

# --- Load YOLOv5 Model ---
print("Loading YOLOv5 model...")
yolov5_model = torch.hub.load('ultralytics/yolov5', pargs.net, pretrained=True, _verbose=False)
yolov5_model = yolov5_model.to(device).eval()
print("YOLOv5 model loaded successfully.")

patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)

# --- Data Preparation Step (if requested) ---
if pargs.prepare_data:
    print("--- Preparing Test Data using YOLOv5 ---")
    conf_thresh = 0.5  # Confidence threshold for generating pseudo-labels
    iou_thresh = 0.45  # IoU threshold for NMS
    img_ori_dir = './data/INRIAPerson/Test/pos'
    img_dir = './data/test_padded'
    lab_dir = f'./data/test_lab_{pargs.net}'
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], args.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=args.batch_size, shuffle=False, num_workers=8)

    with torch.no_grad():
        for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl), desc="Generating test labels"):
            data = data.to(device)
            
            # --- START OF THE FIX ---
            # 1. Get the raw prediction tensor from the model
            #    The output is a tuple, we need the first element.
            raw_pred = yolov5_model.model(data)[0]

            # 2. Apply Non-Max Suppression (NMS)
            #    This function processes the raw output into a list of detections per image.
            #    The format of each detection is [x1, y1, x2, y2, conf, cls].
            results = non_max_suppression(raw_pred, conf_thres=conf_thresh, iou_thres=iou_thresh)
            # --- END OF THE FIX ---

            for i in range(data.size(0)):
                # `results` is now a list of tensors, one for each image in the batch.
                detections = results[i]
                if detections is None or len(detections) == 0:
                    # If no objects are detected, create an empty file
                    open(os.path.join(lab_dir, labs[i]), 'w').close()
                    continue

                detections = detections.cpu().numpy()
                person_detections = detections[detections[:, 5] == PERSON_CLASS_ID]
                
                # We need to convert from [x1, y1, x2, y2, conf, cls] to the required
                # label format: [cls, x_center, y_center, width, height] (normalized)
                h, w = args.img_size, args.img_size
                x1, y1 = person_detections[:, 0], person_detections[:, 1]
                x2, y2 = person_detections[:, 2], person_detections[:, 3]
                
                # Calculate normalized center coordinates, width, and height
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Combine into the final label format
                # We need the class ID and the confidence for the final label
                final_labels = np.column_stack([
                    person_detections[:, 5],  # class id
                    x_center,
                    y_center,
                    width,
                    height
                ])
                
                # Save labels
                label_path = os.path.join(lab_dir, labs[i])
                np.savetxt(label_path, final_labels, fmt='%f')

                # Save the padded image
                img = unloader(data[i].detach().cpu())
                img_path = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                img.save(img_path)
    print("--- Test Data Preparation Done ---")

# --- Load Test Data ---
img_dir_test = './data/test_padded'
lab_dir_test = f'./data/test_lab_{pargs.net}'
test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, kwargs['max_lab'], args.img_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)
print(f'Evaluation loader has {len(test_loader)} batches.')

def truths_length(truths):
    for i in range(50):
        if truths[i][1] == -1:
            return i
    return 50

def test(model, loader, adv_cloth=None, gan=None, z=None, method='RCA', conf_thresh=0.01, iou_thresh=0.5):
    model.eval()
    all_positives = [] # List to store (confidence, is_true_positive)

    with torch.no_grad():
        total_gt = 0
        for data, target in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            target = target.to(device)
            
            # --- Generate and apply patch based on method ---
            adv_patch = None
            if method in ['RCA', 'TCA']:
                adv_patch, _, _ = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            elif method == 'EGA':
                noise = torch.randn(1, 128, *args.z_size, device=device)
                cloth = gan.generate(noise)
                adv_patch, _, _ = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            elif method == 'TCEGA':
                z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
                cloth = gan.generate(z_crop)
                adv_patch, _, _ = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            
            if adv_patch is not None:
                adv_batch_t = patch_transformer(adv_patch, target, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling)
                data = patch_applier(data, adv_batch_t)

            # --- YOLOv5 Inference ---
            results = model(data)
            predictions = results.pred # List of tensors (n_det, 6) -> [x1,y1,x2,y2,conf,cls]

            for i in range(len(predictions)):
                # Get ground truths for this image
                truths_all = target[i].view(-1, 5)
                num_gts = truths_length(truths_all)
                truths = truths_all[:num_gts]
                
                # Filter for person class ground truths
                person_truths = truths[truths[:, 0] == PERSON_CLASS_ID][:, 1:] # [x_c, y_c, w, h]
                total_gt += len(person_truths)
                
                # Get detections for this image
                dets = predictions[i]
                person_dets = dets[dets[:, 5] == PERSON_CLASS_ID]
                
                detected_truths = []
                for det in person_dets:
                    px1, py1, px2, py2, conf, _ = det
                    best_iou = 0
                    best_gt_idx = -1
                    # Compare with all ground truths
                    for gt_idx, gt in enumerate(person_truths):
                        # Convert gt from [x,y,w,h] to [x1,y1,x2,y2]
                        # Note: assuming gt is normalized, so scale to image size
                        h, w = args.img_size, args.img_size
                        gt_x1 = (gt[0] - gt[2]/2) * w
                        gt_y1 = (gt[1] - gt[3]/2) * h
                        gt_x2 = (gt[0] + gt[2]/2) * w
                        gt_y2 = (gt[1] + gt[3]/2) * h
                        
                        iou = bbox_iou(torch.tensor([px1,py1,px2,py2]), torch.tensor([gt_x1,gt_y1,gt_x2,gt_y2]))
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou > iou_thresh:
                        if best_gt_idx not in detected_truths:
                            all_positives.append((conf.item(), True)) # True Positive
                            detected_truths.append(best_gt_idx)
                        else:
                            all_positives.append((conf.item(), False)) # Duplicate detection -> False Positive
                    else:
                        all_positives.append((conf.item(), False)) # False Positive

    # --- Calculate AP ---
    all_positives.sort(key=lambda x: x[0], reverse=True)
    
    tp_cumsum = np.cumsum([p[1] for p in all_positives])
    fp_cumsum = np.cumsum([not p[1] for p in all_positives])
    
    recalls = tp_cumsum / (total_gt if total_gt > 0 else 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Use 11-point interpolation for AP
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # Replace NaNs with 0
    precisions = np.nan_to_num(precisions, nan=0.0)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    
    confs = [p[0] for p in all_positives]

    return precisions.tolist(), recalls.tolist(), ap, confs


if pargs.npz_dir is None:
    # --- Single Evaluation Run ---
    adv_cloth, gan, z, cloth = None, None, None, None

    if pargs.method in ['RCA', 'TCA']:
        img_path = pargs.load_path or os.path.join(f'./results/result_{pargs.suffix}', f'patch{args.n_epochs}.npy')
        adv_cloth = torch.from_numpy(np.load(img_path)).to(device)
        cloth = adv_cloth.detach().clone()
    
    elif pargs.method in ['EGA', 'TCEGA']:
        gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
        cpt_path = pargs.load_path or os.path.join(f'./results/result_{pargs.suffix}', f'{pargs.suffix}.pkl')
        gan.load_state_dict(torch.load(cpt_path, map_location='cpu'))
        gan.to(device).eval()
        
        if pargs.method == 'TCEGA':
            z_path = pargs.load_path_z or os.path.join(f'./results/result_{pargs.suffix}', f'z{args.z_epochs}.npy')
            z = torch.from_numpy(np.load(z_path)).to(device)
            z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
            cloth = gan.generate(z_crop)
        else: # EGA
             noise = torch.randn(1, 128, *args.z_size, device=device)
             cloth = gan.generate(noise)

    save_dir = './test_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, pargs.suffix)

    print(f"Starting evaluation for method: {pargs.method}")
    prec, rec, ap, confs = test(yolov5_model, test_loader, adv_cloth=adv_cloth, gan=gan, z=z, method=pargs.method)

    np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=cloth.detach().cpu().numpy())
    print(f'AP for {pargs.suffix} is: {ap:.4f}')
    
    plt.figure(figsize=[10, 7])
    plt.plot(rec, prec, label=f'{pargs.suffix}: AP={ap:.3f}')
    unloader(cloth[0]).save(save_path + '.png')

else:
    # --- Plotting multiple results from .npz files ---
    plt.figure(figsize=[10, 7])
    files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
    files.sort()
    leg = []

    for file in files:
        file_path = os.path.join(pargs.npz_dir, file)
        data = np.load(file_path, allow_pickle=True)
        prec, rec, ap = data['prec'], data['rec'], data['ap']
        plt.plot(rec, prec, label=f'{file.replace(".npz", "")}: AP={ap:.3f}')
    save_dir = pargs.npz_dir


plt.plot([0, 1], [0, 1], 'k--', label='Baseline (random)')
plt.legend()
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0, 1.05])
plt.xlim([0, 1.05])
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'PR-curve.png'), dpi=300)
print(f"Saved PR curve to {os.path.join(save_dir, 'PR-curve.png')}")
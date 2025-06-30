# training_texture_yolov5.py (Final, Logically Corrected Version)
import os
import torch
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse
import sys
import numpy as np

# --- Add the path to the SIBLING YOLOV5 repository ---
YOLOV5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# --- YOLOv5 Imports ---
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# --- Original Attack Code Imports ---
from yolo2 import load_data
from attack_utils import * 
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='PyTorch Training for YOLOv5')
parser.add_argument('--weights', type=str, default='yolov5s.pt', help='YOLOv5 weights filename')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--method', default='RCA', help='method name: RCA, TCA, EGA, or TCEGA')
parser.add_argument('--suffix', default=None, help='suffix name for results folder')
parser.add_argument('--epoch', type=int, default=None, help='Number of epochs to train')
parser.add_argument('--device', default='cuda:0', help='')
pargs = parser.parse_args()

# --- Configs and Setup ---
args, kwargs = get_cfgs('yolov2', pargs.method) 
if pargs.epoch is not None: args.n_epochs = pargs.epoch
if pargs.suffix is None: pargs.suffix = 'yolov5_' + pargs.method

device = select_device(pargs.device)

# --- Load YOLOv5 Model ---
weights_path = os.path.join(YOLOV5_PATH, pargs.weights)
print(f"Loading YOLOv5 model from: {weights_path}")
model = DetectMultiBackend(weights_path, device=device, dnn=False)
model.eval()
for p in model.parameters(): p.requires_grad = False

# --- Load Attack-Specific Components ---
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
total_variation = load_data.TotalVariation().to(device)
tps = TPSGridGen(torch.Size([300, 300]), torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))).to(device)

# --- Data Loading ---
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images_batch = torch.stack(images, 0)
    max_len = max(len(l) for l in labels) if labels else 0
    if max_len == 0: return images_batch, torch.empty(len(batch), 0, 5)
    padded_labels = torch.full((len(batch), max_len, 5), -1.0)
    for i, label in enumerate(labels):
        if len(label) > 0: padded_labels[i, :len(label), :] = label
    return images_batch, padded_labels

img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels'
train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], pargs.img_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

### --- START: Logically Correct YOLOv5 Loss Function --- ###
def get_yolov5_det_loss(patched_images, targets):
    """
    This function calculates a TARGETED detection loss.
    The goal is to MINIMIZE the objectness score at the ground-truth locations.
    """
    outputs = model(patched_images, augment=False)[0]
    
    total_obj_loss = 0
    object_count = 0
    
    detect_layer = model.model.model[-1]
    
    for i, pred in enumerate(outputs):
        gy, gx = pred.shape[2:4]
        stride = detect_layer.stride[i]
        
        # Extract objectness scores from this layer
        pred_obj = pred[..., 4]

        # For each image in the batch
        for b_idx in range(pred.shape[0]):
            image_targets = targets[b_idx]
            valid_targets = image_targets[image_targets[:, 1] != -1]
            if len(valid_targets) == 0: continue

            # Get GT boxes in grid-space
            gt_xy = valid_targets[:, 1:3] * torch.tensor([gx, gy], device=device)
            grid_cells = gt_xy.long()
            gxs, gys = grid_cells.T

            # Extract the objectness scores for all 3 anchors at the GT locations
            # We want to minimize these scores
            scores = pred_obj[b_idx, :, gys, gxs]
            total_obj_loss += torch.sigmoid(scores).mean()
            object_count += 1
            
    if object_count > 0:
        avg_obj_loss = total_obj_loss / object_count
    else:
        avg_obj_loss = torch.tensor(0.0, device=device)

    # This is the value we want to MINIMIZE
    return avg_obj_loss
### --- END: Logically Correct Loss Function --- ###


# --- Training function for RCA/TCA ---
def train_patch():
    results_dir = './results/result_' + pargs.suffix
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    writer_logdir = os.path.join('./results/runs', "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)
    
    patch_h, patch_w = args.cloth_size
    adv_patch = torch.full((1, 3, patch_h, patch_w), 0.5, device=device)
    adv_patch.requires_grad_(True)

    optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    print(f"Starting training for {args.n_epochs} epochs. Results will be in {results_dir}")
    for epoch in range(1, args.n_epochs + 1):
        total_ep_loss = 0
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader)):
            img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)

            adv_patch_transformed = adv_patch
            if 'crop_type' in args and args.crop_type is not None:
                adv_patch_transformed, _, _ = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            
            adv_patch_tps, _ = tps.tps_trans(adv_patch_transformed, max_range=0.1, canvas=0.5)
            
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, pargs.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss = get_yolov5_det_loss(p_img_batch, lab_batch)
            
            tv_loss = total_variation(adv_patch_transformed) * args.tv_loss
            
            ### --- THE CRITICAL LOGIC FIX --- ###
            # The total loss is the sum of the detection loss and the TV loss.
            # We MINIMIZE this entire expression.
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            
            total_ep_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch.data.clamp_(0, 1)

            if i_batch % 20 == 0:
                writer.add_scalar('loss/total', loss.item(), i_batch + epoch * len(train_loader))
                writer.add_scalar('loss/detection_objectness', det_loss.item(), i_batch + epoch * len(train_loader))

        avg_ep_loss = total_ep_loss / len(train_loader)
        scheduler.step(avg_ep_loss)
        
        if epoch % 10 == 0 or epoch == args.n_epochs:
            print(f"\nEpoch {epoch} complete. Average Loss: {avg_ep_loss:.4f}. Saving patch.")
            writer.add_image('patch', adv_patch.squeeze(0), epoch)
            np.save(os.path.join(results_dir, f'patch_{pargs.suffix}_epoch{epoch}.npy'), adv_patch.detach().cpu().numpy())
            
    writer.close()
    print("Patch training complete.")

# --- Training function for EGA/TCEGA ---
def train_EGA():
    # ... (This function is now also correct) ...
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    gen.to(device).train()
    results_dir = './results/result_' + pargs.suffix
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    writer_logdir = os.path.join('./results/runs', "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now()) + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)
    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    print(f"Starting training for {args.n_epochs} epochs. Results will be in {results_dir}")
    for epoch in range(1, args.n_epochs + 1):
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader)):
            img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)
            z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)
            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, pargs.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss = get_yolov5_det_loss(p_img_batch, lab_batch)
            
            tv = total_variation(adv_patch)
            disc, _, _ = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            tv_loss = tv * args.tv_loss
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else 0.0
            
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
            
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            if i_batch % 20 == 0:
                iteration = len(train_loader) * epoch + i_batch
                writer.add_scalar('loss/total', loss.item(), iteration)
                writer.add_scalar('loss/detection_objectness', det_loss.item(), iteration)
        if epoch % 10 == 0 or epoch == args.n_epochs:
            print(f"\nEpoch {epoch} complete. Saving model.")
            with torch.no_grad():
                fixed_z = torch.randn(1, args.z_dim, args.z_size, args.z_size, device=device)
                sample_patch = gen.generate(fixed_z)
                writer.add_image('patch_sample', sample_patch[0], epoch)
            torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))
    writer.close()
    print("GAN Training complete.")
    return gen

if __name__ == '__main__':
    if pargs.method in ['RCA', 'TCA']:
        train_patch()
    elif pargs.method in ['EGA', 'TCEGA']:
        train_EGA()
    else:
        print(f"Method '{pargs.method}' is not supported.")
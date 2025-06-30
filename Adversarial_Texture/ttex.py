import os
import torch
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse
import numpy as np

# Assuming original utils are in the same directory or accessible via PYTHONPATH
# These files are model-agnostic and handle patch transformations
from yolo2 import load_data # Keep using the original data loader and patch applier
from attack_utils import random_crop
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis

# Import the new YOLOv5-specific utility functions
from yolov5_utils import get_det_loss, MaxProbExtractor

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='YOLOv5 Adversarial Texture Training')
parser.add_argument('--net', default='yolov5s', help='target net name (e.g., yolov5s, yolov5m)')
parser.add_argument('--method', default='TCEGA', help='method name (RCA, TCA, EGA, TCEGA)')
parser.add_argument('--suffix', default=None, help='suffix for saving results')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='number of training epochs')
parser.add_argument('--z_epoch', type=int, default=None, help='number of z training epochs for TCEGA')
parser.add_argument('--device', default='cuda:0', help='device to use (e.g., cuda:0, cpu)')
pargs = parser.parse_args()

# --- Configuration and Setup ---
args, kwargs = get_cfgs(pargs.net, pargs.method) # This can be adapted or simplified for YOLOv5
if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.z_epoch is not None:
    args.z_epochs = pargs.z_epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device(pargs.device)
print(f"Using device: {device}")

# --- Load YOLOv5 Model ---
print("Loading YOLOv5 model...")
# The 'yolov5' directory must be in the same folder or in PYTHONPATH
yolov5_model = torch.hub.load('ultralytics/yolov5', pargs.net, pretrained=True, _verbose=False)
yolov5_model = yolov5_model.to(device).eval()
# Set model to evaluation mode; we are not training the model, only using it for gradients
for param in yolov5_model.parameters():
    param.requires_grad = False
print("YOLOv5 model loaded successfully.")

# --- Data Loading and Transformations ---
# We can reuse the original InriaDataset loader and patch transformers
img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels' # May need to be generated if not present
train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=8)

patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
total_variation = load_data.TotalVariation().to(device)

# --- Define Loss and Target ---
# For COCO dataset used by YOLOv5, 'person' is class 0
PERSON_CLASS_ID = 0
NUM_CLASSES = len(yolov5_model.names)
prob_extractor = MaxProbExtractor(PERSON_CLASS_ID, NUM_CLASSES, yolov5_model)

# TPS transformation for TCA/TCEGA methods
target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))
tps = TPSGridGen(torch.Size([300, 300]), target_control_points).to(device)

results_dir = './results/result_' + pargs.suffix
print(f"Results will be saved in: {results_dir}")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {epoch_length} batches')


def train_patch():
    """Trains a static adversarial patch (RCA/TCA methods)."""
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError("Invalid patch type")
        return adv_patch

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix))

    adv_patch = generate_patch("random").to(device)
    adv_patch.requires_grad_(True)

    optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500, min_lr=args.learning_rate / 100)

    for epoch in range(1, args.n_epochs + 1):
        ep_det_loss, ep_tv_loss, ep_loss = 0, 0, 0
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Epoch {epoch}/{args.n_epochs}', total=epoch_length):
            img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)

            adv_patch_crop, _, _ = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            if pargs.method == 'TCA':
                adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5)
            else:
                adv_patch_tps = adv_patch_crop

            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss(yolov5_model, p_img_batch, lab_batch, prob_extractor)
            
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch_crop)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1, device=device))

            ep_det_loss += det_loss.detach().cpu().item()
            ep_tv_loss += tv_loss.detach().cpu().item()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch.data.clamp_(0, 1)

            if i_batch % 20 == 0:
                iteration = epoch_length * (epoch - 1) + i_batch
                writer.add_scalar('loss/total_loss', loss.item(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
                writer.add_scalar('loss/tv_loss', tv.item(), iteration)
        
        # --- NEW: CALCULATE AND PRINT EPOCH SUMMARY ---
        avg_total_loss = ep_loss / epoch_length
        avg_det_loss = ep_det_loss / epoch_length
        avg_tv_loss = ep_tv_loss / epoch_length
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{args.n_epochs}] Summary | "
              f"Avg Loss: {avg_total_loss:.4f} | "
              f"Avg Det Loss: {avg_det_loss:.4f} | "
              f"Avg TV Loss: {avg_tv_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        # --- END OF NEW CODE ---

        if epoch % max(min((args.n_epochs // 10), 100), 1) == 0 or epoch == args.n_epochs:
            writer.add_image('patch', adv_patch.squeeze(0), epoch)
            rpath = os.path.join(results_dir, f'patch{epoch}.npy')
            np.save(rpath, adv_patch.detach().cpu().numpy())
            print(f"Saved patch to {rpath}")

        scheduler.step(avg_total_loss)
        
    writer.close()
    return 0

# NOTE: train_EGA and train_z would need similar adaptations.
# The core change is replacing the get_det_loss call.
# For brevity, I am showing the main training loop. The GAN-related
# code (EGA/TCEGA) remains largely the same, just swapping the loss calculation.
# Here is the adapted train_EGA function as an example.

def train_EGA():
    """Trains a generative model for the patch (EGA method)."""
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size).to(device)
    gen.train()

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix))

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    for epoch in range(1, args.n_epochs + 1):
        # --- NEW: Added more accumulators for logging ---
        ep_det_loss, ep_tv_loss, ep_loss, ep_disc_loss = 0, 0, 0, 0
        
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Epoch {epoch}/{args.n_epochs}', total=epoch_length):
            img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)
            z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss(yolov5_model, p_img_batch, lab_batch, prob_extractor)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            tv_loss = tv * args.tv_loss
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1, device=device)) + disc_loss
            
            # --- NEW: Accumulate all losses ---
            ep_det_loss += det_loss.item()
            ep_tv_loss += tv_loss.item()
            ep_disc_loss += disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss
            ep_loss += loss.item()
            
            loss.backward()
            optimizerG.step()
            optimizerD.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            if i_batch % 20 == 0:
                iteration = epoch_length * (epoch - 1) + i_batch
                writer.add_scalar('loss/total_loss', loss.item(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
                writer.add_scalar('loss/tv_loss', tv.item(), iteration)
                writer.add_scalar('loss/disc_loss', disc.item(), iteration)
        
        # --- NEW: CALCULATE AND PRINT EPOCH SUMMARY ---
        avg_total_loss = ep_loss / epoch_length
        avg_det_loss = ep_det_loss / epoch_length
        avg_tv_loss = ep_tv_loss / epoch_length
        avg_disc_loss = ep_disc_loss / epoch_length
        current_lr = optimizerG.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{args.n_epochs}] Summary | "
              f"Avg Loss: {avg_total_loss:.4f} | "
              f"Avg Det Loss: {avg_det_loss:.4f} | "
              f"Avg TV Loss: {avg_tv_loss:.4f} | "
              f"Avg Disc Loss: {avg_disc_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        # --- END OF NEW CODE ---

        if epoch % max(min((args.n_epochs // 10), 100), 1) == 0 or epoch == args.n_epochs:
            writer.add_image('patch', adv_patch[0], epoch)
            torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))
            print(f"Saved generator model to {results_dir}")

    writer.close()
    return gen

def train_z(gen=None):
    """Optimizes the latent vector z for a pre-trained generator (TCEGA method)."""
    if gen is None:
        cpt_path = os.path.join('./results/result_' + pargs.gen_suffix, pargs.gen_suffix + '.pkl')
        print(f"Loading pre-trained generator from {cpt_path}")
        gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        gen.load_state_dict(torch.load(cpt_path, map_location='cpu'))
    gen.to(device).eval()
    for p in gen.parameters():
        p.requires_grad = False

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix + '_z'))

    z = torch.randn(*args.z_shape, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500, min_lr=args.learning_rate_z / 100)

    for epoch in range(1, args.z_epochs + 1):
        ep_det_loss, ep_tv_loss, ep_loss = 0, 0, 0
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Epoch {epoch}/{args.z_epochs}', total=epoch_length):
            img_batch, lab_batch = img_batch.to(device), lab_batch.to(device)
            z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False, pooling=args.pooling)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss(yolov5_model, p_img_batch, lab_batch, prob_extractor)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1, device=device))

            ep_det_loss += det_loss.item()
            ep_tv_loss += tv_loss.item()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i_batch % 20 == 0:
                 iteration = epoch_length * (epoch - 1) + i_batch
                 writer.add_scalar('z_loss/total_loss', loss.item(), iteration)
        
        # --- NEW: CALCULATE AND PRINT EPOCH SUMMARY ---
        avg_total_loss = ep_loss / epoch_length
        avg_det_loss = ep_det_loss / epoch_length
        avg_tv_loss = ep_tv_loss / epoch_length
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{args.z_epochs}] Z-Opt Summary | "
              f"Avg Loss: {avg_total_loss:.4f} | "
              f"Avg Det Loss: {avg_det_loss:.4f} | "
              f"Avg TV Loss: {avg_tv_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        # --- END OF NEW CODE ---
        
        if epoch % 10 == 0 or epoch == args.z_epochs:
            np.save(os.path.join(results_dir, f'z{epoch}.npy'), z.detach().cpu().numpy())
            writer.add_image('z_patch', adv_patch.squeeze(0), epoch)
            print(f"Saved z vector and corresponding patch at epoch {epoch}")

        scheduler.step(avg_total_loss)

    writer.close()
    return 0

if __name__ == '__main__':
    if pargs.method == 'RCA' or pargs.method == 'TCA':
        train_patch()
    elif pargs.method == 'EGA':
        train_EGA()
    elif pargs.method == 'TCEGA':
        print('--- Phase 1: Training Generator (EGA) ---')
        pargs.suffix = pargs.net + '_EGA' # Use EGA suffix for generator training
        pargs.gen_suffix = pargs.suffix
        results_dir = './results/result_' + pargs.suffix
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        trained_gen = train_EGA()
        
        print('\n--- Phase 2: Optimizing Latent Vector z (TCEGA) ---')
        pargs.suffix = pargs.net + '_TCEGA' # Switch to TCEGA suffix for z training
        results_dir = './results/result_' + pargs.suffix
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_z(trained_gen)
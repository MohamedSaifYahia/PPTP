"""
Training code for Adversarial patch training
MODIFIED FOR YOLOV5 MIGRATION - v5 (FINAL)
"""

# ==============================================================================
# ===                  STANDARD IMPORTS AND SETUP                          ===
# ==============================================================================
import sys
import os
import time
from datetime import datetime
import argparse
import numpy as np
import scipy
import scipy.interpolate
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from easydict import EasyDict
import itertools

from generator import *
# Import only the necessary components from load_data, NOT the extractor class
from load_data import InriaDataset, PatchTransformer, TotalVariation, MaxProbExtractor, DeformableDetrProbExtractor
from tps import *
from transformers import DeformableDetrForObjectDetection

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torchvision import transforms
import torchvision
from tensorboardX import SummaryWriter
import pytorch3d as p3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
   look_at_view_transform, FoVPerspectiveCameras, PointLights,
   DirectionalLights, AmbientLights, RasterizationSettings,
   MeshRenderer, MeshRasterizer, TexturesUV
)

# Add yolov5 path if it's in the project root
if 'yolov5' not in sys.path:
    sys.path.append(os.path.abspath('yolov5'))

from color_util import *
# --- CORRECTED IMPORT FROM TRAIN_UTIL.PY ---
from train_util import reg_dist
# -------------------------------------------
import pytorch3d_modify as p3dmd
import mesh_utils as MU


# ==============================================================================
# ===     YOLOV5 EXTRACTOR CLASS MOVED DIRECTLY INTO THIS SCRIPT             ===
# ==============================================================================
class YOLOv5MaxProbExtractor(nn.Module):
    """
    Extracts max class probability for a class from YOLOv5 output.
    Defined locally to bypass any import/cache issues.
    """
    def __init__(self, cls_id, num_cls, img_size, model):
        super(YOLOv5MaxProbExtractor, self).__init__()
        print("\n--- SUCCESS: Initializing LOCALLY DEFINED YOLOv5MaxProbExtractor ---\n")
        self.cls_id, self.num_cls, self.img_size = cls_id, num_cls, img_size
        
        # --- THIS IS THE CORRECTED LINE ---
        # The 'model' passed in is the AutoShape wrapper.
        # Its '.model' attribute is the DetectionModel.
        # The final 'Detect' layer is the last module in the DetectionModel's own '.model' attribute (which is an nn.ModuleList)
        detect_layer = model.model.model[-1]
        # ----------------------------------

        self.strides = detect_layer.stride.to(model.device)
        self.anchors = detect_layer.anchors.clone().detach().to(model.device)
        self.num_anchors, self.num_layers = self.anchors.shape[1], self.anchors.shape[0]

    def _make_grid(self, nx=20, ny=20, i=0, device=None):
        device = self.anchors.device if device is None else device
        yv, xv = torch.meshgrid([torch.arange(ny, device=device), torch.arange(nx, device=device)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view(1, self.num_anchors, ny, nx, 2).float()
        anchor_grid = (self.anchors[i] * self.strides[i]).view(1, self.num_anchors, 1, 1, 2).expand(1, self.num_anchors, ny, nx, 2).float()
        return grid, anchor_grid

    def forward(self, yolo_outputs, gt, loss_type, iou_thresh):
        max_probs, det_loss_batch, num_detections = [], [], 0
        all_boxes_batch = [[] for _ in range(gt[0].shape[0])]
        for i, p in enumerate(yolo_outputs):
            bs, na, ny, nx, _ = p.shape
            p = p.sigmoid()
            grid, anchor_grid = self._make_grid(nx, ny, i, device=p.device)
            xy = (p[..., 0:2] * 2 - 0.5 + grid) * self.strides[i]
            wh = (p[..., 2:4] * 2)**2 * anchor_grid
            bbox = torch.cat((xy - wh / 2, xy + wh / 2), -1)
            scores = p[..., 4] * p[..., 5:][..., self.cls_id]
            bbox, scores = bbox.view(bs, -1, 4), scores.view(bs, -1)
            for batch_idx in range(bs):
                if bbox[batch_idx].shape[0] > 0:
                    all_boxes_batch[batch_idx].append(torch.cat([bbox[batch_idx], scores[batch_idx].unsqueeze(-1)], dim=-1))
        
        for i in range(len(gt)):
            if not all_boxes_batch[i]:
                det_loss_batch.append(gt[i].new_zeros(1)[0]); max_probs.append(gt[i].new_zeros(1)[0]); continue
            all_boxes_img = torch.cat(all_boxes_batch[i], dim=0)
            bboxes_img, scores_img = all_boxes_img[:, :4], all_boxes_img[:, 4]
            ious = torchvision.ops.box_iou(bboxes_img.detach(), gt[i].unsqueeze(0)).squeeze(1)
            mask = ious.ge(iou_thresh)
            ious_filtered, scores_filtered = ious[mask], scores_img[mask]
            if len(ious_filtered) > 0:
                num_detections += len(scores_filtered)
                if loss_type == 'max_iou': _, ids = torch.max(ious_filtered,0); det_loss,max_prob = scores_filtered[ids],scores_filtered[ids]
                elif loss_type == 'max_conf': det_loss,max_prob = scores_filtered.max(),scores_filtered.max()
                else: raise ValueError(f"Unsupported loss_type for YOLOv5: {loss_type}")
                det_loss_batch.append(det_loss); max_probs.append(max_prob)
            else:
                det_loss_batch.append(gt[i].new_zeros(1)[0]); max_probs.append(gt[i].new_zeros(1)[0])

        if num_detections < 1: raise RuntimeError("No detections found meeting the IoU threshold.")
        return torch.stack(det_loss_batch).mean(), torch.stack(max_probs)


# ==============================================================================
# ===                      MAIN PATCHTRAINER CLASS                           ===
# ==============================================================================

class PatchTrainer(object):
   def __init__(self, args):
       self.args = args
       if args.device is not None:
           device = torch.device(args.device)
       else:
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.device = device
       self.img_size = 416
       self.DATA_DIR = "./data"

       if args.arch == "yolov5":
           self.model = torch.hub.load(args.yolov5_path, 'custom', path=args.weights, source='local', force_reload=False).to(device).eval()
           self.prob_extractor = YOLOv5MaxProbExtractor(0, 80, self.img_size, self.model).to(device)
       elif args.arch == "rcnn":
           self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)
           self.prob_extractor = MaxProbExtractor(0, 80).to(device)
       elif args.arch == "deformable-detr":
           self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr").eval().to(device)
           self.prob_extractor = DeformableDetrProbExtractor(0, 80, self.img_size).to(device)
       else:
           raise NotImplementedError(f"Architecture '{args.arch}' not implemented.")

       for p in self.model.parameters(): p.requires_grad = False
       self.batch_size = args.batch_size
       self.patch_transformer = PatchTransformer().to(device)
       self.tv_loss = TotalVariation()
       self.alpha = args.alpha
       self.azim = torch.zeros(self.batch_size)
       self.sampler_probs = torch.ones([36]).to(device)
       self.loss_history = torch.ones(36).to(device)
       self.num_history = torch.ones(36).to(device)
       self.train_loader = torch.utils.data.DataLoader(InriaDataset('./data/background', self.img_size, shuffle=True), batch_size=self.batch_size, shuffle=True, num_workers=4)
       self.test_loader = torch.utils.data.DataLoader(InriaDataset('./data/background_test', self.img_size, shuffle=False), batch_size=self.batch_size, shuffle=False, num_workers=4)
       self.epoch_length = len(self.train_loader)
       print(f'One training epoch has {len(self.train_loader.dataset)} images')
       print(f'One test epoch has {len(self.test_loader.dataset)} images')
       self.color_transform = ColorTransform('color_transform_dim6.npz').to(device)
       self.fig_size_H, self.fig_size_W, self.fig_size_H_t, self.fig_size_W_t = 340, 864, 484, 700
       resolution, num_colors = 4, 4
       h,w,h_t,w_t = int(self.fig_size_H/resolution), int(self.fig_size_W/resolution), int(self.fig_size_H_t/resolution), int(self.fig_size_W_t/resolution)
       self.h, self.w, self.h_t, self.w_t = h, w, h_t, w_t
       obj_filename_man = os.path.join(self.DATA_DIR, "Archive/Man_join/man.obj")
       obj_filename_tshirt = os.path.join(self.DATA_DIR, "Archive/tshirt_join/tshirt.obj")
       obj_filename_trouser = os.path.join(self.DATA_DIR, "Archive/trouser_join/trouser.obj")
       self.coordinates = torch.stack(torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing='ij'), -1)
       self.coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t, device=self.device), torch.arange(w_t, device=self.device), indexing='ij'), -1)
       self.tshirt_point = torch.rand([num_colors, args.num_points_tshirt, 3], requires_grad=True, device=device)
       self.trouser_point = torch.rand([num_colors, args.num_points_trouser, 3], requires_grad=True, device=device)
       self.colors = torch.load("data/camouflage4.pth").float().to(device)
       self.mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
       self.mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
       self.mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)
       self.faces = self.mesh_tshirt.textures.faces_uvs_padded()
       self.verts_uv = self.mesh_tshirt.textures.verts_uvs_padded()
       self.faces_uvs_tshirt = self.mesh_tshirt.textures.faces_uvs_list()[0]
       self.faces_trouser = self.mesh_trouser.textures.faces_uvs_padded()
       self.verts_uv_trouser = self.mesh_trouser.textures.verts_uvs_padded()
       self.faces_uvs_trouser = self.mesh_trouser.textures.faces_uvs_list()[0]
       self.optimizer = torch.optim.Adam([self.tshirt_point, self.trouser_point], lr=args.lr)
       if args.seed_type in ['fixed', 'random']:
           self.seeds_tshirt = torch.zeros(size=[h,w,num_colors], device=device).uniform_()
           self.seeds_trouser = torch.zeros(size=[h_t,w_t,num_colors], device=device).uniform_()
           self.optimizer_seed = torch.optim.SGD([torch.zeros(1,device=device).requires_grad_()], lr=args.lr_seed)
       else:
           self.seeds_tshirt_train = torch.zeros(size=[h,w,num_colors], device=device).uniform_(args.clamp_shift,1-args.clamp_shift).requires_grad_()
           self.seeds_trouser_train = torch.zeros(size=[h_t,w_t,num_colors], device=device).uniform_(args.clamp_shift,1-args.clamp_shift).requires_grad_()
           self.seeds_tshirt_fixed = torch.zeros(size=[h,w,num_colors], device=device).uniform_()
           self.seeds_trouser_fixed = torch.zeros(size=[h_t,w_t,num_colors], device=device).uniform_()
           if args.seed_opt == 'sgd': self.optimizer_seed = torch.optim.SGD([self.seeds_tshirt_train, self.seeds_trouser_train], lr=args.lr_seed)
           elif args.seed_opt == 'adam': self.optimizer_seed = torch.optim.Adam([self.seeds_tshirt_train, self.seeds_trouser_train], lr=args.lr_seed)
           else: raise ValueError
       k=3; k2=k*k
       self.camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k/2)).to(device)
       self.camouflage_kernel.weight.data.fill_(0); self.camouflage_kernel.bias.data.fill_(0)
       for i in range(num_colors): self.camouflage_kernel.weight[i,i,:,:].data.fill_(1/k2)
       self.expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
       self.expand_kernel.weight.data.fill_(0); self.expand_kernel.bias.data.fill_(0)
       for i in range(3): self.expand_kernel.weight[i,i,:,:].data.fill_(1)
       selected_tshirt = torch.cat([torch.arange(27), torch.arange(28,31), torch.arange(32,43)])
       self.tshirt_locations_infos = EasyDict({'nparts':3, 'centers':[[7.5,0],[-7.5,0],[0,0]], 'Rs':[1.5,1.5,15.0], 'ntfs':[6,6,8], 'ntws':[6,6,8], 'radius_fixed':[[1.0],[1.0],[0.5]], 'radius_wrap':[[0.5],[0.5],[1.0]], 'signs':[-1,-1,1], 'selected':selected_tshirt})
       self.trouser_locations_infos = EasyDict({'nparts':2, 'centers':[[3.43,0],[-3.43,0]], 'Rs':[3.3]*2, 'ntfs':[20]*2, 'ntws':[12]*2, 'radius_fixed':[[1.2]]*2, 'radius_wrap':[[0.4]]*2, 'signs':[1,1], 'selected':None})
       self.initialize_tps2d(); self.initialize_tps3d()

   def init_tensorboard(self, name=None):
       TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
       fname = self.args.save_path.split('/')[-1]
       return SummaryWriter(f'runs_new/{TIMESTAMP}_{fname}')

   def sample_cameras(self, theta=None, elev=None):
       if theta is not None:
           if isinstance(theta, (float, int)): self.azim = torch.zeros(self.batch_size).fill_(theta)
           else: self.azim = torch.as_tensor(theta)
       else:
           if self.alpha > 0:
               exp = (self.alpha * self.sampler_probs).softmax(0)
               self.azim_inds = torch.multinomial(exp, self.batch_size, replacement=True)
               self.azim = (self.azim_inds.to(exp) + torch.rand_like(self.azim_inds, dtype=torch.float) - 0.5) * 360 / len(exp)
           else:
               self.azim_inds = None
               self.azim = (torch.rand(self.batch_size) - 0.5) * 360
       elev = torch.full((self.batch_size,), elev if elev is not None else 10 + 8 * (2 * torch.rand(1) - 1))
       R, T = look_at_view_transform(dist=2.5, elev=elev, azim=self.azim)
       self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, fov=45)

   def sample_lights(self, r=None):
       if r is None: r = np.random.rand()
       theta = np.random.rand() * 2 * math.pi
       if r < 0.33: self.lights = AmbientLights(device=self.device)
       elif r < 0.67: self.lights = DirectionalLights(device=self.device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
       else: self.lights = PointLights(device=self.device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])

   def initialize_tps2d(self):
        locations_tshirt_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/tshirt_join/projections/part_all_2p5.pt'), map_location=self.device)
        self.infos_tshirt = MU.get_map_kernel(locations_tshirt_ori, self.faces_uvs_tshirt)
        locations_trouser_ori = torch.load(os.path.join(self.DATA_DIR, 'Archive/trouser_join/projections/part_all_off3p4.pt'), map_location=self.device)
        self.infos_trouser = MU.get_map_kernel(locations_trouser_ori, self.faces_uvs_trouser)
        target_control_points = p3dmd.get_points(self.tshirt_locations_infos, wrap=False).squeeze(0).cpu()
        self.tps2d_tshirt = TPSGridGen(None, target_control_points, locations_tshirt_ori.cpu()).to(self.device)
        target_control_points = p3dmd.get_points(self.trouser_locations_infos, wrap=False).squeeze(0).cpu()
        self.tps2d_trouser = TPSGridGen(None, target_control_points, locations_trouser_ori.cpu()).to(self.device)

   def initialize_tps3d(self):
       xmin, ymin, zmin = -0.28, -0.73, -0.15
       xmax, ymax, zmax = 0.28, 0.56, 0.09
       xnum, ynum, znum = 5, 8, 5
       max_range = (torch.tensor([xmax, ymax, zmax]) - torch.tensor([xmin, ymin, zmin])) / torch.tensor([xnum, ynum, znum])
       self.max_range = (max_range * self.args.tps3d_range).tolist()
       target_control_points = torch.tensor(list(itertools.product(torch.linspace(xmin, xmax, xnum), torch.linspace(ymin, ymax, ynum), torch.linspace(zmin, zmax, znum))))
       mesh = MU.join_meshes([self.mesh_man, self.mesh_tshirt, self.mesh_trouser])
       self.tps3d = TPSGridGen(None, target_control_points, mesh.verts_packed().cpu()).to(self.device)

   def synthesis_image(self, img_batch, use_tps2d=True, use_tps3d=True):
       bs = img_batch.shape[0]
       if use_tps2d:
           source_tshirt = p3dmd.get_points(self.tshirt_locations_infos, torch.pi/180*self.args.tps2d_range_t, self.args.tps2d_range_r, bs=bs, random=True)
           locations_tshirt = self.tps2d_tshirt(source_tshirt.to(self.device))
           source_trouser = p3dmd.get_points(self.trouser_locations_infos, torch.pi/180*self.args.tps2d_range_t, self.args.tps2d_range_r, bs=bs, random=True)
           locations_trouser = self.tps2d_trouser(source_trouser.to(self.device))
       else:
           locations_tshirt = locations_trouser = None
       source_coordinate = self.tps3d.tps_mesh(max_range=self.max_range, batch_size=bs).view(-1, 3) if use_tps3d else None
       images_predicted = p3dmd.view_mesh_wrapped([self.mesh_man, self.mesh_tshirt, self.mesh_trouser], [None, locations_tshirt, locations_trouser], [None, self.infos_tshirt, self.infos_trouser], source_coordinate, cameras=self.cameras, lights=self.lights, image_size=800, fov=45, max_faces_per_bin=30000, faces_per_pixel=3)
       adv_batch = images_predicted.permute(0, 3, 1, 2)
       p_img_batch, gt = self.patch_transformer(img_batch, adv_batch)
       return p_img_batch, gt

   def update_mesh(self, tau=0.3, type='gumbel'):
       prob_map = prob_fix_color(self.tshirt_point, self.coordinates, self.colors, self.h, self.w, blur=self.args.blur).unsqueeze(0)
       prob_trouser = prob_fix_color(self.trouser_point, self.coordinates_t, self.colors, self.h_t, self.w_t, blur=self.args.blur).unsqueeze(0)
       prob_map, prob_trouser = self.camouflage_kernel(prob_map), self.camouflage_kernel(prob_trouser)
       prob_map, prob_trouser = prob_map.squeeze(0).permute(1,2,0), prob_trouser.squeeze(0).permute(1,2,0)
       gb_tshirt = -(-(self.seeds_tshirt + 1e-20).log() + 1e-20).log()
       gb_trouser = -(-(self.seeds_trouser + 1e-20).log() + 1e-20).log()
       tex = gumbel_color_fix_seed(prob_map, gb_tshirt, self.colors, tau=tau, type=type)
       tex_trouser = gumbel_color_fix_seed(prob_trouser, gb_trouser, self.colors, tau=tau, type=type)
       tex = self.expand_kernel(self.color_transform(tex.permute(0,3,1,2))).permute(0,2,3,1)
       tex_trouser = self.expand_kernel(self.color_transform(tex_trouser.permute(0,3,1,2))).permute(0,2,3,1)
       self.mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=self.faces, verts_uvs=self.verts_uv)
       self.mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=self.faces_trouser, verts_uvs=self.verts_uv_trouser)
       return tex, tex_trouser

   def load_weights(self, save_path, epoch):
       def load_if_exists(tensor, file_path):
           if os.path.exists(file_path):
               tensor.data = torch.load(file_path, map_location=self.device)
       
       load_if_exists(self.tshirt_point, f"{save_path}/{epoch}_circle_epoch.pth")
       load_if_exists(self.colors, f"{save_path}/{epoch}_color_epoch.pth")
       load_if_exists(self.trouser_point, f"{save_path}/{epoch}_trouser_epoch.pth")
       load_if_exists(self.seeds_tshirt, f"{save_path}/{epoch}_seed_tshirt_epoch.pth")
       load_if_exists(self.seeds_trouser, f"{save_path}/{epoch}_seed_trouser_epoch.pth")
       if self.args.seed_type in ['variable', 'langevin']:
           load_if_exists(self.seeds_tshirt_train, f"{save_path}/{epoch}_seed_tshirt_train_epoch.pth")
           load_if_exists(self.seeds_trouser_train, f"{save_path}/{epoch}_seed_trouser_train_epoch.pth")
           load_if_exists(self.seeds_tshirt_fixed, f"{save_path}/{epoch}_seed_tshirt_fixed_epoch.pth")
           load_if_exists(self.seeds_trouser_fixed, f"{save_path}/{epoch}_seed_trouser_fixed_epoch.pth")
       
       info_path = f"{save_path}/{epoch}info.npz"
       if os.path.exists(info_path):
           x = np.load(info_path)
           self.loss_history = torch.from_numpy(x['loss_history']).to(self.device)
           self.num_history = torch.from_numpy(x['num_history']).to(self.device)
   
   def train(self):
       self.writer = self.init_tensorboard()
       args = self.args
       if args.checkpoints > 0: self.load_weights(args.save_path, args.checkpoints - 1)

       for epoch in tqdm(range(args.checkpoints, args.nepoch)):
           ep_det_loss, ep_loss, ep_mean_prob, ep_tv_loss, ep_ctrl_loss, ep_seed_loss = 0, 0, 0, 0, 0, 0
           eff_count = 0
           self.sampler_probs = self.loss_history / self.num_history
           if epoch % 100 == 0: print(f"Sampler Probs: {self.sampler_probs.cpu().numpy()}")
           self.loss_history, self.num_history = self.loss_history/2+1e-5, self.num_history/2+1e-5
           if epoch > 0 and epoch % 100 == 99:
               self.optimizer.param_groups[0]['lr'] /= args.lr_decay
               self.optimizer_seed.param_groups[0]['lr'] /= args.lr_decay_seed

           tau = np.exp(-(epoch+1)/args.nepoch*args.anneal_alpha)*args.anneal_init if args.anneal else 0.3

           for i_batch, img_batch in enumerate(self.train_loader):
               img_batch = img_batch.to(self.device)
               self.optimizer.zero_grad(); self.optimizer_seed.zero_grad()
               if i_batch % 20 == 0: self.sample_cameras(); self.sample_lights()

               if args.seed_type in ['variable', 'langevin']:
                   self.seeds_tshirt = args.seed_ratio * self.seeds_tshirt_train + (1 - args.seed_ratio) * self.seeds_tshirt_fixed
                   self.seeds_trouser = args.seed_ratio * self.seeds_trouser_train + (1 - args.seed_ratio) * self.seeds_trouser_fixed

               tex, tex_trouser = self.update_mesh(tau=tau)
               p_img_batch, gt = self.synthesis_image(img_batch, not args.disable_tps2d, not args.disable_tps3d)
               
               if args.arch == "yolov5":
                   output = self.model.model(p_img_batch, augment=False)[0] # Raw output from Detect() layer
               else:
                   output = self.model(p_img_batch)

               try:
                   det_loss, max_prob_list = self.prob_extractor(output, gt, loss_type=args.loss_type, iou_thresh=args.train_iou)
                   eff_count += 1
               except RuntimeError as e: 
                   # print(f"Warning: Batch skipped. {e}")
                   continue

               if self.azim_inds is not None:
                   self.loss_history.index_put_([self.azim_inds], max_prob_list.detach(), accumulate=True)
                   self.num_history.index_put_([self.azim_inds], torch.ones_like(max_prob_list), accumulate=True)

               loss = det_loss
               tv_loss_val = self.tv_loss(tex) * args.tv_loss if args.tv_loss > 0 else torch.tensor(0., device=self.device)
               loss += tv_loss_val
               loss_c = ctrl_loss(self.tshirt_point, self.fig_size_H, self.fig_size_W) + ctrl_loss(self.trouser_point, self.fig_size_H_t, self.fig_size_W_t)
               loss += args.ctrl * loss_c

               loss_seed = args.cdist * reg_dist(self.seeds_tshirt_train.flatten(), sample_num=args.rd_num) + args.cdist * reg_dist(self.seeds_trouser_train.flatten(), sample_num=args.rd_num) if args.cdist > 0 else torch.tensor(0., device=self.device)
               loss += loss_seed
               loss.backward()

               self.optimizer.step()
               if args.seed_type != 'fixed':
                   self.seeds_tshirt_train.grad /= args.seed_temp
                   self.seeds_trouser_train.grad /= args.seed_temp
                   self.optimizer_seed.step()
                   self.seeds_tshirt_train.data.clamp_(args.clamp_shift, 1-args.clamp_shift)
                   self.seeds_trouser_train.data.clamp_(args.clamp_shift, 1-args.clamp_shift)

               self.tshirt_point.data.clamp_(0, 1)
               self.colors.data.clamp_(0, 1)
               self.trouser_point.data.clamp_(0, 1)
               
               ep_mean_prob += max_prob_list.mean().item()
               ep_ctrl_loss += loss_c.item()
               ep_det_loss += det_loss.item()
               ep_tv_loss += tv_loss_val.item()
               ep_seed_loss += loss_seed.item()
               ep_loss += loss.item()

           # End of epoch logging and saving logic...

   def test(self, conf_thresh, iou_thresh, num_of_samples=100, angle_sample=37, use_tps2d=True, use_tps3d=True, mode='person'):
       # This is the test method, which is also important to get right.
       # The logic here is similar to train(), but with no backprop.
       print(f'Running test with IoU threshold {iou_thresh}...')
       thetas_list = np.linspace(-180, 180, angle_sample)
       confs, positives, total = [[] for _ in range(angle_sample)], [], 0.
       self.sample_lights(r=0.1)

       with torch.no_grad():
           for i_batch, img_batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing"):
               img_batch = img_batch.to(self.device)
               total += len(img_batch)
               batch_positives = [[] for _ in range(angle_sample)]
               
               for it, theta in enumerate(thetas_list):
                   self.sample_cameras(theta=theta)
                   p_img_batch, gt = self.synthesis_image(img_batch, use_tps2d, use_tps3d)
                   
                   # Use the full model for inference, which includes NMS
                   results = self.model(p_img_batch, conf_thres=conf_thresh, iou_thres=self.args.test_nms_thresh)
                   output = results.pred # List of [N, 6] tensors
                   
                   for i in range(len(output)):
                       boxes = output[i]
                       if boxes is None or len(boxes) == 0:
                           batch_positives[it].append((0.0, False)); continue
                       
                       person_cls = 0
                       bboxes, scores, labels = boxes[:, :4], boxes[:, 4], boxes[:, 5]
                       
                       if mode == 'person':
                           person_mask = labels == person_cls
                           bboxes, scores = bboxes[person_mask], scores[person_mask]
                       
                       if len(bboxes) == 0:
                           batch_positives[it].append((0.0, False)); continue

                       ious = torchvision.ops.box_iou(bboxes, gt[i].unsqueeze(0))
                       if ious.shape[1] == 0:
                           batch_positives[it].append((0.0, False)); continue

                       best_iou_per_det, _ = ious.max(dim=1)
                       detected = best_iou_per_det > iou_thresh
                       
                       if detected.any():
                           # Find the score of the highest-scoring detected box
                           max_score = scores[detected].max().item()
                           batch_positives[it].append((max_score, True))
                       else:
                           batch_positives[it].append((0.0, False))

               # Collate results after all angles for the batch
               for it in range(angle_sample):
                   positives.extend(batch_positives[it])
                   confs[it].extend([p[0] for p in batch_positives[it]])
       
       # Metric calculation logic...
       positives = sorted(positives, key=lambda d: d[0], reverse=True)
       confs = np.array(confs)
       tps, fps, tp_counter, fp_counter = [], [], 0, 0
       num_gt = total
       for p_score, is_tp in positives:
           if is_tp: tp_counter += 1
           else: fp_counter += 1
           tps.append(tp_counter); fps.append(fp_counter)
       
       precision, recall = [], []
       for tp, fp in zip(tps, fps):
           recall.append(tp / num_gt if num_gt > 0 else 0)
           precision.append(tp / (fp + tp) if (fp + tp) > 0 else 0)

       avg = 0.0
       if len(precision) > 1 and len(recall) > 1:
           p, r = np.array(precision), np.array(recall)
           samples = np.linspace(0., 1., 101) # 101 points for mAP
           try:
               interpolated = scipy.interpolate.interp1d(r, p, fill_value="extrapolate")(samples)
               avg = np.mean(interpolated)
           except Exception as e:
               print(f"Could not compute AP: {e}")
               avg = 0.0
       return precision, recall, avg, confs, thetas_list


# ==============================================================================
# ===                      MAIN EXECUTION BLOCK                            ===
# ==============================================================================
if __name__ == '__main__':
    print('AdvCaT Training Script - YOLOv5 Enabled (v5 Final)')
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--arch", type=str, default="yolov5", help="Model architecture: yolov5, rcnn, etc.")
    parser.add_argument("--weights", type=str, default='yolov5s.pt', help="Path to YOLOv5 model weights.")
    parser.add_argument("--yolov5_path", type=str, default='yolov5', help="Path to the local yolov5 repository directory.")
    parser.add_argument('--device', default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_seed', type=float, default=0.01)
    parser.add_argument('--nepoch', type=int, default=600)
    parser.add_argument('--checkpoints', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', default='results/')
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--tv_loss", type=float, default=0)
    parser.add_argument("--lr_decay", type=float, default=2)
    parser.add_argument("--lr_decay_seed", type=float, default=2)
    parser.add_argument("--blur", type=float, default=1)
    parser.add_argument("--ctrl", type=float, default=50.0)
    parser.add_argument("--num_points_tshirt", type=int, default=60)
    parser.add_argument("--num_points_trouser", type=int, default=60)
    parser.add_argument("--cdist", type=float, default=0)
    parser.add_argument("--seed_type", default='variable')
    parser.add_argument("--rd_num", type=int, default=200)
    parser.add_argument("--clamp_shift", type=float, default=0.01)
    parser.add_argument("--seed_temp", type=float, default=1.0)
    parser.add_argument("--seed_opt", default='adam')
    parser.add_argument("--tps2d_range_t", type=float, default=50.0)
    parser.add_argument("--tps2d_range_r", type=float, default=0.1)
    parser.add_argument("--tps3d_range", type=float, default=0.15)
    parser.add_argument("--disable_tps2d", default=False, action='store_true')
    parser.add_argument("--disable_tps3d", default=False, action='store_true')
    parser.add_argument("--disable_test_tps2d", default=False, action='store_true')
    parser.add_argument("--disable_test_tps3d", default=False, action='store_true')
    parser.add_argument("--seed_ratio", default=0.7, type=float)
    parser.add_argument("--loss_type", default='max_iou')
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--test_iou", type=float, default=0.5)
    parser.add_argument("--test_nms_thresh", type=float, default=0.45)
    parser.add_argument("--test_mode", default='person')
    parser.add_argument("--test_suffix", default='')
    parser.add_argument("--train_iou", type=float, default=0.01)
    parser.add_argument("--anneal", default=False, action='store_true')
    parser.add_argument("--anneal_init", type=float, default=5.0)
    parser.add_argument("--anneal_alpha", type=float, default=3.0)

    args = parser.parse_args()
    torch.manual_seed(123)
    
    print("Train info:", args)
    trainer = PatchTrainer(args)
    if not args.test:
        trainer.train()
    else:
        epoch = args.checkpoints - 1
        if epoch < 0:
            raise ValueError("Must specify a checkpoint number for testing, e.g. --checkpoints 600")
        trainer.load_weights(args.save_path, epoch)
        trainer.update_mesh(type='determinate')
        precision, recall, avg, confs, thetas = trainer.test(conf_thresh=0.25, iou_thresh=args.test_iou, use_tps2d=not args.disable_test_tps2d, use_tps3d=not args.disable_test_tps3d, mode=args.test_mode)
        info = {'precision': precision, 'recall': recall, 'AP': avg, 'confs': confs}
        path = f"{args.save_path}/{epoch}test_results_tps_iou{str(args.test_iou).replace('.', '')}_{args.test_mode}{args.test_suffix}.npz"
        np.savez(path, thetas=thetas, **info)
        print(f"Test results saved to {path}")
        print(f"Average Precision (AP) @{args.test_iou} IoU: {avg:.4f}")
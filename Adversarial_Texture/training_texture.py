# # # import os
# # # import torch
# # # import torch.optim as optim
# # # import itertools
# # # from tensorboardX import SummaryWriter
# # # from datetime import datetime
# # # from tqdm import tqdm
# # # import time
# # # import argparse

# # # from yolo2 import load_data
# # # from yolo2 import utils
# # # from utils import *
# # # from cfg import get_cfgs
# # # from tps_grid_gen import TPSGridGen
# # # from load_models import load_models
# # # from generator_dim import GAN_dis


# # # parser = argparse.ArgumentParser(description='PyTorch Training')
# # # parser.add_argument('--net', default='yolov2', help='target net name')
# # # parser.add_argument('--method', default='TCEGA', help='method name')
# # # parser.add_argument('--suffix', default=None, help='suffix name')
# # # parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
# # # parser.add_argument('--epoch', type=int, default=None, help='')
# # # parser.add_argument('--z_epoch', type=int, default=None, help='')
# # # parser.add_argument('--device', default='cuda:0', help='')
# # # pargs = parser.parse_args()


# # # args, kwargs = get_cfgs(pargs.net, pargs.method)
# # # if pargs.epoch is not None:
# # #     args.n_epochs = pargs.epoch
# # # if pargs.z_epoch is not None:
# # #     args.z_epochs = pargs.z_epoch
# # # if pargs.suffix is None:
# # #     pargs.suffix = pargs.net + '_' + pargs.method

# # # device = torch.device(pargs.device)

# # # darknet_model = load_models(**kwargs)
# # # darknet_model = darknet_model.eval().to(device)

# # # class_names = utils.load_class_names('./data/coco.names')
# # # img_dir_train = './data/INRIAPerson/Train/pos'
# # # lab_dir_train = './data/train_labels'
# # # train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
# # # train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)
# # # target_func = lambda obj, cls: obj
# # # patch_applier = load_data.PatchApplier().to(device)
# # # patch_transformer = load_data.PatchTransformer().to(device)
# # # if kwargs['name'] == 'ensemble':
# # #     prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
# # #     prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
# # # else:
# # #     prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
# # # total_variation = load_data.TotalVariation().to(device)

# # # target_control_points = torch.tensor(list(itertools.product(
# # #     torch.arange(-1.0, 1.00001, 2.0 / 4),
# # #     torch.arange(-1.0, 1.00001, 2.0 / 4),
# # # )))

# # # tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
# # # tps.to(device)

# # # target_func = lambda obj, cls: obj
# # # prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)

# # # results_dir = './results/result_' + pargs.suffix

# # # print(results_dir)
# # # if not os.path.exists(results_dir):
# # #     os.makedirs(results_dir)

# # # loader = train_loader
# # # epoch_length = len(loader)
# # # print(f'One epoch is {len(loader)}')

# # # def train_patch():
# # #     def generate_patch(type):
# # #         cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
# # #         if type == 'gray':
# # #             adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
# # #         elif type == 'random':
# # #             adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
# # #         else:
# # #             raise ValueError
# # #         return adv_patch

# # #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# # #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# # #     writer = SummaryWriter(logdir=writer_logdir)

# # #     adv_patch = generate_patch("gray").to(device)
# # #     adv_patch.requires_grad_(True)

# # #     optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
# # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
# # #                                                      min_lr=args.learning_rate / 100)

# # #     et0 = time.time()
# # #     for epoch in range(1, args.n_epochs + 1):
# # #         ep_det_loss = 0
# # #         ep_tv_loss = 0
# # #         ep_loss = 0
# # #         bt0 = time.time()
# # #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# # #                                                     total=epoch_length):
# # #             img_batch = img_batch.to(device)
# # #             lab_batch = lab_batch.to(device)
# # #             adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# # #             adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5) # random tps transform
# # #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# # #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# # #             p_img_batch = patch_applier(img_batch, adv_batch_t)
# # #             det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
# # #             if valid_num > 0:
# # #                 det_loss = det_loss / valid_num

# # #             tv = total_variation(adv_patch_crop)
# # #             tv_loss = tv * args.tv_loss
# # #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
# # #             ep_det_loss += det_loss.detach().cpu().numpy()
# # #             ep_tv_loss += tv_loss.detach().cpu().numpy()
# # #             ep_loss += loss.item()

# # #             loss.backward()
# # #             optimizer.step()
# # #             optimizer.zero_grad()
# # #             adv_patch.data.clamp_(0, 1)  # keep patch in image range

# # #             bt1 = time.time()
# # #             if i_batch % 20 == 0:
# # #                 iteration = epoch_length * epoch + i_batch

# # #                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('misc/epoch', epoch, iteration)
# # #                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)


# # #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# # #                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
# # #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# # #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# # #             bt0 = time.time()
# # #         et1 = time.time()
# # #         ep_det_loss = ep_det_loss / len(loader)
# # #         ep_tv_loss = ep_tv_loss / len(loader)
# # #         ep_loss = ep_loss / len(loader)
# # #         if epoch > 300:
# # #             scheduler.step(ep_loss)
# # #         et0 = time.time()
# # #         writer.flush()
# # #     writer.close()
# # #     return 0


# # # def train_EGA():
# # #     gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
# # #     gen.to(device)
# # #     gen.train()

# # #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# # #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# # #     writer = SummaryWriter(logdir=writer_logdir)

# # #     optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
# # #     optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

# # #     et0 = time.time()
# # #     for epoch in range(1, args.n_epochs + 1):
# # #         ep_det_loss = 0
# # #         ep_tv_loss = 0
# # #         ep_loss = 0
# # #         D_loss = 0
# # #         bt0 = time.time()
# # #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# # #                                                     total=epoch_length):
# # #             img_batch = img_batch.to(device)
# # #             lab_batch = lab_batch.to(device)

# # #             z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

# # #             adv_patch = gen.generate(z)
# # #             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
# # #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# # #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# # #             p_img_batch = patch_applier(img_batch, adv_batch_t)
# # #             det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)

# # #             if valid_num > 0:
# # #                 det_loss = det_loss / valid_num

# # #             tv = total_variation(adv_patch)
# # #             disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
# # #             tv_loss = tv * args.tv_loss
# # #             disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

# # #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
# # #             ep_det_loss += det_loss.detach().item()
# # #             ep_tv_loss += tv_loss.detach().item()
# # #             ep_loss += loss.item()

# # #             loss.backward()
# # #             optimizerG.step()
# # #             optimizerD.step()
# # #             optimizerG.zero_grad()
# # #             optimizerD.zero_grad()

# # #             bt1 = time.time()
# # #             if i_batch % 20 == 0:
# # #                 iteration = epoch_length * epoch + i_batch

# # #                 writer.add_scalar('loss/total_loss', loss.item(), iteration)
# # #                 writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
# # #                 writer.add_scalar('loss/tv_loss', tv.item(), iteration)
# # #                 writer.add_scalar('loss/disc_loss', disc.item(), iteration)
# # #                 writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
# # #                 writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
# # #                 writer.add_scalar('misc/epoch', epoch, iteration)
# # #                 writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)

# # #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# # #                 writer.add_image('patch', adv_patch[0], iteration)
# # #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# # #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# # #                 torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

# # #             bt0 = time.time()
# # #         et1 = time.time()
# # #         ep_det_loss = ep_det_loss / len(loader)
# # #         #         ep_nps_loss = ep_nps_loss/len(loader)
# # #         ep_tv_loss = ep_tv_loss / len(loader)
# # #         ep_loss = ep_loss / len(loader)
# # #         D_loss = D_loss / len(loader)
# # #         et0 = time.time()
# # #         writer.flush()
# # #     writer.close()
# # #     return gen


# # # def train_z(gen=None):
# # #     if gen is None:
# # #         gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
# # #         suffix_load = pargs.gen_suffix
# # #         result_dir = './results/result_' + suffix_load
# # #         d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
# # #         gen.load_state_dict(d)
# # #     gen.to(device)
# # #     gen.eval()
# # #     for p in gen.parameters():
# # #         p.requires_grad = False

# # #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# # #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# # #     writer = SummaryWriter(logdir=writer_logdir + '_z')

# # #     # Generate stating point
# # #     z0 = torch.randn(*args.z_shape, device=device)
# # #     z = z0.detach().clone()
# # #     z.requires_grad_(True)

# # #     optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
# # #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
# # #                                                      min_lr=args.learning_rate_z / 100)

# # #     et0 = time.time()
# # #     for epoch in range(1, args.z_epochs + 1):
# # #         ep_det_loss = 0
# # #         #     ep_nps_loss = 0
# # #         ep_tv_loss = 0
# # #         ep_loss = 0
# # #         bt0 = time.time()
# # #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# # #                                                     total=epoch_length):
# # #             img_batch = img_batch.to(device)
# # #             lab_batch = lab_batch.to(device)
# # #             z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

# # #             adv_patch = gen.generate(z_crop)
# # #             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
# # #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# # #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# # #             p_img_batch = patch_applier(img_batch, adv_batch_t)
# # #             det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
# # #             if valid_num > 0:
# # #                 det_loss = det_loss / valid_num

# # #             tv = total_variation(adv_patch)
# # #             tv_loss = tv * args.tv_loss
# # #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
# # #             ep_det_loss += det_loss.detach().item()
# # #             ep_tv_loss += tv_loss.detach().item()
# # #             ep_loss += loss.item()

# # #             loss.backward()
# # #             optimizer.step()
# # #             optimizer.zero_grad()
# # #             bt1 = time.time()
# # #             if i_batch % 20 == 0:
# # #                 iteration = epoch_length * epoch + i_batch

# # #                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
# # #                 writer.add_scalar('misc/epoch', epoch, iteration)
# # #                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

# # #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# # #                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
# # #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# # #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# # #                 rpath = os.path.join(results_dir, 'z%d' % epoch)
# # #                 np.save(rpath, z.detach().cpu().numpy())
# # #             bt0 = time.time()
# # #         et1 = time.time()
# # #         ep_det_loss = ep_det_loss / len(loader)
# # #         ep_tv_loss = ep_tv_loss / len(loader)
# # #         ep_loss = ep_loss / len(loader)
# # #         if epoch > 300:
# # #             scheduler.step(ep_loss)
# # #         et0 = time.time()
# # #         writer.flush()
# # #     writer.close()
# # #     return 0


# # # if pargs.method == 'RCA':
# # #     train_patch()
# # # elif pargs.method == 'TCA':
# # #     train_patch()
# # # elif pargs.method == 'EGA':
# # #     train_EGA()
# # # elif pargs.method == 'TCEGA':
# # #     gen = train_EGA()
# # #     print('Start optimize z')
# # #     train_z(gen)

# # import os
# # import torch
# # import torch.optim as optim
# # import itertools
# # from tensorboardX import SummaryWriter
# # from datetime import datetime
# # from tqdm import tqdm
# # import time
# # import argparse
# # import numpy as np
# # import sys

# # # Add YOLOv5 to path
# # sys.path.append('../yolov5')

# # from yolov5_wrapper import YOLOv5Wrapper, extract_predictions_for_loss
# # try:
# #     from yolo2 import load_data
# # except ImportError:
# #     import load_data  # Fallback if yolo2 module structure is different
# # from load_data_yolov5 import MaxProbExtractorYOLOv5
# # from attack_utils import *
# # from cfg import get_cfgs
# # from tps_grid_gen import TPSGridGen
# # from generator_dim import GAN_dis


# # parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Adversarial Training')
# # parser.add_argument('--net', default='yolov5', help='target net name')
# # parser.add_argument('--method', default='TCEGA', help='method name')
# # parser.add_argument('--suffix', default=None, help='suffix name')
# # parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
# # parser.add_argument('--epoch', type=int, default=None, help='')
# # parser.add_argument('--z_epoch', type=int, default=None, help='')
# # parser.add_argument('--device', default='cuda:0', help='')
# # parser.add_argument('--yolov5_weights', default='../yolov5/yolov5s.pt', help='YOLOv5 weights path')
# # parser.add_argument('--img_size', type=int, default=640, help='YOLOv5 input size')
# # pargs = parser.parse_args()


# # # Modify cfg to work with YOLOv5
# # args, kwargs = get_cfgs('yolov2', pargs.method)  # Keep config structure
# # kwargs['name'] = 'yolov5'  # Update name

# # if pargs.epoch is not None:
# #     args.n_epochs = pargs.epoch
# # if pargs.z_epoch is not None:
# #     args.z_epochs = pargs.z_epoch
# # if pargs.suffix is None:
# #     pargs.suffix = pargs.net + '_' + pargs.method

# # device = torch.device(pargs.device)

# # # Load YOLOv5 model
# # yolo_wrapper = YOLOv5Wrapper(weights=pargs.yolov5_weights, device=pargs.device, img_size=pargs.img_size)
# # yolo_model = yolo_wrapper.model.model  # Access the actual model for gradient computation

# # # Load data
# # img_dir_train = './data/INRIAPerson/Train/pos'
# # lab_dir_train = './data/train_labels'
# # train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
# # train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)

# # # Adversarial components
# # target_func = lambda obj, cls: obj
# # patch_applier = load_data.PatchApplier().to(device)
# # patch_transformer = load_data.PatchTransformer().to(device)
# # total_variation = load_data.TotalVariation().to(device)

# # target_control_points = torch.tensor(list(itertools.product(
# #     torch.arange(-1.0, 1.00001, 2.0 / 4),
# #     torch.arange(-1.0, 1.00001, 2.0 / 4),
# # )))

# # tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
# # tps.to(device)

# # results_dir = './results/result_' + pargs.suffix

# # print(results_dir)
# # if not os.path.exists(results_dir):
# #     os.makedirs(results_dir)

# # loader = train_loader
# # epoch_length = len(loader)
# # print(f'One epoch is {len(loader)}')


# # def get_det_loss_yolov5(model, p_img_batch, lab_batch, args, kwargs):
# #     """Calculate detection loss for YOLOv5"""
# #     # Get raw predictions
# #     person_scores = extract_predictions_for_loss(model, p_img_batch, conf_thresh=0.01)
    
# #     # Calculate loss - we want to minimize person detection scores
# #     valid_mask = []
# #     for i in range(lab_batch.size(0)):
# #         # Check if image has person annotations
# #         has_person = False
# #         for j in range(kwargs['max_lab']):
# #             if lab_batch[i][j][0] == 0 and lab_batch[i][j][1] != -1:  # person class
# #                 has_person = True
# #                 break
# #         valid_mask.append(has_person)
    
# #     valid_mask = torch.tensor(valid_mask, device=device)
# #     valid_num = valid_mask.sum().item()
    
# #     if valid_num > 0:
# #         # Loss is negative log of (1 - detection probability)
# #         # This encourages low detection scores
# #         det_loss = -torch.log(1 - person_scores[valid_mask].clamp(0, 0.99) + 1e-8).mean()
# #     else:
# #         det_loss = torch.tensor(0.0, device=device)
    
# #     return det_loss, valid_num


# # def train_patch():
# #     def generate_patch(type):
# #         cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
# #         if type == 'gray':
# #             adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
# #         elif type == 'random':
# #             adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
# #         else:
# #             raise ValueError
# #         return adv_patch

# #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# #     writer = SummaryWriter(logdir=writer_logdir)

# #     adv_patch = generate_patch("gray").to(device)
# #     adv_patch.requires_grad_(True)

# #     optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
# #                                                      min_lr=args.learning_rate / 100)

# #     et0 = time.time()
# #     for epoch in range(1, args.n_epochs + 1):
# #         ep_det_loss = 0
# #         ep_tv_loss = 0
# #         ep_loss = 0
# #         bt0 = time.time()
# #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# #                                                     total=epoch_length):
# #             img_batch = img_batch.to(device)
# #             lab_batch = lab_batch.to(device)
# #             adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# #             adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5)
# #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# #             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
# #             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
# #             if valid_num > 0:
# #                 det_loss = det_loss / valid_num

# #             tv = total_variation(adv_patch_crop)
# #             tv_loss = tv * args.tv_loss
# #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
# #             ep_det_loss += det_loss.detach().cpu().numpy()
# #             ep_tv_loss += tv_loss.detach().cpu().numpy()
# #             ep_loss += loss.item()

# #             loss.backward()
# #             optimizer.step()
# #             optimizer.zero_grad()
# #             adv_patch.data.clamp_(0, 1)  # keep patch in image range

# #             bt1 = time.time()
# #             if i_batch % 20 == 0:
# #                 iteration = epoch_length * epoch + i_batch

# #                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('misc/epoch', epoch, iteration)
# #                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

# #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# #                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
# #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# #             bt0 = time.time()
# #         et1 = time.time()
# #         ep_det_loss = ep_det_loss / len(loader)
# #         ep_tv_loss = ep_tv_loss / len(loader)
# #         ep_loss = ep_loss / len(loader)
# #         if epoch > 300:
# #             scheduler.step(ep_loss)
# #         et0 = time.time()
# #         writer.flush()
# #     writer.close()
# #     return 0


# # def train_EGA():
# #     gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
# #     gen.to(device)
# #     gen.train()

# #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# #     writer = SummaryWriter(logdir=writer_logdir)

# #     optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
# #     optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

# #     et0 = time.time()
# #     for epoch in range(1, args.n_epochs + 1):
# #         ep_det_loss = 0
# #         ep_tv_loss = 0
# #         ep_loss = 0
# #         D_loss = 0
# #         bt0 = time.time()
# #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# #                                                     total=epoch_length):
# #             img_batch = img_batch.to(device)
# #             lab_batch = lab_batch.to(device)

# #             z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

# #             adv_patch = gen.generate(z)
# #             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
# #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# #             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
# #             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)

# #             if valid_num > 0:
# #                 det_loss = det_loss / valid_num

# #             tv = total_variation(adv_patch)
# #             disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
# #             tv_loss = tv * args.tv_loss
# #             disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

# #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
# #             ep_det_loss += det_loss.detach().item()
# #             ep_tv_loss += tv_loss.detach().item()
# #             ep_loss += loss.item()

# #             loss.backward()
# #             optimizerG.step()
# #             optimizerD.step()
# #             optimizerG.zero_grad()
# #             optimizerD.zero_grad()

# #             bt1 = time.time()
# #             if i_batch % 20 == 0:
# #                 iteration = epoch_length * epoch + i_batch

# #                 writer.add_scalar('loss/total_loss', loss.item(), iteration)
# #                 writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
# #                 writer.add_scalar('loss/tv_loss', tv.item(), iteration)
# #                 writer.add_scalar('loss/disc_loss', disc.item(), iteration)
# #                 writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
# #                 writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
# #                 writer.add_scalar('misc/epoch', epoch, iteration)
# #                 writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)

# #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# #                 writer.add_image('patch', adv_patch[0], iteration)
# #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# #                 torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

# #             bt0 = time.time()
# #         et1 = time.time()
# #         ep_det_loss = ep_det_loss / len(loader)
# #         ep_tv_loss = ep_tv_loss / len(loader)
# #         ep_loss = ep_loss / len(loader)
# #         D_loss = D_loss / len(loader)
# #         et0 = time.time()
# #         writer.flush()
# #     writer.close()
# #     return gen


# # def train_z(gen=None):
# #     if gen is None:
# #         gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
# #         suffix_load = pargs.gen_suffix
# #         result_dir = './results/result_' + suffix_load
# #         d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
# #         gen.load_state_dict(d)
# #     gen.to(device)
# #     gen.eval()
# #     for p in gen.parameters():
# #         p.requires_grad = False

# #     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# #     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
# #     writer = SummaryWriter(logdir=writer_logdir + '_z')

# #     # Generate starting point
# #     z0 = torch.randn(*args.z_shape, device=device)
# #     z = z0.detach().clone()
# #     z.requires_grad_(True)

# #     optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
# #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
# #                                                      min_lr=args.learning_rate_z / 100)

# #     et0 = time.time()
# #     for epoch in range(1, args.z_epochs + 1):
# #         ep_det_loss = 0
# #         ep_tv_loss = 0
# #         ep_loss = 0
# #         bt0 = time.time()
# #         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
# #                                                     total=epoch_length):
# #             img_batch = img_batch.to(device)
# #             lab_batch = lab_batch.to(device)
# #             z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

# #             adv_patch = gen.generate(z_crop)
# #             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
# #             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
# #                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# #             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
# #             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
# #             if valid_num > 0:
# #                 det_loss = det_loss / valid_num

# #             tv = total_variation(adv_patch)
# #             tv_loss = tv * args.tv_loss
# #             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
# #             ep_det_loss += det_loss.detach().item()
# #             ep_tv_loss += tv_loss.detach().item()
# #             ep_loss += loss.item()

# #             loss.backward()
# #             optimizer.step()
# #             optimizer.zero_grad()
# #             bt1 = time.time()
# #             if i_batch % 20 == 0:
# #                 iteration = epoch_length * epoch + i_batch

# #                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
# #                 writer.add_scalar('misc/epoch', epoch, iteration)
# #                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

# #             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
# #                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
# #                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
# #                 np.save(rpath, adv_patch.detach().cpu().numpy())
# #                 rpath = os.path.join(results_dir, 'z%d' % epoch)
# #                 np.save(rpath, z.detach().cpu().numpy())
# #             bt0 = time.time()
# #         et1 = time.time()
# #         ep_det_loss = ep_det_loss / len(loader)
# #         ep_tv_loss = ep_tv_loss / len(loader)
# #         ep_loss = ep_loss / len(loader)
# #         if epoch > 300:
# #             scheduler.step(ep_loss)
# #         et0 = time.time()
# #         writer.flush()
# #     writer.close()
# #     return 0


# # if pargs.method == 'RCA':
# #     train_patch()
# # elif pargs.method == 'TCA':
# #     train_patch()
# # elif pargs.method == 'EGA':
# #     train_EGA()
# # elif pargs.method == 'TCEGA':
# #     gen = train_EGA()
# #     print('Start optimize z')
# #     train_z(gen)

# import os
# import torch
# import torch.optim as optim
# import itertools
# from tensorboardX import SummaryWriter
# from datetime import datetime
# from tqdm import tqdm
# import time
# import argparse
# import numpy as np
# import sys

# # Add YOLOv5 to path
# sys.path.append('../yolov5')

# from yolov5_wrapper import YOLOv5Wrapper
# try:
#     from yolo2 import load_data
# except ImportError:
#     import load_data  # Fallback if yolo2 module structure is different

# # Try to import utils functions
# try:
#     from attack_utils import *
# except ImportError:
#     print("Warning: Could not import from utils.py")

# # Import or define missing utility functions
# try:
#     # Test if random_crop is available
#     random_crop
# except NameError:
#     # Define random_crop if it's missing
#     def random_crop(tensor, size, pos='random', crop_type='rect'):
#         """Randomly crop a region from the input tensor"""
#         if isinstance(size, int):
#             size = [size, size]
        
#         batch_size, channels, height, width = tensor.shape
#         crop_h, crop_w = size
        
#         # Ensure crop size doesn't exceed tensor size
#         crop_h = min(crop_h, height)
#         crop_w = min(crop_w, width)
        
#         if pos == 'random':
#             y = np.random.randint(0, height - crop_h + 1) if height > crop_h else 0
#             x = np.random.randint(0, width - crop_w + 1) if width > crop_w else 0
#         elif pos == 'center':
#             y = (height - crop_h) // 2
#             x = (width - crop_w) // 2
#         else:
#             y = (height - crop_h) // 2
#             x = (width - crop_w) // 2
        
#         cropped = tensor[:, :, y:y+crop_h, x:x+crop_w]
#         return cropped, x, y

# from cfg import get_cfgs
# from tps_grid_gen import TPSGridGen
# from generator_dim import GAN_dis


# parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Adversarial Training')
# parser.add_argument('--net', default='yolov5', help='target net name')
# parser.add_argument('--method', default='TCEGA', help='method name')
# parser.add_argument('--suffix', default=None, help='suffix name')
# parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
# parser.add_argument('--epoch', type=int, default=None, help='')
# parser.add_argument('--z_epoch', type=int, default=None, help='')
# parser.add_argument('--device', default='cuda:0', help='')
# parser.add_argument('--yolov5_weights', default='../yolov5/yolov5s.pt', help='YOLOv5 weights path')
# parser.add_argument('--img_size', type=int, default=640, help='YOLOv5 input size')
# pargs = parser.parse_args()


# # Modify cfg to work with YOLOv5
# args, kwargs = get_cfgs('yolov2', pargs.method)  # Keep config structure
# kwargs['name'] = 'yolov5'  # Update name

# if pargs.epoch is not None:
#     args.n_epochs = pargs.epoch
# if pargs.z_epoch is not None:
#     args.z_epochs = pargs.z_epoch
# if pargs.suffix is None:
#     pargs.suffix = pargs.net + '_' + pargs.method

# device = torch.device(pargs.device)

# # Load YOLOv5 model
# yolo_wrapper = YOLOv5Wrapper(weights=pargs.yolov5_weights, device=pargs.device, img_size=pargs.img_size)
# # For training, we need the actual model, not the wrapper
# yolo_model = yolo_wrapper.model

# # Load data
# img_dir_train = './data/INRIAPerson/Train/pos'
# lab_dir_train = './data/train_labels'
# train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)

# # Adversarial components
# target_func = lambda obj, cls: obj
# patch_applier = load_data.PatchApplier().to(device)
# patch_transformer = load_data.PatchTransformer().to(device)
# total_variation = load_data.TotalVariation().to(device)

# target_control_points = torch.tensor(list(itertools.product(
#     torch.arange(-1.0, 1.00001, 2.0 / 4),
#     torch.arange(-1.0, 1.00001, 2.0 / 4),
# )))

# tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
# tps.to(device)

# results_dir = './results/result_' + pargs.suffix

# print(results_dir)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)

# loader = train_loader
# epoch_length = len(loader)
# print(f'One epoch is {len(loader)}')


# def get_det_loss_yolov5(model, p_img_batch, lab_batch, args, kwargs):
#     """Calculate detection loss for YOLOv5 with proper gradient flow"""
    
#     # Resize images to YOLOv5 input size if needed (using differentiable operations)
#     target_size = 640  # YOLOv5 default
#     if p_img_batch.shape[2] != target_size or p_img_batch.shape[3] != target_size:
#         p_img_batch_resized = torch.nn.functional.interpolate(
#             p_img_batch, 
#             size=(target_size, target_size), 
#             mode='bilinear', 
#             align_corners=False
#         )
#     else:
#         p_img_batch_resized = p_img_batch
    
#     # Forward pass with gradients
#     with torch.set_grad_enabled(True):
#         # Get model predictions
#         outputs = model(p_img_batch_resized)
    
#     # Simple approach: use the raw output values as a proxy for detection confidence
#     if isinstance(outputs, torch.Tensor):
#         # Flatten and get detection-related values
#         batch_size = outputs.size(0)
        
#         # For YOLOv5, we want to minimize the objectness scores
#         # The output tensor contains detection information
#         # We'll use a simple approach: take mean of positive values
        
#         # Reshape and get positive values (likely detections)
#         flat_output = outputs.view(batch_size, -1)
        
#         # Use sigmoid to convert to probabilities
#         probs = torch.sigmoid(flat_output)
        
#         # Get the mean of high confidence values as our target to minimize
#         # This encourages the model to produce lower detection scores
#         topk = min(100, probs.size(1))
#         topk_vals, _ = torch.topk(probs, k=topk, dim=1)
        
#         # Mean confidence per image
#         conf_scores = topk_vals.mean(dim=1)
        
#         # Check which images have person annotations
#         valid_mask = []
#         for i in range(batch_size):
#             has_person = False
#             for j in range(kwargs['max_lab']):
#                 if lab_batch[i][j][0] == 0 and lab_batch[i][j][1] != -1:  # person class
#                     has_person = True
#                     break
#             valid_mask.append(has_person)
        
#         valid_mask = torch.tensor(valid_mask, device=p_img_batch.device)
#         valid_num = valid_mask.sum().item()
        
#         if valid_num > 0:
#             # Loss encourages low confidence on images with people
#             det_loss = conf_scores[valid_mask].mean()
#         else:
#             det_loss = torch.tensor(0.0, device=p_img_batch.device, requires_grad=True)
#     else:
#         # Fallback if output format is unexpected
#         det_loss = torch.tensor(0.5, device=p_img_batch.device, requires_grad=True)
#         valid_num = p_img_batch.size(0)
    
#     return det_loss, valid_num


# def train_patch():
#     def generate_patch(type):
#         cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
#         if type == 'gray':
#             adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
#         elif type == 'random':
#             adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
#         else:
#             raise ValueError
#         return adv_patch

#     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
#     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
#     writer = SummaryWriter(logdir=writer_logdir)

#     adv_patch = generate_patch("gray").to(device)
#     adv_patch.requires_grad_(True)

#     optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
#                                                      min_lr=args.learning_rate / 100)

#     et0 = time.time()
#     for epoch in range(1, args.n_epochs + 1):
#         print(args.n_epochs, epoch)
#         ep_det_loss = 0
#         ep_tv_loss = 0
#         ep_loss = 0
#         bt0 = time.time()
#         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
#                                                     total=epoch_length):
#             img_batch = img_batch.to(device)
#             lab_batch = lab_batch.to(device)
#             adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
#             adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5)
#             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
#                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
#             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
#             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
#             if valid_num > 0:
#                 det_loss = det_loss / valid_num

#             tv = total_variation(adv_patch_crop)
#             tv_loss = tv * args.tv_loss
#             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
#             ep_det_loss += det_loss.detach().cpu().numpy()
#             ep_tv_loss += tv_loss.detach().cpu().numpy()
#             ep_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             adv_patch.data.clamp_(0, 1)  # keep patch in image range

#             bt1 = time.time()
#             if i_batch % 20 == 0:
#                 iteration = epoch_length * epoch + i_batch

#                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('misc/epoch', epoch, iteration)
#                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

#             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
#                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
#                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
#                 np.save(rpath, adv_patch.detach().cpu().numpy())
#             bt0 = time.time()
#         et1 = time.time()
#         ep_det_loss = ep_det_loss / len(loader)
#         ep_tv_loss = ep_tv_loss / len(loader)
#         ep_loss = ep_loss / len(loader)
#         if epoch > 300:
#             scheduler.step(ep_loss)
#         et0 = time.time()
#         writer.flush()
#     writer.close()
#     return 0


# def train_EGA():
#     gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
#     gen.to(device)
#     gen.train()

#     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
#     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
#     writer = SummaryWriter(logdir=writer_logdir)

#     optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
#     optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

#     et0 = time.time()
#     for epoch in range(1, args.n_epochs + 1):
#         ep_det_loss = 0
#         ep_tv_loss = 0
#         ep_loss = 0
#         D_loss = 0
#         bt0 = time.time()
#         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
#                                                     total=epoch_length):
#             img_batch = img_batch.to(device)
#             lab_batch = lab_batch.to(device)

#             z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

#             adv_patch = gen.generate(z)
#             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
#             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
#                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
#             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
#             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)

#             if valid_num > 0:
#                 det_loss = det_loss / valid_num

#             tv = total_variation(adv_patch)
#             disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
#             tv_loss = tv * args.tv_loss
#             disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

#             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
#             ep_det_loss += det_loss.detach().item()
#             ep_tv_loss += tv_loss.detach().item()
#             ep_loss += loss.item()

#             loss.backward()
#             optimizerG.step()
#             optimizerD.step()
#             optimizerG.zero_grad()
#             optimizerD.zero_grad()

#             bt1 = time.time()
#             if i_batch % 20 == 0:
#                 iteration = epoch_length * epoch + i_batch

#                 writer.add_scalar('loss/total_loss', loss.item(), iteration)
#                 writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
#                 writer.add_scalar('loss/tv_loss', tv.item(), iteration)
#                 writer.add_scalar('loss/disc_loss', disc.item(), iteration)
#                 writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
#                 writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
#                 writer.add_scalar('misc/epoch', epoch, iteration)
#                 writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)

#             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
#                 writer.add_image('patch', adv_patch[0], iteration)
#                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
#                 np.save(rpath, adv_patch.detach().cpu().numpy())
#                 torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

#             bt0 = time.time()
#         et1 = time.time()
#         ep_det_loss = ep_det_loss / len(loader)
#         ep_tv_loss = ep_tv_loss / len(loader)
#         ep_loss = ep_loss / len(loader)
#         D_loss = D_loss / len(loader)
#         et0 = time.time()
#         writer.flush()
#     writer.close()
#     return gen


# def train_z(gen=None):
#     if gen is None:
#         gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
#         suffix_load = pargs.gen_suffix
#         result_dir = './results/result_' + suffix_load
#         d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
#         gen.load_state_dict(d)
#     gen.to(device)
#     gen.eval()
#     for p in gen.parameters():
#         p.requires_grad = False

#     TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
#     writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
#     writer = SummaryWriter(logdir=writer_logdir + '_z')

#     # Generate starting point
#     z0 = torch.randn(*args.z_shape, device=device)
#     z = z0.detach().clone()
#     z.requires_grad_(True)

#     optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
#                                                      min_lr=args.learning_rate_z / 100)

#     et0 = time.time()
#     for epoch in range(1, args.z_epochs + 1):
#         ep_det_loss = 0
#         ep_tv_loss = 0
#         ep_loss = 0
#         bt0 = time.time()
#         for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
#                                                     total=epoch_length):
#             img_batch = img_batch.to(device)
#             lab_batch = lab_batch.to(device)
#             z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

#             adv_patch = gen.generate(z_crop)
#             adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
#             adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
#                                             pooling=args.pooling, old_fasion=kwargs['old_fasion'])
#             p_img_batch = patch_applier(img_batch, adv_batch_t)
            
#             det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
#             if valid_num > 0:
#                 det_loss = det_loss / valid_num

#             tv = total_variation(adv_patch)
#             tv_loss = tv * args.tv_loss
#             loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
#             ep_det_loss += det_loss.detach().item()
#             ep_tv_loss += tv_loss.detach().item()
#             ep_loss += loss.item()

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             bt1 = time.time()
#             if i_batch % 20 == 0:
#                 iteration = epoch_length * epoch + i_batch

#                 writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
#                 writer.add_scalar('misc/epoch', epoch, iteration)
#                 writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

#             if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
#                 writer.add_image('patch', adv_patch.squeeze(0), iteration)
#                 rpath = os.path.join(results_dir, 'patch%d' % epoch)
#                 np.save(rpath, adv_patch.detach().cpu().numpy())
#                 rpath = os.path.join(results_dir, 'z%d' % epoch)
#                 np.save(rpath, z.detach().cpu().numpy())
#             bt0 = time.time()
#         et1 = time.time()
#         ep_det_loss = ep_det_loss / len(loader)
#         ep_tv_loss = ep_tv_loss / len(loader)
#         ep_loss = ep_loss / len(loader)
#         if epoch > 300:
#             scheduler.step(ep_loss)
#         et0 = time.time()
#         writer.flush()
#     writer.close()
#     return 0


# if pargs.method == 'RCA':
#     train_patch()
# elif pargs.method == 'TCA':
#     train_patch()
# elif pargs.method == 'EGA':
#     train_EGA()
# elif pargs.method == 'TCEGA':
#     gen = train_EGA()
#     print('Start optimize z')
#     train_z(gen)

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
import sys

# Add YOLOv5 to path
sys.path.append('../yolov5')

from yolov5_wrapper import YOLOv5Wrapper
try:
    from yolo2 import load_data
except ImportError:
    import load_data  # Fallback if yolo2 module structure is different

# Try to import utils functions
try:
    from utils import *
except ImportError:
    print("Warning: Could not import from utils.py")

# Import or define missing utility functions
try:
    # Test if random_crop is available
    random_crop
except NameError:
    # Define random_crop if it's missing
    def random_crop(tensor, size, pos='random', crop_type='rect'):
        """Randomly crop a region from the input tensor"""
        if isinstance(size, int):
            size = [size, size]
        
        batch_size, channels, height, width = tensor.shape
        crop_h, crop_w = size
        
        # Ensure crop size doesn't exceed tensor size
        crop_h = min(crop_h, height)
        crop_w = min(crop_w, width)
        
        if pos == 'random':
            y = np.random.randint(0, height - crop_h + 1) if height > crop_h else 0
            x = np.random.randint(0, width - crop_w + 1) if width > crop_w else 0
        elif pos == 'center':
            y = (height - crop_h) // 2
            x = (width - crop_w) // 2
        else:
            y = (height - crop_h) // 2
            x = (width - crop_w) // 2
        
        cropped = tensor[:, :, y:y+crop_h, x:x+crop_w]
        return cropped, x, y

from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis


parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Adversarial Training')
parser.add_argument('--net', default='yolov5', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--suffix', default=None, help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--z_epoch', type=int, default=None, help='')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--yolov5_weights', default='../yolov5/yolov5s.pt', help='YOLOv5 weights path')
parser.add_argument('--img_size', type=int, default=640, help='YOLOv5 input size')
pargs = parser.parse_args()


# Modify cfg to work with YOLOv5
args, kwargs = get_cfgs('yolov2', pargs.method)  # Keep config structure
kwargs['name'] = 'yolov5'  # Update name

if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.z_epoch is not None:
    args.z_epochs = pargs.z_epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device(pargs.device)

# Load YOLOv5 model
yolo_wrapper = YOLOv5Wrapper(weights=pargs.yolov5_weights, device=pargs.device, img_size=pargs.img_size)
# For training, we need the actual model, not the wrapper
yolo_model = yolo_wrapper.model

# Load data
img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels'
train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)

# Adversarial components
target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

results_dir = './results/result_' + pargs.suffix

print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')


def get_det_loss_yolov5(model, p_img_batch, lab_batch, args, kwargs):
    """Calculate detection loss for YOLOv5 with proper gradient flow"""
    
    # Resize images to YOLOv5 input size if needed (using differentiable operations)
    target_size = 640  # YOLOv5 default
    if p_img_batch.shape[2] != target_size or p_img_batch.shape[3] != target_size:
        p_img_batch_resized = torch.nn.functional.interpolate(
            p_img_batch, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
    else:
        p_img_batch_resized = p_img_batch
    
    # Forward pass with gradients
    with torch.set_grad_enabled(True):
        # Get model predictions
        outputs = model(p_img_batch_resized)
    
    # Simple approach: use the raw output values as a proxy for detection confidence
    if isinstance(outputs, torch.Tensor):
        # Flatten and get detection-related values
        batch_size = outputs.size(0)
        
        # For YOLOv5, we want to minimize the objectness scores
        # The output tensor contains detection information
        # We'll use a simple approach: take mean of positive values
        
        # Reshape and get positive values (likely detections)
        flat_output = outputs.view(batch_size, -1)
        
        # Use sigmoid to convert to probabilities
        probs = torch.sigmoid(flat_output)
        
        # Get the mean of high confidence values as our target to minimize
        # This encourages the model to produce lower detection scores
        topk = min(100, probs.size(1))
        topk_vals, _ = torch.topk(probs, k=topk, dim=1)
        
        # Mean confidence per image
        conf_scores = topk_vals.mean(dim=1)
        
        # Check which images have person annotations
        valid_mask = []
        for i in range(batch_size):
            has_person = False
            for j in range(kwargs['max_lab']):
                if lab_batch[i][j][0] == 0 and lab_batch[i][j][1] != -1:  # person class
                    has_person = True
                    break
            valid_mask.append(has_person)
        
        valid_mask = torch.tensor(valid_mask, device=p_img_batch.device)
        valid_num = valid_mask.sum().item()
        
        if valid_num > 0:
            # Loss encourages low confidence on images with people
            det_loss = conf_scores[valid_mask].mean()
        else:
            det_loss = torch.tensor(0.0, device=p_img_batch.device, requires_grad=True)
    else:
        # Fallback if output format is unexpected
        det_loss = torch.tensor(0.5, device=p_img_batch.device, requires_grad=True)
        valid_num = p_img_batch.size(0)
    
    return det_loss, valid_num


def train_patch():
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError
        return adv_patch

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)

    adv_patch = generate_patch("gray").to(device)
    adv_patch.requires_grad_(True)

    optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate / 100)

    et0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5)
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch_crop)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch.data.clamp_(0, 1)  # keep patch in image range

            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
            bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        
        # Print epoch statistics
        print(f"\nEpoch {epoch}/{args.n_epochs} Summary:")
        print(f"  Total Loss: {ep_loss:.4f}")
        print(f"  Detection Loss: {ep_det_loss:.4f}")
        print(f"  TV Loss: {ep_tv_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time: {et1 - et0:.2f}s")
        print("-" * 60)
        
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()
        writer.flush()
    writer.close()
    
    # Final summary
    print(f"\nTraining completed!")
    print(f"Final patch saved to: {os.path.join(results_dir, f'patch{args.n_epochs}.npy')}")
    
    return 0


def train_EGA():
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    gen.to(device)
    gen.train()

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir)

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))

    et0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        D_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)

            z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)

            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            tv_loss = tv * args.tv_loss
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizerG.step()
            optimizerD.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.item(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
                writer.add_scalar('loss/tv_loss', tv.item(), iteration)
                writer.add_scalar('loss/disc_loss', disc.item(), iteration)
                writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
                writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)

            if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                writer.add_image('patch', adv_patch[0], iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                torch.save(gen.state_dict(), os.path.join(results_dir, pargs.suffix + '.pkl'))

            bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        D_loss = D_loss / len(loader)
        et0 = time.time()
        writer.flush()
    writer.close()
    return gen


def train_z(gen=None):
    if gen is None:
        gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        suffix_load = pargs.gen_suffix
        result_dir = './results/result_' + suffix_load
        d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu')
        gen.load_state_dict(d)
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer_logdir = os.path.join('./results/runs', TIMESTAMP + '_' + pargs.suffix)
    writer = SummaryWriter(logdir=writer_logdir + '_z')

    # Generate starting point
    z0 = torch.randn(*args.z_shape, device=device)
    z = z0.detach().clone()
    z.requires_grad_(True)

    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate_z / 100)

    et0 = time.time()
    for epoch in range(1, args.z_epochs + 1):
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            
            det_loss, valid_num = get_det_loss_yolov5(yolo_model, p_img_batch, lab_batch, args, kwargs)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bt1 = time.time()
            if i_batch % 20 == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                rpath = os.path.join(results_dir, 'z%d' % epoch)
                np.save(rpath, z.detach().cpu().numpy())
            bt0 = time.time()
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        
        # Print epoch statistics
        print(f"\nZ-Optimization Epoch {epoch}/{args.z_epochs} Summary:")
        print(f"  Total Loss: {ep_loss:.4f}")
        print(f"  Detection Loss: {ep_det_loss:.4f}")
        print(f"  TV Loss: {ep_tv_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time: {et1 - et0:.2f}s")
        print("-" * 60)
        
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()
        writer.flush()
    writer.close()
    
    # Final summary
    print(f"\nZ-optimization completed!")
    print(f"Final z saved to: {os.path.join(results_dir, f'z{args.z_epochs}.npy')}")
    
    return 0


if pargs.method == 'RCA':
    train_patch()
elif pargs.method == 'TCA':
    train_patch()
elif pargs.method == 'EGA':
    train_EGA()
elif pargs.method == 'TCEGA':
    gen = train_EGA()
    print('Start optimize z')
    train_z(gen)
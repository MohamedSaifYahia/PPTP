# # import os
# # import torch
# # import itertools
# # from tqdm import tqdm
# # import argparse
# # from scipy.interpolate import interp1d
# # from torchvision import transforms
# # import numpy as np
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import fnmatch
# # import re
# # import sys

# # # Add yolov5 to path, assuming it's in a 'yolov5' subdirectory
# # sys.path.insert(0, './yolov5')

# # # --- Original Attack Code Imports ---
# # from yolo2 import load_data
# # from cfg import get_cfgs
# # from tps_grid_gen import TPSGridGen
# # from generator_dim import GAN_dis

# # # --- YOLOv5 Imports ---
# # from models.common import DetectMultiBackend
# # from yolov5.utils.general import (non_max_suppression, xywh2xyxy, xyxy2xywh)
# # from yolov5.utils.torch_utils import select_device
# # from yolov5.utils.metrics import box_iou

# # unloader = transforms.ToPILImage()

# # # --- Argument Parser ---
# # parser = argparse.ArgumentParser(description='YOLOv5 Adversarial Texture Evaluation')
# # # YOLOv5 specific
# # parser.add_argument('--weights', type=str, default='yolov5/yolov5s.pt', help='YOLOv5 weights path')
# # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

# # # Attack specific
# # parser.add_argument('--method', default='TCEGA', help='method name: RCA, TCA, EGA, TCEGA')
# # parser.add_argument('--suffix', default=None, help='suffix for saving results')
# # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# # parser.add_argument('--prepare_data', action='store_true', help='generate test labels using the model')
# # parser.add_argument('--load_path', default=None, help='path to saved patch (.npy) or GAN (.pkl)')
# # parser.add_argument('--load_path_z', default=None, help='path to saved z for TCEGA (.npy)')
# # parser.add_argument('--npz_dir', default=None, help='directory of .npz files to plot PR curve')

# # pargs = parser.parse_args()

# # # --- Configs and Setup ---
# # if pargs.suffix is None:
# #     pargs.suffix = 'yolov5_' + pargs.method

# # batch_size = get_cfgs('yolov2', pargs.method, 'test')[0].batch_size
# # device = select_device(pargs.device, batch_size=batch_size)

# # # --- Load YOLOv5 Model ---
# # print("Loading YOLOv5 model...")
# # model = DetectMultiBackend(pargs.weights, device=device, dnn=False)
# # model.eval()
# # stride = model.stride
# # names = model.names

# # # --- Load Attack-Specific Components ---
# # args, kwargs = get_cfgs('yolov2', pargs.method, 'test')

# # patch_applier = load_data.PatchApplier().to(device)
# # patch_transformer = load_data.PatchTransformer().to(device)

# # # --- UTILITY FUNCTIONS ---

# # def custom_collate_fn(batch):
# #     images = [item[0] for item in batch]
# #     labels = [item[1] for item in batch]
# #     images_batch = torch.stack(images, 0)
# #     max_len = max(len(l) for l in labels) if labels else 0
# #     if max_len == 0:
# #         return images_batch, torch.empty(len(batch), 0, 5)
# #     padded_labels = torch.full((len(batch), max_len, 5), -1.0)
# #     for i, label in enumerate(labels):
# #         num_objects = len(label)
# #         if num_objects > 0:
# #             padded_labels[i, :num_objects, :] = label
# #     return images_batch, padded_labels

# # def random_crop(x_in, size, pos=None, crop_type='random'):
# #     if size is None: raise ValueError("Crop size cannot be None in random_crop")
# #     if len(x_in.shape) != 4: raise ValueError("random_crop expects a 4D tensor [B, C, H, W]")
# #     h_c, w_c = size
# #     if crop_type == 'recursive':
# #         tiled_x = x_in.repeat(1, 1, 3, 3) 
# #         x = tiled_x[0]
# #         c, h, w = x.shape
# #         if h_c > h or w_c > w: raise ValueError(f"Recursive crop size ({h_c}, {w_c}) > tiled image ({h}, {w})")
# #         top = np.random.randint(0, h - h_c + 1); left = np.random.randint(0, w - w_c + 1)
# #     else:
# #         x = x_in[0]
# #         c, h, w = x.shape
# #         if h_c > h or w_c > w: raise ValueError(f"Crop size ({h_c}, {w_c}) > image ({h}, {w})")
# #         if crop_type == 'random':
# #             if pos is None: top = np.random.randint(0, h - h_c + 1); left = np.random.randint(0, w - w_c + 1)
# #             else: top = np.random.randint(pos[0], pos[1]); left = np.random.randint(pos[2], pos[3])
# #         elif crop_type == 'center': top = (h - h_c) // 2; left = (w - w_c) // 2
# #         else: raise NotImplementedError(f"crop_type '{crop_type}' not recognized.")
# #     return x[:, top:top+h_c, left:left+w_c].unsqueeze(0), left, top

# # # --- Data Preparation ---
# # if pargs.prepare_data:
# #     # ... (code is correct, omitting for brevity)
# #     conf_thresh = 0.25; iou_thresh = 0.45
# #     img_ori_dir = './data/INRIAPerson/Test/pos'; img_dir = './data/test_padded'; lab_dir = './data/test_lab_yolov5'
# #     data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], pargs.img_size, shuffle=False)
# #     loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=batch_size, shuffle=False, num_workers=4)
# #     if not os.path.exists(lab_dir): os.makedirs(lab_dir)
# #     if not os.path.exists(img_dir): os.makedirs(img_dir)
# #     print('Preparing the test data with YOLOv5...');
# #     with torch.no_grad():
# #         for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
# #             data = data.to(device)
# #             pred = model(data, augment=False, visualize=False)
# #             pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)
# #             for i, det in enumerate(pred):
# #                 img_path = labs[i]; lab_path = os.path.join(lab_dir, os.path.basename(img_path))
# #                 img_save_path = os.path.join(img_dir, os.path.basename(img_path).replace('.txt', '.png'))
# #                 unloader(data[i].cpu()).save(img_save_path)
# #                 if len(det):
# #                     h, w = data[i].shape[1:]
# #                     boxes_xywh = xyxy2xywh(det[:, :4])
# #                     boxes_xywh[:, 0] /= w; boxes_xywh[:, 1] /= h; boxes_xywh[:, 2] /= w; boxes_xywh[:, 3] /= h
# #                     labels_to_save = torch.cat((det[:, -1].unsqueeze(1), boxes_xywh), 1)
# #                     np.savetxt(lab_path, labels_to_save.cpu().numpy(), fmt='%f')
# #                 else:
# #                     open(lab_path, 'w').close()
# #     print('Preparation done.'); sys.exit(0)

# # # --- Evaluation Function ---
# # def truths_length(truths):
# #     for i in range(truths.shape[0]):
# #         if truths[i][1] == -1: return i
# #     return truths.shape[0]

# # def test(model, loader, adv_cloth=None, gan=None, z=None, type=None,
# #          conf_thresh=0.01, iou_thresh=0.5):
# #     model.eval()
# #     total_gts = 0.0; positives = []
# #     with torch.no_grad():
# #         for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader), position=0):
# #             data = data.to(device); target = target.to(device)
            
# #             # --- THIS IS THE FINAL FIX ---
# #             if type == 'gan': # EGA method
# #                 z_rand = torch.randn(1, 128, *args.z_size, device=device)
# #                 adv_patch = gan.generate(z_rand)
# #             elif type == 'z': # TCEGA method
# #                 z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
# #                 adv_patch = gan.generate(z_crop)
# #             elif type == 'patch': # RCA/TCA methods
# #                 if args.crop_type is not None:
# #                     adv_patch, _, _ = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# #                 else:
# #                     adv_patch = adv_cloth
# #             else: adv_patch = None

# #             if adv_patch is not None:
# #                 adv_batch_t = patch_transformer(adv_patch, target, pargs.img_size, do_rotate=True, rand_loc=False,
# #                                                 pooling=args.pooling, old_fasion=kwargs['old_fasion'])
# #                 data = patch_applier(data, adv_batch_t)

# #             pred = model(data, augment=False, visualize=False)
# #             pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)
# #             for i, det in enumerate(pred):
# #                 truths = target[i]; num_gts = truths_length(truths); truths = truths[:num_gts]
# #                 truths = truths[truths[:, 0] == 0]; total_gts += len(truths)
# #                 if len(det) == 0: continue
# #                 if len(truths) > 0:
# #                     gt_boxes_xywh = truths[:, 1:]; gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
# #                     h, w = data[i].shape[1:]; gt_boxes_xyxy[:, [0, 2]] *= w; gt_boxes_xyxy[:, [1, 3]] *= h
# #                     pred_boxes_xyxy = det[:, :4]; iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
# #                     matched_gts = set()
# #                     for j in range(len(det)):
# #                         iou_scores = iou_matrix[j, :]
# #                         if len(iou_scores) == 0: positives.append((det[j, 4].item(), False)); continue
# #                         best_iou, best_gt_idx = torch.max(iou_scores, 0)
# #                         if best_iou > iou_thresh and best_gt_idx.item() not in matched_gts:
# #                             positives.append((det[j, 4].item(), True)); matched_gts.add(best_gt_idx.item())
# #                         else: positives.append((det[j, 4].item(), False))
# #                 else:
# #                     for j in range(len(det)): positives.append((det[j, 4].item(), False))
# #     if total_gts == 0: return [], [], 0.0, []
# #     positives = sorted(positives, key=lambda d: d[0], reverse=True)
# #     tps_arr, fps_arr, confs = [], [], []; tp_counter, fp_counter = 0, 0
# #     for conf, is_tp in positives:
# #         if is_tp: tp_counter += 1
# #         else: fp_counter += 1
# #         tps_arr.append(tp_counter); fps_arr.append(fp_counter); confs.append(conf)
# #     precision, recall = [], []
# #     for tp, fp in zip(tps_arr, fps_arr):
# #         if (fp + tp) > 0: recall.append(tp / total_gts); precision.append(tp / (fp + tp))
# #     if len(precision) > 1 and len(recall) > 1:
# #         p_interp = np.interp(np.linspace(0, 1, 101), np.flip(recall), np.flip(precision)); avg = p_interp.mean()
# #     elif len(precision) > 0: avg = precision[0] * recall[0]
# #     else: avg = 0.0
# #     return precision, recall, avg, confs

# # # --- Main Execution Logic ---
# # if pargs.npz_dir is None:
# #     lab_dir_test = './data/test_lab_yolov5'
# #     if not os.path.exists(lab_dir_test):
# #         print(f"Error: Label directory {lab_dir_test} not found.")
# #         print("Please run with --prepare_data first to generate labels for YOLOv5."); sys.exit(1)
        
# #     test_data = load_data.InriaDataset('./data/test_padded', lab_dir_test, kwargs['max_lab'], pargs.img_size, shuffle=False)
# #     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
# #     print(f'Test loader has {len(test_loader)} batches.')

# #     test_cloth, test_gan, test_z, cloth = None, None, None, None; test_type = None
# #     if pargs.method in ['RCA', 'TCA']:
# #         if not pargs.load_path: raise ValueError("Provide --load_path for RCA/TCA")
# #         cloth = torch.from_numpy(np.load(pargs.load_path)[:1]).to(device)
# #         test_cloth = cloth.detach().clone(); test_type = 'patch'
# #     elif pargs.method in ['EGA', 'TCEGA']:
# #         if not pargs.load_path: raise ValueError("Provide --load_path for EGA/TCEGA")
# #         gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
# #         gan.load_state_dict(torch.load(pargs.load_path, map_location=device)); gan.to(device).eval()
# #         for p in gan.parameters(): p.requires_grad = False
# #         test_gan = gan
# #         if pargs.method == 'EGA':
# #             test_type, test_z = 'gan', None; cloth = None # Cloth will be generated in test loop
# #         else: # TCEGA
# #             if not pargs.load_path_z: raise ValueError("Provide --load_path_z for TCEGA")
# #             z = torch.from_numpy(np.load(pargs.load_path_z)).to(device)
# #             test_z, test_type = z, 'z'; cloth = None # Cloth will be generated in test loop
# #     else: print("Evaluating clean model (no attack)...")
# #     save_dir = './test_results'
# #     if not os.path.exists(save_dir): os.makedirs(save_dir)
# #     save_path = os.path.join(save_dir, pargs.suffix)
# #     plt.figure(figsize=[10, 7])
# #     prec, rec, ap, confs = test(model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type=test_type, conf_thresh=0.01)
# #     save_data = {'prec': prec, 'rec': rec, 'ap': ap, 'confs': confs}
# #     if test_type is not None: # A patch was generated/loaded, we should save something
# #         # Re-generate one sample to save, since the ones in the loop are random
# #         if test_type == 'gan': final_cloth = test_gan.generate(torch.randn(1, 128, *args.z_size, device=device))
# #         elif test_type == 'z': z_crop, _, _ = random_crop(test_z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type); final_cloth = test_gan.generate(z_crop)
# #         else: final_cloth = test_cloth
# #         save_data['adv_patch'] = final_cloth.detach().cpu().numpy()
# #         unloader(final_cloth[0]).save(save_path + '.png')
# #     np.savez(save_path, **save_data)
# #     print(f'AP is {ap:.4f}')
# #     plt.plot(rec, prec, label=f'{pargs.suffix}: ap {ap:.3f}')
# #     plt.legend(); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
# #     plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(save_path + '_PR.png', dpi=300)
# # else:
# #     # ... (code is correct, omitting for brevity)
# #     files = fnmatch.filter(os.listdir(pargs.npz_dir), '*yolov5*.npz')
# #     order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
# #     files.sort(key=lambda x: order.get(re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group(), 1e5) if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) else 1e5)
# #     plt.figure(figsize=[10, 7])
# #     for file in files:
# #         save_path = os.path.join(pargs.npz_dir, file)
# #         try:
# #             with np.load(save_path, allow_pickle=True) as data:
# #                 plt.plot(data['rec'], data['prec'], label=f"{file.replace('.npz', '')}, ap: {data['ap']:.3f}")
# #                 if 'adv_patch' in data:
# #                     unloader(torch.from_numpy(data['adv_patch'][0])).save(save_path.replace('.npz', '.png'))
# #         except Exception as e: print(f"Could not process {file}: {e}")
# #     plt.plot([0, 1], [0, 1], 'k--')
# #     plt.legend(loc='lower left'); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
# #     plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(os.path.join(pargs.npz_dir, 'PR-curve-yolov5.png'), dpi=300)

# import os
# import torch
# import itertools
# from tqdm import tqdm
# import argparse
# from scipy.interpolate import interp1d
# from torchvision import transforms
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import fnmatch
# import re
# import sys

# # Add yolov5 to path, assuming it's in a 'yolov5' subdirectory
# sys.path.insert(0, './yolov5')

# # --- Original Attack Code Imports ---
# from yolo2 import load_data
# from cfg import get_cfgs
# from tps_grid_gen import TPSGridGen
# from generator_dim import GAN_dis

# # --- YOLOv5 Imports ---
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.general import (non_max_suppression, xywh2xyxy, xyxy2xywh)
# from yolov5.utils.torch_utils import select_device
# from yolov5.utils.metrics import box_iou

# unloader = transforms.ToPILImage()

# # --- Argument Parser ---
# parser = argparse.ArgumentParser(description='YOLOv5 Adversarial Texture Evaluation')
# # YOLOv5 specific
# parser.add_argument('--weights', type=str, default='yolov5/yolov5s.pt', help='YOLOv5 weights path')
# parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

# # Attack specific
# parser.add_argument('--method', default='TCEGA', help='method name: RCA, TCA, EGA, TCEGA')
# parser.add_argument('--suffix', default=None, help='suffix for saving results')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--prepare_data', action='store_true', help='generate test labels using the model')
# parser.add_argument('--load_path', default=None, help='path to saved patch (.npy) or GAN (.pkl)')
# parser.add_argument('--load_path_z', default=None, help='path to saved z for TCEGA (.npy)')
# parser.add_argument('--npz_dir', default=None, help='directory of .npz files to plot PR curve')

# pargs = parser.parse_args()

# # --- Configs and Setup ---
# if pargs.suffix is None:
#     pargs.suffix = 'yolov5_' + pargs.method

# batch_size = get_cfgs('yolov2', pargs.method, 'test')[0].batch_size
# device = select_device(pargs.device, batch_size=batch_size)

# # --- Load YOLOv5 Model ---
# print("Loading YOLOv5 model...")
# model = DetectMultiBackend(pargs.weights, device=device, dnn=False)
# model.eval()
# stride = model.stride
# names = model.names

# # --- Load Attack-Specific Components ---
# args, kwargs = get_cfgs('yolov2', pargs.method, 'test')

# patch_applier = load_data.PatchApplier().to(device)
# patch_transformer = load_data.PatchTransformer().to(device)

# # --- UTILITY FUNCTIONS ---

# def custom_collate_fn(batch):
#     images = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
#     images_batch = torch.stack(images, 0)
#     max_len = max(len(l) for l in labels) if labels else 0
#     if max_len == 0:
#         return images_batch, torch.empty(len(batch), 0, 5)
#     padded_labels = torch.full((len(batch), max_len, 5), -1.0)
#     for i, label in enumerate(labels):
#         num_objects = len(label)
#         if num_objects > 0:
#             padded_labels[i, :num_objects, :] = label
#     return images_batch, padded_labels

# def random_crop(x_in, size, pos=None, crop_type='random'):
#     if size is None: raise ValueError("Crop size cannot be None in random_crop")
#     if len(x_in.shape) != 4: raise ValueError("random_crop expects a 4D tensor [B, C, H, W]")
#     h_c, w_c = size
#     if crop_type == 'recursive':
#         tiled_x = x_in.repeat(1, 1, 3, 3) 
#         x = tiled_x[0]
#         c, h, w = x.shape
#         if h_c > h or w_c > w: raise ValueError(f"Recursive crop size ({h_c}, {w_c}) > tiled image ({h}, {w})")
#         top = np.random.randint(0, h - h_c + 1); left = np.random.randint(0, w - w_c + 1)
#     else:
#         x = x_in[0]
#         c, h, w = x.shape
#         if h_c > h or w_c > w: raise ValueError(f"Crop size ({h_c}, {w_c}) > image ({h}, {w})")
#         if crop_type == 'random':
#             if pos is None: top = np.random.randint(0, h - h_c + 1); left = np.random.randint(0, w - w_c + 1)
#             else: top = np.random.randint(pos[0], pos[1]); left = np.random.randint(pos[2], pos[3])
#         elif crop_type == 'center': top = (h - h_c) // 2; left = (w - w_c) // 2
#         else: raise NotImplementedError(f"crop_type '{crop_type}' not recognized.")
#     return x[:, top:top+h_c, left:left+w_c].unsqueeze(0), left, top

# # --- Data Preparation ---
# if pargs.prepare_data:
#     # ... (code is correct, omitting for brevity)
#     conf_thresh = 0.25; iou_thresh = 0.45
#     img_ori_dir = './data/INRIAPerson/Test/pos'; img_dir = './data/test_padded'; lab_dir = './data/test_lab_yolov5'
#     data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], pargs.img_size, shuffle=False)
#     loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=batch_size, shuffle=False, num_workers=4)
#     if not os.path.exists(lab_dir): os.makedirs(lab_dir)
#     if not os.path.exists(img_dir): os.makedirs(img_dir)
#     print('Preparing the test data with YOLOv5...');
#     with torch.no_grad():
#         for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
#             data = data.to(device)
#             pred = model(data, augment=False, visualize=False)
#             pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)
#             for i, det in enumerate(pred):
#                 img_path = labs[i]; lab_path = os.path.join(lab_dir, os.path.basename(img_path))
#                 img_save_path = os.path.join(img_dir, os.path.basename(img_path).replace('.txt', '.png'))
#                 unloader(data[i].cpu()).save(img_save_path)
#                 if len(det):
#                     h, w = data[i].shape[1:]
#                     boxes_xywh = xyxy2xywh(det[:, :4])
#                     boxes_xywh[:, 0] /= w; boxes_xywh[:, 1] /= h; boxes_xywh[:, 2] /= w; boxes_xywh[:, 3] /= h
#                     labels_to_save = torch.cat((det[:, -1].unsqueeze(1), boxes_xywh), 1)
#                     np.savetxt(lab_path, labels_to_save.cpu().numpy(), fmt='%f')
#                 else:
#                     open(lab_path, 'w').close()
#     print('Preparation done.'); sys.exit(0)

# # --- Evaluation Function ---
# def truths_length(truths):
#     for i in range(truths.shape[0]):
#         if truths[i][1] == -1: return i
#     return truths.shape[0]

# def test(model, loader, adv_cloth=None, gan=None, z=None, type=None,
#          conf_thresh=0.01, iou_thresh=0.5):
#     model.eval()
#     total_gts = 0.0; positives = []
#     with torch.no_grad():
#         for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader), position=0):
#             data = data.to(device); target = target.to(device)
            
#             # --- THIS IS THE FINAL FIX ---
#             if type == 'gan': # EGA method
#                 z_rand = torch.randn(1, 128, *args.z_size, device=device)
#                 adv_patch = gan.generate(z_rand)
#             elif type == 'z': # TCEGA method
#                 z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
#                 adv_patch = gan.generate(z_crop)
#             elif type == 'patch': # RCA/TCA methods
#                 if args.crop_type is not None:
#                     adv_patch, _, _ = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
#                 else:
#                     adv_patch = adv_cloth
#             else: adv_patch = None

#             if adv_patch is not None:
#                 adv_batch_t = patch_transformer(adv_patch, target, pargs.img_size, do_rotate=True, rand_loc=False,
#                                                 pooling=args.pooling, old_fasion=kwargs['old_fasion'])
#                 data = patch_applier(data, adv_batch_t)

#             pred = model(data, augment=False, visualize=False)
#             pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)
#             for i, det in enumerate(pred):
#                 truths = target[i]; num_gts = truths_length(truths); truths = truths[:num_gts]
#                 truths = truths[truths[:, 0] == 0]; total_gts += len(truths)
#                 if len(det) == 0: continue
#                 if len(truths) > 0:
#                     gt_boxes_xywh = truths[:, 1:]; gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
#                     h, w = data[i].shape[1:]; gt_boxes_xyxy[:, [0, 2]] *= w; gt_boxes_xyxy[:, [1, 3]] *= h
#                     pred_boxes_xyxy = det[:, :4]; iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
#                     matched_gts = set()
#                     for j in range(len(det)):
#                         iou_scores = iou_matrix[j, :]
#                         if len(iou_scores) == 0: positives.append((det[j, 4].item(), False)); continue
#                         best_iou, best_gt_idx = torch.max(iou_scores, 0)
#                         if best_iou > iou_thresh and best_gt_idx.item() not in matched_gts:
#                             positives.append((det[j, 4].item(), True)); matched_gts.add(best_gt_idx.item())
#                         else: positives.append((det[j, 4].item(), False))
#                 else:
#                     for j in range(len(det)): positives.append((det[j, 4].item(), False))
#     if total_gts == 0: return [], [], 0.0, []
#     positives = sorted(positives, key=lambda d: d[0], reverse=True)
#     tps_arr, fps_arr, confs = [], [], []; tp_counter, fp_counter = 0, 0
#     for conf, is_tp in positives:
#         if is_tp: tp_counter += 1
#         else: fp_counter += 1
#         tps_arr.append(tp_counter); fps_arr.append(fp_counter); confs.append(conf)
#     precision, recall = [], []
#     for tp, fp in zip(tps_arr, fps_arr):
#         if (fp + tp) > 0: recall.append(tp / total_gts); precision.append(tp / (fp + tp))
#     if len(precision) > 1 and len(recall) > 1:
#         p_interp = np.interp(np.linspace(0, 1, 101), np.flip(recall), np.flip(precision)); avg = p_interp.mean()
#     elif len(precision) > 0: avg = precision[0] * recall[0]
#     else: avg = 0.0
#     return precision, recall, avg, confs

# # --- Main Execution Logic ---
# if pargs.npz_dir is None:
#     lab_dir_test = './data/test_lab_yolov5'
#     if not os.path.exists(lab_dir_test):
#         print(f"Error: Label directory {lab_dir_test} not found.")
#         print("Please run with --prepare_data first to generate labels for YOLOv5."); sys.exit(1)
        
#     test_data = load_data.InriaDataset('./data/test_padded', lab_dir_test, kwargs['max_lab'], pargs.img_size, shuffle=False)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
#     print(f'Test loader has {len(test_loader)} batches.')

#     test_cloth, test_gan, test_z, cloth = None, None, None, None; test_type = None
#     if pargs.method in ['RCA', 'TCA']:
#         if not pargs.load_path: raise ValueError("Provide --load_path for RCA/TCA")
#         cloth = torch.from_numpy(np.load(pargs.load_path)[:1]).to(device)
#         test_cloth = cloth.detach().clone(); test_type = 'patch'
#     elif pargs.method in ['EGA', 'TCEGA']:
#         if not pargs.load_path: raise ValueError("Provide --load_path for EGA/TCEGA")
#         gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
#         gan.load_state_dict(torch.load(pargs.load_path, map_location=device)); gan.to(device).eval()
#         for p in gan.parameters(): p.requires_grad = False
#         test_gan = gan
#         if pargs.method == 'EGA':
#             test_type, test_z = 'gan', None; cloth = None # Cloth will be generated in test loop
#         else: # TCEGA
#             if not pargs.load_path_z: raise ValueError("Provide --load_path_z for TCEGA")
#             z = torch.from_numpy(np.load(pargs.load_path_z)).to(device)
#             test_z, test_type = z, 'z'; cloth = None # Cloth will be generated in test loop
#     else: print("Evaluating clean model (no attack)...")
#     save_dir = './test_results'
#     if not os.path.exists(save_dir): os.makedirs(save_dir)
#     save_path = os.path.join(save_dir, pargs.suffix)
#     plt.figure(figsize=[10, 7])
#     prec, rec, ap, confs = test(model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type=test_type, conf_thresh=0.01)
#     save_data = {'prec': prec, 'rec': rec, 'ap': ap, 'confs': confs}
#     if test_type is not None: # A patch was generated/loaded, we should save something
#         # Re-generate one sample to save, since the ones in the loop are random
#         if test_type == 'gan': final_cloth = test_gan.generate(torch.randn(1, 128, *args.z_size, device=device))
#         elif test_type == 'z': z_crop, _, _ = random_crop(test_z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type); final_cloth = test_gan.generate(z_crop)
#         else: final_cloth = test_cloth
#         save_data['adv_patch'] = final_cloth.detach().cpu().numpy()
#         unloader(final_cloth[0]).save(save_path + '.png')
#     np.savez(save_path, **save_data)
#     print(f'AP is {ap:.4f}')
#     plt.plot(rec, prec, label=f'{pargs.suffix}: ap {ap:.3f}')
#     plt.legend(); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
#     plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(save_path + '_PR.png', dpi=300)
# else:
#     # ... (code is correct, omitting for brevity)
#     files = fnmatch.filter(os.listdir(pargs.npz_dir), '*yolov5*.npz')
#     order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
#     files.sort(key=lambda x: order.get(re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group(), 1e5) if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) else 1e5)
#     plt.figure(figsize=[10, 7])
#     for file in files:
#         save_path = os.path.join(pargs.npz_dir, file)
#         try:
#             with np.load(save_path, allow_pickle=True) as data:
#                 plt.plot(data['rec'], data['prec'], label=f"{file.replace('.npz', '')}, ap: {data['ap']:.3f}")
#                 if 'adv_patch' in data:
#                     unloader(torch.from_numpy(data['adv_patch'][0])).save(save_path.replace('.npz', '.png'))
#         except Exception as e: print(f"Could not process {file}: {e}")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.legend(loc='lower left'); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
#     plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(os.path.join(pargs.npz_dir, 'PR-curve-yolov5.png'), dpi=300)

# evaluation_texture_yolov5.py (Corrected Pathing)
import os
import torch
import itertools
from tqdm import tqdm
import argparse
from scipy.interpolate import interp1d
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fnmatch
import re
import sys

### --- START: Corrected Pathing Logic --- ###
# This ensures we correctly find the SIBLING yolov5 directory
YOLOV5_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)
### --- END: Corrected Pathing Logic --- ###

# --- Original Attack Code Imports ---
from yolo2 import load_data
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis
# Make sure you have renamed your original utils.py to attack_utils.py
from attack_utils import *

# --- YOLOv5 Imports ---
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, xywh2xyxy, xyxy2xywh)
from utils.torch_utils import select_device
from utils.metrics import box_iou

unloader = transforms.ToPILImage()

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='YOLOv5 Adversarial Texture Evaluation')
# YOLOv5 specific
# The default path now correctly points to the sibling directory structure
parser.add_argument('--weights', type=str, default=os.path.join(YOLOV5_PATH, 'yolov5s.pt'), help='YOLOv5 weights path')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')

# Attack specific
parser.add_argument('--method', default='RCA', help='method name: RCA, TCA, EGA, TCEGA')
parser.add_argument('--suffix', default=None, help='suffix for saving results')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--prepare_data', action='store_true', help='generate test labels using the model')
parser.add_argument('--load_path', default=None, help='path to saved patch (.npy) or GAN (.pkl)')
parser.add_argument('--load_path_z', default=None, help='path to saved z for TCEGA (.npy)')
parser.add_argument('--npz_dir', default=None, help='directory of .npz files to plot PR curve')

pargs = parser.parse_args()

# --- Configs and Setup ---
if pargs.suffix is None:
    pargs.suffix = 'yolov5_' + pargs.method

batch_size = get_cfgs('yolov2', pargs.method, 'test')[0].batch_size
device = select_device(pargs.device, batch_size=batch_size)

# --- Load YOLOv5 Model ---
print(f"Loading YOLOv5 model from: {pargs.weights}")
model = DetectMultiBackend(pargs.weights, device=device, dnn=False)
model.eval()
stride = model.stride
names = model.names

# --- Load Attack-Specific Components ---
args, kwargs = get_cfgs('yolov2', pargs.method, 'test')

patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)

# --- UTILITY FUNCTIONS ---
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    images_batch = torch.stack(images, 0)
    max_len = max(len(l) for l in labels) if labels else 0
    if max_len == 0:
        return images_batch, torch.empty(len(batch), 0, 5)
    padded_labels = torch.full((len(batch), max_len, 5), -1.0)
    for i, label in enumerate(labels):
        num_objects = len(label)
        if num_objects > 0:
            padded_labels[i, :num_objects, :] = label
    return images_batch, padded_labels

# --- Data Preparation ---
if pargs.prepare_data:
    conf_thresh = 0.25; iou_thresh = 0.45
    img_ori_dir = './data/INRIAPerson/Test/pos'; img_dir = './data/test_padded'; lab_dir = './data/test_lab_yolov5'
    data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], pargs.img_size, shuffle=False)
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=batch_size, shuffle=False, num_workers=4)
    if not os.path.exists(lab_dir): os.makedirs(lab_dir)
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    print('Preparing the test data with YOLOv5...');
    with torch.no_grad():
        for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
            data = data.to(device)
            pred = model(data, augment=False)
            pred = non_max_suppression(pred[0], conf_thresh, iou_thresh, classes=[0], agnostic=False)
            for i, det in enumerate(pred):
                img_path = labs[i]; lab_path = os.path.join(lab_dir, os.path.basename(img_path))
                img_save_path = os.path.join(img_dir, os.path.basename(img_path).replace('.txt', '.png'))
                unloader(data[i].cpu()).save(img_save_path)
                if len(det):
                    h, w = data[i].shape[1:]
                    boxes_xywh = xyxy2xywh(det[:, :4])
                    boxes_xywh[:, 0] /= w; boxes_xywh[:, 1] /= h; boxes_xywh[:, 2] /= w; boxes_xywh[:, 3] /= h
                    labels_to_save = torch.cat((det[:, -1].unsqueeze(1), boxes_xywh), 1)
                    np.savetxt(lab_path, labels_to_save.cpu().numpy(), fmt='%f')
                else:
                    open(lab_path, 'w').close()
    print('Preparation done.'); sys.exit(0)

# --- Evaluation Function ---
def truths_length(truths):
    for i in range(truths.shape[0]):
        if truths[i][1] == -1: return i
    return truths.shape[0]

def test(model, loader, adv_cloth=None, gan=None, z=None, type=None,
         conf_thresh=0.01, iou_thresh=0.5):
    model.eval()
    total_gts = 0.0; positives = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader), position=0):
            data = data.to(device); target = target.to(device)
            
            if type == 'gan':
                z_rand = torch.randn(1, 128, *args.z_size, device=device)
                adv_patch = gan.generate(z_rand)
            elif type == 'z':
                z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
                adv_patch = gan.generate(z_crop)
            elif type == 'patch':
                if 'crop_type' in args and args.crop_type is not None:
                    adv_patch, _, _ = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
                else:
                    adv_patch = adv_cloth
            else: adv_patch = None

            if adv_patch is not None:
                adv_batch_t = patch_transformer(adv_patch, target, pargs.img_size, do_rotate=True, rand_loc=False,
                                                pooling=args.pooling, old_fasion=kwargs['old_fasion'])
                data = patch_applier(data, adv_batch_t)

            pred = model(data, augment=False)[0]
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0], agnostic=False)
            for i, det in enumerate(pred):
                truths = target[i]; num_gts = truths_length(truths); truths = truths[:num_gts]
                truths = truths[truths[:, 0] == 0]; total_gts += len(truths)
                if len(det) == 0: continue
                if len(truths) > 0:
                    gt_boxes_xywh = truths[:, 1:]; gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
                    h, w = data[i].shape[1:]; gt_boxes_xyxy[:, [0, 2]] *= w; gt_boxes_xyxy[:, [1, 3]] *= h
                    pred_boxes_xyxy = det[:, :4]; iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                    matched_gts = set()
                    for j in range(len(det)):
                        iou_scores = iou_matrix[j, :]
                        if len(iou_scores) == 0: positives.append((det[j, 4].item(), False)); continue
                        best_iou, best_gt_idx = torch.max(iou_scores, 0)
                        if best_iou > iou_thresh and best_gt_idx.item() not in matched_gts:
                            positives.append((det[j, 4].item(), True)); matched_gts.add(best_gt_idx.item())
                        else: positives.append((det[j, 4].item(), False))
                else:
                    for j in range(len(det)): positives.append((det[j, 4].item(), False))
    if total_gts == 0: return [], [], 0.0, []
    positives = sorted(positives, key=lambda d: d[0], reverse=True)
    tps_arr, fps_arr, confs = [], [], []; tp_counter, fp_counter = 0, 0
    for conf, is_tp in positives:
        if is_tp: tp_counter += 1
        else: fp_counter += 1
        tps_arr.append(tp_counter); fps_arr.append(fp_counter); confs.append(conf)
    precision, recall = [], []
    for tp, fp in zip(tps_arr, fps_arr):
        if (fp + tp) > 0: recall.append(tp / total_gts); precision.append(tp / (fp + tp))
    if len(precision) > 1 and len(recall) > 1:
        p_interp = np.interp(np.linspace(0, 1, 101), np.flip(recall), np.flip(precision)); avg = p_interp.mean()
    elif len(precision) > 0: avg = precision[0] * recall[0]
    else: avg = 0.0
    return precision, recall, avg, confs

# --- Main Execution Logic ---
if pargs.npz_dir is None:
    lab_dir_test = './data/test_lab_yolov5'
    if not os.path.exists(lab_dir_test):
        print(f"Error: Label directory {lab_dir_test} not found.")
        print("Please run with --prepare_data first to generate labels for YOLOv5."); sys.exit(1)
        
    test_data = load_data.InriaDataset('./data/test_padded', lab_dir_test, kwargs['max_lab'], pargs.img_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    print(f'Test loader has {len(test_loader)} batches.')

    test_cloth, test_gan, test_z, cloth = None, None, None, None; test_type = None
    if pargs.method in ['RCA', 'TCA']:
        if not pargs.load_path: raise ValueError("Provide --load_path for RCA/TCA")
        cloth = torch.from_numpy(np.load(pargs.load_path)).to(device)
        test_cloth = cloth.detach().clone(); test_type = 'patch'
    elif pargs.method in ['EGA', 'TCEGA']:
        if not pargs.load_path: raise ValueError("Provide --load_path for EGA/TCEGA")
        gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        gan.load_state_dict(torch.load(pargs.load_path, map_location=device)); gan.to(device).eval()
        for p in gan.parameters(): p.requires_grad = False
        test_gan = gan
        if pargs.method == 'EGA':
            test_type, test_z = 'gan', None
        else: # TCEGA
            if not pargs.load_path_z: raise ValueError("Provide --load_path_z for TCEGA")
            z = torch.from_numpy(np.load(pargs.load_path_z)).to(device)
            test_z, test_type = z, 'z'
    else: print("Evaluating clean model (no attack)...")
    save_dir = './test_results'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, pargs.suffix if pargs.suffix else f"eval_{pargs.method}")
    plt.figure(figsize=[10, 7])
    prec, rec, ap, confs = test(model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type=test_type, conf_thresh=0.01)
    
    # Generate a sample cloth for saving if it wasn't loaded directly
    if test_type is not None:
        if test_type == 'gan': final_cloth = test_gan.generate(torch.randn(1, 128, *args.z_size, device=device))
        elif test_type == 'z': z_crop, _, _ = random_crop(test_z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type); final_cloth = test_gan.generate(z_crop)
        else: final_cloth = test_cloth
        save_data = {'prec': prec, 'rec': rec, 'ap': ap, 'confs': confs, 'adv_patch': final_cloth.detach().cpu().numpy()}
        unloader(final_cloth[0]).save(save_path + '.png')
    else:
        save_data = {'prec': prec, 'rec': rec, 'ap': ap, 'confs': confs}
        
    np.savez(save_path, **save_data)
    print(f'AP is {ap:.4f}')
    plt.plot(rec, prec, label=f'{pargs.suffix if pargs.suffix else pargs.method}: ap {ap:.3f}')
    plt.legend(); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
    plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(save_path + '_PR.png', dpi=300)
else:
    files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
    order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
    files.sort(key=lambda x: order.get(re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group(), 1e5) if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) else 1e5)
    plt.figure(figsize=[10, 7])
    for file in files:
        save_path = os.path.join(pargs.npz_dir, file)
        try:
            with np.load(save_path, allow_pickle=True) as data:
                plt.plot(data['rec'], data['prec'], label=f"{file.replace('.npz', '')}, ap: {data['ap']:.3f}")
                if 'adv_patch' in data:
                    unloader(torch.from_numpy(data['adv_patch'][0])).save(save_path.replace('.npz', '.png'))
        except Exception as e: print(f"Could not process {file}: {e}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower left'); plt.title('PR-curve for YOLOv5'); plt.ylabel('Precision'); plt.xlabel('Recall')
    plt.ylim([0, 1.05]); plt.xlim([0, 1.05]); plt.savefig(os.path.join(pargs.npz_dir, 'PR-curve-yolov5.png'), dpi=300)
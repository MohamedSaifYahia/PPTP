# # # import os
# # # import torch
# # # import itertools
# # # from tqdm import tqdm
# # # import argparse
# # # from scipy.interpolate import interp1d
# # # from torchvision import transforms
# # # unloader = transforms.ToPILImage()
# # # import matplotlib
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt
# # # import fnmatch
# # # import re

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
# # # parser.add_argument('--device', default='cuda:0', help='')
# # # parser.add_argument('--prepare_data', default=False, action='store_true', help='')
# # # parser.add_argument('--epoch', type=int, default=None, help='')
# # # parser.add_argument('--load_path', default=None, help='')
# # # parser.add_argument('--load_path_z', default=None, help='')
# # # parser.add_argument('--npz_dir', default=None, help='')
# # # # parser.add_argument('--eval_times', type=int, default=1, help='evaluate multiple times')
# # # pargs = parser.parse_args()


# # # args, kwargs = get_cfgs(pargs.net, pargs.method, 'test')
# # # if pargs.epoch is not None:
# # #     args.n_epochs = pargs.epoch
# # # if pargs.suffix is None:
# # #     pargs.suffix = pargs.net + '_' + pargs.method

# # # device = torch.device(pargs.device)

# # # darknet_model = load_models(**kwargs)
# # # darknet_model = darknet_model.eval().to(device)

# # # class_names = utils.load_class_names('./data/coco.names')

# # # target_func = lambda obj, cls: obj
# # # patch_applier = load_data.PatchApplier().to(device)
# # # patch_transformer = load_data.PatchTransformer().to(device)
# # # prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
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

# # # if pargs.prepare_data:
# # #     conf_thresh = 0.5
# # #     nms_thresh = 0.4
# # #     img_ori_dir = './data/INRIAPerson/Test/pos'
# # #     img_dir = './data/test_padded'
# # #     lab_dir = './data/test_lab_%s' % kwargs['name']
# # #     data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], args.img_size, shuffle=False)
# # #     loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=args.batch_size, shuffle=False, num_workers=10)
# # #     if lab_dir is not None:
# # #         if not os.path.exists(lab_dir):
# # #             os.makedirs(lab_dir)
# # #     if img_dir is not None:
# # #         if not os.path.exists(img_dir):
# # #             os.makedirs(img_dir)
# # #     print('preparing the test data')
# # #     with torch.no_grad():
# # #         for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
# # #             data = data.to(device)
# # #             output = darknet_model(data)
# # #             all_boxes = utils.get_region_boxes_general(output, darknet_model, conf_thresh, kwargs['name'])
# # #             for i in range(data.size(0)):
# # #                 boxes = all_boxes[i]
# # #                 boxes = utils.nms(boxes, nms_thresh)
# # #                 new_boxes = boxes[:, [6, 0, 1, 2, 3]]
# # #                 new_boxes = new_boxes[new_boxes[:, 0] == 0]
# # #                 new_boxes = new_boxes.detach().cpu().numpy()
# # #                 if lab_dir is not None:
# # #                     save_dir = os.path.join(lab_dir, labs[i])
# # #                     np.savetxt(save_dir, new_boxes, fmt='%f')
# # #                     img = unloader(data[i].detach().cpu())
# # #                 if img_dir is not None:
# # #                     save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
# # #                     img.save(save_dir)
# # #     print('preparing done')

# # # img_dir_test = './data/test_padded'
# # # lab_dir_test = './data/test_lab_%s' % kwargs['name']
# # # test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, kwargs['max_lab'], args.img_size, shuffle=False)
# # # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=10)
# # # loader = test_loader
# # # epoch_length = len(loader)
# # # print(f'One epoch is {len(loader)}')


# # # def truths_length(truths):
# # #     for i in range(50):
# # #         if truths[i][1] == -1:
# # #             return i


# # # def label_filter(truths, labels=None):
# # #     if labels is not None:
# # #         new_truths = truths.new(truths.shape).fill_(-1)
# # #         c = 0
# # #         for t in truths:
# # #             if t[0].item() in labels:
# # #                 new_truths[c] = t
# # #                 c = c + 1
# # #         return new_truths


# # # def test(model, loader, adv_cloth=None, gan=None, z=None, type=None, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
# # #          old_fasion=True):
# # #     model.eval()
# # #     total = 0.0
# # #     proposals = 0.0
# # #     correct = 0.0
# # #     batch_num = len(loader)

# # #     with torch.no_grad():
# # #         positives = []
# # #         for batch_idx, (data, target) in tqdm(enumerate(loader), total=batch_num, position=0):
# # #             data = data.to(device)
# # #             if type == 'gan':
# # #                 z = torch.randn(1, 128, *args.z_size, device=device)
# # #                 cloth = gan.generate(z)
# # #                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# # #             elif type =='z':
# # #                 z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
# # #                 cloth = gan.generate(z_crop)
# # #                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# # #             elif type == 'patch':
# # #                 adv_patch, x, y = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# # #             elif type is not None:
# # #                 raise ValueError

# # #             if adv_patch is not None:
# # #                 target = target.to(device)
# # #                 adv_batch_t = patch_transformer(adv_patch, target, args.img_size, do_rotate=True, rand_loc=False,
# # #                                                 pooling=args.pooling, old_fasion=old_fasion)
# # #                 data = patch_applier(data, adv_batch_t)
# # #             output = model(data)
# # #             all_boxes = utils.get_region_boxes_general(output, model, conf_thresh, kwargs['name'])
# # #             for i in range(len(all_boxes)):
# # #                 boxes = all_boxes[i]
# # #                 boxes = utils.nms(boxes, nms_thresh)
# # #                 truths = target[i].view(-1, 5)
# # #                 truths = label_filter(truths, labels=[0])
# # #                 num_gts = truths_length(truths)
# # #                 truths = truths[:num_gts, 1:]
# # #                 truths = truths.tolist()
# # #                 total = total + num_gts
# # #                 for j in range(len(boxes)):
# # #                     if boxes[j][6].item() == 0:
# # #                         best_iou = 0
# # #                         best_index = 0

# # #                         for ib, box_gt in enumerate(truths):
# # #                             iou = utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
# # #                             if iou > best_iou:
# # #                                 best_iou = iou
# # #                                 best_index = ib
# # #                         if best_iou > iou_thresh:
# # #                             del truths[best_index]
# # #                             positives.append((boxes[j][4].item(), True))
# # #                         else:
# # #                             positives.append((boxes[j][4].item(), False))
# # #         positives = sorted(positives, key=lambda d: d[0], reverse=True)

# # #         tps = []
# # #         fps = []
# # #         confs = []
# # #         tp_counter = 0
# # #         fp_counter = 0
# # #         for pos in positives:
# # #             if pos[1]:
# # #                 tp_counter += 1
# # #             else:
# # #                 fp_counter += 1
# # #             tps.append(tp_counter)
# # #             fps.append(fp_counter)
# # #             confs.append(pos[0])

# # #         precision = []
# # #         recall = []
# # #         for tp, fp in zip(tps, fps):
# # #             recall.append(tp / total)
# # #             precision.append(tp / (fp + tp))

# # #     if len(precision) > 1 and len(recall) > 1:
# # #         p = np.array(precision)
# # #         r = np.array(recall)
# # #         p_start = p[np.argmin(r)]
# # #         samples = np.arange(0., 1., 1.0 / num_of_samples)
# # #         interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
# # #         avg = sum(interpolated) / len(interpolated)
# # #     elif len(precision) > 0 and len(recall) > 0:
# # #         avg = precision[0] * recall[0]
# # #     else:
# # #         avg = float('nan')

# # #     return precision, recall, avg, confs

# # # if pargs.npz_dir is None:
# # #     if pargs.method == 'RCA' or pargs.method == 'TCA':
# # #         if pargs.load_path is None:
# # #             result_dir = './results/result_' + pargs.net + '_' + pargs.method
# # #             img_path = os.path.join(result_dir, 'patch%d.npy' % args.n_epochs)
# # #         else:
# # #             img_path = pargs.load_path
# # #         cloth = torch.from_numpy(np.load(img_path)[:1]).to(device)
# # #         test_cloth = cloth.detach().clone()
# # #         test_gan = None
# # #         test_z = None
# # #         test_type = 'patch'

# # #     elif pargs.method == 'EGA' or pargs.method == 'TCEGA':
# # #         gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
# # #         if pargs.load_path is None:
# # #             result_dir = './results/result_' + pargs.net + '_' + pargs.method
# # #             cpt = os.path.join(result_dir, pargs.net + '_' + pargs.method + '.pkl')
# # #         else:
# # #             cpt = pargs.load_path
# # #         d = torch.load(cpt, map_location='cpu')
# # #         gan.load_state_dict(d)
# # #         gan.to(device)
# # #         gan.eval()
# # #         for p in gan.parameters():
# # #             p.requires_grad = False
# # #         test_cloth = None
# # #         test_gan = gan
# # #         if pargs.method == 'EGA':
# # #             test_z = None
# # #             test_type = 'gan'
# # #             cloth = gan.generate(torch.randn(1, 128, *args.z_size, device=device))
# # #         else:
# # #             if pargs.load_path_z is None:
# # #                 result_dir = './results/result_' + pargs.net + '_' + pargs.method
# # #                 z_path = os.path.join(result_dir, 'z2000.npy')
# # #             else:
# # #                 z_path = pargs.load_path_z
# # #             z = np.load(z_path)
# # #             z = torch.from_numpy(z).to(device)
# # #             test_z = z
# # #             test_type = 'z'
# # #             z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
# # #             cloth = gan.generate(z_crop)
# # #     else:
# # #         raise ValueError

# # #     save_dir = './test_results'
# # #     if not os.path.exists(save_dir):
# # #         os.makedirs(save_dir)
# # #     save_path = os.path.join(save_dir, pargs.suffix)

# # #     plt.figure(figsize=[15, 10])
# # #     prec, rec, ap, confs = test(darknet_model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, type=test_type, conf_thresh=0.01, old_fasion=kwargs['old_fasion'])

# # #     np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=cloth.detach().cpu().numpy())
# # #     print('AP is %.4f'% ap)
# # #     plt.plot(rec, prec)
# # #     leg = [pargs.suffix + ': ap %.3f' % ap]
# # #     unloader(cloth[0]).save(save_path + '.png')
# # # else:
# # #     files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
# # #     order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
# # #     files.sort()
# # #     files.sort(key=lambda x: order[re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group()] if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) is not None else 1e5)

# # #     leg = []
# # #     for file in files:
# # #         save_path = os.path.join(pargs.npz_dir, file)
# # #         save_data = np.load(save_path, allow_pickle=True)
# # #         save_data = save_data.values()
# # #         prec, rec, ap, confs, clothi = list(save_data)
# # #         plt.plot(rec, prec)
# # #         leg.append(file.replace('.npz', '') + ', ap: %.3f' % ap)
# # #         unloader(torch.from_numpy(clothi[0])).save(save_path.replace('.npz', '.png'))
# # #     save_dir = pargs.npz_dir

# # # plt.plot([0, 1], [0, 1], 'k--')
# # # plt.legend(leg, loc=4)
# # # plt.title('PR-curve')
# # # plt.ylabel('Precision')
# # # plt.xlabel('Recall')
# # # plt.ylim([0, 1.05])
# # # plt.xlim([0, 1.05])
# # # plt.savefig(os.path.join(save_dir, 'PR-curve.png'), dpi=300)


# # import os
# # import torch
# # import itertools
# # from tqdm import tqdm
# # import argparse
# # from scipy.interpolate import interp1d
# # from torchvision import transforms
# # unloader = transforms.ToPILImage()
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import fnmatch
# # import re
# # import numpy as np
# # import sys

# # # Add YOLOv5 to path
# # sys.path.append('../yolov5')

# # from yolov5_wrapper import YOLOv5Wrapper, extract_predictions_for_loss
# # from yolo2 import load_data
# # from load_data_yolov5 import MaxProbExtractorYOLOv5
# # from attack_utils import *
# # from cfg import get_cfgs
# # from tps_grid_gen import TPSGridGen
# # from generator_dim import GAN_dis

# # parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Adversarial Evaluation')
# # parser.add_argument('--net', default='yolov5', help='target net name')
# # parser.add_argument('--method', default='TCEGA', help='method name')
# # parser.add_argument('--suffix', default=None, help='suffix name')
# # parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
# # parser.add_argument('--device', default='cuda:0', help='')
# # parser.add_argument('--prepare_data', default=False, action='store_true', help='')
# # parser.add_argument('--epoch', type=int, default=None, help='')
# # parser.add_argument('--load_path', default=None, help='')
# # parser.add_argument('--load_path_z', default=None, help='')
# # parser.add_argument('--npz_dir', default=None, help='')
# # parser.add_argument('--yolov5_weights', default='../yolov5/yolov5s.pt', help='YOLOv5 weights path')
# # parser.add_argument('--img_size', type=int, default=640, help='YOLOv5 input size')
# # pargs = parser.parse_args()

# # # Modify cfg to work with YOLOv5
# # args, kwargs = get_cfgs('yolov2', pargs.method, 'test')  # Keep config structure
# # kwargs['name'] = 'yolov5'  # Update name

# # if pargs.epoch is not None:
# #     args.n_epochs = pargs.epoch
# # if pargs.suffix is None:
# #     pargs.suffix = pargs.net + '_' + pargs.method

# # device = torch.device(pargs.device)

# # # Load YOLOv5 model
# # yolo_model = YOLOv5Wrapper(weights=pargs.yolov5_weights, device=pargs.device, img_size=pargs.img_size)

# # # Load components for adversarial patches
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

# # if pargs.prepare_data:
# #     conf_thresh = 0.5
# #     nms_thresh = 0.4
# #     img_ori_dir = './data/INRIAPerson/Test/pos'
# #     img_dir = './data/test_padded'
# #     lab_dir = './data/test_lab_yolov5'
    
# #     data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], args.img_size, shuffle=False)
# #     loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
# #     if lab_dir is not None:
# #         if not os.path.exists(lab_dir):
# #             os.makedirs(lab_dir)
# #     if img_dir is not None:
# #         if not os.path.exists(img_dir):
# #             os.makedirs(img_dir)
    
# #     print('preparing the test data for YOLOv5')
# #     with torch.no_grad():
# #         for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
# #             data = data.to(device)
            
# #             # Get YOLOv5 detections
# #             all_boxes = yolo_model.get_detections(data, conf_thresh, nms_thresh)
            
# #             for i in range(data.size(0)):
# #                 boxes = all_boxes[i]
                
# #                 # Convert to numpy array format expected by evaluation
# #                 new_boxes = []
# #                 for box in boxes:
# #                     # box format: [cx, cy, w, h, conf, conf, class]
# #                     if box[6] == 0:  # person class
# #                         new_boxes.append([box[6], box[0], box[1], box[2], box[3]])
                
# #                 new_boxes = np.array(new_boxes) if new_boxes else np.array([]).reshape(0, 5)
                
# #                 if lab_dir is not None:
# #                     save_dir = os.path.join(lab_dir, labs[i])
# #                     np.savetxt(save_dir, new_boxes, fmt='%f')
                    
# #                 if img_dir is not None:
# #                     img = unloader(data[i].detach().cpu())
# #                     save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
# #                     img.save(save_dir)
# #     print('preparing done')

# # img_dir_test = './data/test_padded'
# # lab_dir_test = './data/test_lab_yolov5'
# # test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, kwargs['max_lab'], args.img_size, shuffle=False)
# # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=10)
# # loader = test_loader
# # epoch_length = len(loader)
# # print(f'One epoch is {len(loader)}')


# # def truths_length(truths):
# #     for i in range(50):
# #         if truths[i][1] == -1:
# #             return i


# # def label_filter(truths, labels=None):
# #     if labels is not None:
# #         new_truths = truths.new(truths.shape).fill_(-1)
# #         c = 0
# #         for t in truths:
# #             if t[0].item() in labels:
# #                 new_truths[c] = t
# #                 c = c + 1
# #         return new_truths


# # def bbox_iou_yolov5(box1, box2, x1y1x2y2=False):
# #     """Calculate IoU between two boxes"""
# #     if x1y1x2y2:
# #         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
# #         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
# #     else:
# #         # Convert from center format to corner format
# #         b1_x1, b1_x2 = box1[0] - box1[2]/2, box1[0] + box1[2]/2
# #         b1_y1, b1_y2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
# #         b2_x1, b2_x2 = box2[0] - box2[2]/2, box2[0] + box2[2]/2
# #         b2_y1, b2_y2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2
    
# #     # Intersection area
# #     inter_area = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)).clamp(0) * \
# #                  (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)).clamp(0)
    
# #     # Union Area
# #     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
# #     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
# #     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    
# #     return inter_area / union_area


# # def test(model, loader, adv_cloth=None, gan=None, z=None, type=None, conf_thresh=0.5, nms_thresh=0.4, 
# #          iou_thresh=0.5, num_of_samples=100, old_fasion=True):
# #     model.model.eval()
# #     total = 0.0
# #     proposals = 0.0
# #     correct = 0.0
# #     batch_num = len(loader)

# #     with torch.no_grad():
# #         positives = []
# #         for batch_idx, (data, target) in tqdm(enumerate(loader), total=batch_num, position=0):
# #             data = data.to(device)
# #             if type == 'gan':
# #                 z = torch.randn(1, 128, *args.z_size, device=device)
# #                 cloth = gan.generate(z)
# #                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# #             elif type =='z':
# #                 z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
# #                 cloth = gan.generate(z_crop)
# #                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# #             elif type == 'patch':
# #                 adv_patch, x, y = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
# #             elif type is not None:
# #                 raise ValueError

# #             if adv_patch is not None:
# #                 target = target.to(device)
# #                 adv_batch_t = patch_transformer(adv_patch, target, args.img_size, do_rotate=True, rand_loc=False,
# #                                                 pooling=args.pooling, old_fasion=old_fasion)
# #                 data = patch_applier(data, adv_batch_t)
            
# #             # Get YOLOv5 detections
# #             all_boxes = model.get_detections(data, conf_thresh, nms_thresh)
            
# #             for i in range(len(all_boxes)):
# #                 boxes = all_boxes[i]
# #                 truths = target[i].view(-1, 5)
# #                 truths = label_filter(truths, labels=[0])
# #                 num_gts = truths_length(truths)
# #                 truths = truths[:num_gts, 1:]
# #                 truths = truths.tolist()
# #                 total = total + num_gts
                
# #                 for j in range(len(boxes)):
# #                     if boxes[j][6] == 0:  # person class
# #                         best_iou = 0
# #                         best_index = 0

# #                         for ib, box_gt in enumerate(truths):
# #                             iou = bbox_iou_yolov5(box_gt, boxes[j], x1y1x2y2=False)
# #                             if iou > best_iou:
# #                                 best_iou = iou
# #                                 best_index = ib
                        
# #                         if best_iou > iou_thresh:
# #                             del truths[best_index]
# #                             positives.append((boxes[j][4], True))
# #                         else:
# #                             positives.append((boxes[j][4], False))
        
# #         positives = sorted(positives, key=lambda d: d[0], reverse=True)

# #         tps = []
# #         fps = []
# #         confs = []
# #         tp_counter = 0
# #         fp_counter = 0
# #         for pos in positives:
# #             if pos[1]:
# #                 tp_counter += 1
# #             else:
# #                 fp_counter += 1
# #             tps.append(tp_counter)
# #             fps.append(fp_counter)
# #             confs.append(pos[0])

# #         precision = []
# #         recall = []
# #         for tp, fp in zip(tps, fps):
# #             recall.append(tp / total)
# #             precision.append(tp / (fp + tp))

# #     if len(precision) > 1 and len(recall) > 1:
# #         p = np.array(precision)
# #         r = np.array(recall)
# #         p_start = p[np.argmin(r)]
# #         samples = np.arange(0., 1., 1.0 / num_of_samples)
# #         interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
# #         avg = sum(interpolated) / len(interpolated)
# #     elif len(precision) > 0 and len(recall) > 0:
# #         avg = precision[0] * recall[0]
# #     else:
# #         avg = float('nan')

# #     return precision, recall, avg, confs


# # if pargs.npz_dir is None:
# #     if pargs.method == 'RCA' or pargs.method == 'TCA':
# #         if pargs.load_path is None:
# #             result_dir = './results/result_' + pargs.net + '_' + pargs.method
# #             img_path = os.path.join(result_dir, 'patch%d.npy' % args.n_epochs)
# #         else:
# #             img_path = pargs.load_path
# #         cloth = torch.from_numpy(np.load(img_path)[:1]).to(device)
# #         test_cloth = cloth.detach().clone()
# #         test_gan = None
# #         test_z = None
# #         test_type = 'patch'

# #     elif pargs.method == 'EGA' or pargs.method == 'TCEGA':
# #         gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
# #         if pargs.load_path is None:
# #             result_dir = './results/result_' + pargs.net + '_' + pargs.method
# #             cpt = os.path.join(result_dir, pargs.net + '_' + pargs.method + '.pkl')
# #         else:
# #             cpt = pargs.load_path
# #         d = torch.load(cpt, map_location='cpu')
# #         gan.load_state_dict(d)
# #         gan.to(device)
# #         gan.eval()
# #         for p in gan.parameters():
# #             p.requires_grad = False
# #         test_cloth = None
# #         test_gan = gan
# #         if pargs.method == 'EGA':
# #             test_z = None
# #             test_type = 'gan'
# #             cloth = gan.generate(torch.randn(1, 128, *args.z_size, device=device))
# #         else:
# #             if pargs.load_path_z is None:
# #                 result_dir = './results/result_' + pargs.net + '_' + pargs.method
# #                 z_path = os.path.join(result_dir, 'z2000.npy')
# #             else:
# #                 z_path = pargs.load_path_z
# #             z = np.load(z_path)
# #             z = torch.from_numpy(z).to(device)
# #             test_z = z
# #             test_type = 'z'
# #             z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
# #             cloth = gan.generate(z_crop)
# #     else:
# #         raise ValueError

# #     save_dir = './test_results'
# #     if not os.path.exists(save_dir):
# #         os.makedirs(save_dir)
# #     save_path = os.path.join(save_dir, pargs.suffix)

# #     plt.figure(figsize=[15, 10])
# #     prec, rec, ap, confs = test(yolo_model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, 
# #                                  type=test_type, conf_thresh=0.01, old_fasion=kwargs['old_fasion'])

# #     np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=cloth.detach().cpu().numpy())
# #     print('AP is %.4f'% ap)
# #     plt.plot(rec, prec)
# #     leg = [pargs.suffix + ': ap %.3f' % ap]
# #     unloader(cloth[0]).save(save_path + '.png')
# # else:
# #     files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
# #     order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
# #     files.sort()
# #     files.sort(key=lambda x: order[re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group()] if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) is not None else 1e5)

# #     leg = []
# #     for file in files:
# #         save_path = os.path.join(pargs.npz_dir, file)
# #         save_data = np.load(save_path, allow_pickle=True)
# #         save_data = save_data.values()
# #         prec, rec, ap, confs, clothi = list(save_data)
# #         plt.plot(rec, prec)
# #         leg.append(file.replace('.npz', '') + ', ap: %.3f' % ap)
# #         unloader(torch.from_numpy(clothi[0])).save(save_path.replace('.npz', '.png'))
# #     save_dir = pargs.npz_dir

# # plt.plot([0, 1], [0, 1], 'k--')
# # plt.legend(leg, loc=4)
# # plt.title('PR-curve')
# # plt.ylabel('Precision')
# # plt.xlabel('Recall')
# # plt.ylim([0, 1.05])
# # plt.xlim([0, 1.05])
# # plt.savefig(os.path.join(save_dir, 'PR-curve.png'), dpi=300)


# import os
# import torch
# import itertools
# from tqdm import tqdm
# import argparse
# from scipy.interpolate import interp1d
# from torchvision import transforms
# unloader = transforms.ToPILImage()
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import fnmatch
# import re
# import numpy as np
# import sys

# # Add YOLOv5 to path
# sys.path.append('../yolov5')

# from yolov5_wrapper import YOLOv5Wrapper, extract_predictions_for_loss
# try:
#     from yolo2 import load_data
# except ImportError:
#     import load_data  # Fallback if yolo2 module structure is different
# from attack_utils import *
# from cfg import get_cfgs
# from tps_grid_gen import TPSGridGen
# from generator_dim import GAN_dis

# parser = argparse.ArgumentParser(description='PyTorch YOLOv5 Adversarial Evaluation')
# parser.add_argument('--net', default='yolov5', help='target net name')
# parser.add_argument('--method', default='TCEGA', help='method name')
# parser.add_argument('--suffix', default=None, help='suffix name')
# parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
# parser.add_argument('--device', default='cuda:0', help='')
# parser.add_argument('--prepare_data', default=False, action='store_true', help='')
# parser.add_argument('--epoch', type=int, default=None, help='')
# parser.add_argument('--load_path', default=None, help='')
# parser.add_argument('--load_path_z', default=None, help='')
# parser.add_argument('--npz_dir', default=None, help='')
# parser.add_argument('--yolov5_weights', default='../yolov5/yolov5s.pt', help='YOLOv5 weights path')
# parser.add_argument('--img_size', type=int, default=640, help='YOLOv5 input size')
# pargs = parser.parse_args()

# # Modify cfg to work with YOLOv5
# args, kwargs = get_cfgs('yolov2', pargs.method, 'test')  # Keep config structure
# kwargs['name'] = 'yolov5'  # Update name

# if pargs.epoch is not None:
#     args.n_epochs = pargs.epoch
# if pargs.suffix is None:
#     pargs.suffix = pargs.net + '_' + pargs.method

# device = torch.device(pargs.device)

# # Load YOLOv5 model
# yolo_model = YOLOv5Wrapper(weights=pargs.yolov5_weights, device=pargs.device, img_size=pargs.img_size)

# # Load components for adversarial patches
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

# if pargs.prepare_data:
#     conf_thresh = 0.5
#     nms_thresh = 0.4
#     img_ori_dir = './data/INRIAPerson/Test/pos'
#     img_dir = './data/test_padded'
#     lab_dir = './data/test_lab_yolov5'
    
#     data_nl = load_data.InriaDataset(img_ori_dir, None, kwargs['max_lab'], args.img_size, shuffle=False)
#     loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
#     if lab_dir is not None:
#         if not os.path.exists(lab_dir):
#             os.makedirs(lab_dir)
#     if img_dir is not None:
#         if not os.path.exists(img_dir):
#             os.makedirs(img_dir)
    
#     print('preparing the test data for YOLOv5')
#     with torch.no_grad():
#         for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
#             data = data.to(device)
            
#             # Get YOLOv5 detections
#             all_boxes = yolo_model.get_detections(data, conf_thresh, nms_thresh)
            
#             for i in range(data.size(0)):
#                 boxes = all_boxes[i]
                
#                 # Convert to numpy array format expected by evaluation
#                 new_boxes = []
#                 for box in boxes:
#                     # box format: [cx, cy, w, h, conf, conf, class]
#                     if box[6] == 0:  # person class
#                         new_boxes.append([box[6], box[0], box[1], box[2], box[3]])
                
#                 new_boxes = np.array(new_boxes) if new_boxes else np.array([]).reshape(0, 5)
                
#                 if lab_dir is not None:
#                     save_dir = os.path.join(lab_dir, labs[i])
#                     np.savetxt(save_dir, new_boxes, fmt='%f')
                    
#                 if img_dir is not None:
#                     img = unloader(data[i].detach().cpu())
#                     save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
#                     img.save(save_dir)
#     print('preparing done')

# img_dir_test = './data/test_padded'
# lab_dir_test = './data/test_lab_yolov5'
# test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, kwargs['max_lab'], args.img_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=10)
# loader = test_loader
# epoch_length = len(loader)
# print(f'One epoch is {len(loader)}')


# def truths_length(truths):
#     for i in range(50):
#         if truths[i][1] == -1:
#             return i


# def label_filter(truths, labels=None):
#     if labels is not None:
#         new_truths = truths.new(truths.shape).fill_(-1)
#         c = 0
#         for t in truths:
#             if t[0].item() in labels:
#                 new_truths[c] = t
#                 c = c + 1
#         return new_truths


# def bbox_iou_yolov5(box1, box2, x1y1x2y2=False):
#     """Calculate IoU between two boxes"""
#     if x1y1x2y2:
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#     else:
#         # Convert from center format to corner format
#         b1_x1, b1_x2 = box1[0] - box1[2]/2, box1[0] + box1[2]/2
#         b1_y1, b1_y2 = box1[1] - box1[3]/2, box1[1] + box1[3]/2
#         b2_x1, b2_x2 = box2[0] - box2[2]/2, box2[0] + box2[2]/2
#         b2_y1, b2_y2 = box2[1] - box2[3]/2, box2[1] + box2[3]/2
    
#     # Intersection area
#     inter_x1 = max(b1_x1, b2_x1)
#     inter_y1 = max(b1_y1, b2_y1)
#     inter_x2 = min(b1_x2, b2_x2)
#     inter_y2 = min(b1_y2, b2_y2)
    
#     # Check if there's an intersection
#     if inter_x2 < inter_x1 or inter_y2 < inter_y1:
#         return 0.0
    
#     inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
#     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    
#     return inter_area / union_area


# def test(model, loader, adv_cloth=None, gan=None, z=None, type=None, conf_thresh=0.5, nms_thresh=0.4, 
#          iou_thresh=0.5, num_of_samples=100, old_fasion=True):
#     model.model.eval()
#     total = 0.0
#     proposals = 0.0
#     correct = 0.0
#     batch_num = len(loader)

#     with torch.no_grad():
#         positives = []
#         for batch_idx, (data, target) in tqdm(enumerate(loader), total=batch_num, position=0):
#             data = data.to(device)
#             if type == 'gan':
#                 z = torch.randn(1, 128, *args.z_size, device=device)
#                 cloth = gan.generate(z)
#                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
#             elif type =='z':
#                 z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
#                 cloth = gan.generate(z_crop)
#                 adv_patch, x, y = random_crop(cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
#             elif type == 'patch':
#                 adv_patch, x, y = random_crop(adv_cloth, args.crop_size, pos=args.pos, crop_type=args.crop_type)
#             elif type is not None:
#                 raise ValueError

#             if adv_patch is not None:
#                 target = target.to(device)
#                 adv_batch_t = patch_transformer(adv_patch, target, args.img_size, do_rotate=True, rand_loc=False,
#                                                 pooling=args.pooling, old_fasion=old_fasion)
#                 data = patch_applier(data, adv_batch_t)
            
#             # Get YOLOv5 detections
#             all_boxes = model.get_detections(data, conf_thresh, nms_thresh)
            
#             for i in range(len(all_boxes)):
#                 boxes = all_boxes[i]
#                 truths = target[i].view(-1, 5)
#                 truths = label_filter(truths, labels=[0])
#                 num_gts = truths_length(truths)
#                 truths = truths[:num_gts, 1:]
#                 truths = truths.tolist()
#                 total = total + num_gts
                
#                 for j in range(len(boxes)):
#                     if boxes[j][6] == 0:  # person class
#                         best_iou = 0
#                         best_index = 0

#                         for ib, box_gt in enumerate(truths):
#                             iou = bbox_iou_yolov5(box_gt, boxes[j], x1y1x2y2=False)
#                             if iou > best_iou:
#                                 best_iou = iou
#                                 best_index = ib
                        
#                         if best_iou > iou_thresh:
#                             del truths[best_index]
#                             positives.append((boxes[j][4], True))
#                         else:
#                             positives.append((boxes[j][4], False))
        
#         positives = sorted(positives, key=lambda d: d[0], reverse=True)

#         tps = []
#         fps = []
#         confs = []
#         tp_counter = 0
#         fp_counter = 0
#         for pos in positives:
#             if pos[1]:
#                 tp_counter += 1
#             else:
#                 fp_counter += 1
#             tps.append(tp_counter)
#             fps.append(fp_counter)
#             confs.append(pos[0])

#         precision = []
#         recall = []
#         for tp, fp in zip(tps, fps):
#             recall.append(tp / total)
#             precision.append(tp / (fp + tp))

#     if len(precision) > 1 and len(recall) > 1:
#         p = np.array(precision)
#         r = np.array(recall)
#         p_start = p[np.argmin(r)]
#         samples = np.arange(0., 1., 1.0 / num_of_samples)
#         interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
#         avg = sum(interpolated) / len(interpolated)
#     elif len(precision) > 0 and len(recall) > 0:
#         avg = precision[0] * recall[0]
#     else:
#         avg = float('nan')

#     return precision, recall, avg, confs


# if pargs.npz_dir is None:
#     if pargs.method == 'RCA' or pargs.method == 'TCA':
#         if pargs.load_path is None:
#             result_dir = './results/result_' + pargs.net + '_' + pargs.method
#             img_path = os.path.join(result_dir, 'patch%d.npy' % args.n_epochs)
#         else:
#             img_path = pargs.load_path
#         cloth = torch.from_numpy(np.load(img_path)[:1]).to(device)
#         test_cloth = cloth.detach().clone()
#         test_gan = None
#         test_z = None
#         test_type = 'patch'

#     elif pargs.method == 'EGA' or pargs.method == 'TCEGA':
#         gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
#         if pargs.load_path is None:
#             result_dir = './results/result_' + pargs.net + '_' + pargs.method
#             cpt = os.path.join(result_dir, pargs.net + '_' + pargs.method + '.pkl')
#         else:
#             cpt = pargs.load_path
#         d = torch.load(cpt, map_location='cpu')
#         gan.load_state_dict(d)
#         gan.to(device)
#         gan.eval()
#         for p in gan.parameters():
#             p.requires_grad = False
#         test_cloth = None
#         test_gan = gan
#         if pargs.method == 'EGA':
#             test_z = None
#             test_type = 'gan'
#             cloth = gan.generate(torch.randn(1, 128, *args.z_size, device=device))
#         else:
#             if pargs.load_path_z is None:
#                 result_dir = './results/result_' + pargs.net + '_' + pargs.method
#                 z_path = os.path.join(result_dir, 'z2000.npy')
#             else:
#                 z_path = pargs.load_path_z
#             z = np.load(z_path)
#             z = torch.from_numpy(z).to(device)
#             test_z = z
#             test_type = 'z'
#             z_crop, _, _ = random_crop(z, args.z_crop_size, pos=args.z_pos, crop_type=args.z_crop_type)
#             cloth = gan.generate(z_crop)
#     else:
#         raise ValueError

#     save_dir = './test_results'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     save_path = os.path.join(save_dir, pargs.suffix)

#     plt.figure(figsize=[15, 10])
#     prec, rec, ap, confs = test(yolo_model, test_loader, adv_cloth=test_cloth, gan=test_gan, z=test_z, 
#                                  type=test_type, conf_thresh=0.01, old_fasion=kwargs['old_fasion'])

#     np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, adv_patch=cloth.detach().cpu().numpy())
#     print('AP is %.4f'% ap)
#     plt.plot(rec, prec)
#     leg = [pargs.suffix + ': ap %.3f' % ap]
#     unloader(cloth[0]).save(save_path + '.png')
# else:
#     files = fnmatch.filter(os.listdir(pargs.npz_dir), '*.npz')
#     order = {'RCA': 0, 'TCA': 1, 'EGA': 2, 'TCEGA': 3}
#     files.sort()
#     files.sort(key=lambda x: order[re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x).group()] if re.search('(RCA)|(TCA)|(EGA)|(TCEGA)', x) is not None else 1e5)

#     leg = []
#     for file in files:
#         save_path = os.path.join(pargs.npz_dir, file)
#         save_data = np.load(save_path, allow_pickle=True)
#         save_data = save_data.values()
#         prec, rec, ap, confs, clothi = list(save_data)
#         plt.plot(rec, prec)
#         leg.append(file.replace('.npz', '') + ', ap: %.3f' % ap)
#         unloader(torch.from_numpy(clothi[0])).save(save_path.replace('.npz', '.png'))
#     save_dir = pargs.npz_dir

# plt.plot([0, 1], [0, 1], 'k--')
# plt.legend(leg, loc=4)
# plt.title('PR-curve')
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.ylim([0, 1.05])
# plt.xlim([0, 1.05])
# plt.savefig(os.path.join(save_dir, 'PR-curve.png'), dpi=300)

# evaluation_texture.py
import os
import fnmatch
import re
import argparse
import itertools
import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
from torchvision import transforms
unloader = transforms.ToPILImage()

# patch/gan machinery & dataset
from yolo2 import load_data                      # for InriaDataset, PatchApplier, etc.
from attack_utils import random_crop, get_det_loss       # your existing helpers
from tps_grid_gen import TPSGridGen
from generator_dim import GAN_dis
#from easy_dict import EasyDict                    

# Load COCO names if needed for logging
CLASS_NAMES = [l.strip() for l in open('data/coco.names')]

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='TCEGA', choices=['RCA','TCA','EGA','TCEGA'])
parser.add_argument('--suffix', default=None)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--prepare_data', action='store_true')
parser.add_argument('--load_path', default=None)
parser.add_argument('--load_path_z', default=None)
parser.add_argument('--npz_dir', default=None)
parser.add_argument('--conf_thres', type=float, default=0.01)
parser.add_argument('--iou_thres', type=float, default=0.5)
args = parser.parse_args()

# default suffix
if args.suffix is None:
    args.suffix = f'yolov5_{args.method}'

device = torch.device(args.device)

# 1) Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.conf = args.conf_thres  # confidence threshold
model.iou  = args.iou_thres   # NMS IoU threshold
model.eval()

# 2) Prepare Inria test data if requested
if args.prepare_data:
    print('Preparing test data...')
    data_nl = load_data.InriaDataset(
        img_dir='./data/INRIAPerson/Test/pos',
        label_dir=None,
        max_lab=50,
        img_size=(640,640),
        shuffle=False
    )
    loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=16, num_workers=10)
    os.makedirs('./data/test_padded', exist_ok=True)
    os.makedirs(f'./data/test_lab_yolov5', exist_ok=True)

    with torch.no_grad():
        for imgs, labs in tqdm(loader_nl):
            imgs = imgs.to(device)
            results = model(imgs)  # inference
            for i, (img, lab_name) in enumerate(zip(imgs, labs)):
                # save padded image
                unloader(img.cpu()).save(f'./data/test_padded/{lab_name.replace(".txt",".png")}')
                # save ground-truth (from original Inria labels)
                gt = data_nl.annotations_for(lab_name)  # implement this in your InriaDataset
                np.savetxt(f'./data/test_lab_yolov5/{lab_name}', gt, fmt='%f')
    print('Done.')

# 3) Build test loader
test_data = load_data.InriaDataset(
    img_dir='./data/test_padded',
    label_dir=f'./data/test_lab_yolov5',
    max_lab=50,
    img_size=(640,640),
    shuffle=False
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, num_workers=10)

# 4) TPS & transformers
ctrl_pts = torch.tensor(list(itertools.product(
    torch.linspace(-1,1,5), torch.linspace(-1,1,5)
)))
tps = TPSGridGen(torch.Size([300,300]), ctrl_pts).to(device)
patch_applier    = load_data.PatchApplier().to(device)
patch_transformer= load_data.PatchTransformer().to(device)
total_variation  = load_data.TotalVariation().to(device)

# 5) Test function
def test(adv_patch=None, gan=None, z=None, mode='patch'):
    tp_fp = []
    total_gts = 0
    with torch.no_grad():
        for imgs, truths in tqdm(test_loader):
            imgs = imgs.to(device)

            # generate adversarial patch
            if mode=='patch':
                patch, x,y = random_crop(adv_patch, (324,324), pos=(0,0))
            elif mode=='gan':
                z_sample = torch.randn(1,128,324,324, device=device)
                gen_patch = gan.generate(z_sample)
                patch, x,y = random_crop(gen_patch, (324,324), pos=(0,0))
            else: raise

            # apply TPS + patch
            patch_tps,_ = tps.tps_trans(patch, max_range=0.1, canvas=0.5)
            batch_t = patch_transformer(patch_tps, truths, (640,640), do_rotate=True, rand_loc=False)
            adv_imgs = patch_applier(imgs, batch_t)

            # inference on adversarial
            results = model(adv_imgs)
            for i, pred in enumerate(results.pred):
                # pred: [N×6] (x1,y1,x2,y2,conf,cls)
                gt_boxes = truths[i].cpu().numpy()  # Nx4
                total_gts += len(gt_boxes)
                detected = []
                for *xyxy, conf, cls in pred.cpu().numpy():
                    if int(cls)==0:  # person class
                        ious = [iou_xyxy(xyxy, gt) for gt in gt_boxes]
                        if max(ious, default=0) > args.iou_thres:
                            tp_fp.append((conf, True))
                        else:
                            tp_fp.append((conf, False))

    # compute precision/recall
    tp_fp = sorted(tp_fp, key=lambda x: -x[0])
    tps,fps,confs = [],[],[]
    tp=f=0
    for conf, is_tp in tp_fp:
        if is_tp: tp+=1
        else:     f+=1
        tps.append(tp); fps.append(f); confs.append(conf)
    prec = [tp/(tp+fp) for tp,fp in zip(tps,fps)]
    rec  = [tp/total_gts for tp in tps]
    # AP via interpolation
    if len(prec)>1:
        samples = np.linspace(0,1,100)
        ap = float(np.mean(interp1d(rec, prec, bounds_error=False, fill_value=(0,0))(samples)))
    else:
        ap = 0.0
    return prec, rec, ap

# 6) Load adv texture / GAN
if args.method in ['RCA','TCA']:
    cloth = torch.from_numpy(np.load(args.load_path))[None].to(device)
    mode='patch'
    adv_patch=cloth
    gan=None; z=None
elif args.method in ['EGA','TCEGA']:
    gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324,324)).to(device)
    state = torch.load(args.load_path, map_location=device)
    gan.load_state_dict(state); gan.eval()
    if args.method=='EGA':
        mode='gan'; adv_patch=None; z=None
    else:
        z = torch.from_numpy(np.load(args.load_path_z)).to(device)
        mode='gan'; adv_patch=None
else:
    raise ValueError

# 7) Run test & save
os.makedirs('test_results', exist_ok=True)
prec, rec, ap = test(adv_patch, gan, z, mode=mode)
print(f'AP = {ap:.4f}')

np.savez(f'test_results/{args.suffix}.npz', prec=prec, rec=rec, ap=ap)
plt = __import__('matplotlib.pyplot')
plt.plot(rec, prec)
plt.savefig(f'test_results/{args.suffix}.png', dpi=300)

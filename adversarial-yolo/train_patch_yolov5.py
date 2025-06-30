"""
Training code for Adversarial patch training, migrated to YOLOv5.
"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

# ==========================================================================================
# NEW CLASS: YOLOv5ProbExtractor
# This class replaces the old MaxProbExtractor. It is designed to work with the
# output of YOLOv5 models.
# ==========================================================================================
# ==========================================================================================
# NEW CLASS: YOLOv5ProbExtractor (Corrected Version)
# This version correctly processes raw YOLOv5 logits by applying a sigmoid function.
# ==========================================================================================
class YOLOv5ProbExtractor(nn.Module):
    """
    Extracts the maximum objectness score * class probability for a given class from raw YOLOv5 output.
    """
    def __init__(self, class_id, config):
        super(YOLOv5ProbExtractor, self).__init__()
        self.class_id = class_id
        self.config = config

    def forward(self, yolov5_output):
        # Raw model output is a tuple. The first element contains the predictions.
        output = yolov5_output[0] 

        # The output tensor is (batch, num_predictions, 5 + num_classes)
        # Columns 0-4 are: x, y, w, h, objectness_logit
        # Columns 5+ are class_logits

        # Apply sigmoid to convert logits to probabilities
        objectness_prob = torch.sigmoid(output[:, :, 4])
        class_probs = torch.sigmoid(output[:, :, 5:])
        
        # Get the probability for the target class ('person' is class 0 in COCO)
        person_prob = class_probs[:, :, self.class_id]
        
        # Calculate the final detection probability for the 'person' class
        detection_prob = objectness_prob * person_prob
        
        # Get the maximum probability for each image in the batch
        max_det_prob, _ = torch.max(detection_prob, dim=1)
        
        return max_det_prob

# ==========================================================================================
# Main Trainer Class (Modified for YOLOv5)
# ==========================================================================================
class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        # ==================================================================
        # CHANGED: Model Loading
        # We now load a pretrained YOLOv5s model from torch.hub.
        # The old Darknet model is removed.
        # ==================================================================
        # ==================================================================
        # CHANGED: Model Loading (Corrected)
        # We specify the device directly in the hub.load call for robustness.
        # ==================================================================
        print("Loading YOLOv5 model...")
        device = '0' if torch.cuda.is_available() else 'cpu'
        self.yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False, device=device)
        
        # The model is now loaded on the correct device.
        # It's good practice to explicitly set it to eval mode.
        self.yolov5_model.eval()
        
        # Make sure the model does not update its own gradients
        for param in self.yolov5_model.parameters():
            param.requires_grad = False
        print("YOLOv5 model loaded.")

        # These helper classes from the original repo are model-agnostic and can be reused.
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        
        # ==================================================================
        # CHANGED: Probability Extractor
        # We use our new YOLOv5ProbExtractor. '0' is the class ID for 'person' in the COCO dataset.
        # ==================================================================
        self.prob_extractor = YOLOv5ProbExtractor(0, self.config).cuda()
        
        # These loss components are calculated on the patch itself, so they don't need to be changed.
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        """

        # CHANGED: Set image size to a value suitable for YOLOv5 (e.g., 640)
        img_size = 640
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14 # This is related to the dataset labels, no change needed.

        # Generate starting point
        adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu.requires_grad_(True)

        # The dataset loading remains the same
        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)} batches')

        # The optimizer and scheduler setup remains the same
        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    adv_patch = adv_patch_cpu.cuda()

                    # The patch application and transformation logic is unchanged
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    
                    # CHANGED: Input resizing for the model
                    # We resize the batch to the required input size of YOLOv5
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    # ==================================================================
                    # CHANGED: Forward pass through the model
                    # We now pass the patched images through the YOLOv5 model.
                    # We access the raw output via model.forward() which is needed for backprop.
                    # ==================================================================
                    output = self.yolov5_model(p_img_batch)
                    
                    # The rest of the loop is the same, but now uses the new prob_extractor
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(max_prob)
                    
                    # The total loss function remains the same
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss.detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # Keep patch in [0,1] range

                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image('patch', adv_patch_cpu, iteration)

                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()

            et1 = time.time()
            ep_det_loss /= len(train_loader)
            ep_nps_loss /= len(train_loader)
            ep_tv_loss /= len(train_loader)
            ep_loss /= len(train_loader)

            scheduler.step(ep_loss)
            
            print(f'EPOCH {epoch} SUMMARY:')
            print(f'  Epoch Loss: {ep_loss:.4f}')
            print(f'  Detection Loss: {ep_det_loss:.4f}')
            print(f'  NPS Loss: {ep_nps_loss:.4f}')
            print(f'  TV Loss: {ep_tv_loss:.4f}')
            print(f'  Epoch Time: {et1 - et0:.2f}s')
            
            # Save the patch image periodically
            if epoch % 5 == 0:
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                im.save(f"yolov5_patch_epoch_{epoch}.png")

            del output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
            torch.cuda.empty_cache()
            et0 = time.time()

    # The utility functions generate_patch and read_image remain unchanged.
    def generate_patch(self, type):
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
        return adv_patch_cpu

    def read_image(self, path):
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu

def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Example: python train_patch_yolov5.py paper_obj')
        print('Possible modes are:', list(patch_config.patch_configs.keys()))
        return

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()
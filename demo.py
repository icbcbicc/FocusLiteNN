import argparse
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from PIL import Image


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="focuslitenn",
                        help="options: 'focuslitenn, 'eonss', 'densenet13', 'resnet10', 'resnet50', 'resnet101'")
    parser.add_argument("--num_channel", type=int, default=1, help='num of channels for the FocusLiteNN model')
    parser.add_argument('--img', type=str, default="imgs/TCGA@Focus_patch_i_9651_j_81514.png", help='name of the image')
    parser.add_argument("--heatmap", type=bool, default=False, help='value normalized to [0, 1]')
    parser.add_argument("--save_result", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=False)
    return parser.parse_args()


def get_patches(image, output_size, stride):
    w, h = image.size[:2]
    new_h, new_w = output_size, output_size
    stride_h, stride_w = stride, stride

    h_start = np.arange(0, h - new_h + 1, stride_h)
    w_start = np.arange(0, w - new_w + 1, stride_w)

    patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in w_start]

    to_tensor = torchvision.transforms.ToTensor()
    patches = [to_tensor(patch) for patch in patches]
    patches = torch.stack(patches, dim=0)
    return patches


class TestingSingle():
    def __init__(self, config):
        self.config = config
        self.use_cuda = torch.cuda.is_available() and self.config.use_cuda

        # initialize the model
        if config.arch.lower() == "focuslitenn":
            from model.focuslitenn import FocusLiteNN
            self.model = FocusLiteNN(num_channel=config.num_channel)
        elif config.arch.lower() == "eonss":
            from model.eonss import EONSS
            self.model = EONSS()
        elif config.arch.lower() in ["densenet13", "densenet"]:
            self.model = torchvision.models.DenseNet(block_config=(1, 1, 1, 1), num_classes=1)
        elif config.arch.lower() in ["resnet10", "resnet"]:
            from torchvision.models.resnet import BasicBlock
            self.model = torchvision.models.ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=1)
        elif config.arch.lower() == "resnet50":
            self.model = torchvision.models.resnet50(num_classes=1)
        elif config.arch.lower() == "resnet101":
            self.model = torchvision.models.resnet101(num_classes=1)
        else:
            raise NotImplementedError(f"[****] '{config.arch}' is not a valid architecture")
        self.model_name = type(self.model).__name__
        self.model.eval()

        if self.use_cuda:
            print("[*] Using GPU")
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        else:
            print("[*] Using CPU")
            self.model.cpu()
        print("[*] Model %s initialized" % self.model_name)

        # load the model
        if config.arch.lower() == "focuslitenn":
            config.ckpt = os.path.join("pretrained_model", f"focuslitenn-{config.num_channel}kernel.pt")
        else:
            config.ckpt = os.path.join("pretrained_model", config.arch.lower() + ".pt")
        self._load_checkpoint(config.ckpt)
        print("[*] Checkpoint %s loaded" % config.ckpt)

    def eval(self):

        if os.path.isfile(self.config.img):
            image = Image.open(self.config.img)
        else:
            raise Exception("[!] no image found at '{}'".format(self.config.img))

        t1 = time.time()

        stride = 128
        image_patches = get_patches(image, 235, stride)

        image_patches = torch.autograd.Variable(image_patches)
        if self.use_cuda:
            image_patches = image_patches.cuda()

        if self.config.heatmap:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from skimage.transform import resize
            import matplotlib.pyplot as plt

            original_image = np.asarray(image.convert('L'))

            original_h, original_w = original_image.shape[0], original_image.shape[1]
            new_h = math.floor((original_h - 235) / stride) + 1
            new_w = math.floor((original_w - 235) / stride) + 1

            num_patches = int(new_w * new_h)
            heatmap = np.zeros(num_patches)
            for i in range(num_patches):
                heatmap[i] = torch.squeeze(self.model(image_patches[i][None, :, :, :]).cpu().data).numpy()
            score_predict_mean = np.mean(heatmap)
            heatmap = heatmap.reshape([new_h, new_w])

            # normalize
            heatmap -= heatmap.min()
            heatmap /= heatmap.max()

            # interpolate
            heatmap_interpolated = resize(heatmap, (original_h, original_w))

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=200)
            ax.imshow(original_image, cmap='gray')
            im = ax.imshow(heatmap_interpolated, cmap='jet', alpha=0.2, vmin=0.0, vmax=1.0)
            ax.axis('off')
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax1)
            cbar.ax.tick_params(labelsize=12)
            if not os.path.exists("heatmap"):
                os.mkdir("heatmap")
            if self.config.arch.lower() == "focuslitenn":
                heatmap_name = f"heatmap/heatmap_{self.config.arch}_{self.config.num_channel}kernel.png"
            else:
                heatmap_name = f"heatmap/heatmap_{self.config.arch}.png"
            plt.savefig(heatmap_name, bbox_inches='tight', dpi='figure', quality=70)

        else:
            score_predict = self.model(image_patches).cpu().data
            score_predict = torch.squeeze(score_predict, dim=1).numpy()
            score_predict_mean = np.mean(score_predict)

        t2 = time.time()

        print("[-] Image name:\t\t", self.config.img)
        print("[-] %s score:\t%f" % (self.config.arch, score_predict_mean))
        print("[-] Time consumed:\t %.4f s" % ((t2 - t1)))

        if self.config.save_result:
            with open("release/%s_result.txt" % self.model_name, 'w') as txt_file:
                txt_file.write("Image name:\t\t" + str(self.config.img) + "\n" + "%s score:\t" % self.model_name + str(score_predict_mean))

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            if not torch.cuda.is_available():
                checkpoint = torch.load(ckpt, map_location='cpu')
            else:
                checkpoint = torch.load(ckpt)
            model_has_module = (list(self.model.state_dict().keys())[0].lower().find("module") != -1)
            checkpoint_has_module = (list(checkpoint['state_dict'].keys())[0].lower().find("module") != -1)
            if model_has_module and not checkpoint_has_module:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = "module." + k  # add `module.` in the state_dict which is saved with the "nn.DataParallel()"
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            elif not model_has_module and checkpoint_has_module:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove `module.` in the state_dict which is saved with the "nn.DataParallel()"
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))


if __name__ == "__main__":
    cfg = parse_config()
    t = TestingSingle(cfg)
    t.eval()

import argparse
import os
import time
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import FocusDataset
from plcc_loss import PLCCLoss


def parse_config():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--use_cuda", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--seed", type=int, default=2020)

    # CNN architecture
    parser.add_argument("--arch", type=str, default="FocusLiteNN", help='options: FocusLiteNN, EONSS, DenseNet13, ResNet10, ResNet50, ResNet101')
    parser.add_argument("--num_channel", type=int, default=1, help='num of channels for the FocusLiteNN model')

    # training dataset
    parser.add_argument("--trainset", type=str, default="data/FocusPath_full/")
    parser.add_argument("--train_csv", type=str, default="data/FocusPath_full_split1.txt")

    # hyperparameters
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--initial_lr", type=float, default=1e-2)
    parser.add_argument("--decay_interval", type=int, default=60)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="plcc", choices=["plcc", "mse", "mae"])

    # utils
    parser.add_argument("--num_workers", type=int, default=4, help='num of threads to load data')
    parser.add_argument("--epochs_per_save", type=int, default=30)
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str, metavar='PATH', help='path to checkpoints')
    parser.add_argument('--board', default='./board', type=str, help='tensorboard log file path')

    return parser.parse_args()


class Trainer(object):

    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.use_cuda = torch.cuda.is_available() and config.use_cuda

        # dataset
        self.train_transform = transforms.Compose([transforms.RandomCrop(size=235), transforms.ToTensor()])
        self.train_batch_size = config.batch_size
        self.train_data = FocusDataset(csv_file=config.train_csv, root_dir=config.trainset, transform=self.train_transform)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=config.num_workers)
        self.train_data_size = len(self.train_loader.dataset)
        self.num_steps_per_epoch = len(self.train_loader)

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
        num_param = sum([p.numel() for p in self.model.parameters()])
        print(f"[*] Initilizing model: {self.model_name}, num of params: {num_param}")

        if torch.cuda.device_count() > 1 and config.use_cuda:
            print("[*] GPU #", torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        if self.use_cuda:
            self.model.cuda()

        # loss function
        self.loss_type = config.loss_type.lower()
        if self.loss_type == "plcc":
            self.crit_loss = PLCCLoss()
        elif self.loss_type == "mse":
            self.crit_loss = nn.MSELoss(reduction="mean")
        elif self.loss_type == "mae":
            self.crit_loss = nn.L1Loss(reduction="mean")
        else:
            raise NotImplementedError(f"[*] '{self.loss_type}' is not a valid loss type")

        if self.use_cuda:
            self.crit_loss = self.crit_loss.cuda()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.initial_lr)

        # lr scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)

        self.max_epochs = config.max_epochs
        self.epochs_per_save = config.epochs_per_save

        self.ckpt_path = os.path.join(config.ckpt_path, config.arch)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        self.writer = SummaryWriter(log_dir=os.path.join(config.board, config.arch))

        self.arch = config.arch

    def fit(self):
        for epoch in range(self.max_epochs):
            self._train_single_epoch(epoch)

    def _train_single_epoch(self, epoch):
        self.current_epoch = epoch
        local_counter = epoch * self.num_steps_per_epoch + 1
        start_time = time.time()

        # start training
        for step, sample_batched in enumerate(self.train_loader, 0):
            images_batch, score_batch = sample_batched['image'], sample_batched['score']

            image = Variable(images_batch)  # shape: (batch_size, channel, H, W)
            score = Variable(score_batch.float())  # shape: (batch_size)

            if self.use_cuda:
                score = score.cuda()
                image = image.cuda()

            self.optimizer.zero_grad()
            q = self.model(image)

            batch_size = int(q.nelement() / 1)
            q_avg = q.view(batch_size, 1).mean(1)  # shape: (batch_size)
            
            self.loss = self.crit_loss(q_avg, score)
            if self.loss_type == "plcc":
                self.loss = -1 * self.loss

            self.loss.backward()
            self.optimizer.step()

            if self.arch.lower() == "eonss":
                if torch.cuda.device_count() > 1 and self.use_cuda:
                    self.model.module._gdn_param_proc()
                else:
                    self.model._gdn_param_proc()

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/TrainLoss', self.loss.item(), local_counter)
            self.writer.add_scalar('lr', lr, local_counter)

            current_time = time.time()
            duration = current_time - start_time
            examples_per_sec = self.train_batch_size / duration

            format_str = '(E:%d, S:%d) [loss = %.4f, lr = %.6e] (%.1f samples/sec; %.3f sec/batch)'
            print_str = format_str % (epoch, step, self.loss.item(), lr, examples_per_sec, duration)
            print(print_str)

            local_counter += 1
            start_time = time.time()

        self.scheduler.step()

        if (epoch + 1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            if hasattr(self.model, 'module'):
                self._save_checkpoint({'state_dict': self.model.module.state_dict()}, model_name)
            else:
                self._save_checkpoint({'state_dict': self.model.state_dict()}, model_name)

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename):
        torch.save(state, filename)


if __name__ == "__main__":
    cfg = parse_config()
    t = Trainer(cfg)
    t.fit()

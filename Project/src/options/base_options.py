import torch
import os
from torch.utils.tensorboard import SummaryWriter
import utils


class BaseOptions():

    def __init__(self, is_train=True, device=None, config_path=None, gpu_ids=None,
                 logs_dir="raw_dataset/logs", tb_dir="raw_dataset/tb", name="Model", lr=1e-4, batch_size=12, n_epochs=4, print_freq=1, nepochs_decay=200, save_epoch_frequency=3,
                 logs_modeldir="raw_dataset/model", display_freq=6, save_latest_freq=5000
                 ):
        self.is_train = is_train
        self.config_path = config_path
        self.gpu_ids = gpu_ids
        self.logs_dir = logs_dir
        self.name = name
        self.lr = lr
        self.batch_size = batch_size
        self.nepochs = n_epochs
        self.tb_dir = tb_dir
        self.print_freq = print_freq
        self.nepochs_decay = nepochs_decay
        self.save_epoch_freq = save_epoch_frequency
        self.logs_modeldir = logs_modeldir
        self.gpu_ids_str = ""
        self.display_freq = display_freq
        self.save_latest_freq = save_latest_freq

        if (is_train):
            expr_dir = os.path.join(self.logs_dir, self.name)
            # tensorboard writer
            tb_dir = '%s/tboard' % expr_dir
            if not os.path.exists(tb_dir):
                os.makedirs(tb_dir)
            self.tb_dir = tb_dir
            writer = SummaryWriter(log_dir=tb_dir)
            self.writer = writer

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

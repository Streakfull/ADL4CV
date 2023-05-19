import torch


class BaseOptions():

    def __init__(self, is_train=True, device=None, config_path=None, gpu_ids=None,
                 logs_dir="", name="Model", lr=1e-4, batch_size=12, n_epochs=200):
        self.is_train = is_train
        self.config_path = config_path
        self.gpu_ids = gpu_ids
        self.logs_dir = logs_dir
        self.name = name
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        if device is not None:
            self.device = device
        else:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

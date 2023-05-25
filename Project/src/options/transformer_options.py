from options.base_options import BaseOptions


class TransformerOptions(BaseOptions):
    def __init__(self, is_train=True, device=None, config_path=None, gpu_ids=None, logs_dir="raw_dataset/logs",
                 tb_dir="raw_dataset/tb", name="Model", lr=0.0001, batch_size=12, n_epochs=4, print_freq=1,
                 nepochs_decay=200, save_epoch_frequency=3, logs_modeldir="raw_dataset/model",
                 display_freq=6, save_latest_freq=5000, tf_config=None, vq_ckpt=None):
        super().__init__(is_train, device, config_path, gpu_ids, logs_dir, tb_dir, name, lr, batch_size, n_epochs,
                         print_freq, nepochs_decay, save_epoch_frequency, logs_modeldir, display_freq, save_latest_freq)
        self.tf_config = tf_config
        self.vq_ckpt = vq_ckpt

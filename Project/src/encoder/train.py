import sys  # nopep8
import os
sys.path.append(os.path.abspath(os.path.join('..', '')))  # nopep8

from encoder.pvqae import PVQVAE  # nopep8
from options.base_options import BaseOptions
from pathlib import Path
from utils.visualizer import Visualizer
from datasets.shape_net import ShapenetDataset
from utils import util
from torch import profiler
from tqdm import tqdm
from termcolor import colored, cprint
import torch
import inspect
import time
from datasets.shape_net import ShapenetDataset
from dataset_preprocessing.constants import DATA_SET_PATH, FULL_DATA_SET_PATH
from torch.utils.data import DataLoader

# from datasets.shape_net import ShapenetDataset

# from options.train_options import TrainOptions


# from models.base_model import create_model


def get_data_generator(loader):
    while True:
        for data in loader:
            yield data


seed = 512
util.seed_everything(seed)

cwd = Path.cwd()
print(cwd, "CWD")
root_folder = ".."
shape_dir = f"{root_folder}/{DATA_SET_PATH}"
full_dataset_path = f"{root_folder}/{FULL_DATA_SET_PATH}"

shapenet = ShapenetDataset(shape_dir, full_dataset_path,
                           resolution=64, cat="chairs")
train_ds, test_ds = torch.utils.data.random_split(
    shapenet, [0.8, 0.2])
vq_cfg = f"{root_folder}/configs/pvqae_configs.yaml"
options = BaseOptions(config_path=vq_cfg, name="pvqae-6",
                      batch_size=4,
                      n_epochs=30, nepochs_decay=5)

train_dl = DataLoader(train_ds, batch_size=options.batch_size, shuffle=False, sampler=None,
                      batch_sampler=None, num_workers=0, collate_fn=None,
                      pin_memory=False, drop_last=False, timeout=0,
                      worker_init_fn=None,  prefetch_factor=2,
                      persistent_workers=False)

test_dl = DataLoader(test_ds, batch_size=options.batch_size, shuffle=False, sampler=None,
                     batch_sampler=None, num_workers=0, collate_fn=None,
                     pin_memory=False, drop_last=False, timeout=0,
                     worker_init_fn=None,  prefetch_factor=2,
                     persistent_workers=False)


test_dg = get_data_generator(test_ds)
dataset_size = len(train_ds)


cprint('[*] # training images = %d' % len(train_ds), 'yellow')
cprint('[*] # testing images = %d' % len(test_ds), 'yellow')


pvqvae = PVQVAE()
pvqvae.initialize(options)

cprint(f'[*] "{pvqvae.name()}" initialized.', 'cyan')

visualizer = Visualizer(options)
expr_dir = '%s/%s' % (options.logs_modeldir, options.name)
model_f = inspect.getfile(pvqvae.__class__)
dset_f = inspect.getfile(train_ds.__class__)
cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
# os.system(f'cp {model_f} {modelf_out}')
# os.system(f'cp {dset_f} {dsetf_out}')


cprint("[*] Using pytorch's profiler...", 'blue')
tensorboard_trace_handler = profiler.tensorboard_trace_handler(options.tb_dir)
schedule_args = {'wait': 2, 'warmup': 2, 'active': 6, 'repeat': 1}
schedule = profiler.schedule(**schedule_args)
activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]


def train_one_epoch(pt_profiler=None):
    global total_steps

    epoch_iter = 0
    for i, data in tqdm(enumerate(train_dl), total=len(train_dl)):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += options.batch_size
        epoch_iter += options.batch_size

        pvqvae.set_input(data)

        pvqvae.optimize_parameters(total_steps)

        nBatches_has_trained = total_steps // options.batch_size

        # if total_steps % options.print_freq == 0:
        if nBatches_has_trained % options.print_freq == 0:
            errors = pvqvae.get_current_errors()

            t = (time.time() - iter_start_time) / options.batch_size
            visualizer.print_current_errors(
                epoch, epoch_iter, total_steps, errors, t)

        if (nBatches_has_trained % options.display_freq == 0) or i == 0:
            # eval
            pvqvae.inference(data)
            visualizer.display_current_results(
                pvqvae.get_current_visuals(), total_steps, phase='train')

            # pvqvae.set_input(next(test_dg))
            test_data = next(test_dg)
            pvqvae.inference(test_data.unsqueeze(0))
            visualizer.display_current_results(
                pvqvae.get_current_visuals(), total_steps, phase='test')

        if total_steps % options.save_latest_freq == 0:
            cprint('saving the latest pvqvae (epoch %d, total_steps %d)' %
                   (epoch, total_steps), 'blue')
            latest_name = f'epoch-latest'
            pvqvae.save(latest_name)

        if pt_profiler is not None:
            pt_profiler.step()


cprint('[*] Start training. name: %s' % options.name, 'blue')
total_steps = 0
for epoch in range(options.nepochs + options.nepochs_decay):
    epoch_start_time = time.time()
    # epoch_iter = 0

    # profile
    with profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=tensorboard_trace_handler,
        record_shapes=True,
        with_stack=True,
    ) as pt_profiler:
        train_one_epoch(pt_profiler)

    if epoch % options.save_epoch_freq == 0:
        cprint('saving the model at the end of epoch %d, iters %d' %
               (epoch, total_steps), 'blue')
        latest_name = f'epoch-latest'
        pvqvae.save(latest_name)
        cur_name = f'epoch-{epoch}'
        pvqvae.save(cur_name)

    # eval every 3 epoch
    if epoch % options.save_epoch_freq == 0:
        metrics = pvqvae.eval_metrics(test_dl)
        visualizer.print_current_metrics(epoch, metrics, phase='test')
        print(metrics)

    cprint(f'[*] End of epoch %d / %d \t Time Taken: %d sec \n%s' %
           (
               epoch, options.nepochs + options.nepochs_decay,
               time.time() - epoch_start_time,
               os.path.abspath(os.path.join(options.logs_dir, options.name))
           ), 'blue', attrs=['bold']
           )
    pvqvae.update_learning_rate()

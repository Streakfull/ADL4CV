import torch
from models.base_model import BaseModel
import omegaconf
# from utils.util_3d import init_mesh_renderer
from utils.util import iou
from einops import rearrange
from encoder.auto_encoder import AutoEncoder
from tqdm import tqdm
from collections import OrderedDict
from models.pvqvae_networks.losses import VQLoss
import os
from torch import optim
from termcolor import colored
#from utils.util_3d import render_sdf, init_mesh_renderer


class PVQVAE(BaseModel):
    def name(self):
        return 'PVQVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.is_train
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.config_path is not None
        configs = omegaconf.OmegaConf.load(opt.config_path)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        self.n_embed = n_embed

        n_down = len(ddconfig.ch_mult) - 1

        self.vqvae = AutoEncoder(ddconfig, n_embed, embed_dim)
        self.vqvae.to(opt.device)

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            lossconfig = configs.lossconfig
            lossparams = lossconfig.params
            self.loss_vq = VQLoss(**lossparams).to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam(
                self.vqvae.parameters(), lr=opt.lr, betas=(0.5, 0.9))

            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 30, 0.5,)
            # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        resolution = configs.model.params.ddconfig['resolution']
        self.resolution = resolution

        # setup hyper-params
        nC = resolution
        self.cube_size = 2 ** n_down  # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        # assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        dist, elev, azim = 1.7, 20, 20
        # self.renderer = init_mesh_renderer(
        #     image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        self.best_iou = -1e12

    @staticmethod
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """ 
            assume x.shape: b, c, d, h, w 
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(
            3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(
            x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    @staticmethod
    # def fold_to_voxels(self, x_cubes, batch_size, ncubes_per_dim):
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                      p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def set_input(self, input):
        '''Samples at training time'''
        # import pdb; pdb.set_trace()
        x = input
        self.x = x
        self.input = x
        self.cur_bs = x.shape[0]  # to handle last batch

        self.x_cubes = self.unfold_to_cubes(x, self.cube_size, self.stride)
        vars_list = ['x', 'x_cubes']

        self.tocuda(var_names=vars_list)

    def encode_indices(self, sdf):
        n_batch = sdf.shape[0]
        self.set_input(sdf)
        indices = self.vqvae.encode(self.x_cubes, return_indices=True)
        indices = indices.reshape(
            n_batch, self.ncubes_per_dim,  self.ncubes_per_dim,  self.ncubes_per_dim)
        return indices

    def forward(self):
        # qloss: codebook loss
        self.zq_cubes, self.qloss, _ = self.vqvae.encode(
            self.x_cubes)  # zq_cubes: ncubes X zdim X 1 X 1 X 1
        # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        self.zq_voxels = self.fold_to_voxels(
            self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
        self.x_recon = self.vqvae.decode(self.zq_voxels)

    def inference(self, data, should_render=False, verbose=False):
        self.vqvae.eval()
        self.set_input(data)

        # make sure it has the same name as forward
        with torch.no_grad():
            self.zq_cubes, _, self.info = self.vqvae.encode(self.x_cubes)
            self.zq_voxels = self.fold_to_voxels(
                self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
            self.x_recon = self.vqvae.decode(self.zq_voxels)
            # _, _, quant_ix = info
            #

            # if should_render:
            #     self.image = render_sdf(self.renderer, self.x)
            #     self.image_recon = render_sdf(self.renderer, self.x_recon)

        self.vqvae.train()

    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        # self.set_input(data)

        self.vqvae.eval()
        self.inference(data, should_render=False)
        self.vqvae.train()

        x = self.x
        x_recon = self.x_recon

        iou_result = iou(x, x_recon, thres)

        return iou_result

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()

        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):

                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()

        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name)

        self.train()
        return ret

    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        aeloss, log_dict_ae = self.loss_vq(self.qloss, self.x, self.x_recon)

        self.loss = aeloss

        self.loss_codebook = log_dict_ae['loss_codebook']
        self.loss_nll = log_dict_ae['loss_nll']
        self.loss_rec = log_dict_ae['loss_rec']
        self.loss_p = log_dict_ae['loss_p']
        self.loss.backward()

    def optimize_parameters(self, total_steps=None):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()

    def get_current_visuals(self):

        with torch.no_grad():
            self.image = render_sdf(self.renderer, self.x)
            self.image_recon = render_sdf(self.renderer, self.x_recon)

        vis_tensor_names = [
            'image',
            'image_recon',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        # vis_tensor_names = ['%s/%s' % (phase, n) for n in vis_tensor_names]
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)
        # return OrderedDict()

    def get_current_errors(self):

        ret = OrderedDict([
            ('codebook', self.loss_codebook.data),
            ('nll', self.loss_nll.data),
            ('rec', self.loss_rec.data),
            ('p', self.loss_p.data),
        ])

        return ret

    def save(self, label):
        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        if torch.cuda.is_available():
            torch.save(self.vqvae.cpu().state_dict(), save_path)
            self.vqvae.cuda()
        else:
            torch.save(self.vqvae.cpu().state_dict(), save_path)

    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from models.pvqvae_networks.modules import Encoder3D, Decoder3D
from models.pvqvae_networks.quantizer import VectorQuantizer
from utils.model_utils import init_weights
from einops import rearrange


class AutoEncoder(nn.Module):
    def __init__(
            self,
            ddconfig,
            n_embed,
            embed_dim,
            remap=None,
            sane_index_shape=False
    ):
        """_summary_

            Args:
                ddconfig (object): configuration passed to encoder and decoder modules
                n_embed (_type_): number of embeddings for the Z codebook
                embed_dim (_type_): embedding dimensions for the Z codebook
                remap (_type_, optional): input Codebook. Defaults to None.
                sane_index_shape (bool, optional): Defaults to False.
        """

        super(AutoEncoder, self).__init__()
        self.ddconfig = ddconfig

        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                        remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], 1)
        init_weights(self.encoder, 'normal', 0.02)
        init_weights(self.decoder, 'normal', 0.02)
        init_weights(self.quant_conv, 'normal', 0.02)
        init_weights(self.post_quant_conv, 'normal', 0.02)

    def encode(self, x):
        """_summary_

            Args:
                x (patches): _description_

            Returns:
                _type_: _description_
            """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_from_quant(self, quant_code):
        embed_from_code = self.quantize.embedding(quant_code)
        return embed_from_code

    def decode_enc_idices(self, enc_indices, z_spatial_dim=8):

        # for transformer
        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices)  # (bs t) zd
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3',
                        d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, verbose=False):
        quant, diff, info = self.encode(input)
        dec = self.decode(quant)

        if verbose:
            return dec, quant, diff, info
        else:
            return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

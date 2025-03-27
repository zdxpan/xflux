import math
from typing import Tuple, List, Union, Generator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as InnerAutoencoderKL
from diffusers.models.autoencoders.vae import Decoder, DecoderOutput
from diffusers.models.resnet import ResnetBlock2D
# from dl_models.modules.utils import create_np_image
from tensor_util import create_np_image
from torch import Tensor
from torch.nn import GroupNorm

Tile = Var = Mean = Tensor
TaskRet = Union[Tuple[GroupNorm, Tile, Tuple[Var, Mean]], Tile]
TaskGen = Generator[TaskRet, None, None]
BBox = Tuple[int, int, int, int]
sync_approx: bool = False  # True: apply, False: collect
sync_approx_plan: List[Tuple[Var, Mean]] = []
sync_approx_pc: int = 0


def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c / num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])

    var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    if torch.isinf(var).any():
        var, mean = torch.var_mean(input_reshaped.float(), dim=[0, 2, 3, 4], unbiased=False)
        var, mean = var.to(input_reshaped.dtype), mean.to(input_reshaped.dtype)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c / num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None, training=False, momentum=0, eps=eps)
    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None: out *= weight.view(1, -1, 1, 1)
    if bias is not None: out += bias.view(1, -1, 1, 1)
    return out


def nonlinearity(x: Tensor) -> Tensor:
    return F.silu(x, inplace=True)


def GroupNorm_forward(gn: GroupNorm, h: Tensor) -> TaskGen:
    if sync_approx:  # apply
        global sync_approx_pc
        var, mean = sync_approx_plan[sync_approx_pc]
        h = custom_group_norm(h, gn.num_groups, mean, var, gn.weight, gn.bias, gn.eps)
        sync_approx_pc = (sync_approx_pc + 1) % len(sync_approx_plan)
    else:  # collect
        var, mean = get_var_mean(h, gn.num_groups, gn.eps)
        if var.dtype == torch.bfloat16 and var.isinf().any():
            fp32_tile = h.float()
            var, mean = get_var_mean(fp32_tile, 32)
        sync_approx_plan.append((var, mean))
        h = gn(h)
    yield h


def Resblock_forward(self: ResnetBlock2D, x: Tensor) -> TaskGen:
    h = x
    for item in GroupNorm_forward(self.norm1, h):
        if isinstance(item, Tensor):
            h = item
        else:
            yield item
    h = nonlinearity(h)
    h: Tensor = self.conv1(h)
    for item in GroupNorm_forward(self.norm2, h):
        if isinstance(item, Tensor):
            h = item
        else:
            yield item
    h = nonlinearity(h)
    h = self.conv2(h)
    if self.in_channels != self.out_channels:
        x = self.conv_shortcut(x)
    yield x + h


def AttnBlock_forward(self: Attention, x: Tensor) -> TaskGen:
    h = x
    for item in GroupNorm_forward(self.group_norm, h):
        if isinstance(item, Tensor):
            h = item
        else:
            yield item
    h = h.permute(0, 2, 3, 1)
    q = self.to_q(h)
    k = self.to_k(h)
    v = self.to_v(h)
    B, H, W, C = q.shape
    q = q.reshape(B, C, H * W)
    q = q.permute(0, 2, 1)  # b,hw,c
    k = k.reshape(B, C, H * W)  # b,c,hw
    w = torch.bmm(q, k)
    w = w * (int(C) ** (-0.5))
    w = torch.nn.functional.softmax(w, dim=2)
    v = v.reshape(B, C, H * W)
    w = w.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
    h = torch.bmm(v, w)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h = h.reshape(B, C, H, W)
    h = h.permute(0, 2, 3, 1)
    h: Tensor = self.to_out[0](h)
    h = h.permute(0, 3, 1, 2)
    yield x + h


def _mid_forward(self, x: Tensor) -> TaskGen:
    for item in Resblock_forward(self.mid_block.resnets[0], x):
        if isinstance(item, Tensor):
            x = item
        else:
            yield item
    for item in AttnBlock_forward(self.mid_block.attentions[0], x):
        if isinstance(item, Tensor):
            x = item
        else:
            yield item
    for item in Resblock_forward(self.mid_block.resnets[1], x):
        if isinstance(item, Tensor):
            x = item
        else:
            yield item
    yield x


def Decoder_forward(self: Decoder, x: Tensor) -> TaskGen:
    x = self.conv_in(x)  # [B, C=4, H, W] => [B, C=512, H, W]
    # skip_dec = skip_infer_plan if skip_infer else skip_infer_plan_dummy
    for item in _mid_forward(self, x):
        if isinstance(item, Tensor):
            x = item
        else:
            yield item
    if self.vae_type == "upscaler_vae":
        self.num_resolutions = 3
    else:
        self.num_resolutions = 4
    self.block_id = len(self.up_blocks[0].resnets)
    self.condition_id = self.num_resolutions - 1
    for i_level in range(self.num_resolutions):
        for i_block in range(self.block_id):
            for item in Resblock_forward(self.up_blocks[i_level].resnets[i_block], x):
                if isinstance(item, Tensor):
                    x = item
                else:
                    yield item
        if i_level != self.condition_id:
            x = self.up_blocks[i_level].upsamplers[0](x)
    for item in GroupNorm_forward(self.conv_norm_out, x):
        if isinstance(item, Tensor):
            x = item
        else:
            yield item

    x = nonlinearity(x)
    x = self.conv_out(x)
    yield x.cpu()


def get_real_tile_config(z: Tensor, tile_size: int) -> Tuple[int, int, int, int]:
    B, C, H, W = z.shape
    tile_size_H = tile_size_W = tile_size
    n_tiles_H = math.ceil(H / tile_size_H)
    n_tiles_W = math.ceil(W / tile_size_W)
    return tile_size_H, tile_size_W, n_tiles_H, n_tiles_W


def make_bbox(n_tiles_H: int, n_tiles_W: int, tile_size_H: int, tile_size_W: int, H: int, W: int, P: int,
              scaler: Union[int, float]) -> Tuple[List[BBox], List[BBox]]:
    bbox_inputs: List[BBox] = []
    bbox_outputs: List[BBox] = []

    x = 0
    for _ in range(n_tiles_H):
        y = 0
        for _ in range(n_tiles_W):
            bbox_inputs.append((
                x, min(x + tile_size_H, H) + 2 * P,
                y, min(y + tile_size_W, W) + 2 * P,
            ))
            bbox_outputs.append((
                int(x * scaler), int(min(x + tile_size_H, H) * scaler),
                int(y * scaler), int(min(y + tile_size_W, W) * scaler),
            ))
            y += tile_size_W
        x += tile_size_H

    return bbox_inputs, bbox_outputs


def VAE_forward_tile(self, z, tile_size, P):
    """
    Decode a latent vector z into an image in a tiled manner.
    @param z: latent vector
    @return: image
    """
    global sync_approx, sync_approx_plan
    if len(self.block_out_channels) == 4:
        self.decoder.vae_type = "normal_vae"
    else:
        self.decoder.vae_type = "upscaler_vae"
    self = self.decoder.to(torch.bfloat16)
    z = z.to(torch.bfloat16)
    B, C, H, W = z.shape
    if self.vae_type == "upscaler_vae":
        scaler = 4
    else:
        scaler = 8

    ch = 3
    result = None
    zigzag_dir = True
    tile_size_H, tile_size_W, n_tiles_H, n_tiles_W = get_real_tile_config(z, tile_size)
    ''' split tiles '''
    if P != 0: z = F.pad(z, (P, P, P, P), mode='reflect')  # [B, C, H+2*pad, W+2*pad]
    bbox_inputs, bbox_outputs = make_bbox(n_tiles_H, n_tiles_W, tile_size_H, tile_size_W, H, W, P, scaler)
    workers: List[TaskGen] = []
    for bbox in bbox_inputs:
        Hs, He, Ws, We = bbox
        tile = z[:, :, Hs:He, Ws:We]
        workers.append(Decoder_forward(self, tile))
    del z
    n_workers = len(workers)
    ''' start workers '''
    # steps =   n_workers
    # pbar = tqdm(total=steps, desc=f'VAE tile decoding')
    while True:
        outputs: List[TaskRet] = [None] * n_workers
        for i in (reversed if zigzag_dir else iter)(range(n_workers)):
            outputs[i] = next(workers[i])
            # pbar.update()
            if isinstance(outputs[i], Tile): workers[i] = None  # trick: release resource when done
        zigzag_dir = not zigzag_dir
        assert len(bbox_outputs) == len(outputs), 'n_tiles != n_bbox_outputs'
        result = torch.zeros([B, ch, int(H * scaler), int(W * scaler)], dtype=outputs[0].dtype)
        crop_pad = lambda x, P: x if P == 0 else x[:, :, P:-P, P:-P]
        for i, bbox in enumerate(bbox_outputs):
            Hs, He, Ws, We = bbox
            result[:, :, Hs:He, Ws:We] += crop_pad(outputs[i], int(P * scaler))
        # pbar.close()
        break
    ''' finish '''
    return DecoderOutput(sample=result.to(torch.float32))


class AutoencoderKL(InnerAutoencoderKL):
    def tiled_decode(self, z, return_dict: bool = True):
        global sync_approx, sync_approx_plan
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
            # z = self.post_quant_conv(z)
        tile_size = 128
        pad_size = 11
        sync_approx = False
        sync_approx_plan.clear()

        z_float32 = z.to(torch.float32)
        z_hat: Tensor = F.interpolate(z_float32, size=(tile_size, tile_size),
                                      mode='nearest')  # NOTE: do NOT interp in order to keep stats
        z_hat = z_hat.to(torch.float32)
        std_src, mean_src = torch.std_mean(z_hat, dim=[0, 2, 3], keepdim=True)
        std_tgt, mean_tgt = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
        z_hat = (z_hat - mean_src) / std_src
        z_hat = z_hat * std_tgt + mean_tgt
        z_hat = z_hat.clamp_(z.min(), z.max())
        del std_src, mean_src, std_tgt, mean_tgt
        x_hat = VAE_forward_tile(self, z_hat, tile_size, pad_size)
        del z_hat, x_hat
        sync_approx = True
        result = VAE_forward_tile(self, z, tile_size, pad_size)
        self.decoder.to(torch.float16)
        return result

    def decode_image(self, latent, height=None, width=None):
        latent = (
                         1 / self.config.scaling_factor
                 ) * latent + self.config.shift_factor
        x_samples = self.decode(latent).sample
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.float()
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        if height != None and width != None:
            x_samples = x_samples[:, :height, :width, :]
        samples = [
            Image.fromarray(x) for x in (x_samples * 255).round().astype(np.uint8)
        ]
        return samples

    def encode_image(self, image, batch_size=1):
        latent = self.encode(
            create_np_image(image).to(self.device, dtype=self.dtype)
        ).latent_dist.sample()
        latent = (
                         latent - self.config.shift_factor
                 ) * self.config.scaling_factor
        if batch_size > 1:
            latent = latent.repeat(batch_size, 1, 1, 1)
        return latent

    @property
    def vae_scale_factor(self):
        return 2 ** (len(self.config.block_out_channels))

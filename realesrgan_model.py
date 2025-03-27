import os
import sys

import numpy as np
import torch

sys.path.append('../..')

from log import statistical_runtime
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGAN:
    MODEL_NAME = 'realesr-general-x4v3.pth'

    @torch.no_grad()
    def __call__(self, image, upscale=2):
        with statistical_runtime(f'RealESRGAN x{upscale} upscaling'):
            image = image.convert('RGB')
            image = np.array(image, dtype=np.uint8)
            x_image, _ = self.upsampler.enhance(
                image,
                outscale=upscale
            )
            image = Image.fromarray(x_image)
        return image

    @torch.no_grad()
    def __init__(self, model_dir, device, bg_tile=0, denoise_strength=1):
        use_half_precision = True if str(device) == 'cuda' else False
        model_path = os.path.join(model_dir, self.MODEL_NAME)
        dni_weight = None

        if denoise_strength != 1:
            model_path = [model_path, model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=dni_weight,
            model=SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type='prelu'
            ),
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half_precision
        )


import sys
__file = '/home/dell/workspace/js-ai-svc/src/xflux'
sys.path.append(__file)

import torch
from dataclasses import dataclass, field, asdict
from log import logger, statistical_runtime
statistical_runtime.reset_collection()
from flux_pipe import FluxPipeline, FluxNormalInput, normalize_size
from text_encoder import PromptEncoder
import os, copy
device = 'cuda'
model_id = '/data/models/'

DEFAULT_INDEX = {
    "annotator": model_id+"annotator",
    "clip_vit_large": model_id+"clip_vit_large",
    "safety_check": model_id+"safety_check",
    "flux": {
        "root": model_id + "flux.1-dev/",
    },
}

# annotator = Annotator(self.index["_"]["annotator"], self.device)

text_encoder = PromptEncoder(
    base=DEFAULT_INDEX["flux"],
    device=device,
    dtype=torch.bfloat16,
)
pipe = FluxPipeline(base = DEFAULT_INDEX['flux']['root'], device=device, dtype=torch.bfloat16,esrgan=None, 
            controlnet=None, encoder_hid_proj=None, 
            prompt_encoder=text_encoder, annotator=None
        )
pipe.initiate()

default_params = FluxNormalInput(task='generate')
default_template: dict = asdict(default_params)



def test_tile():
    from PIL import Image
    img = Image.open('/home/dell/workspace/img/lowq/scenary.jpeg')

    # img = res['images'][0]
    args = {
        "prompt": "beautiful scenary at seaside, beach, birds view from top, blue ocen",
        "resizer": 'crop_middle',
        "steps": 30, "n_samples": 1,
        "tile_percent": 0, "enable_tile": True, "new_tile": True,   # 启用分块放大算法~
        "width": 3840, "height": 2880,
        "file": [img], "image": 0,  "strength": 0.65,
        "mask": None,  "scale": 4.5,
        "seed": [-1], "variant": 0.0, "variant_seed": [-1],
    }
    args['prompt'] = '##'.join([args['prompt']]*4)
    default_template.update(args)
    normal_input = FluxNormalInput(**default_template)
    normal_input.normal_width, normal_input.normal_height = normalize_size(
            normal_input.width, normal_input.height, 4096)
    res = pipe(normal_input)

if __name__ == '__main__':
    test_tile()
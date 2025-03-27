import sys
__file = '/home/dell/workspace/js-ai-svc/src/xflux'
sys.path.append(__file)

import torch

import numpy as np
from PIL import Image
from typing import Tuple, List, Union
from torch import Tensor
import torch
import time, os
import random
import math, cv2
import gradio as gr
from gradio_imageslider import ImageSlider

from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.models import SD3Transformer2DModel

from dataclasses import dataclass, field, asdict
from log import logger, statistical_runtime
statistical_runtime.reset_collection()
from flux_pipe import FluxPipeline, FluxNormalInput, normalize_size
from text_encoder import PromptEncoder
import os, copy
from llm import QwenClient, MinicpmClient
from realesrgan_model import RealESRGAN
import matplotlib.pyplot as plt

from clip_inter_rogator import ClipInterrogator
from ttp_tile import concatenate_columnwise, concatenate_rowwise, tensor2pil, pil2tensor
from ttp_tile import TTP
from diffusers.utils import make_image_grid

dtype=torch.bfloat16
device = "cuda"
model_id = '/data/models/'
sd3_modelid = '/home/dell/models/stable-diffusion-3.5/'
sd3turbo_modelid = '/home/dell/models/stable-diffusion-3.5-large-turbo/'

clip_vit_cache_path = '/data/comfy_model/clip_interrogator/'
blip_image_caption_path = '/home/dell/comfy_model/blip_image_caption'

LLM_CONFIGS = {
    'qwen': { 'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1", 'api_key': os.getenv("DASHSCOPE_API_KEY") },
    # 'minicpm': { 'base_url': "http://prod.tair.a1-llm.xiaopiu.com//api/llava/generate", 'api_key': '' }
    'minicpm': { 'base_url': "http://192.168.1.4:5001/api/minicpm/generate", 'api_key': '' }        # ok pass    
}
LLM_MODELS = {
    'qwen': QwenClient,
    'minicpm': MinicpmClient
}

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
f1pipe = FluxPipeline(base = DEFAULT_INDEX['flux']['root'], device=device, dtype=torch.bfloat16,esrgan=None, 
            controlnet=None, encoder_hid_proj=None, 
            prompt_encoder=text_encoder, annotator=None
        )
f1pipe.initiate()
clip_rogator = ClipInterrogator()
clip_rogator.load_model(clip_vit_cache_path, blip_image_caption_path)  # 1.9G


default_params = FluxNormalInput(task='generate')
default_template: dict = asdict(default_params)



class MultiplySigmas:
    # sigmas: input  
    # factor 0 ~ 100   "step": 0.001  default - 1.0
    # start:  0, 0~1  0.001  same with end
    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, factor, start, end):
        # Clone the sigmas to ensure the input is not modified (stateless)
        # sigmas = sigmas.clone()
        
        total_sigmas = len(sigmas)
        start_idx = int(start * total_sigmas)
        end_idx = int(end * total_sigmas)

        for i in range(start_idx, end_idx):
            sigmas[i] *= factor
        return sigmas


def timer_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ceildiv(big, small):
    # Correct ceiling division that avoids floating-point errors and importing math.ceil.
    return -(big // -small)

class BBox:
    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)
        self.prompt = ''

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]


def split_4bboxes(w:int, h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = 2
    rows = 2
    tile_w = w // 2 + overlap
    tile_h = h // 2 + overlap
    bbox_1 = BBox(0,          0,        tile_w, tile_h)
    bbox_2 = BBox(w - tile_w, 0,        tile_w, tile_h)
    bbox_3 = BBox(0,          h-tile_h, tile_w, tile_h)
    bbox_4 = BBox(w - tile_w, h-tile_h, tile_w, tile_h)
    
    bbox_list = [bbox_1, bbox_2, bbox_3, bbox_4]
    weight = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    for bbox in bbox_list:
        weight[bbox.slicer] += init_weight
    return bbox_list, weight

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight, cols, rows

@torch.no_grad()
def generate_llm(prompt, image, model='minicpm-v', width=1024, height=1024, resizer='middle_crop'):
    '''
        使用llava vlm 获取图片描述
    '''
    w,h = image.size
    image = image.resize(size = (w//2,h//2)) if h > 768 else image

    kwargs = {}

    if model in ['qwen-vl-plus', 'qwen-turbo']:
        # ground_prompt = chat_qwen(image, prompt = prompt, model=model)
        client = LLM_MODELS.get('qwen')(**LLM_CONFIGS['qwen'])
    else:
        # ground_prompt = vlm_minicpm([image], prompt = prompt)[0]
        client = LLM_MODELS.get('minicpm')(**LLM_CONFIGS['minicpm'])

    msg = client.send(image=image,prompt=prompt,**kwargs)
    ground_prompt = msg['content']
    print('>> ground_prompt:', ground_prompt)
    return ground_prompt


class LazyLoadPipeline:
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        pipe = StableDiffusion3Pipeline.from_pretrained(
            sd3_modelid, 
            torch_dtype=torch.bfloat16
        ).to(device)
                # load turvo model
        from diffusers.models import SD3Transformer2DModel
        # transformer_turbo = SD3Transformer2DModel.from_pretrained(
        #     sd3turbo_modelid,  subfolder='transformer', #"diffusers/controlnet-canny-sdxl-1.0",
        #     torch_dtype=dtype, use_safetensors=True,
        # ).to(device)
        transformer = SD3Transformer2DModel.from_pretrained(
            sd3_modelid,  subfolder='transformer', #"diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=dtype, use_safetensors=True,
        ).to(device)
        pipe.transformer = transformer
        # self.transformer_turbo = transformer_turbo
        self.transformer = transformer

        self.sd3pipe = StableDiffusion3Pipeline(
            transformer=pipe.transformer, 
            scheduler=pipe.scheduler, 
            vae = pipe.vae, 
            text_encoder = pipe.text_encoder, 
            text_encoder_2 = pipe.text_encoder_2, 
            text_encoder_3 = pipe.text_encoder_3, 
            tokenizer = pipe.tokenizer,
            tokenizer_2 = pipe.tokenizer_2,
            tokenizer_3 = pipe.tokenizer_3)
        self.sd3i2ipipe = StableDiffusion3Img2ImgPipeline(
            transformer=pipe.transformer, 
            scheduler=pipe.scheduler, 
            vae = pipe.vae, 
            text_encoder = pipe.text_encoder, 
            text_encoder_2 = pipe.text_encoder_2, 
            text_encoder_3 = pipe.text_encoder_3, 
            tokenizer = pipe.tokenizer,
            tokenizer_2 = pipe.tokenizer_2,
            tokenizer_3 = pipe.tokenizer_3)

        # self.pipe.vae.enable_tiling()
        # self.pipe.vae.enable_slicing()
        # pipe.load_lora_weights(model_index['lora1'])   #    "models/Lora/SDXLrender_v2.0.safetensors")
        # pipe.fuse_lora(lora_scale=0.5)
        # pipe.load_lora_weights(model_index['lora2'])  #    "models/Lora/more_details.safetensors")
        # pipe.fuse_lora(lora_scale=1.)
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        return pipe
    
    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)


class LazyRealESRGAN:
    def __init__(self, device, scale):
        self.device = device
        self.scale = scale
        self.model = None

    def load_model(self):
        gan_model_id = '/home/dell/models/realesrgan/weights/'
        if self.model is None:
            self.model = RealESRGAN(model_dir=gan_model_id, device=device,bg_tile=512, denoise_strength=0.5)
            # self.model.load_weights(f'models/upscalers/RealESRGAN_x{self.scale}.pth', download=False)
    def predict(self, img, scale=2):
        self.load_model()
        return self.model(img, scale)

lazy_realesrgan_x2 = LazyRealESRGAN(device=device, scale=4)

@timer_func
def resize_and_upscale(input_image, resolution):
    scale = 2 if resolution <= 2048 else 4
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H = int(round(H * k / 64.0)) * 64
    W = int(round(W * k / 64.0)) * 64
    img = lazy_realesrgan_x2.predict(img)  # use gan, get better result~
    img = img.resize((W, H), resample=Image.Resampling.LANCZOS)
    # img = img.resize(size = (int(W*2), int(H*2)))
    # if scale == 2:
    #     img = lazy_realesrgan_x2.predict(img)
    # else:
    #     img = lazy_realesrgan_x2.predict(img)
    #     img = lazy_realesrgan_x2.predict(img)
    return img


@timer_func
def create_hdr_effect(original_image, hdr):
    if hdr == 0:
        return original_image
    cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    factors = [1.0 - 0.9 * hdr, 1.0 - 0.7 * hdr, 1.0 - 0.45 * hdr,
              1.0 - 0.25 * hdr, 1.0, 1.0 + 0.2 * hdr,
              1.0 + 0.4 * hdr, 1.0 + 0.6 * hdr, 1.0 + 0.8 * hdr]
    images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]
    merge_mertens = cv2.createMergeMertens()
    hdr_image = merge_mertens.process(images)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

@timer_func
def progressive_upscale(input_image, target_resolution, steps=3):
    current_image = input_image.convert("RGB")
    current_size = max(current_image.size)
    
    for _ in range(steps):
        if current_size >= target_resolution:
            break
        
        scale_factor = min(2, target_resolution / current_size)
        new_size = (int(current_image.width * scale_factor), int(current_image.height * scale_factor))
        
        if scale_factor <= 1.5:
            current_image = current_image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            current_image = lazy_realesrgan_x2.predict(current_image)
            # current_image = current_image.resize((int(current_image.width * 2), int(current_image.height * 2)), Image.Resampling.LANCZOS)
        
        current_size = max(current_image.size)
    
    # Final resize to exact target resolution
    if current_size != target_resolution:
        aspect_ratio = current_image.width / current_image.height
        if current_image.width > current_image.height:
            new_size = (target_resolution, int(target_resolution / aspect_ratio))
        else:
            new_size = (int(target_resolution * aspect_ratio), target_resolution)
        current_image = current_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return current_image

def prepare_image(input_image, resolution, hdr):
    upscaled_image = progressive_upscale(input_image, resolution)
    return create_hdr_effect(upscaled_image, hdr)

def create_gaussian_weight(tile_size, sigma=0.3):
    x = np.linspace(-1, 1, tile_size)
    y = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, y)
    gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian_weight

def adaptive_tile_size(image_size, base_tile_size=512, max_tile_size=1024):
    w, h = image_size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        tile_w = min(w, max_tile_size)
        tile_h = min(int(tile_w / aspect_ratio), max_tile_size)
    else:
        tile_h = min(h, max_tile_size)
        tile_w = min(int(tile_h * aspect_ratio), max_tile_size)
    return max(tile_w, base_tile_size), max(tile_h, base_tile_size)

def process_tile(tile, num_inference_steps, strength, guidance_scale, controlnet_strength):
    prompt = "masterpiece, best quality, highres"
    negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality, verybadimagenegative_v1.3, JuggernautNegative-neg"
    
    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": tile,
        "control_image": tile,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": float(controlnet_strength),
        "generator": torch.Generator(device=device).manual_seed(random.randint(0, 2147483647)),
    }
    return np.array(lazy_pipe(**options).images[0])



# print('>> Image load done')
# --- preprare model~ -----
lazy_pipe = LazyLoadPipeline()
if 0:
    if is_turbo:
        lazy_pipe.transformer = lazy_pipe.transformer_turbo
    else:
        lazy_pipe.transformer = lazy_pipe.transformer
# lazy_pipe.load()


# @spaces.GPU
@timer_func
def gradio_process_image(input_image, resolution, steps, strength, hdr, 
    tile_rows, tile_cols, sigma_scale, sigma_start, sigma_end,
    guidance_scale, controlnet_strength, scheduler_name, is_turbo):
    statistical_runtime.reset_collection()

    print("Starting image processing...")
    OVERLAP = 64
    tile_processor = TTP()
    condition_image = prepare_image(input_image, resolution, hdr)
    # condition_image = resize_and_upscale(condition_image, resolution)   # make sure 64  can diffusion
    input_image = condition_image
    W, H = condition_image.size
    # 3840, "height": 2304,
    if 1:
        # resolution = 3840
        TILE_SIZE = 1280
        tile_width = W // 2 + OVERLAP
        tile_height = H // 2 + OVERLAP
        bbox_list, weight, cols, rows = split_bboxes(W, H, tile_width, tile_height, OVERLAP)  # 不能均等切割~
    elif 0:  # no same tile 
        tile_width, tile_height = adaptive_tile_size((W, H), base_tile_size=1280, max_tile_size=1920)  # max_tile_size=1920
        bbox_list, weight, cols, rows = split_bboxes(W, H, tile_width, tile_height, OVERLAP)  # 不能均等切割~
    elif 0:
        # way || TTP tile v2
        #  ttplant tile v2: image 3840,2880
        height_scale = tile_rows
        width_scale = tile_cols
        tile_width, tile_height = tile_processor.image_width_height(input_image, width_scale, height_scale, OVERLAP)   #  
        bbox_list, (num_cols, num_rows) = tile_processor.tile_image(input_image, tile_width, tile_height)
        cols, rows = num_cols, num_rows
        # for tile_box in box_tiles:
        #     tile_im = image.crop(tile_box.box)
    print(f'>> tile_width, tile_height: {tile_width, tile_height}')

    for bbox in bbox_list:
        left, top, right, bottom = bbox.box
        tile_w, tile_h = right - left, bottom - top
        tile = input_image.crop((left, top, right, bottom))
        # bbox.prompt = generate_llm('describe this image in English', tile, model='qwen-vl-plus')
        res = clip_rogator.run(image=tile, prompt_mode='fast', image_analysis='off')
        bbox.prompt = res['result'][0][0]
        print(f'>> __tile__ {(left, top, right, bottom)} size {tile_w, tile_h} prompt:{len(bbox.prompt)}')

    result = np.zeros((H, W, 3), dtype=np.float32)
    result_list = []
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)
    # Create gaussian weight
    tile_width, tile_height = bbox_list[0].box[2:]
    gaussian_weight = create_gaussian_weight(max(tile_width, tile_height))

    for bbox in bbox_list:
        left, top, right, bottom = bbox.box
        tile = input_image.crop((left, top, right, bottom))
        # Adjust tile size if it's at the edge
        current_tile_size = (bottom - top, right - left)  # h, w

        print(f'>> process tile  coordinate {(left, top, right, bottom)} current_tile_size {tile.size} ')

        # Process the tile
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=dtype):
                if 0:
                    result_tile = lazy_pipe.sd3i2ipipe(
                        prompt=bbox.prompt,height=tile_height, width=tile_width, num_inference_steps=num_inference_steps, 
                        image=tile, strength=strength, guidance_scale=guidance_scale).images[0]
                else:
                    steps = steps
                    sigmas=np.linspace(1.0, 1 / steps, steps)
                    mult_sgma = MultiplySigmas()
                    sigmas=mult_sgma.simple_output(sigmas, sigma_scale, sigma_start, sigma_end)
                    args = {
                        "prompt": bbox.prompt, "resizer": 'crop_middle', "steps": steps, "n_samples": 1,
                        "tile_percent": 0, "enable_tile": False, "new_tile": False,   # 启用分块放大算法~
                        "width": tile_width, "height": tile_height,  "file": [tile], "image": 0,  "strength": strength, 'sigmas': sigmas,
                        "mask": None,  "scale": 3.5, "seed": [-1], "variant": 0.0, "variant_seed": [-1],
                    }
                    default_template.update(args)
                    normal_input = FluxNormalInput(**default_template)
                    normal_input.normal_width, normal_input.normal_height = normalize_size(
                            normal_input.width, normal_input.height, 4096)
                    result_pipe = f1pipe(normal_input)
                    result_tile = result_pipe['images'][0]
                    result_list.append(result_tile)
                                    
            # Apply gaussian weighting
            tile_weight = gaussian_weight[:current_tile_size[0], :current_tile_size[1]]  # h, w
            # Add the tile to the result with gaussian weighting
            result[top:bottom, left:right] += result_tile * tile_weight[:, :, np.newaxis]
            weight_sum[top:bottom, left:right] += tile_weight[:, :, np.newaxis]
    
    # Normal liner result WAY |
    debug_im = make_image_grid(
        result_list, cols=cols, rows=rows
    )
    debug_im.save('/home/dell/workspace/js-ai-svc/src/xflux/1_debug_tile_neeTTP_.png')
    result_list = [pil2tensor(im_).permute(0, 3, 1,2) for im_ in result_list]
    result_list = [
        result_list[i: i + cols] for i in range(0, len(result_list), cols)
    ]
    # 线性拼接~ overlap
    stitched_image_matrix_tensor = [
        concatenate_rowwise(elem, OVERLAP * 2) for elem in result_list
    ]
    stitched_image_matrix_tensor = concatenate_columnwise(stitched_image_matrix_tensor, OVERLAP * 2)
    stitched_image = tensor2pil(
        stitched_image_matrix_tensor.permute(0, 2, 3 ,1)
    )

    # Normalize the result WAY ||  较大的tile 存在overlap伪影~
    final_result = (result / weight_sum).astype(np.uint8)
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    output_pil = Image.fromarray(final_result)
    print("Image processing completed successfully")
    return gr.update(value = [input_image,stitched_image]), gr.update(value=stitched_image)


if __name__ == "__main__":
    title = """<h1 align="center">Tile Upscaler V3 flux.1 pp sd3 </h1>"""
    """
    <p align="center">Creative version of Tile Upscaler. The main ideas come from</p>
    <p><center>
    <a href="https://huggingface.co/spaces/gokaygokay/Tile-Upscaler" target="_blank">[Tile Upscaler]</a>
    <a href="https://github.com/philz1337x/clarity-upscaler" target="_blank">[philz1337x]</a>
    <a href="https://github.com/BatouResearch/controlnet-tile-upscale" target="_blank">[Pau-Lozano]</a>
    </center></p>
    """

    with gr.Blocks() as demo:
        gr.HTML(title)
        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(type="pil", label="Input Image")
                run_button = gr.Button("Enhance Image")
                with gr.Row():
                    resolution = gr.Slider(minimum=128, maximum=4096, value=3840, step=128, label="Resolution Mx len")
                    num_inference_steps = gr.Slider(minimum=1, maximum=50, value=16, step=1, label="Steps")
                with gr.Row():
                    strength = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.01, label="Strength")
                    hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        tile_rows = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="tile Rows")
                        tile_cols = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="tile Cols")
                    with gr.Row():
                        sigma_scale = gr.Slider(minimum=0.8, maximum=2.0, value=1, step=0.05, label="sigma-scale")
                        sigma_start = gr.Slider(minimum=0.0, maximum=0.8, value=0.1, step=0.05, label="sigma-start")
                        sigma_end = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="sigma-end")
                    with gr.Row():
                        is_turbo = gr.Checkbox(label='speed_turbo only 4 sd3', value=False)
                        guidance_scale = gr.Slider(minimum=0, maximum=20, value=6, step=0.5, label="Guidance Scale")
                    controlnet_strength = gr.Slider(minimum=0.0, maximum=2.0, value=0.75, step=0.05, label="ControlNet Strength only 4 sd1.5")
                    scheduler_name = gr.Dropdown(
                        choices=["DDIM", "DPM++ 3M SDE Karras", "DPM++ 3M Karras"],
                        value="DDIM",
                        label="Scheduler"
                    )

            with gr.Column(scale=3):
                output_image = gr.Image(label="After", type="numpy")
                output_slider = ImageSlider(label="Before / After", type="numpy")
                # with gr.Column():

        run_button.click(fn=gradio_process_image, 
                        inputs=[input_image, resolution, num_inference_steps, strength, hdr,
                                tile_rows, tile_cols, sigma_scale, sigma_start, sigma_end,
                             guidance_scale, controlnet_strength, scheduler_name, is_turbo],
                        outputs=[output_slider, output_image])

        # gr.Examples(
        #     examples=[
        #         [1536, 20, 0.4, 0, 6, 0.75, "DDIM"],
        #         [512, 20, 0.55, 0, 6, 0.6, "DDIM"],
        #         [1024, 20, 0.3, 0, 6, 0.65, "DDIM"]
        #     ],
        #     inputs=[resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name],
        #     outputs=[],
        #     fn=gradio_process_image,
        #     cache_examples=True,
        # )

    demo.launch(debug=False, share=False, server_name='0.0.0.0', server_port=5003)


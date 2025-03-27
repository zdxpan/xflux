# import spaces
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple, List, Union
from torch import Tensor
import time, os
import random
import math, cv2
import gradio as gr
from gradio_imageslider import ImageSlider
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.models import SD3Transformer2DModel


# from RealESRGAN import RealESRGAN
import sys
sys.path.append('/home/dell/workspace/js-sd-svc/src/stable-diffusion/')
from log import statistical_runtime
from modules.realesrgan import RealESRGAN
statistical_runtime.reset_collection()
from modules.llm import QwenClient, MinicpmClient

LLM_CONFIGS = {
    'qwen': { 'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1", 'api_key': os.getenv("DASHSCOPE_API_KEY") },
    # 'minicpm': { 'base_url': "http://prod.tair.a1-llm.xiaopiu.com//api/llava/generate", 'api_key': '' }
    'minicpm': { 'base_url': "http://192.168.1.4:5001/api/minicpm/generate", 'api_key': '' }        # ok pass    
}
LLM_MODELS = {
    'qwen': QwenClient,
    'minicpm': MinicpmClient
}

sd3_modelid = '/home/dell/models/stable-diffusion-3.5/'
sd3turbo_modelid = '/home/dell/models/stable-diffusion-3.5-large-turbo/'
dtype=torch.bfloat16
device = "cuda"


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

    return bbox_list, weight

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
        model_id = '/home/dell/models/realesrgan/weights/'
        if self.model is None:
            self.model = RealESRGAN(model_dir=model_id, device=device, scale=4 ,bg_tile=512, denoise_strength=0.5)
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
    img = input_image.resize((W, H), resample=Image.Resampling.LANCZOS)
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

# im = Image.open('/home/dell/workspace/js-sd-svc/img/lowq/02.png').resize(size=(512, 512))
# resize_and_upscale(im, 1024)

print('>> Image load done')
lazy_pipe = LazyLoadPipeline()
lazy_pipe.load()

print('>> lazy_pipe load done')

# @spaces.GPU
@timer_func
def gradio_process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name, is_turbo):
    statistical_runtime.reset_collection()

    print("Starting image processing...")
    # 参数设置（与原始问题一致）
    # im = Image.open('/home/dell/workspace/flux_f16/test/tuchong1500_1k/1812708067026796552.png')
    # condition_image = prepare_image(im, resolution, hdr)
    if is_turbo:
        lazy_pipe.transformer = lazy_pipe.transformer_turbo
    else:
        lazy_pipe.transformer = lazy_pipe.transformer

    condition_image = prepare_image(input_image, resolution, hdr)
    condition_image = resize_and_upscale(condition_image, resolution)   # make sure 64  can diffusion

    input_image = condition_image
    W, H = condition_image.size
    tile_width, tile_height = adaptive_tile_size((W, H), base_tile_size=1024, max_tile_size=1280)

    print('im', input_image.size, 'resolution', resolution, 'condition_image_size', condition_image.size, 'tile_w_H', (tile_width, tile_height))

    TILE_SIZE = 1280
    # tile_width = TILE_SIZE
    # tile_height = TILE_SIZE
    OVERLAP = 64
    bbox_list, weight = split_bboxes(W, H, tile_width, tile_height, OVERLAP)

    result = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)

    # Create gaussian weight
    gaussian_weight = create_gaussian_weight(max(tile_width, tile_height))

    for bbox in bbox_list:
        left, top, right, bottom = bbox.box
        
        tile = input_image.crop((left, top, right, bottom))
        if bbox.prompt != '':
            pass
        else:
            bbox.prompt = generate_llm('describe this image in English', tile, model='qwen-vl-plus')
        # Adjust tile size if it's at the edge
        current_tile_size = (bottom - top, right - left)
        
        # tile = tile.resize((tile_width, tile_height))
        print(f'>> process tile  coordinate {(left, top, right, bottom)} current_tile_size {tile.size} ')

        # Process the tile
        with torch.no_grad():  # 推理时不需要计算梯度
            with torch.autocast(device_type='cuda', dtype=dtype):
                # result_tile = process_tile(tile, num_inference_steps, strength, guidance_scale, controlnet_strength)
                # result_tile = sd3pipe(
                #     prompt="a photo of a a girl hold a cat , and the cat holding a sign that says hello world",
                #     negative_prompt="smooth, blur, digtal",
                #     num_inference_steps=6,
                #     height=tile_height,
                #     width=tile_width,
                #     guidance_scale=1.11,
                # ).images[0]
                result_tile = lazy_pipe.sd3i2ipipe(
                    prompt=bbox.prompt,height=tile_height, width=tile_width, num_inference_steps=num_inference_steps, 
                    image=tile, strength=strength, guidance_scale=guidance_scale).images[0]
                
            # Apply gaussian weighting
            tile_weight = gaussian_weight[:current_tile_size[0], :current_tile_size[1]]
            
            # Add the tile to the result with gaussian weighting
            result[top:bottom, left:right] += result_tile * tile_weight[:, :, np.newaxis]
            weight_sum[top:bottom, left:right] += tile_weight[:, :, np.newaxis]

    # Normalize the result
    final_result = (result / weight_sum).astype(np.uint8)
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    output_pil = Image.fromarray(final_result)
    print("Image processing completed successfully")


    return [input_image,output_pil]


if __name__ == "__main__":
    title = """<h1 align="center">Tile Upscaler V3 sd3</h1>
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
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                run_button = gr.Button("Enhance Image")
            with gr.Column():
                output_slider = ImageSlider(label="Before / After", type="numpy")
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                resolution = gr.Slider(minimum=128, maximum=4096, value=1280, step=128, label="Resolution")
                num_inference_steps = gr.Slider(minimum=1, maximum=50, value=8, step=1, label="Number of Inference Steps")
            with gr.Row():
                strength = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.01, label="Strength")
                hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
            with gr.Row():
                is_turbo = gr.Checkbox(label='speed_turbo', value=False)
                guidance_scale = gr.Slider(minimum=0, maximum=20, value=6, step=0.5, label="Guidance Scale")
            controlnet_strength = gr.Slider(minimum=0.0, maximum=2.0, value=0.75, step=0.05, label="ControlNet Strength")
            scheduler_name = gr.Dropdown(
                choices=["DDIM", "DPM++ 3M SDE Karras", "DPM++ 3M Karras"],
                value="DDIM",
                label="Scheduler"
            )

        run_button.click(fn=gradio_process_image, 
                        inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name, is_turbo],
                        outputs=output_slider)

    demo.launch(debug=False, share=False, server_name='0.0.0.0', server_port=5003)


# import spaces
import os
import time

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.models import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0

from PIL import Image
import cv2
import numpy as np

# from RealESRGAN import RealESRGAN
from modules.realesrgan import RealESRGAN
from log import statistical_runtime
statistical_runtime.reset_collection()

import random
import math

import gradio as gr
from gradio_imageslider import ImageSlider

from huggingface_hub import hf_hub_download
import torch


USE_TORCH_COMPILE = False
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.bfloat16

model_id = '/ssd/models/tile_upscale_v15/'
realesrgan_index = {"realesrgan": "/home/dell/models/realesrgan/weights",}

tile_upscale_v15 = model_id
# vae:       vae-ft-mse-840000-ema-pruned.safetensors   ~/models/stable-diffusion-v1-5/vae-ft-mse-840000-ema-pruned/diffusion_pytorch_model.fp16.safetensors 
model_index = {
    "model":  "juggernaut_reborn.safetensors",
    "CONTROLNET":  "control_v11f1e_sd15_tile.pth",
    "lora1": "SDXLrender_v2.0.safetensors", 
    "lora2": "more_details.safetensors", 
    "NEGATIVE_1": "JuggernautNegative-neg.pt",
    "NEGATIVE_2": "verybadimagenegative_v1.3.pt",
    "vae": "/home/dell/models/stable-diffusion-v1-5/vae-ft-mse-840000-ema-pruned/"
}
model_index = {k: tile_upscale_v15 + v  if 'vae' not in k else  v  for k, v in model_index.items() }

print('>> model_index', model_index)

def download_models():
    models = {
        # "MODEL": ("dantea1118/juggernaut_reborn", "juggernaut_reborn.safetensors", "models/models/Stable-diffusion"),
        # "UPSCALER_X2": ("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth", "models/upscalers/"),
        # "UPSCALER_X4": ("ai-forever/Real-ESRGAN", "RealESRGAN_x4.pth", "models/upscalers/"),
        "NEGATIVE_1": ("philz1337x/embeddings", "verybadimagenegative_v1.3.pt", "models/embeddings"),
        # "NEGATIVE_2": ("philz1337x/embeddings", "JuggernautNegative-neg.pt", "models/embeddings"),
        "LORA_1": ("philz1337x/loras", "SDXLrender_v2.0.safetensors", "models/Lora"),
        "LORA_2": ("philz1337x/loras", "more_details.safetensors", "models/Lora"),
        "CONTROLNET": ("lllyasviel/ControlNet-v1-1", "control_v11f1e_sd15_tile.pth", "models/ControlNet"),
        # "VAE": ("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors", "models/VAE"),
    }

    for model, (repo_id, filename, local_dir) in models.items():
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

# download_models()

def timer_func(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_scheduler(scheduler_name, config):
    if scheduler_name == "DDIM":
        return DDIMScheduler.from_config(config)
    elif scheduler_name == "DPM++ 3M SDE Karras":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
    elif scheduler_name == "DPM++ 3M Karras":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

class LazyLoadPipeline:
    def __init__(self):
        self.pipe = None

    @timer_func
    def load(self):
        if self.pipe is None:
            print("Starting to load the pipeline...")
            self.pipe = self.setup_pipeline()
            print(f"Moving pipeline to device: {device}")
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                self.pipe.to(device)
            if USE_TORCH_COMPILE:
                print("Compiling the model...")
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)

    @timer_func
    def setup_pipeline(self):
        print("Setting up the pipeline...")
        controlnet = ControlNetModel.from_single_file(
            # "models/ControlNet/control_v11f1e_sd15_tile.pth", torch_dtype=torch.float16
            model_index['CONTROLNET'], torch_dtype=dtype
        ).to(device)
        model_path = "models/models/Stable-diffusion/juggernaut_reborn.safetensors"
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            # model_path,
            model_index['model'],
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            model_index['vae'],
            use_safetensors=True, variant='fp16',
            torch_dtype=dtype
        ).to(device)
        vae.to(device)
        vae.enable_tiling()
        vae.enable_slicing()
        pipe.vae = vae
        pipe.load_textual_inversion(model_index['NEGATIVE_1'])   #  "models/embeddings/verybadimagenegative_v1.3.pt")
        pipe.load_textual_inversion(model_index['NEGATIVE_2'])   #  "models/embeddings/JuggernautNegative-neg.pt")
        pipe.load_lora_weights(model_index['lora1'])   #    "models/Lora/SDXLrender_v2.0.safetensors")
        pipe.fuse_lora(lora_scale=0.5)
        pipe.load_lora_weights(model_index['lora2'])  #    "models/Lora/more_details.safetensors")
        pipe.fuse_lora(lora_scale=1.)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        return pipe

    def set_scheduler(self, scheduler_name):
        if self.pipe is not None:
            self.pipe.scheduler = get_scheduler(scheduler_name, self.pipe.scheduler.config)

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
    img = img.resize(size = (int(W*2), int(H*2)))
    # if scale == 2:
    #     img = lazy_realesrgan_x2.predict(img)
    # else:
    #     img = lazy_realesrgan_x4.predict(img)
    return img

im = Image.open('/home/dell/workspace/js-sd-svc/img/lowq/02.png').resize(size=(512, 512))
resize_and_upscale(im, 1024)
# resize_and_upscale(im, 2048*2048)  #  占用极高的内存，到底是什么原因？  

print('>> Image load done')

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


lazy_pipe = LazyLoadPipeline()
lazy_pipe.load()

print('>> lazy_pipe load done')

# lazy_realesrgan_x2 = LazyRealESRGAN(device, scale=2)
# lazy_realesrgan_x4 = LazyRealESRGAN(device, scale=4)
print('>> lazy_realesrgan load done')


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

# @spaces.GPU
@timer_func
def gradio_process_image(input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name):
    print("Starting image processing...")
    # torch.cuda.empty_cache()
    statistical_runtime.reset_collection()
    lazy_pipe.set_scheduler(scheduler_name)
    
    # Convert input_image to numpy array
    input_array = np.array(input_image)
    
    # Prepare the condition image
    condition_image = prepare_image(input_image, resolution, hdr)
    W, H = condition_image.size
    
    # Adaptive tiling
    tile_width, tile_height = adaptive_tile_size((W, H))
    
    # Calculate the number of tiles
    overlap = min(64, tile_width // 8, tile_height // 8)  # Adaptive overlap
    num_tiles_x = math.ceil((W - overlap) / (tile_width - overlap))
    num_tiles_y = math.ceil((H - overlap) / (tile_height - overlap))
    
    # Create a blank canvas for the result
    result = np.zeros((H, W, 3), dtype=np.float32)
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)
    
    # Create gaussian weight
    gaussian_weight = create_gaussian_weight(max(tile_width, tile_height))
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Calculate tile coordinates
            left = j * (tile_width - overlap)
            top = i * (tile_height - overlap)
            right = min(left + tile_width, W)
            bottom = min(top + tile_height, H)
            
            # Adjust tile size if it's at the edge
            current_tile_size = (bottom - top, right - left)
            
            tile = condition_image.crop((left, top, right, bottom))
            tile = tile.resize((tile_width, tile_height))
            
            # Process the tile
            with torch.no_grad():  # 推理时不需要计算梯度
                with torch.autocast(device_type='cuda', dtype=dtype):
                    result_tile = process_tile(tile, num_inference_steps, strength, guidance_scale, controlnet_strength)
            
            # Apply gaussian weighting
            if current_tile_size != (tile_width, tile_height):
                result_tile = cv2.resize(result_tile, current_tile_size[::-1])
                tile_weight = cv2.resize(gaussian_weight, current_tile_size[::-1])
            else:
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
    
    return [input_image, output_pil]

if __name__ == "__main__":
    title = """<h1 align="center">Tile Upscaler V2</h1>
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
            resolution = gr.Slider(minimum=128, maximum=4096, value=1536, step=128, label="Resolution")
            num_inference_steps = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Number of Inference Steps")
            strength = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.01, label="Strength")
            hdr = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="HDR Effect")
            guidance_scale = gr.Slider(minimum=0, maximum=20, value=6, step=0.5, label="Guidance Scale")
            controlnet_strength = gr.Slider(minimum=0.0, maximum=2.0, value=0.75, step=0.05, label="ControlNet Strength")
            scheduler_name = gr.Dropdown(
                choices=["DDIM", "DPM++ 3M SDE Karras", "DPM++ 3M Karras"],
                value="DDIM",
                label="Scheduler"
            )

        run_button.click(fn=gradio_process_image, 
                        inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name],
                        outputs=output_slider)

        # gr.Examples(
        #     examples=[
        #         ["image1.jpg", 1536, 20, 0.4, 0, 6, 0.75, "DDIM"],
        #         ["image2.png", 512, 20, 0.55, 0, 6, 0.6, "DDIM"],
        #         ["image3.png", 1024, 20, 0.3, 0, 6, 0.65, "DDIM"]
        #     ],
        #     inputs=[input_image, resolution, num_inference_steps, strength, hdr, guidance_scale, controlnet_strength, scheduler_name],
        #     outputs=output_slider,
        #     fn=gradio_process_image,
        #     cache_examples=True,
        # )

    demo.launch(debug=False, share=False, server_name='0.0.0.0', server_port=5003)


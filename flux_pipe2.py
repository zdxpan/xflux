
from log import logger, statistical_runtime
# from services.input_checker import FluxNormalInput
statistical_runtime.reset_collection()

import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import inspect
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from argparse import Namespace
from text_encoder import PromptEncoder


import diffusers
import torch
from torch import nn


# from transformers      import CLIPTextModel, T5EncoderModel
# from transformers                 import CLIPTokenizer, T5Tokenizer
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, PreTrainedModel
from diffusers.image_processor import VaeImageProcessor
from diffusers import FluxControlNetModel, FluxTransformer2DModel
from tile_vae_processer import AutoencoderKL
from realesrgan_model import RealESRGAN
from transformer_model import QuantizedFluxTransformer2DModel, load_origin_parameters, zero_parameters, load_state_dict,load_gguf_q8_0_quantized_parameters
from attention_processor import FluxAttnProcessor2_0
# from dl_models.flux.tensor_loader.dev_tensor_loader import FluxDevGenerateTensorLoader

from tensor_util import (
    pack_latents, unpack_latents, prepare_image_ids, 
    concatenate_rowwise, concatenate_columnwise,
    create_np_mask, create_random_tensors, normalize_size,slerp,has_transparency,
    pack_latents, unpack_latents, prepare_image_ids, calculate_shift, resize_image,
    resize_numpy_image_long, seed_to_int
)
from clip_inter_rogator import ClipInterrogator
# tile _ttp
from ttp_tile import TTP, split_4bboxes, split_bboxes


FLUX_1DEV_MODEL_WEIGHTS = {
    "gguf_8steps_q8_0": "flux.1dev_fuse_8steps_lora_gguf.safetensors",
    "gguf_q8_0": "flux1.dev_q80_gguf.safetensors",
    # "nf4": "diffusion_pytorch_model.hyper8.nf4.safetensors",
    'origin_dev': 'diffusion_pytorch_model.safetensors'
}
clip_vit_cache_path = '/data/comfy_model/clip_interrogator/'
blip_image_caption_path = '/home/dell/comfy_model/blip_image_caption'


@dataclass
class Control:
    index: list[int]
    annotator: str
    name: str
    scale: float
    range: float
    image: Image.Image    
    conds: list[torch.Tensor] = field(default_factory=list)

@dataclass
class FluxNormalInput:
    '''
    flux模型全部输入参数的定义
    '''
    task: str
    trace_id: str = field(default='')
    req_id: str = field(default='')
    family: str = field(default='flux.1dev')
    sampler: str = field(default='flow_euler')
    base_shift: float = field(default=0.5)
    max_shift: float = field(default=1.15)
    # detial sigmas
    sigmas: list[float] = field(default_factory=lambda: None)
    # detail_scale: float = field(default=1.15)
    # detail_start
    file: list[Image.Image | str | Path] = field(default_factory=list)
    annotate: list[tuple[int, str]] = field(default_factory=list)
    init_model: str = field(default='gguf_8steps_q8_0')
    use_scaled: bool = field(default=True)
    image: int | None = field(default=None)
    mask: int | None = field(default=None)
    enable_tile: bool = field(default=False)
    new_tile: bool = field(default=False)
    tile_percent: float = field(default=0.4)
    mixin: list[tuple[str | Path, int]] = field(default_factory=list)
    control: list[tuple[list[int], str, str, float, float]] = field(default_factory=list)
    width: int = field(default=1024)
    height: int = field(default=1024)
    n_samples: int = field(default=1)
    clip_skip: int = field(default=2)
    resizer: str = field(default='crop_middle')
    steps: int = field(default=8)
    strength: float = field(default=0.75)
    scale: float = field(default=3.5)
    prompt: str = field(default='')
    mask_blur: float = field(default=0.1)
    output_name: str | None = field(default=None)
    output_dir: str | None = field(default=None)
    image_output_encoding: str = field(default='base64')
    image_save_path: list[str] = field(default_factory=lambda: [""])
    image_output_format: list[str] = field(default_factory=lambda: ["png"])
    seed: list[int] = field(default_factory=lambda: [-1])
    variant_seed: list[int] = field(default_factory=lambda: [-1])
    variant: float = field(default=0)
    # 来自text encoder的输入参数，由text encoder处理好,此处仅用来定义
    pooled_prompt_embeds: torch.Tensor = field(default=None)
    prompt_embeds: torch.Tensor = field(default=None)
    text_ids: torch.Tensor = field(default=None)
    normal_width: int = field(default=1024)
    normal_height: int = field(default=1024)


@dataclass
class FluxDevGeneratePreparedTensors:
    image_ids: torch.Tensor = field(default=None)
    init_noise: torch.Tensor = field(default=None)
    timesteps: torch.Tensor = field(default=None)
    init_latent: torch.Tensor = field(default=None)
    image_latents: torch.Tensor = field(default=None)
    latent_mask: torch.Tensor = field(default=None)
    images: list[Image.Image] = field(default_factory=list)
    ip_pulid_controls: list[Control] = field(default_factory=list)
    ip_adapter_controls: list[Control] = field(default_factory=list)
    xlabel_controls: list[Control] = field(default_factory=list)


class FluxDevGenerateTensorLoader:
    '''
    flux 负责flux服务的预处理与输入
    ps: 推理过程中创建的临时张量由推理pipeline自己创建
    '''
    FieldProcessor = namedtuple('FieldProcessor', ['method', 'dependence'])

    def __init__(self, device, dtype, scheduler, vae, pulid_model, esrgan, clip_image_processor, image_encoder,
                 image_processor, annotator):
        self.device = device
        self.dtype = dtype
        self.scheduler = scheduler
        self.vae = vae
        self.pulid_model = pulid_model
        self.esrgan = esrgan
        self.clip_image_processor = clip_image_processor
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.annotator = annotator
        self.factor = 2 ** (self.vae.config.out_channels)
        self.channels = self.vae.config.latent_channels

    @torch.no_grad()
    def prepare(self, opt: FluxNormalInput) -> FluxDevGeneratePreparedTensors:
        result = {}
        field_processors = (
            ('images', self.FieldProcessor(prepare_images, dependence=None)),
            ('image_ids', self.FieldProcessor(prepare_input_image_ids, dependence=None)),
            ('init_noise', self.FieldProcessor(prepare_init_noise, dependence=None)),
            ('timesteps', self.FieldProcessor(prepare_timesteps, dependence=None)),
            ('latent_mask', self.FieldProcessor(prepare_latent_mask, dependence=['images'])),
            ('image_latents', self.FieldProcessor(prepare_image_latent, dependence=['images'])),
            ('init_latent',
             self.FieldProcessor(prepare_init_latent, dependence=['timesteps', 'image_latents', 'init_noise'])),
            ('ip_pulid_controls', self.FieldProcessor(prepare_ip_pulid, dependence=['timesteps'])),
            ('ip_adapter_controls', self.FieldProcessor(prepare_ip_adapters, dependence=None)),
            ('xlabel_controls', self.FieldProcessor(prepare_xlabel_controls, dependence=['timesteps'])),
        )
        for idx, (key_name, (field_processor, dependence)) in enumerate(field_processors):
            # 有些方法依赖前面处理好的数据
            if dependence:
                self.check_dependence(field_processors, idx)
                params = [opt]
                for key in dependence:
                    v = result.get(key)
                    params.append(v)
                v = field_processor(self, *params)
            else:
                v = field_processor(self, opt)
            result[key_name] = v
        return FluxDevGeneratePreparedTensors(**result)

    def check_dependence(self, field_processors, idx):
        current_key_name, (_, current_dependence) = field_processors[idx]
        if not current_dependence:
            return
        # 检测依赖，当前依赖的参数值不能在后面才定义，防止程序出现问题
        for key_name, _ in field_processors[idx:]:
            for item in current_dependence:
                assert item != key_name, f'{key_name} must be defined before {current_key_name}'

    def encode_mask(self, mask, blur=0.0, channels=16, factor=8, batch_size=1):
        return nn.functional.interpolate(
            create_np_mask(mask, blur).to(self.device, dtype=self.dtype),
            scale_factor=(1 / factor),
        ).repeat(batch_size, channels, 1, 1)

    def prepare_image(
            self,
            image,
            width,
            height,
            batch_size,
            device,
            dtype,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image = image.repeat_interleave(batch_size, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image


def prepare_images(tensor_loader: FluxDevGenerateTensorLoader, opt):
    images = [
        resize_image(image, opt.width, opt.height, tensor_loader.esrgan, opt.resizer)
        for image in opt.file
    ]
    return images


def prepare_input_image_ids(tensor_loader: FluxDevGenerateTensorLoader, opt):
    image_ids = prepare_image_ids(
        opt.normal_height // tensor_loader.factor, opt.normal_width // tensor_loader.factor, opt.n_samples
    ).to(tensor_loader.device, dtype=tensor_loader.dtype)
    return image_ids


def prepare_init_noise(tensor_loader: FluxDevGenerateTensorLoader, opt):
    base_x = create_random_tensors(
        [tensor_loader.channels, opt.normal_height // tensor_loader.factor, opt.normal_width // tensor_loader.factor],
        opt.seed
    )
    if opt.variant == 0.0:
        init_noise = base_x
    else:
        variant_x = create_random_tensors(
            [tensor_loader.channels, opt.normal_height // tensor_loader.factor,
             opt.normal_width // tensor_loader.factor], opt.variant_seed
        )
        init_noise = slerp(max(0.0, min(1.0, opt.variant)), base_x, variant_x)
    init_noise = init_noise.to(tensor_loader.device, dtype=tensor_loader.dtype)
    return init_noise


def prepare_timesteps(tensor_loader: FluxDevGenerateTensorLoader, opt):
    sche_params = set(
        inspect.signature(tensor_loader.scheduler.set_timesteps).parameters.keys()
    )
    sigmas = opt.sigmas if opt.sigmas is not None else np.linspace(1.0, 1 / opt.steps, opt.steps)
        
    if "sigmas" in sche_params and "mu" in sche_params:
        tensor_loader.scheduler.set_timesteps(
            sigmas=sigmas,
            mu=calculate_shift(
                (min(opt.normal_height, 1024) // tensor_loader.factor // 2) * (
                            min(opt.normal_width, 1024) // tensor_loader.factor // 2),
                tensor_loader.scheduler.config.base_image_seq_len,
                tensor_loader.scheduler.config.max_image_seq_len,
                tensor_loader.scheduler.config.base_shift,
                tensor_loader.scheduler.config.max_shift,
            ),
            device=tensor_loader.device,
        )
    elif "sigmas" in sche_params:
        tensor_loader.scheduler.set_timesteps(
            sigmas=sigmas, device=tensor_loader.device
        )
    else:
        tensor_loader.scheduler.set_timesteps(opt.steps, device=tensor_loader.device)

    timesteps = tensor_loader.scheduler.timesteps
    # logger.info(f"init timesteps:{timesteps.tolist()}")
    num_inference_steps = opt.steps
    strength = 1.0 if opt.image is None else opt.strength
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    timesteps = tensor_loader.scheduler.timesteps[t_start * tensor_loader.scheduler.order:]
    # logger.info(f"curr timesteps:{timesteps.tolist()}")
    return timesteps


def prepare_latent_mask(tensor_loader: FluxDevGenerateTensorLoader, opt, images):
    alpha_mask = None
    for item in [opt.image]:
        if item is not None and has_transparency(images[item]):
            mask = tensor_loader.encode_mask(
                images[item], 0, tensor_loader.channels, tensor_loader.factor, opt.n_samples
            )
            alpha_mask = mask if alpha_mask is None else alpha_mask * mask
    if alpha_mask is not None:
        alpha_mask = pack_latents(alpha_mask)
        logger.debug("Use alpha mask")

    latent_mask = None
    if opt.mask is not None:
        latent_mask = tensor_loader.encode_mask(
            images[opt.mask], opt.mask_blur, tensor_loader.channels, tensor_loader.factor, opt.n_samples
        )
        latent_mask = pack_latents(latent_mask)
        if alpha_mask is not None:
            latent_mask *= alpha_mask
    return latent_mask


def prepare_image_latent(tensor_loader: FluxDevGenerateTensorLoader, opt, images):
    if opt.image is not None:
        image_latents = tensor_loader.vae.encode_image(images[opt.image], opt.n_samples)
        return image_latents


def prepare_init_latent(tensor_loader: FluxDevGenerateTensorLoader, opt, timesteps, image_latents, init_noise):
    if image_latents is not None:
        latent_timestep = timesteps[:1].repeat(opt.n_samples)
        init_latent = tensor_loader.scheduler.scale_noise(
            image_latents, latent_timestep, init_noise
        ) if len(timesteps) > 0 else image_latents
    else:
        init_latent = init_noise.clone()
    init_latent = pack_latents(init_latent)
    return init_latent


def prepare_ip_pulid(tensor_loader: FluxDevGenerateTensorLoader, opt, timesteps):
    ip_controls = [ct for ct in opt.control if ct[2].startswith("ip_pulid")]

    controls = []
    for index, [
        controlnet_cond_inx,
        annotator,
        controlnet_name,
        controlnet_scale,
        controlnet_range,
    ] in enumerate(ip_controls):
        controlnet_conds = [opt.file[idx] for idx in controlnet_cond_inx]
        image_conds = []
        if controlnet_name == "ip_pulid":
            for controlnet_cond in controlnet_conds:
                id_image = np.array(controlnet_cond.convert("RGB"))
                id_image = resize_numpy_image_long(id_image, 1024)
                id_embedding, _ = tensor_loader.pulid_model.get_id_embedding(
                    id_image, cal_uncond=False
                )
                image_conds.append(
                    id_embedding.repeat_interleave(opt.n_samples, dim=0)
                )
        ip_range = int((1 - controlnet_range) * len(timesteps)) if controlnet_name == "ip_pulid" else int(
            controlnet_range * len(timesteps))
        controls.append(
            Control(
                index=controlnet_cond_inx,
                name=controlnet_name,
                annotator=annotator,
                range=ip_range,
                scale=controlnet_scale,
                conds=image_conds
            )
        )
    return controls


def prepare_ip_adapters(tensor_loader: FluxDevGenerateTensorLoader, opt):
    ip_adapters = [ct for ct in opt.control if ct[2].startswith("ip_adapter")]
    controls = []
    for index, [
        ip_adapter_cond_inx,
        annotator,
        ip_adapter_name,
        ip_adapter_scale,
        ip_adapter_range,
    ] in enumerate(ip_adapters):
        conds = []
        ip_adapter_cond = opt.file[ip_adapter_cond_inx[0]].convert("RGB")
        ip_adapter_cond = tensor_loader.clip_image_processor(
            images=ip_adapter_cond, return_tensors="pt"
        ).pixel_values
        ip_adapter_cond = ip_adapter_cond.to(tensor_loader.device, tensor_loader.image_encoder.dtype)
        image_embed = tensor_loader.image_encoder(ip_adapter_cond).image_embeds.to(
            device=tensor_loader.device, dtype=tensor_loader.dtype
        )
        conds.append(
            image_embed.repeat_interleave(opt.n_samples, dim=0)
        )
        controls.append(
            Control(
                index=index,
                name=ip_adapter_name,
                annotator=annotator,
                range=ip_adapter_range,
                scale=ip_adapter_scale,
                conds=conds
            ),
        )
    return controls


def prepare_xlabel_controls(tensor_loader: FluxDevGenerateTensorLoader, opt, timesteps):
    xlabs_controls = [
        ct for ct in opt.control if ct[2].startswith(("canny", "depth", "upscale"))
    ]
    controls = []
    for index, [
        controlnet_cond_inx,
        annotator,
        controlnet_name,
        controlnet_scale,
        controlnet_range,
    ] in enumerate(xlabs_controls):
        conds = []
        controlnet_cond = opt.file[controlnet_cond_inx[0]]
        controlnet_cond = resize_image(
            controlnet_cond, opt.width, opt.height, None, opt.resizer
        )
        cond_image = copy.deepcopy(controlnet_cond)
        controlnet_cond = tensor_loader.annotator.annotate(
            annotator, controlnet_cond
        ).convert("RGB")
        controlnet_cond = tensor_loader.prepare_image(
            image=controlnet_cond,
            width=opt.normal_width,
            height=opt.normal_height,
            batch_size=opt.n_samples,
            device=tensor_loader.device,
            dtype=tensor_loader.dtype,
        )
        conds.append(controlnet_cond)
        controls.append(
            Control(
                index=index,
                name=controlnet_name,
                annotator=annotator,
                range=int(controlnet_range * len(timesteps)),
                scale=controlnet_scale,
                image=cond_image,
                conds=conds,
            )
        )
    return controls


class FluxPipeline:

    @torch.no_grad()
    def __init__(
            self,
            base,
            device,
            dtype,
            esrgan: RealESRGAN,
            controlnet,
            encoder_hid_proj,
            prompt_encoder,
            annotator
        ):
        self.base = base
        self.device = device
        self.dtype = dtype
        self.annotator = annotator
        self.esrgan = esrgan
        self.controlnet = controlnet
        self.encoder_hid_proj = encoder_hid_proj
        self.prompt_encoder = prompt_encoder
        self.clip_image_processor = CLIPImageProcessor()
        self._initiated = False
        self.is_cleared = False
        self.default_args = {
            "prompt": "",
            "resizer": 'crop_middle',
            "steps": 20,
            "n_samples": 1,
            "width": 1024,
            "height": 1024,
            "file": [],
            "image": None,
            "mask": None,
            "strength": 0.75,
            "scale": 3.5,
            "seed": [-1],
            "variant": 0.0,
            "variant_seed": [-1],
        }

    def initiate(self):
        if self._initiated:
            return
        logger.debug(f'Initiating FLUX on the first request...')

        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.base,
            subfolder='scheduler'
        )

        if not self.prompt_encoder._initiated:
            self.prompt_encoder.initiate()
        self.clip_rogator = ClipInterrogator()
        self.clip_rogator.load_model(clip_vit_cache_path, blip_image_caption_path)  # 1.9G

        #     self.tokenizer_1 = self.prompt_encoder.clip_tokenizer
        #     self.tokenizer_2 = self.prompt_encoder.t5_tokenizer
        #     self.text_encoder_1    = self.prompt_encoder.clip_encoder
        #     self.text_encoder_2    = self.prompt_encoder.t5_encoder
        # else:
            # self.text_encoder_1 = CLIPTextModel.from_pretrained(
            #     self.base,subfolder='text_encoder',torch_dtype=self.dtype).to(self.device)
            # self.text_encoder_2 = T5EncoderModel.from_pretrained(self.base,subfolder='text_encoder_2_full_weight',
            #     torch_dtype=self.dtype).to(self.device)
            # self.tokenizer_1 = CLIPTokenizer.from_pretrained(self.base,
            #     subfolder='tokenizer',torch_dtype=self.dtype)
            # self.tokenizer_2 = T5Tokenizer.from_pretrained(self.base,subfolder='tokenizer_2',
            #     torch_dtype=self.dtype)

        self.vae = AutoencoderKL.from_pretrained(
            self.base, subfolder="vae", torch_dtype=self.dtype
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels))
            if hasattr(self, "vae") and self.vae is not None
            else 16
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.vae_scale_factor)
        # Apply optimization
        logger.debug('Enbalbe VAE tiling and slicing')
        self.vae.enable_tiling()
        # logger.debug('Enbalbe VAE slicing')
        self.vae.enable_slicing()
        # prompt encoder initiate
        self.tensor_loader = FluxDevGenerateTensorLoader(
            self.device,
            self.dtype,
            self.scheduler,
            self.vae,
            None,
            self.esrgan,
            self.clip_image_processor,
            None,
            self.image_processor,
            self.annotator
        )

        self.transformer = QuantizedFluxTransformer2DModel.from_pretrained(
            # '/ssd/flux1-dev-transformer',
            self.base,
            subfolder='transformer',
            init_model = "origin_dev",
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.models = [self.transformer, self.vae]

        for model in self.models:
            model.requires_grad_ = False

        self._initiated = True

    @statistical_runtime("transformer forward")
    @torch.no_grad()
    def __call__(self, opt: FluxNormalInput, prepared_tensors=None):
        statistical_runtime.reset_collection()
        start_time = time.time()

        # try to use cllip to invert image2image 
        if 0:
            from PIL import  Image
            im = Image.open('/home/dell/workspace/img/girl_seaside.png')
            res = self.clip_rogator.run(image=im, prompt_mode='fast', image_analysis='off')
            res = res['prompt_result']

        opt.seed = seed_to_int(opt.seed, opt.n_samples)
        self.prepare_models(init_model='origin_dev', mixin=opt.mixin)

        if prepared_tensors is None:
            FluxDevGenerateTensorLoader.prepare
            prepared_tensors = self.tensor_loader.prepare(opt)

        # 1. Prepare arguments
        width, height = opt.normal_width, opt.normal_height
        factor = 2 ** (self.vae.config.out_channels)
        channels = self.vae.config.latent_channels

        # tile
        enable_tile = opt.enable_tile
        enable_new_tile = opt.new_tile
        tile_percent = opt.tile_percent
        overlap = 64 // factor
        # for ttp tile lego~
        width_scale = 3
        height_scale = 2
        tile_processor = TTP()
        images = prepared_tensors.images
        image_index = opt.image
        tile_images = []
        if image_index is not None and enable_new_tile:
            image = images[image_index]
            #  ttplant tile v2: image 3840,2880
            # tile_width, tile_height = tile_processor.image_width_height(image, width_scale, height_scale, 0.1)   #  
            # new_tiles, (num_cols, num_rows) = tile_processor.tile_image(image, tile_width, tile_height, factor=factor)
            # for tile_box in new_tiles:
            #     tile_im = image.crop(tile_box.box)
            #     tile_box.prompt = self.clip_rogator.run(image=tile_im, prompt_mode='fast', image_analysis='off')
            # don`t know how to process with the overlap. seemed each crop apply each`s condition 
        
            # use tile diffusion v1: make tile  get eache tile_prompt
            width, height = image.size
            tile_images, tiles_weight = split_4bboxes(width, height, overlap=64, device=self.device, factor=factor)
            for tile_box in tile_images:
                tile_im = image.crop(tile_box.box)
                tile_box.prompt = self.clip_rogator.run(image=tile_im, prompt_mode='fast', image_analysis='off')['result'][0][0]
            prompt_all = opt.prompt + '##' + '##'.join([tile.prompt for tile in tile_images])
            opt.prompt = prompt_all if '##' not in opt.prompt else opt.prompt

        # 2. Encode prompts
        newopt = Namespace(
            prompt = opt.prompt, mixin = [], n_samples = 1, family = 'flux1.dev',
            enable_tile = enable_tile, new_tile = enable_new_tile, 
        )
        PromptEncoder.__call__
        text_embedding_dc = self.prompt_encoder(newopt)   # about 5 sec
        # for k,v in text_embedding_dc.items():
        #     print(f'{k}: {v.shape}')
        self.prompt_encoder.clear_gpu_model()
        prompt_embeds, pooled_prompt_embeds, text_ids = (
            text_embedding_dc['prompt_embeds'].to(self.device),
            text_embedding_dc['pooled_prompt_embeds'].to(self.device),
            text_embedding_dc['text_ids'].to(self.device),
        )
        # prompt_embeds, pooled_prompt_embeds, text_ids = (
        #     opt.prompt_embeds.to(self.device),
        #     opt.pooled_prompt_embeds.to(self.device),
        #     opt.text_ids.to(self.device),
        # )
        # put embedding to each tile
        if len(tile_images) > 0:
            for inx_, tile_box in enumerate(tile_images):
                # transformer_args['encoder_hidden_states'] = prompt_embeds[-tile_batch:][inx_]
                # transformer_args['pooled_projections'] = pooled_prompt_embeds[-tile_batch:][inx_]
                tile_box.prompt_embeds = prompt_embeds[inx_ + 1]
                tile_box.pooled_prompt_embeds = pooled_prompt_embeds[inx_ + 1]
                tile_box.text_ids = text_ids[inx_ + 1]

        # 3. Prepare image ids
        image_ids = prepared_tensors.image_ids

        # 4. Prepare init noise
        init_noise = prepared_tensors.init_noise

        # 5. Prepare timesteps
        timesteps = prepared_tensors.timesteps

        # 6. Prepare init image and mask
        latent_mask = prepared_tensors.latent_mask
        image_latents = prepared_tensors.image_latents
        init_latent = prepared_tensors.init_latent
        # 7. prepare control
        # ip-pulid
        ip_controls = prepared_tensors.ip_pulid_controls

        ip_names = []
        ip_embeddings = []
        ip_scales = []
        ip_ranges = []
        for control in ip_controls:
            ip_names.append(control.name)
            ip_embeddings.extend(control.conds)
            ip_scales.append(control.scale)
            ip_ranges.append(control.range)

        # ip-adapter
        ip_adapters = prepared_tensors.ip_adapter_controls
        ip_adapter_image_embeds = []
        for control in ip_adapters:
            ip_adapter_image_embeds.extend(control.conds)
        # encode image
        ip_projected_image_embeds = self.encoder_hid_proj(
            ip_adapter_image_embeds) if self.encoder_hid_proj is not None else None

        # xlabs controlnet
        controlnet_conds = []
        controlnet_scales = []
        controlnet_ranges = []
        controlnet_images = []
        xlabs_controls = prepared_tensors.xlabel_controls
        for control in xlabs_controls:
            controlnet_conds.extend(control.conds)
            controlnet_scales.append(control.scale)
            controlnet_ranges.append(control.range)
            controlnet_images.append(control.image)


        # 8. Denoising loop
        loop_start_time = time.time()
        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([opt.scale], device=self.device)
            guidance = guidance.expand(opt.n_samples)
        else:
            guidance = None

        # prepare for controlnet upscaler
        control_image = None
        if hasattr(self.controlnet, 'input_hint_block') and self.controlnet.input_hint_block is None:
            # vae encode
            control_image = self.vae.encode(controlnet_conds[0]).latent_dist.sample()
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            # pack
            control_image = pack_latents(control_image)
            self.controlnet.to(self.device)

        latent = init_latent
        with tqdm(total=len(timesteps), disable=None) as pbar:
            for i, t in enumerate(timesteps):
                # ip-pulid
                tmp_ip_scales = ip_scales.copy()
                for idx, ip_range in enumerate(ip_ranges):
                    if i < ip_range:
                        tmp_ip_scales[idx] = 0.0

                ip_params = {
                    "ip_names": ip_names,
                    "ip_embeddings": ip_embeddings,
                    "ip_scales": tmp_ip_scales,
                }

                # xlabs controlnet
                control_mode = None
                cnet_args = {'hidden_states':latent, 'controlnet_cond': control_image, 'controlnet_mode':control_mode,'conditioning_scale':controlnet_scales[0] if len(controlnet_scales) > 0 else 1.0,
                                #  'idx':i, 'controlnet_range':controlnet_ranges,
                             'return_dict':False,
                             'encoder_hidden_states':prompt_embeds,'pooled_projections':pooled_prompt_embeds,
                             'timestep':t.expand(latent.shape[0]).to(latent.dtype) / 1000,'img_ids':image_ids,'txt_ids':text_ids,'guidance':guidance,'joint_attention_kwargs':None}
                use_cnet = self.controlnet is not None and len(controlnet_ranges) > 0 and i <= controlnet_ranges[0]
                controlnet_blocks_repeat = False if hasattr(self.controlnet, 'input_hint_block') and self.controlnet.input_hint_block is None else True

                controlnet_block_samples, controlnet_single_block_samples = self.controlnet(**cnet_args) if use_cnet else (None, None)
                pulid_ca = None
                # tile
                transformer_args = {'hidden_states': latent, 'timestep': t.expand(latent.shape[0]).to(latent.dtype) / 1000,
                                'guidance': guidance, 'pooled_projections': pooled_prompt_embeds[0],
                                'encoder_hidden_states': prompt_embeds[0], 'txt_ids': text_ids[0], 
                                'img_ids':image_ids, 'ip_params':ip_params, 'joint_attention_kwargs': None, 
                                'controlnet_block_samples': controlnet_block_samples, 
                                'controlnet_single_block_samples': controlnet_single_block_samples,
                                'return_dict': False, 'controlnet_blocks_repeat': controlnet_blocks_repeat, 
                                'pulid_ca': pulid_ca, 'ip_projected_image_embeds': ip_projected_image_embeds,
                                }
                if enable_tile and len(tile_images) > 0:
                    tiles_weight = tiles_weight.to(latent.dtype)
                    x_latent = unpack_latents(                 #  torch.Size([1, 43200, 64])
                        latent, height // factor, width // factor, channels
                    )  #  down sample 8 scale  [1, 16, 360, 480]
                    x_noise = torch.zeros_like(x_latent)
                    noises = []
                    tile_batch = 4
                    k_col = 2
                    _, _, h, w = x_latent.shape
                    for inx_, tile in enumerate(tile_images):
                        image_ids = prepare_image_ids(
                            tile.h // factor,  tile.w // factor, 1,
                        ).to(self.device, dtype=self.dtype)
                        # self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)
                        sub_latent = x_latent[tile.latent_slicer]       #  [1, 16, 188, 248] -> final: torch.Size([1, 16, 188, 248])
                        
                        sub_latent = pack_latents(torch.cat([sub_latent], dim=0))   # torch.Size([1, 11656, 64])
                        tile_guidance = guidance.expand(sub_latent.shape[0])
                        transformer_args['hidden_states'] = sub_latent
                        transformer_args['guidance'] = guidance
                        transformer_args['img_ids'] = image_ids
                        transformer_args['timestep'] = t.expand(sub_latent.shape[0]).to(latent.dtype) / 1000
                        transformer_args['encoder_hidden_states'] = tile.prompt_embeds # prompt_embeds[-tile_batch:][inx_] !shape       [512, 4096]
                        transformer_args['pooled_projections'] =  tile.pooled_prompt_embeds # pooled_prompt_embeds[-tile_batch:][inx_]  [768]
                        noise = self.transformer(**transformer_args)[0]        # [1, 11656, 64])
                        noise = unpack_latents(noise, tile.h // factor, tile.w // factor, channels)
                        noises.append(noise)
                        x_noise[tile.latent_slicer] += noise * tiles_weight[tile.latent_slicer]
                    # x_noise = x_noise / tiles_weight
                    noises = [
                        noises[i: i + k_col] for i in range(0, len(noises), k_col)
                    ]

                    noises = [
                        concatenate_rowwise(elem, overlap * 2) for elem in noises
                    ]
                    x_noise = concatenate_columnwise(noises, overlap * 2)

                    x_noise = pack_latents(x_noise)

                elif 0:
                # if enable_tile:
                    if i > len(timesteps) * tile_percent:
                        image_ids = prepare_image_ids(
                            height // factor // 2 + 64 // factor,
                            width // factor // 2 + 64 // factor,
                            1,
                        ).to(self.device, dtype=self.dtype)

                        x_latent = unpack_latents(
                            latent, height // factor, width // factor, channels
                        )
                        x_noise = torch.zeros_like(x_latent)
                        _, _, h, w = x_latent.shape

                        latent0 = x_latent[
                                  :, :, 0: h // 2 + overlap, 0: w // 2 + overlap
                                  ]
                        latent1 = x_latent[
                                  :, :, 0: h // 2 + overlap, w // 2 - overlap:
                                  ]
                        latent2 = x_latent[
                                  :, :, h // 2 - overlap:, 0: w // 2 + overlap
                                  ]
                        latent3 = x_latent[:, :, h // 2 - overlap:, w // 2 - overlap:]
                        x_latent = torch.cat(
                            [latent0, latent1, latent2, latent3], dim=0
                        )
                        x_latent = pack_latents(x_latent)

                        guidance = guidance.expand(x_latent.shape[0])

                        tile_batch = 4
                        k_col = 2

                        transformer_args = {'hidden_states': x_latent, 'timestep': t.expand(x_latent.shape[0]).to(latent.dtype) / 1000,
                                        'guidance': guidance, 'pooled_projections': pooled_prompt_embeds[-tile_batch:],
                                        'encoder_hidden_states': prompt_embeds[-tile_batch:], 'txt_ids': text_ids[0], 
                                        'img_ids':image_ids, 'ip_params':ip_params, 'joint_attention_kwargs': None, 
                                        'controlnet_block_samples': controlnet_block_samples, 
                                        'controlnet_single_block_samples': controlnet_single_block_samples,
                                        'return_dict': False, 'controlnet_blocks_repeat': controlnet_blocks_repeat, 
                                        'pulid_ca': pulid_ca, 'ip_projected_image_embeds': ip_projected_image_embeds,
                                        }
                        # if enable_new_tile:
                        noise = self.transformer(**transformer_args)[0] # [4, 11656, 64])

                        noise = unpack_latents(
                            noise,
                            height // factor // 2 + overlap,
                            width // factor // 2 + overlap,
                            channels,
                        )  # torch.Size([4, 16, 188, 248])

                        noises = [elem for elem in torch.split(noise, 1, dim=0)]  # 4 [] torch.Size([1, 16, 188, 248])

                        noises = [
                            noises[i: i + k_col] for i in range(0, len(noises), k_col)
                        ]

                        noises = [
                            concatenate_rowwise(elem, overlap * 2) for elem in noises
                        ]
                        x_noise = concatenate_columnwise(noises, overlap * 2)

                        x_noise = pack_latents(x_noise)
                    else:
                        x_noise = self.transformer(
                            hidden_states=latent,
                            timestep=t.expand(latent.shape[0]).to(latent.dtype) / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds[:1],
                            encoder_hidden_states=prompt_embeds[:1],
                            txt_ids=text_ids,
                            img_ids=image_ids,
                            ip_params=ip_params,
                            joint_attention_kwargs=None,
                            controlnet_block_samples=controlnet_block_samples,
                            controlnet_single_block_samples=controlnet_single_block_samples,
                            return_dict=False,
                            controlnet_blocks_repeat=controlnet_blocks_repeat,
                            pulid_ca=pulid_ca,
                            ip_projected_image_embeds=ip_projected_image_embeds,
                        )[0]
                else:
                    x_noise = self.transformer(
                        hidden_states=latent,
                        timestep=t.expand(latent.shape[0]).to(latent.dtype) / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=image_ids,
                        ip_params=ip_params,
                        joint_attention_kwargs=None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                        pulid_ca=pulid_ca,
                        ip_projected_image_embeds=ip_projected_image_embeds,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latent = self.scheduler.step(x_noise, t, latent, return_dict=False)[0]

                # inpaint
                if latent_mask is not None:
                    init_latents_proper = image_latents
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.scale_noise(
                            init_latents_proper,
                            torch.tensor([noise_timestep]),
                            init_noise,
                        )
                    init_latents_proper = pack_latents(init_latents_proper)
                    latent = init_latents_proper * latent_mask + (1.0 - latent_mask) * latent

                # update progress bar
                pbar.update()

        loop_end_time = time.time()
        # logger.debug(
        #     f"Denoising loop with {opt.steps} steps finished in {opt.steps / (loop_end_time - loop_start_time):.2f}it/s"
        # )

        # 8. Decode latent
        latent = unpack_latents(
            latent, height // factor, width // factor, channels
        )
        samples = self.vae.decode_image(latent, opt.height, opt.width)

        inference_complete_time = time.time()
        # logger.debug(f"Inference costs {inference_complete_time - start_time:.2f}s")

        # 9. Done
        return {
            "images": samples,
            "prompt": opt.prompt,
            "seed": opt.seed,
            "variant": opt.variant,
            "steps": opt.steps,
            "scale": opt.scale,
        }

    def load_lora(self):
        from safetensors.torch import load_file
        from lora_conversion_utils import LoRAModel, LoRALinearLayer
        #lora_sd = load_file('test/VibrantTech3D_v1.safetensors')
        lora_sd = load_file('test/amateurphoto-v6-forcu-hf.safetensors')
        #lora_sd = load_file('test/Hyper-FLUX.1-dev-8steps-lora.safetensors')
        for name, module in self.transformer.named_modules():
            if isinstance(module, LoRALinearLayer):
                up_name = f'transformer.{name}.lora_B.weight'
                down_name = f'transformer.{name}.lora_A.weight'
                #up_name = f'transformer.{name}.lora_up.weight'
                #down_name = f'transformer.{name}.lora_down.weight'
                #up_name = f'lora_unet_{name.replace(".", "_")}.lora_up.weight'
                #down_name = f'lora_unet_{name.replace(".", "_")}.down_name.weight'
                if not down_name in lora_sd:
                    continue
                assert up_name in lora_sd
                down_weight = lora_sd.pop(down_name)
                up_weight = lora_sd.pop(up_name)

                rank, _ = down_weight.shape
                #out_features, _ = up_weight.shape

                lora_layer = LoRALinearLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank
                ).to(self.device, dtype=self.dtype)

                lora_layer.load_state_dict({
                    'down.weight': down_weight,
                    'up.weight': up_weight
                })

                module.set_lora_layer(lora_layer)

        print('lora_sd', lora_sd)

    def prepare_models(
            self,
            init_model,
            mixin=None,
            controlnets=None,
            ip_adapters=None,
    ):
        self.initiate()
        if self.prompt_encoder.is_cleared:
            self.prompt_encoder.prepare_models(mixin=[])
        if self.is_cleared:
            tf_weight_path = os.path.join(
                self.base,
                "transformer",
                FLUX_1DEV_MODEL_WEIGHTS[init_model],
            )
            tf_state_dict = load_state_dict(tf_weight_path)

            if init_model == "origin_dev":
                load_origin_parameters(self.transformer, tf_state_dict)
            elif init_model == "gguf_8steps_q8_0":
                load_gguf_q8_0_quantized_parameters(self.transformer, tf_state_dict)
            self.is_cleared = False
        # self.load_lora(mixin)
        # self.load_controlnets(controlnets)
        # self.load_ip_adapters(ip_adapters)
        print('>>  enable sage attn speed')
        self.transformer.set_attn_processor(FluxAttnProcessor2_0())

        for model in self.models:
            model.to(self.device)

        self.vae.to(self.dtype)

    def unload(self):
        if self.is_cleared:
            return

        if self._initiated:
            try:
                zero_parameters(self.transformer)
                # for model_name in ["controlnet", "pulid_model", "image_encoder"]:
                #     model = getattr(self, model_name, None)
                #     if model is not None:
                #         zero_parameters(model)
            finally:
                for model in self.models:
                    model.to("cpu")
                self.is_cleared = True

    @statistical_runtime("transformer load lora")
    def load_lora(self, mixin):
        if self.is_load_lora:
            self.unload_lora()
        if not mixin:
            return
        for i, (pathname, ratio) in enumerate(mixin):
            lora_sd = load_state_dict(
                pathname, device=str(self.device), dtype=self.dtype
            )
            self.load_single_lora(lora_sd, ratio)
        self.is_load_lora = True

    def load_single_lora(self, lora_sd, scale):
        tf_lora_sd = {}

        key_list = list(lora_sd.keys())
        for k in key_list:
            if k.startswith("text"):
                lora_sd.pop(k)
            elif k.startswith("encoder"):
                lora_sd.pop(k)
            elif k.startswith("transformer"):
                tf_lora_sd[k] = lora_sd.pop(k)

        if len(lora_sd) > 0:
            logger.warn(f"unmatched keys are: {list(lora_sd.keys())}")

        if len(tf_lora_sd) > 0:
            self.load_tf_lora(tf_lora_sd, scale)

    def unload_lora(self):
        self.unload_tf_lora()
        self.is_load_lora = False

    def load_tf_lora(self, lora_state_dict, scale):
        prefix = "transformer."
        lora_state_dict_keys = list(lora_state_dict.keys())

        for name, module in self.transformer.named_modules():
            if prefix + name + ".lora_A.weight" in lora_state_dict_keys:
                down = lora_state_dict.pop(prefix + name + ".lora_A.weight")
                up = lora_state_dict.pop(prefix + name + ".lora_B.weight")
                in_dim = down.shape[1]
                out_dim = up.shape[0]
                rank = down.shape[0]
                if prefix + name + ".alpha" in lora_state_dict_keys:
                    alpha = lora_state_dict.pop(prefix + name + ".alpha").item()
                else:
                    alpha = None

                lora_layer = LoRALinearLayer(
                    in_dim, out_dim, rank, scale, alpha, self.device, self.dtype
                )
                lora_layer.up.weight = torch.nn.Parameter(
                    up.to(self.device, self.dtype)
                )
                lora_layer.down.weight = torch.nn.Parameter(
                    down.to(self.device, self.dtype)
                )
                lora_layer.requires_grad_(False)

                if isinstance(module, LoRAModel):
                    module.set_lora_layer(lora_layer)
                else:
                    new_module = LoRAModel(module)
                    new_module.set_lora_layer(lora_layer)
                    parent_name, _ = name.rsplit(".", 1)
                    parent_module = self.transformer.get_submodule(parent_name)
                    setattr(parent_module, name.split(".")[-1], new_module)

        unmatch_lora_keys = list(lora_state_dict.keys())
        if len(unmatch_lora_keys) > 0:
            logger.info(f"unmatch_lora_keys: {unmatch_lora_keys}")

    def unload_tf_lora(self):
        for n, module in self.transformer.named_modules():
            if isinstance(module, LoRAModel):
                module.remove_lora_layers()


if __name__ == '__main__':
# if 1:
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
            "esrgan": model_id+"realesrgan",
            "controlnet": {
                "canny": model_id+"controlnet/flux-canny-controlnet-v3-converted.safetensors",
                "depth": model_id+"controlnet/flux-depth-controlnet-v3-converted.safetensors",
                "upscal": model_id+"controlnet/flux-depth-controlnet-v3-converted.safetensors",
            },
            "image_encoder": model_id+"image_encoder",
            "ip_adapter": model_id+"ip-adapter/flux-ip-adapter-converted.safetensors",
        },
    }

    # annotator = Annotator(self.index["_"]["annotator"], self.device)

    text_encoder = PromptEncoder(
        base=DEFAULT_INDEX["flux"],
        device=device,
        dtype=torch.bfloat16,
    )
    # from argparse import Namespace
    # opt = Namespace(
    #     prompt = '', mixin = [], n_samples = 1, family = 'flux1.dev',
    #     enable_tile = False, new_tile = False, 
    # )
    # res = text_encoder(opt)
    # for k,v in res.items():
    #     print(k, v.shape)

    pipe = FluxPipeline(base = DEFAULT_INDEX['flux']['root'], device=device, dtype=torch.bfloat16,esrgan=None, 
                controlnet=None, encoder_hid_proj=None, 
                prompt_encoder=text_encoder, annotator=None
            )
    pipe.initiate()
    # res = pipe.prompt_encoder(opt)

    default_params = FluxNormalInput(task='generate')
    default_template: dict = asdict(default_params)
    args = {
        "prompt": "",
        "resizer": 'crop_middle',
        "steps": 20,
        "n_samples": 1,
        "width": 1024,
        "height": 1024,
        "file": [],
        "image": None,
        "mask": None,
        "strength": 0.75,
        "scale": 3.5,
        "seed": [-1],
        "variant": 0.0,
        "variant_seed": [-1],
    }
    default_template.update(args)
    normal_input = FluxNormalInput(**default_template)
    normal_input.normal_width, normal_input.normal_height = normalize_size(
            normal_input.width, normal_input.height, 4096)
    for i in range(1):
        res = pipe(normal_input)
    

    # test tile ~~~~~~  13s/it   batch 部分太夸张了，显存占用的也比较多~ 
    # 构建全局构图布局，第一步，慢一些，4k生成，稍微理解
    from PIL import Image
    img = Image.open('/home/dell/workspace/img/lowq/scenary.jpeg')

    # img = res['images'][0]
    args = {
        "prompt": "beautiful scenary at seaside, beach, birds view from top, blue ocen",
        "resizer": 'crop_middle',
        "steps": 30, "n_samples": 1,
        "tile_percent": 0, "enable_tile": True, "new_tile": True,   # 启用分块放大算法~
        "width": 3840, "height": 2880,
        "file": [img], "image": 0,  "strength": 0.35,
        "mask": None,  "scale": 4.5,
        "seed": [-1], "variant": 0.0, "variant_seed": [-1],
    }
    # args['prompt'] = '##'.join([args['prompt']]*5)
    default_template.update(args)
    normal_input = FluxNormalInput(**default_template)
    normal_input.normal_width, normal_input.normal_height = normalize_size(
            normal_input.width, normal_input.height, 4096)
    res = pipe(normal_input)
    
    res['images'][0].save('1_test_fluxe_new_tile.jpeg')

    # the result is not clearly

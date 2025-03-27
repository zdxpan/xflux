import os
import numpy as np
from typing import Dict, Union, Optional, Any

import accelerate
import bitsandbytes as bnb
import torch
from diffusers import FluxTransformer2DModel
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers, logging
from tensor_util  import load_parameters
from log import statistical_runtime
from safetensors.torch import load_file
from torch import nn as nn

logger_diff = logging.get_logger(__name__)

FLUX_1DEV_MODEL_WEIGHTS = {
    "gguf_8steps_q8_0": "flux.1dev_fuse_8steps_lora_gguf.safetensors",
    "gguf_q8_0": "flux1.dev_q80_gguf.safetensors",
    # "nf4": "diffusion_pytorch_model.hyper8.nf4.safetensors",
    'origin_dev': 'flux1.dev.hf.bf16.safetensors'  #'diffusion_pytorch_model.safetensors'
}


class QuantizedFluxTransformer2DModel(FluxTransformer2DModel):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            subfolder,
            init_model="",
            use_scaled=True,
            torch_dtype=torch.float32,
    ):
        config, unused_kwargs, commit_hash = cls.load_config(
            pretrained_model_name_or_path,
            return_unused_kwargs=True,
            return_commit_hash=True,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        )

        with accelerate.init_empty_weights():
            model = cls.from_config(config, **unused_kwargs)
            model.register_to_config(_name_or_path=pretrained_model_name_or_path)
            model.eval()

        if init_model == "origin_dev":
            path = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                FLUX_1DEV_MODEL_WEIGHTS[init_model],
            )
            state_dict = load_file(path)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    has_bias = module.bias is not None

                    weight = state_dict.pop(f"{name}.weight")
                    bias = state_dict.pop(f"{name}.bias", None)

                    module.weight = torch.nn.Parameter(weight, requires_grad=False)
                    if has_bias:
                        module.bias = torch.nn.Parameter(bias, requires_grad=False)
                else:
                    weight = state_dict.pop(f"{name}.weight", None)
                    if weight is not None:
                        module.load_state_dict({"weight": weight}, assign=True)
        elif init_model == "nf4":
            path = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                FLUX_1DEV_MODEL_WEIGHTS[init_model],
            )
            state_dict = load_file(path)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    in_features = module.in_features
                    out_features = module.out_features
                    has_bias = module.bias is not None

                    weight = state_dict.pop(f"{name}.weight")
                    bias = state_dict.pop(f"{name}.bias", None)
                    quantized_stats = {
                        k: state_dict.pop(f"{name}.{k}")
                        for k in (
                            "weight.absmax",
                            "weight.nested_absmax",
                            "weight.nested_quant_map",
                            "weight.quant_map",
                            "weight.quant_state.bitsandbytes__nf4",
                        )
                        if f"{name}.{k}" in state_dict
                    }
                    quantized_module = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        bias=has_bias,
                        compress_statistics="weight.nested_absmax" in quantized_stats,
                        compute_dtype=torch_dtype,
                        quant_type="nf4",
                        device="meta",
                    )

                    quantized_module.weight = bnb.nn.Params4bit.from_prequantized(
                        quantized_stats=quantized_stats,
                        data=weight,
                        require_grad=False,
                        module=module,
                    )
                    quantized_module.quant_state = quantized_module.weight.quant_state
                    if has_bias:
                        quantized_module.bias = torch.nn.Parameter(bias)

                    parent = model
                    name = name.split(".")
                    for n in name[:-1]:
                        parent = getattr(parent, n)
                    setattr(parent, name[-1], quantized_module)
                else:
                    weight = state_dict.pop(f"{name}.weight", None)
                    if weight is not None:
                        module.load_state_dict({"weight": weight}, assign=True)
        elif init_model in ["gguf_8steps_q8_0", "gguf_q8_0"]:
            path = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                FLUX_1DEV_MODEL_WEIGHTS[init_model],
            )
            state_dict = load_file(path)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    has_bias = module.bias is not None

                    weight = state_dict.pop(f"{name}.weight")
                    bias = state_dict.pop(f"{name}.bias", None)

                    out_features, in_features = weight.shape

                    is_ggml_quantized = weight.dtype == torch.uint8
                    gguf_module = GGUFLinear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=has_bias,
                        is_ggml_quantized=is_ggml_quantized,
                    )
                    # gguf_module.requires_grad_ = False
                    gguf_module.weight = torch.nn.Parameter(weight, requires_grad=False)
                    if has_bias:
                        gguf_module.bias = torch.nn.Parameter(bias, requires_grad=False)

                    parent_name = name.rsplit(".", 1)
                    if len(parent_name) == 1:
                        parent_module = model
                    else:
                        parent_module = model.get_submodule(parent_name[0])
                    setattr(parent_module, name.rsplit(".", 1)[-1], gguf_module)
                else:
                    weight = state_dict.pop(f"{name}.weight", None)
                    if weight is not None:
                        module.load_state_dict({"weight": weight}, assign=True)
        else:
            if use_scaled:
                path = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    "diffusion_pytorch_model_schnell_use_scaled.safetensors",
                )
            else:
                path = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    "diffusion_pytorch_model_schnell_no_scaled.safetensors",
                )
            state_dict = load_file(path)
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    has_bias = module.bias is not None

                    w_f8 = state_dict.pop(f"{name}.w_f8")
                    w_inv_s = (
                        state_dict.pop(f"{name}.w_inv_s", None) if use_scaled else None
                    )
                    bias = state_dict.pop(f"{name}.bias", None)

                    fp8_module = Fp8Linear(w_f8, w_inv_s, bias, use_scaled)
                    parent_name = name.rsplit(".", 1)
                    if len(parent_name) == 1:
                        parent_module = model
                    else:
                        parent_module = model.get_submodule(parent_name[0])
                    setattr(parent_module, name.rsplit(".", 1)[-1], fp8_module)

                else:
                    weight = state_dict.pop(f"{name}.weight", None)
                    if weight is not None:
                        module.load_state_dict({"weight": weight}, assign=True)

        return model

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
                name: str,
                module: torch.nn.Module,
                processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
            self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            ip_params: dict = {},
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            interval_control: int = 2,
            return_dict: bool = True,
            ip_projected_image_embeds=[],
            pulid_ca: torch.nn.Module = None,
            controlnet_single_block_samples = None,
            controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                    joint_attention_kwargs is not None
                    and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger_diff.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # ip-pulid
        ip_names = ip_params.pop("ip_names", [])
        ip_embeddings = ip_params.pop("ip_embeddings", [])
        ip_scales = ip_params.pop("ip_scales", [])
        ca_idx = [0] * len(ip_names)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,  # 1, 22400, 3072
                encoder_hidden_states=(
                    (encoder_hidden_states, ip_projected_image_embeds)
                    if ip_projected_image_embeds
                    else encoder_hidden_states  # 1, 512, 3072
                ),
                temb=temb,  # 1,3972
                image_rotary_emb=image_rotary_emb,  # (22912,128) * 2
            )

            for cnt, (ip_name, ip_embedding, ip_scale) in enumerate(
                    zip(ip_names, ip_embeddings, ip_scales)
            ):
                if (
                        ip_name == "ip_pulid" and pulid_ca is not None
                        and index_block % pulid_ca.pulid_double_interval == 0
                        and ip_embedding is not None
                ):
                    hidden_states = hidden_states + ip_scale * pulid_ca[ca_idx[cnt]](
                        ip_embedding, hidden_states
                    )
                    ca_idx[cnt] += 1

            if controlnet_block_samples is not None:
                hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % interval_control]
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            for cnt, (ip_name, ip_embedding, ip_scale) in enumerate(
                    zip(ip_names, ip_embeddings, ip_scales)
            ):
                if (
                        ip_name == "ip_pulid" and pulid_ca is not None
                        and index_block % pulid_ca.pulid_single_interval == 0
                        and ip_embedding is not None
                ):
                    hidden_states = hidden_states + ip_scale * pulid_ca[ca_idx[cnt]](
                        ip_embedding, hidden_states
                    )
                    ca_idx[cnt] += 1
        
        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def load_bnb4_quantized_parameters(model, state_dict):
    with statistical_runtime(f"Load quantized {model.__class__.__name__} weight"):
        for name, module in model.named_modules():
            if hasattr(module, "base_layer"):
                continue
            elif isinstance(module, bnb.nn.Linear4bit):
                name = name.replace(".base_layer", "")
                weight = state_dict.pop(f"{name}.weight")
                bias = state_dict.pop(f"{name}.bias", None)
                quantized_stats = {
                    "weight.absmax": state_dict.pop(f"{name}.weight.absmax"),
                    "weight.quant_map": state_dict.pop(f"{name}.weight.quant_map"),
                    "weight.quant_state.bitsandbytes__nf4": state_dict.pop(
                        f"{name}.weight.quant_state.bitsandbytes__nf4"
                    ),
                }
                if f"{name}.weight.nested_absmax" in state_dict:
                    quantized_stats["weight.nested_absmax"] = state_dict.pop(
                        f"{name}.weight.nested_absmax"
                    )
                    quantized_stats["weight.nested_quant_map"] = state_dict.pop(
                        f"{name}.weight.nested_quant_map"
                    )

                module.weight = bnb.nn.Params4bit.from_prequantized(
                    quantized_stats=quantized_stats,
                    data=weight,
                    require_grad=False,
                    module=module,
                )
                module.quant_state = module.weight.quant_state
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(bias, requires_grad=False)
            else:
                name = name.replace(".base_layer", "")
                module_sd = {}
                for sk in ["weight", "bias"]:
                    weight_value = state_dict.pop(f"{name}.{sk}", None)
                    if weight_value is not None:
                        module_sd[sk] = weight_value
                if module_sd:
                    load_parameters(module, module_sd)


def load_gguf_q8_0_quantized_parameters(model, state_dict):
    with statistical_runtime(f"Load quantized {model.__class__.__name__} weight"):
        for name, module in model.named_modules():
            if hasattr(module, "base_layer"):
                continue
            elif isinstance(module, GGUFLinear):
                name = name.replace(".base_layer", "")
                weight = state_dict.pop(f"{name}.weight")
                bias = state_dict.pop(f"{name}.bias", None)

                module.weight = torch.nn.Parameter(weight, requires_grad=False)
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(bias, requires_grad=False)
            else:
                name = name.replace(".base_layer", "")
                module_sd = {}
                for sk in ["weight", "bias"]:
                    weight_value = state_dict.pop(f"{name}.{sk}", None)
                    if weight_value is not None:
                        module_sd[sk] = weight_value
                if module_sd:
                    load_parameters(module, module_sd)


def load_schnell_use_scaled_parameters(model, state_dict, use_scaled=True):
    with statistical_runtime(f"Load quantized {model.__class__.__name__} weight"):
        for name, module in model.named_modules():
            if isinstance(module, Fp8Linear):

                w_f8 = state_dict.pop(f"{name}.w_f8")
                w_inv_s = (
                    state_dict.pop(f"{name}.w_inv_s", None) if use_scaled else None
                )
                bias = state_dict.pop(f"{name}.bias", None)

                fp8_module = Fp8Linear(w_f8, w_inv_s, bias, use_scaled)

                parent_name = name.rsplit(".", 1)
                if len(parent_name) == 1:
                    parent_module = model
                else:
                    parent_module = model.get_submodule(parent_name[0])
                setattr(parent_module, name.rsplit(".", 1)[-1], fp8_module)
            else:
                module_sd = {}
                for sk in ["weight", "bias"]:
                    weight_value = state_dict.pop(f"{name}.{sk}", None)
                    if weight_value is not None:
                        module_sd[sk] = weight_value
                if module_sd:
                    load_parameters(module, module_sd)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_blocks_Q8_0(blocks, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def quant_shape_from_byte_shape(shape, block_size, type_size):
    if shape[-1] % type_size != 0:
        raise ValueError(f"Quantized tensor bytes per row ({shape[-1]})")
    return (*shape[:-1], shape[-1] // type_size * block_size)


def dequantize(data, dtype=None):
    block_size, type_size = (32, 34)
    oshape = quant_shape_from_byte_shape(data.shape, block_size, type_size)

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks_Q8_0(blocks, dtype)
    return blocks.reshape(oshape)


class GGUFLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            is_ggml_quantized=False,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.is_ggml_quantized = is_ggml_quantized

    def get_weight(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor

        return dequantize(tensor.data, dtype)

    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.bfloat16)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device

        bias = None
        non_blocking = True
        if self.bias is not None:
            bias = self.get_weight(self.bias.to(device), dtype)
            bias = bias.to(device=device, dtype=dtype, non_blocking=non_blocking)

        weight = self.get_weight(self.weight.to(device), dtype)
        weight = weight.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return weight, bias

    def ggml_forward(self, input):
        weight, bias = self.cast_bias_weight(input)
        return torch.nn.functional.linear(input, weight, bias)

    def forward(self, *args, **kwargs):
        if self.is_ggml_quantized:
            return self.ggml_forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


class Fp8Linear(nn.Module):

    def __init__(
            self,
            w_f8,
            w_inv_s=None,
            bias=None,
            use_scaled=True,
            calc_dtype=torch.float8_e4m3fn,
    ):
        super().__init__()
        self.w_f8 = torch.nn.Parameter(w_f8.t(), requires_grad=False)
        self.w_inv_s = (
            torch.nn.Parameter(w_inv_s, requires_grad=False)
            if w_inv_s is not None
            else None
        )
        self.bias = (
            torch.nn.Parameter(bias, requires_grad=False) if bias is not None else None
        )

        self.use_scaled = use_scaled
        self.calc_dtype = calc_dtype

    def use_scaled_forward(self, x):
        o_shape = x.shape[:-1] + (self.w_f8.shape[-1],)
        x_f8, x_inv_s = to_float8(x, self.calc_dtype)
        x_f8_reshaped = x_f8.view(-1, x_f8.shape[-1])

        y, _ = torch._scaled_mm(
            x_f8_reshaped,
            self.w_f8,
            bias=self.bias,
            out_dtype=x.dtype,
            scale_a=x_inv_s,
            scale_b=self.w_inv_s,
        )
        return y.reshape(o_shape)

    def fp8_forward(self, x):
        o_shape = x.shape[:-1] + (self.w_f8.shape[-1],)
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])

        out_dtype = x.dtype
        x = x.to(self.calc_dtype)
        scale_weight = torch.ones((1), device=x.device, dtype=torch.float32)
        scale_input = torch.ones((1), device=x.device, dtype=torch.float32)

        y, _ = torch._scaled_mm(
            x,
            self.w_f8,
            out_dtype=out_dtype,
            bias=self.bias,
            scale_a=scale_input,
            scale_b=scale_weight,
        )
        return y.reshape(o_shape)

    def forward(self, x):
        if self.use_scaled:
            return self.use_scaled_forward(x)

        return self.fp8_forward(x)


def load_origin_parameters(model, state_dict):
    with statistical_runtime(f"Load quantized {model.__class__.__name__} weight"):
        for name, module in model.named_modules():
            if hasattr(module, "base_layer"):
                continue
            elif isinstance(module, nn.Linear):
                name = name.replace(".base_layer", "")
                weight = state_dict.pop(f"{name}.weight")
                bias = state_dict.pop(f"{name}.bias", None)

                module.weight = torch.nn.Parameter(weight, requires_grad=False)
                if module.bias is not None:
                    module.bias = torch.nn.Parameter(bias, requires_grad=False)
            else:
                name = name.replace(".base_layer", "")
                module_sd = {}
                for sk in ["weight", "bias"]:
                    weight_value = state_dict.pop(f"{name}.{sk}", None)
                    if weight_value is not None:
                        module_sd[sk] = weight_value
                if module_sd:
                    load_parameters(module, module_sd)

def zero_parameters(model: torch.nn.Module):
    """
    In order to save gpu memory, the weight of the model is remade to an empty tensor
    """
    if hasattr(model, "device") and hasattr(model, "dtype"):
        empty = torch.Tensor().to(model.device).to(model.dtype)
    else:
        empty = torch.Tensor()
    for _, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(recurse=False):
            module.register_parameter(
                parameter_name, torch.nn.Parameter(empty, requires_grad=False)
            )


def load_state_dict(*pathnames, device="cpu", dtype=None):
    if device == "cuda":
        # Fix the issue of loading the correct index number when using multiple GPUs.
        device = f"cuda:{torch.cuda.current_device()}"
    assert len(pathnames) > 0
    state_dict = []
    for pathname in pathnames:
        sd = None
        if pathname is not None:
            with statistical_runtime(f"Load state dict from {pathname}"):
                if pathname.endswith(".safetensors.zst"):
                    try:
                        with open(pathname, "rb") as f:
                            data = f.read()
                        dctx = zstandard.ZstdDecompressor()
                        data = dctx.decompress(data)
                        pl_sd = load(data)
                    except Exception as e:
                        raise Exception(f"load {pathname} safetensors.zst error: {e}")
                elif pathname.endswith(".safetensors"):
                    try:
                        pl_sd = load_file(pathname, device=device)
                    except Exception as e:
                        raise Exception(f"load {pathname} safetensors error: {e}")
                else:
                    try:
                        pl_sd = torch.load(pathname, map_location=device)
                    except Exception as e:
                        raise Exception(f"torch load {pathname} eeror: {e}")
                sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
        if sd and dtype:
            for key, value in sd.items():
                sd[key] = value.to(dtype)
        state_dict.append(sd)
    return state_dict if len(state_dict) > 1 else state_dict[0]


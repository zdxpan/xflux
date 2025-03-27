import re

import torch
import torch.nn as nn
from diffusers.utils import is_peft_version, logging

logger = logging.get_logger(__name__)


class LoRALinearLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            rank=4,
            scale=1.0,
            network_alpha=None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        self.network_alpha = network_alpha
        self.rank = rank
        self.scale = scale

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        hidden_states = self.down(hidden_states.to(dtype))
        hidden_states = self.up(hidden_states)

        if self.network_alpha is not None:
            hidden_states *= self.scale * self.network_alpha / self.rank
        else:
            hidden_states *= self.scale

        return hidden_states.to(orig_dtype)


class LoRAModel(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.lora_layers = nn.ModuleList()

    def set_lora_layer(self, lora_layer):
        self.lora_layers.append(lora_layer)

    def remove_lora_layers(self):
        self.lora_layers = nn.ModuleList()

    def forward(self, hidden_states):
        out = self.base_layer(hidden_states)
        for layer in self.lora_layers:
            out += layer(hidden_states)
        return out


def convert_tf_lora_state_dict(
        state_dict,
        return_alphas: bool = False,
):
    unmatch_state_dict = {}

    is_unkown1 = any(
        k.startswith("diffusion_model") and k.endswith(".lora_down.weight")
        for k in state_dict
    )
    if is_unkown1:
        state_dict, unmatch_state_dict = convert_unkown1_flux_lora_to_diffusers(
            state_dict
        )
        return (
            (state_dict, unmatch_state_dict, None)
            if return_alphas
            else (state_dict, unmatch_state_dict)
        )

    is_kohya = any(".lora_down.weight" in k for k in state_dict)
    if is_kohya:
        state_dict, unmatch_state_dict = convert_kohya_flux_lora_to_diffusers(
            state_dict
        )
        # Kohya already takes care of scaling the LoRA parameters with alpha.
        return (
            (state_dict, unmatch_state_dict, None)
            if return_alphas
            else (state_dict, unmatch_state_dict)
        )

    is_xlabs = any("processor" in k for k in state_dict)
    if is_xlabs:
        state_dict, unmatch_state_dict = convert_xlabs_flux_lora_to_diffusers(
            state_dict
        )
        # xlabs doesn't use `alpha`.
        return (
            (state_dict, unmatch_state_dict, None)
            if return_alphas
            else (state_dict, unmatch_state_dict)
        )

    # For state dicts like
    # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA
    keys = list(state_dict.keys())
    network_alphas = {}
    for k in keys:
        if "alpha" in k:
            alpha_value = state_dict.get(k)
            if (
                    torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)
            ) or isinstance(alpha_value, float):
                network_alphas[k] = state_dict.pop(k)
            else:
                raise ValueError(
                    f"The alpha key ({k}) seems to be incorrect. If you think this error is unexpected, please open as issue."
                )

    if return_alphas:
        return state_dict, unmatch_state_dict, network_alphas
    else:
        return state_dict, unmatch_state_dict


# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are taken from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
# All credits go to `kohya-ss`.
def convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
        scale = (
                alpha / rank
        )  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_B.weight"] = (
                sds_sd.pop(sds_key + ".lora_up.weight") * scale_up
        )

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        alpha = sds_sd.pop(sds_key + ".alpha")
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i: i + dims[j], k * ait_rank: (k + 1) * ait_rank]
                        == 0
                    )
                i += dims[j]
            if is_sparse:
                logger.info(f"weight is sparse: {sds_key}")

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update({k: down_weight for k in ait_down_keys})

            # up_weight is split to each split
            ait_sd.update(
                {k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))}
            )  # noqa: C416
        else:
            # down_weight is chunked to each split
            ait_sd.update(
                {
                    k: v
                    for k, v in zip(
                    ait_down_keys, torch.chunk(down_weight, num_splits, dim=0)
                )
                }
            )  # noqa: C416

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[
                                         i: i + dims[j], j * ait_rank: (j + 1) * ait_rank
                                         ].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        if len(sds_sd) > 0:
            logger.warning(f"Unsuppored keys for ai-toolkit: {sds_sd.keys()}")

        return ait_sd, sds_sd

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


# Adapted from https://gist.github.com/Leommm-byte/6b331a1e9bd53271210b26543a7065d6
# Some utilities were reused from
# https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
def convert_xlabs_flux_lora_to_diffusers(old_state_dict):
    new_state_dict = {}
    orig_keys = list(old_state_dict.keys())

    def handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        down_weight = sds_sd.pop(sds_key)
        up_weight = sds_sd.pop(sds_key.replace(".down.weight", ".up.weight"))

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]

        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split
        ait_sd.update(
            {k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))}
        )  # noqa: C416

    for old_key in orig_keys:
        # Handle double_blocks
        if old_key.startswith(("diffusion_model.double_blocks", "double_blocks")):
            block_num = re.search(r"double_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.transformer_blocks.{block_num}"

            if "processor.proj_lora1" in old_key:
                new_key += ".attn.to_out.0"
            elif "processor.proj_lora2" in old_key:
                new_key += ".attn.to_add_out"
            # Handle text latents.
            elif "processor.qkv_lora2" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.add_q_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_k_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_v_proj",
                    ],
                )
                # continue
            # Handle image latents.
            elif "processor.qkv_lora1" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.transformer_blocks.{block_num}.attn.to_v",
                    ],
                )
                # continue

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        # Handle single_blocks
        elif old_key.startswith("diffusion_model.single_blocks", "single_blocks"):
            block_num = re.search(r"single_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.single_transformer_blocks.{block_num}"

            if "proj_lora1" in old_key or "proj_lora2" in old_key:
                new_key += ".proj_out"
            elif "qkv_lora1" in old_key or "qkv_lora2" in old_key:
                new_key += ".norm.linear"

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        else:
            # Handle other potential key patterns here
            new_key = old_key

        # Since we already handle qkv above.
        if "qkv" not in old_key:
            new_state_dict[new_key] = old_state_dict.pop(old_key)

    if len(old_state_dict) > 0:
        raise ValueError(
            f"`old_state_dict` should be at this point but has: {list(old_state_dict.keys())}."
        )

    return new_state_dict, old_state_dict


def convert_unkown1_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        if sds_key + ".alpha" in sds_sd:
            alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
        else:
            alpha = rank
        scale = (
                alpha / rank
        )  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_B.weight"] = (
                sds_sd.pop(sds_key + ".lora_up.weight") * scale_up
        )

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        if sds_key + ".alpha" in sds_sd:
            alpha = sds_sd.pop(sds_key + ".alpha")
        else:
            alpha = sd_lora_rank
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i: i + dims[j], k * ait_rank: (k + 1) * ait_rank]
                        == 0
                    )
                i += dims[j]
            if is_sparse:
                logger.info(f"weight is sparse: {sds_key}")

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update({k: down_weight for k in ait_down_keys})

            # up_weight is split to each split
            ait_sd.update(
                {k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))}
            )  # noqa: C416
        else:
            # down_weight is chunked to each split
            ait_sd.update(
                {
                    k: v
                    for k, v in zip(
                    ait_down_keys, torch.chunk(down_weight, num_splits, dim=0)
                )
                }
            )  # noqa: C416

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[
                                         i: i + dims[j], j * ait_rank: (j + 1) * ait_rank
                                         ].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.img_attn.proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.img_attn.qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.img_mlp_0",  # mod
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.img_mlp_2",  # mod
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.img_mod_lin",  # mod
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.txt_attn.proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.txt_attn.qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.txt_mlp_0",  # mod
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.txt_mlp_2",  # mod
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.double_blocks.{i}.txt_mod_lin",  # mod
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"diffusion_model.single_blocks.{i}.linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.single_blocks.{i}.linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"diffusion_model.single_blocks.{i}.modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        if len(sds_sd) > 0:
            logger.warning(f"Unsuppored keys for ai-toolkit: {sds_sd.keys()}")

        return ait_sd, sds_sd

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


def convert_te_lora_to_diffusers(state_dict, text_encoder_name="text_encoder"):
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}
    network_alphas_2 = {}

    # Check for DoRA-enabled LoRAs.
    dora_present_in_te = any(
        "dora_scale" in k and ("lora_te_" in k or "lora_te1_" in k) for k in state_dict
    )
    dora_present_in_te2 = any(
        "dora_scale" in k and "lora_te2_" in k for k in state_dict
    )
    if dora_present_in_te or dora_present_in_te2:
        if is_peft_version("<", "0.9.0"):
            raise ValueError(
                "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
            )

    all_lora_keys = list(state_dict.keys())
    for key in all_lora_keys:
        if not key.startswith("text_encoder."):
            continue
        new_key = key
        new_key = new_key.replace("lora_linear_layer", "lora")
        new_key = new_key.replace("to_q_lora", "q_proj.lora")
        new_key = new_key.replace("to_k_lora", "k_proj.lora")
        new_key = new_key.replace("to_v_lora", "v_proj.lora")
        state_dict[new_key] = state_dict.pop(key)

    # Iterate over all LoRA weights.
    all_lora_keys = list(state_dict.keys())
    for key in all_lora_keys:
        if not key.endswith("lora_down.weight") and not key.endswith(
                "lora.down.weight"
        ):
            continue

        # Extract LoRA name.
        if key.endswith("lora_down.weight"):
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
        else:
            lora_name = key.replace(".lora.down.weight", "")
            lora_name_up = lora_name + ".lora.up.weight"

        lora_name_alpha = lora_name + ".alpha"

        # Handle text encoder LoRAs.
        if lora_name.startswith(
                ("lora_te_", "lora_te1_", "lora_te2_", "text_encoder.")
        ):
            diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

            # Store down and up weights for te or te2.
            if lora_name.startswith(("lora_te_", "lora_te1_", "text_encoder.")):
                te_state_dict[diffusers_name] = state_dict.pop(key)
                te_state_dict[diffusers_name.replace(".down.", ".up.")] = (
                    state_dict.pop(lora_name_up)
                )
            else:
                te2_state_dict[diffusers_name] = state_dict.pop(key)
                te2_state_dict[diffusers_name.replace(".down.", ".up.")] = (
                    state_dict.pop(lora_name_up)
                )

            # Store DoRA scale if present.
            if dora_present_in_te or dora_present_in_te2:
                dora_scale_key_to_replace_te = (
                    "_lora.down."
                    if "_lora.down." in diffusers_name
                    else ".lora_linear_layer."
                )
                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    te_state_dict[
                        diffusers_name.replace(
                            dora_scale_key_to_replace_te, ".lora_magnitude_vector."
                        )
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))
                elif lora_name.startswith("lora_te2_"):
                    te2_state_dict[
                        diffusers_name.replace(
                            dora_scale_key_to_replace_te, ".lora_magnitude_vector."
                        )
                    ] = state_dict.pop(key.replace("lora_down.weight", "dora_scale"))

        # Store alpha if present.
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha)
            if lora_name_alpha.startswith(("lora_te_", "lora_te1_", "text_encoder.")):
                network_alphas.update(
                    _get_alpha_name(lora_name_alpha, diffusers_name, alpha)
                )
            else:
                network_alphas_2.update(
                    _get_alpha_name(lora_name_alpha, diffusers_name, alpha)
                )

    # Check if any keys remain.
    if len(state_dict) > 0:
        raise ValueError(
            f"The following keys have not been correctly renamed: \n\n {', '.join(state_dict.keys())}"
        )

    logger.info("Non-diffusers checkpoint detected.")

    # Construct final state dict.
    te_state_dict = {
        f"{text_encoder_name}.{module_name}": params
        for module_name, params in te_state_dict.items()
    }
    te2_state_dict = (
        {
            f"text_encoder_2.{module_name}": params
            for module_name, params in te2_state_dict.items()
        }
        if len(te2_state_dict) > 0
        else {}
    )

    return te_state_dict, network_alphas, te2_state_dict, network_alphas_2, state_dict


def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_", "text_encoder")):
        if lora_name.startswith("lora_te_"):
            key_to_replace = "lora_te_"
        elif lora_name.startswith("lora_te1_"):
            key_to_replace = "lora_te1_"
        else:
            key_to_replace = "text_encoder."
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "q_proj.lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "k_proj.lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "v_proj.lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "out_proj.lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        # diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
        ...
    return diffusers_name


def _get_alpha_name(lora_name_alpha, diffusers_name, alpha):
    """
    Gets the correct alpha name for the Diffusers model.
    """
    if lora_name_alpha.startswith("lora_unet_"):
        prefix = "unet."
    elif lora_name_alpha.startswith(("lora_te_", "lora_te1_", "text_encoder.")):
        prefix = "text_encoder."
    else:
        prefix = "text_encoder_2."
    new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
    return {new_name: alpha}

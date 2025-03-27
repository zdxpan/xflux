import os
import functools
import re
import numpy as np

import torch
from tensor_util import load_parameters, load_state_dict, zero_parameters
from log import statistical_runtime, logger
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer

from lora_conversion_utils import LoRALinearLayer, LoRAModel

RE_ESCAPED = re.compile(r"&.*?;")  # match &plus; &minus; &lb; &rb;

ESCAPE_MAP = {
    # '(': '&lb;',
    # ')': '&rb;',
    # '+': '&plus;',
    # '-': '&minus;',
    # '\\': '\\',
    "&minus;": "-",
    "&plus;": "+",
    "&rb;": ")",
    "&lb;": "(",
}

RE_EXPLICIT_EXPONENT = re.compile(r"(\+|\-)(\d+)")  # match word+num

RE_IMPLICIT_SCOPE = re.compile(
    r"(\b\w+\b)(\++|\-+)"
)  # word add left bracket and right bracket

RE_ATTENTION = re.compile(
    r"""
    (
    \++\)       |  # right bracket with plus symbol
    \-+\)       |  # right bracket with minus symbol
    \)          |  # right bracket
    \(          |  # left bracket
    .           |  # other
    \n          |  # newline
    )
    """,
    re.X,
)


def escape(prompt):
    """
    Escape special characters
    eg:
    origin:  &plus; &minus; &lb; &rb;
    escaped  +       -      (     )
    """
    return RE_ESCAPED.sub(lambda x: ESCAPE_MAP.get(x.group(0), x.group(0)), prompt)


def normal_operator(x, threshold):
    return x.group(1) * min(int(x.group(2)), threshold)


def normalize(prompt, threshold):
    """
    word pattern normalize
    """
    # word+2 => word++
    prompt = RE_EXPLICIT_EXPONENT.sub(
        functools.partial(normal_operator, threshold=threshold), prompt
    )
    # word++ => (word)++
    prompt = RE_IMPLICIT_SCOPE.sub(r"(\g<1>)\g<2>", prompt)
    return prompt


def parse_prompt(prompt, multiplier_base, max_exponent):
    prompt = escape(prompt)
    prompt = normalize(prompt, max_exponent)
    # revert the prompt string to simplify the parsing method
    prompt = prompt[::-1]

    piece = []
    fragments = []
    multipliers = []

    def flush():
        if len(piece) > 0:
            f = "".join(piece)
            f = f[::-1]  # unrevert the fragment string
            f = escape(f)
            fragments.append([f, np.prod(multipliers)])
            piece.clear()

    index = 0
    length = len(prompt)
    while index < length:
        result = RE_ATTENTION.match(prompt[index:])
        text = result[0]
        if text.startswith("+") and text.endswith(")"):  # end of strong fragment
            flush()
            multipliers.append(multiplier_base ** min(max_exponent, len(text) - 1))
        elif text.startswith("-") and text.endswith(")"):  # end of weak fragment
            flush()
            multipliers.append(1 / multiplier_base ** min(max_exponent, len(text) - 1))
        elif text == ")":  # end of normal fragment
            flush()
            multipliers.append(1.0)
        elif text == "(":  # start of fragment
            flush()
            # If there is an extra left bracket, an error will be reported
            # eg: (A cat ((rides)+2 a horse+1 on Mars)+2
            if len(multipliers):
                multipliers.pop()
        else:  # prompt text
            piece.append(text)
        index += result.span()[1]

    flush()  # flush the last fragment

    return fragments[::-1]  # restore fragments order to ltr


class PromptEncoder:
    def __init__(self, base, device, dtype, multiplier_base=1.1, max_exponent=9):
        self.base = base
        self.device = device
        self.dtype = dtype
        self.multiplier_base = multiplier_base
        self.max_exponent = max_exponent
        self._initiated = False
        self.is_cleared = False
        self.is_load_lora = False

    @statistical_runtime("first initiate the text encoder")
    def initiate(self):
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            self.base["root"], subfolder="tokenizer", torch_dtype=self.dtype
        )
        self.clip_max_length = self.clip_tokenizer.model_max_length
        self.clip_encoder = CLIPTextModel.from_pretrained(
            self.base["root"], subfolder="text_encoder", torch_dtype=self.dtype
        )

        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            self.base["root"], subfolder="tokenizer_2", torch_dtype=self.dtype
        )
        logger.debug(f'load t5encoder from text_encoder_2_full_weight  : {self.base["root"]}')
        self.t5_encoder = T5EncoderModel.from_pretrained(
            self.base["root"],
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
        )

        self.models = [self.clip_encoder, self.t5_encoder]
        for model in self.models:
            model.requires_grad_ = False

        self._initiated = True

    # @statistical_runtime("Text encoder(clip t5) load lora")
    def load_lora(self, mixin):
        if self.is_load_lora:
            self.unload_lora()

        if not mixin:
            return

        for _, (pathname, ratio) in enumerate(mixin):
            lora_sd = load_state_dict(
                pathname, device=str(self.device), dtype=self.dtype
            )
            self.load_single_lora(lora_sd, ratio)
        self.is_load_lora = True

    def load_single_lora(self, lora_sd, scale):
        te1_lora_sd, te2_lora_sd = {}, {}

        key_list = list(lora_sd.keys())
        for k in key_list:
            if k.startswith("text"):
                te1_lora_sd[k] = lora_sd.pop(k)
            elif k.startswith("encoder"):
                te2_lora_sd[k] = lora_sd.pop(k)
            elif k.startswith("transformer"):
                lora_sd.pop(k)

        if len(lora_sd) > 0:
            logger.warn(f"unmatched keys are: {list(lora_sd.keys())}")

        if len(te1_lora_sd) > 0:
            self.load_te_lora(te1_lora_sd, scale)

        if len(te2_lora_sd) > 0:
            logger.warn(f"please add the T5 lora load function")

    def load_te_lora(self, lora_state_dict, scale):
        prefix = "text_encoder."
        lora_state_dict_keys = list(lora_state_dict.keys())

        for name, module in self.clip_encoder.named_modules():
            if prefix + name + ".lora.down.weight" in lora_state_dict_keys:
                down = lora_state_dict.pop(prefix + name + ".lora.down.weight")
                up = lora_state_dict.pop(prefix + name + ".lora.up.weight")
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
                    up.to(self.device, self.dtype), requires_grad=False
                )
                lora_layer.down.weight = torch.nn.Parameter(
                    down.to(self.device, self.dtype), requires_grad=False
                )
                lora_layer.requires_grad_(False)

                if isinstance(module, LoRAModel):
                    module.set_lora_layer(lora_layer)
                else:
                    new_module = LoRAModel(module)
                    new_module.set_lora_layer(lora_layer)
                    parent_name, _ = name.rsplit(".", 1)
                    parent_module = self.clip_encoder.get_submodule(parent_name)
                    setattr(parent_module, name.split(".")[-1], new_module)

        unmatch_lora_keys = list(lora_state_dict.keys())
        if len(unmatch_lora_keys) > 0:
            logger.info(f"unmatch_lora_keys: {unmatch_lora_keys}")

    def unload_lora(self):
        self.unload_te_lora()
        self.is_load_lora = False

    def unload_te_lora(self):
        for _, module in self.clip_encoder.named_modules():
            if isinstance(module, LoRAModel):
                module.remove_lora_layers()

    # @statistical_runtime("prepare the text encoder model")
    def prepare_models(self, mixin):
        if not self._initiated:
            self.initiate()
        elif self.is_cleared:
            load_parameters(
                self.t5_encoder,
                load_state_dict(
                    os.path.join(
                        self.base["root"],
                        "text_encoder_2",   # "text_encoder_2_full_weight",
                        "model.safetensors",
                    )
                ),
            )
            self.is_cleared = False

        self.load_lora(mixin)

        for model in self.models:
            model.to(self.device)

    def clear_gpu_model(self):
        if self.is_cleared:
            return

        if self._initiated:
            try:
                zero_parameters(self.t5_encoder)
            finally:
                for model in self.models:
                    model.to("cpu")
                self.is_cleared = True

    def get_clip_embeds(self, prompt):
        return self.clip_encoder(
            self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.clip_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.clip_encoder.device),
            output_hidden_states=False,
        ).pooler_output

    def get_t5_embeds(self, fragments, max_sequence_length):
        eos_id = self.t5_tokenizer.eos_token_id
        pad_id = self.t5_tokenizer.pad_token_id

        ground_embs = self.t5_encoder(
            self.t5_tokenizer(
                "",
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids.to(self.t5_encoder.device),
            output_hidden_states=False,
        )[0]

        ground_embs = ground_embs.to(self.t5_encoder.device)
        # tokenize prompt and calculate weights
        input_ids = []
        input_weights = []
        for f in fragments:
            f_ids = self.t5_tokenizer(
                f[0], add_special_tokens=False, padding=False, truncation=False
            ).input_ids
            input_ids += f_ids
            input_weights += [[f[1]]] * len(f_ids)

        # truncate ids
        input_ids = input_ids[: max_sequence_length - 1]
        input_weights = input_weights[: max_sequence_length - 1]

        # append eos id
        input_ids += [eos_id]
        input_weights += [[1.0]]

        # append pad id
        padding_length = max_sequence_length - len(input_ids)
        input_ids += [pad_id] * padding_length
        input_weights += [[1.0]] * padding_length

        # convert to tensor
        input_ids = torch.tensor(
            [input_ids], dtype=torch.int64, device=self.t5_encoder.device
        )
        input_weights = torch.tensor(
            input_weights, dtype=self.t5_encoder.dtype, device=self.t5_encoder.device
        )

        # encode prompt
        embs = self.t5_encoder(
            input_ids,
            output_hidden_states=False,
        )[0]

        # apply prompt weight
        embs = ground_embs + (embs - ground_embs) * input_weights

        return embs

    @statistical_runtime("generate prompt embeddings")
    @torch.no_grad
    def __call__(self, opt):
        prompts, mixin, batch_size, family = (
            opt.prompt,
            opt.mixin,
            opt.n_samples,
            opt.family,
        )
        if opt.enable_tile and opt.new_tile:
            prompts = prompts.split("##")
        else:
            prompts = [prompts]

        max_sequence_length = 512
        if family == "flux.1schnell":
            max_sequence_length = 256

        self.prepare_models(mixin)

        prompt_embeds_, pooled_prompt_embeds_, text_ids_ = [], [], []
        for prompt in prompts:
            prompt_embeds = self.get_t5_embeds(
                parse_prompt(prompt, self.multiplier_base, self.max_exponent),
                max_sequence_length,
            ).repeat(batch_size, 1, 1)

            pooled_prompt_embeds = self.get_clip_embeds(prompt).repeat(batch_size, 1)

            text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
                prompt_embeds.device, dtype=prompt_embeds.dtype
            )
            prompt_embeds_.append(prompt_embeds)
            pooled_prompt_embeds_.append(pooled_prompt_embeds)
            text_ids_.append(text_ids)

        prompt_embeds, pooled_prompt_embeds, text_ids = (
            torch.cat(prompt_embeds_),
            torch.cat(pooled_prompt_embeds_),
            torch.cat(text_ids_),
        )

        return {
            "prompt_embeds": prompt_embeds.to("cpu"),
            "pooled_prompt_embeds": pooled_prompt_embeds.to("cpu"),
            "text_ids": text_ids.to("cpu"),
        }

if __name__ == '__main__':
    device = 'cuda'
    statistical_runtime.reset_collection()
    DEFAULT_INDEX = {
        "flux": {
            "root": "/ssd/models/flux.1-dev",
        },
    }
    text_encoder = PromptEncoder(
        base=DEFAULT_INDEX["flux"],
        device=device,
        dtype=torch.bfloat16,
    )
    from argparse import Namespace
    opt = Namespace(
        prompt = '', mixin = [], n_samples = 1, family = 'flux1.dev',
        enable_tile = False, new_tile = False, 
    )
    res = text_encoder(opt)
    for k,v in res.items():
        print(k, v.shape)

    # prompt_embeds torch.Size([1, 512, 4096])
    # pooled_prompt_embeds torch.Size([1, 768])
    # text_ids torch.Size([1, 512, 3])

    opt = Namespace(
        prompt = '1##2##3##4##5', mixin = [], n_samples = 1, family = 'flux1.dev',
        enable_tile = True, new_tile = True, 
    )
    res = text_encoder(opt)
    for k,v in res.items():
        print(k, v.shape)
    # prompt_embeds torch.Size([5, 512, 4096])
    # pooled_prompt_embeds torch.Size([5, 768])
    # text_ids torch.Size([5, 512, 3]



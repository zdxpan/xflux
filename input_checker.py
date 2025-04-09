import os
import re
from abc import abstractmethod, ABC
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
# from code_status import InvalidFamilyBadRequest, InvalidSamplerBadRequest, InvalidFileBadRequest, \
#     LimitImageSizeBadRequest, InvalidFileIndexBadRequest, InvalidMixinBadRequest, InvalidControlBadRequest, \
#     InvalidNSamplesBadRequest, InvalidResizerBadRequest, InvalidSeedBadRequest, InvalidImageOutputEncodingBadRequest, \
#     InvalidImageSavePathBadRequest, InvalidImageOutputFormatBadRequest, AnnotateBadRequest, InvalidOutputNameBadRequest, \
#     InvalidOutputDirBadRequest
from tensor_util import (
    create_np_mask, create_random_tensors, normalize_size,slerp,has_transparency,
    resize_image,
    resize_numpy_image_long, seed_to_int, RESIZER as resizer_names
)

from log import logger
from werkzeug.datastructures import MultiDict
from flux_pipe import FluxNormalInput


BASE_VALIDATORS = (str, int, float, bool)
MAX_ALLOWED_CONTROL_NET = 2

@dataclass
class OriginInput:
    '''
    定义来自用户端的输入
    '''
    task: str  # task是哪个
    args: dict[str, Any]  # 参数
    # processor: Processor  # 能力接口层

class FluxParameterCheckNormalizer:
    def __init__(self, origin_input=None):
        self.origin_input = origin_input

    def normalize(self, ori_args) -> FluxNormalInput:
        # 先获取全部的默认参数
        default_params = FluxNormalInput(task='generate')
        default_template: dict = asdict(default_params)
        validators = self.validate_generate_task(default_template)
        origin_input = ori_args if isinstance(ori_args, dict) else asdict(ori_args)
        # run_validators(define_validators, self.origin_input, result)
        result = default_template
        for k_name, validator in validators:
            if validator in BASE_VALIDATORS:
                if k_name in origin_input:
                    result[k_name] = validator(origin_input.get(k_name))
            else:
                if isinstance(validator, BaseValidator):
                    validator(k_name, origin_input, result)
                else:
                    validator(origin_input, result)

        normal_input = FluxNormalInput(**result)
        normal_input.normal_width, normal_input.normal_height = normalize_size(normal_input.width, normal_input.height,
                                                                               4096)
        return normal_input

    def validate_generate_task(self, result: dict):
        '''
        定义参数先后处理顺序并执行验证
        '''
        define_validators = (
            ('trace_id', str),
            ('req_id', str),
            ('family', str),
            ('sampler', str),
            ('base_shift', NumberValidator(min_v=0.0, max_v=10.0)),
            ('max_shift', NumberValidator(min_v=0.0, max_v=10.0)),
            ('file', validate_file),
            ('init_model', str),
            ('use_scaled', bool),
            ('image', FileIndexValidator()),
            ('mask', FileIndexValidator()),
            ('enable_tile', bool),
            ('new_tile', bool),
            ('tile_percent', validate_tile_percent),
            # ('control', validate_control),
            ('width', NumberValidator(min_v=0, max_v=4096)),
            ('height', NumberValidator(min_v=0, max_v=4096)),
            ('n_samples', validate_n_samples),
            ('clip_skip', NumberValidator(min_v=1, max_v=4)),
            ('resizer', validate_resizer),
            ('steps', NumberValidator(min_v=1, max_v=1000)),
            ('strength', NumberValidator(min_v=0.0, max_v=1.0)),
            ('scale', NumberValidator(min_v=0.0, max_v=16.0)),
            ('prompt', str),
            ('mask_blur', NumberValidator(min_v=0.0, max_v=50.0)),
            ('seed', SeedValidator()),
            ('variant', NumberValidator(min_v=0.0, max_v=1.0)),
            ('variant_seed', VariantSeedValidator()),
            ('image_output_encoding', validate_image_output_encoding),
            ('image_save_path', validate_image_save_path),
            ('image_output_format', validate_image_output_format),
            ('use_zero_init', bool),
            ('do_true_cfg', bool),
            ('cfg_skip', NumberValidator(min_v=0.0, max_v=10.0)),            
        )
        return define_validators
    
    def validate_annotate_task(self, result: dict):
        '''
        定义参数先后处理顺序并执行验证
        '''
        define_validators = (
            ('trace_id', str),
            ('req_id', str),
            ('family', str),
            ('file', validate_file),
            ('annotate', validate_annotate),
            ('image_output_encoding', validate_image_output_encoding),
            ('image_save_path', validate_image_save_path),
            ('image_output_format', validate_image_output_format),
        )
        run_validators(define_validators, self.origin_input, result)
        return result

    def validate_convert_task(self, result: dict):
        '''
        定义参数先后处理顺序并执行验证
        '''
        define_validators = (
            ('trace_id', str),
            ('req_id', str),
            ('family', str),
            ('file', validate_file),
            ('output_name', validate_output_name),
            ('output_dir', validate_output_dir),
        )
        run_validators(define_validators, self.origin_input, result)
        return result


def convert_args_to_dict(args: MultiDict):
    '''
    将原始的web框架的请求对象转换为python内置对象
    '''
    d = {}
    for k in args:
        if k in ('file', 'annotate', 'mixin', 'control'):
            d[k] = args.getlist(k) if hasattr(args, 'getlist') else args.getall(k)
        else:
            d[k] = args.get(k)
    return d


def pad_or_truncate(some_list, target_len, dummy):
    return some_list[:target_len] + [dummy] * (target_len - len(some_list))


def validate_family(args: dict):
    # FAMILIES
    family = args.get("family", "flux.1dev")
    return family


class BaseValidator(ABC):

    @abstractmethod
    def __call__(self, key_name, origin_input, result):
        raise NotImplementedError


class NumberValidator(BaseValidator):
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v
        assert type(self.min_v) == type(self.max_v)
        self.check_type = type(self.min_v)

    def __call__(self, key_name, origin_input, result):
        if key_name not in origin_input:
            return
        value = self.check_type(origin_input.get(key_name))
        if value < self.min_v or value > self.max_v:
            raise ValueError(f'{key_name} must be in [{self.min_v}, {self.max_v}]')
        result[key_name] = value


def validate_file(origin_input, result):
    file = origin_input.get("file", [])
    for idx, f in enumerate(file):
        if isinstance(f, bytes):
            file[idx] = Image.open(BytesIO(f))
    task = result['task']
    if task in ["convert"] and len(file) == 0:
        raise "Invalid param file, should not be empty."

    for f in file:
        if isinstance(f, str):
            if not os.path.exists(f):
                raise f"Invalid param file, {f} does not exist or inaccessible."
        elif isinstance(f, Image.Image):
            w, h = f.size
            if w * h > 4096 * 4096:
                raise f"Invalid param file, image resolution should be less than 4096 * 4096 but got {w} * {h}."
    result["file"] = file


def validate_annotate(origin_input, result):
    annotate = origin_input.get("annotate", [])
    file = result['file']
    try:
        assert (
                len(annotate) <= origin_input.processor.MAX_ALLOWED_ANNOTATE
        ), f"annotate num should be less than {origin_input.processor.MAX_ALLOWED_ANNOTATE}"
        normalized = []
        for params in annotate:
            params = list(map(str.strip, params.split(",")))
            params = pad_or_truncate(params, 2, "")
            index = int(params[0]) if params[0] else 0
            assert 1 <= index + 1 <= len(file), f"annotate index out of range"
            annotator = params[1]
            assert origin_input.processor.annotator.exist(annotator), f"{annotator} not exists"
            normalized.append([index, annotator])
        annotate = normalized
    except AssertionError as e:
        raise f"Invalid param annotate, {e}."
    except:
        raise f"Invalid param annotate {annotate}."
    result["annotate"] = annotate


class FileIndexValidator(BaseValidator):
    def __call__(self, key_name, origin_input, result):
        index = origin_input.get(key_name, None)
        file = result['file']
        if index is not None:
            index = int(index)
            if not 0 <= index < len(file):
                raise f"Invalid param {key_name}, out of index."
        result[key_name] = index


def validate_tile_percent(origin_input, result):
    tile_percent = float(origin_input.get("tile_percent", 0.4))
    enable_tile = result['enable_tile']
    if enable_tile and tile_percent < 0.0 or tile_percent > 1.0:
        raise ValueError("tile_percent must be in [0.0, 1.0]")
    result["tile_percent"] = tile_percent


def validate_control(origin_input, result):
    control = origin_input.get("control", [])
    try:
        assert (
                len(control) <= MAX_ALLOWED_CONTROL_NET
        ), f"control exceed the limit of {MAX_ALLOWED_CONTROL_NET}"
        family = result["family"]
        file = result['file']
        normalized = []
        for params in control:
            params = list(map(str.strip, params.split(",")))
            params = pad_or_truncate(params, 5, "")
            indexes = [int(index.strip()) for index in params[0].split("|")]
            for index in indexes:
                assert (
                        1 <= int(index) + 1 <= len(file)
                ), "control index out of range"
            # index = int(params[0]) if params[0] else 0
            # assert 1 <= index + 1 <= len(file)
            annotator = params[1]
            assert origin_input.processor.annotator.exist(annotator), "annotator not exist"
            control_name = params[2]
            if family == "flux.1dev":
                assert origin_input.processor.flux_dev.exist(
                    control=control_name
                ), f"{control_name} not exist in {family}"
            control_scale = float(params[3]) if params[3] else 1.0
            assert (
                    control_scale >= 0 and control_scale <= 2
            ), f"control_scale out of range, but got {control_scale}"
            control_range = float(params[4]) if params[4] else 1.0
            assert (
                    control_range >= 0 and control_range <= 1
            ), f"control_range out of range, but got {control_range}"
            normalized.append(
                [indexes, annotator, control_name, control_scale, control_range]
            )
        control = normalized
    except AssertionError as e:
        raise f"Invalid param control, {e}."
    except:
        raise "Invalid param control."
    result["control"] = control


def validate_n_samples(origin_input, result):
    n_samples = int(origin_input.get("n_samples", 1))
    task = result['task']
    width = result['width']
    height = result['height']
    enable_tile = result['enable_tile']
    threshold = 2048 * 2048 if task == "generate" else 1024 * 1024
    m_samples = int(threshold / min(width * height, threshold))
    if enable_tile and n_samples > 1:
        raise f"Invalid param n_samples, if enable_tile n_samples must be 1"
    if n_samples < 1 or n_samples > m_samples:
        raise f"Invalid param n_samples, should be integer and in range (1 to {m_samples}) for given output size"
    result["n_samples"] = n_samples


def validate_resizer(origin_input, result):
    resizer = str(origin_input.get("resizer", "crop_middle"))
    logger.debug(f"resizer_names: {resizer_names}, resizer is : {resizer}")
    if not resizer in resizer_names:
        raise "Invalid param resizer, resizer not exists."
    result["resizer"] = resizer


class SeedValidator(BaseValidator):
    def __call__(self, key_name, origin_input, result):
        length = result['n_samples']
        seed = origin_input.get(key_name, "")
        try:
            if isinstance(seed, int):
                seed = str(seed)
            seed = list(map(str.strip, seed.split(",")))
            assert len(seed) <= length
            for i in range(len(seed)):
                seed[i] = int(seed[i]) if seed[i] else -1
                assert not (seed[i] < -1 or seed[i] > 2 ** 32 - 1)
        except:
            raise f"Invalid param {key_name}"
        result[key_name] = seed_to_int(seed, length)


class VariantSeedValidator(SeedValidator):
    def __call__(self, key_name, origin_input, result):
        super().__call__(key_name, origin_input, result)
        n_samples = result['n_samples']
        variant = result['variant']
        variant_seed = result['variant_seed']
        variant_seed = (
            [0] * n_samples
            if variant == 0
            else (seed_to_int(variant_seed, n_samples))
        )
        result[key_name] = variant_seed


def validate_output_name(origin_input, result):
    output_name = origin_input.get("output_name", None)
    if output_name is not None:
        if re.search("^[a-zA-Z\d\.\_\-]+$", output_name) is None:
            raise "Invalid param output_name, should be combination of [letter|number|_|-|.]"
    result["output_name"] = output_name


def validate_output_dir(origin_input, result):
    output_dir = origin_input.get("output_dir", origin_input.processor.extra_dir)
    if not os.path.isdir(output_dir):
        raise f"Invalid param output_dir, {output_dir} is not a valid directory"
    result["output_dir"] = output_dir


def validate_image_output_encoding(origin_input, result):
    image_output_encoding = origin_input.get(
        "image_output_encoding", "base64").strip()
    if image_output_encoding not in ("base64", "oss_path"):
        raise f"param error: image_output_encoding({image_output_encoding}) not support"
    result["image_output_encoding"] = image_output_encoding


def validate_image_save_path(origin_input, result):
    image_save_path = origin_input.get("image_save_path", "").split(",")
    image_save_path = list(map(str.strip, image_save_path))
    task = result['task']
    n_samples = result['n_samples']
    # only generate annotate ues n_samples
    # upscale task use len(file)
    batch_size = (
        n_samples
        if task in ("generate", "annotate")
        else len(result["file"])
    )
    if len(image_save_path) > batch_size:
        raise f'param error: image_save_path({origin_input["image_save_path"]}) must less than {batch_size}'
    result["image_save_path"] = pad_or_truncate(image_save_path, batch_size, "")


def validate_image_output_format(origin_input, result):
    task = result['task']
    if task == "generate":
        image_output_format = origin_input.get(
            "image_output_format", "png"
        ).split(",")
        image_output_format = list(map(str.strip, image_output_format))
        for image_format in image_output_format:
            if image_format not in ("png", "jpeg"):
                raise f"param error: image_output_format({image_format}) not support"
        result["image_output_format"] = image_output_format
    else:
        # only generate final results that support multiple formats
        result["image_output_format"] = ["png"]



def run_validators(define_validators, origin_input, result: dict):
    for k_name, validator in define_validators:
        if validator in BASE_VALIDATORS:
            if k_name in origin_input:
                result[k_name] = validator(origin_input.get(k_name))
        else:
            if isinstance(validator, BaseValidator):
                validator(k_name, origin_input, result)
            else:
                validator(origin_input, result)

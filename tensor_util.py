import math

import numpy as np
import cv2
import torch
from PIL import ImageFilter, Image
import hashlib
import json
import os
import random

import numpy as np
import zstandard
from PIL import Image
from log import logger, statistical_runtime
from safetensors.torch import load_file, save_file
from torch import load

RESIZER = [
    "crop_middle",
    "crop_start",
    "crop_end",
    "extend_both",
    "extend_start",
    "extend_end",
    "center",
]

def calculate_shift(
        image_seq_len,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def pack_latents(latents, ph=2, pw=2):
    b, c, h, w = latents.shape
    h, w = h // ph, w // pw
    # (b, c, h * ph, w * pw) -> (b, c, h, ph, w, pw)
    latents = latents.view(b, c, h, ph, w, pw)
    # (b, c, h, ph, w, ph) -> (b, h, w, c, ph, pw)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    # (b, h, w, c, ph, pw) - > (b, h * w, c * ph * pw)
    latents = latents.reshape(b, h * w, c * ph * pw)
    return latents


def prepare_image_ids(height, width, batch_size):
    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width // 2)[None, :]
    h, w, c = ids.shape
    ids = ids.reshape(h * w, c)
    ids = ids[None].repeat_interleave(batch_size, dim=0)
    return ids


def unpack_latents(latents, h, w, c, ph=2, pw=2):
    b, _, _ = latents.shape
    h, w = h // ph, w // pw
    # (b, h * w, c * ph * pw) -> (b, h, w, c, ph, pw)
    latents = latents.view(b, h, w, c, ph, pw)
    # (b, h, w, c, ph, pw) -> (b, c, h, ph, w, ph)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    # (b, c, h, ph, w, ph) -> (b, c, h * ph, w * ph)
    latents = latents.reshape(b, c, h * ph, w * ph)
    return latents


def create_random_tensors(shape, seed):
    x = []
    for s in seed:
        torch.manual_seed(s)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        x.append(torch.randn([1] + shape, device="cpu"))
    return torch.cat(x)


def normalize_size(width, height, max_size=2048):
    # the max area of an image should less than max_size * max_size
    coefficient = float(width * height) / (max_size * max_size)
    # resize
    if coefficient > 1:
        width, height = map(lambda x: int(x / math.sqrt(coefficient)), (width, height))
    # normalize to integer multiple of 16
    width, height = map(lambda x: max(math.ceil(x / 16) * 16, 512), (width, height))
    return width, height


def create_np_mask(image, blur=0.0):
    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur * 100))
    if (
            len(image.getbands()) == 4
    ):  # alpha mask, transparent pixels are used as edit area
        # convert into a black/white mask
        mask = Image.new(mode="L", size=image.size, color=255)
        mask.putdata(image.getdata(band=3))
        # logger.debug('writing the mask to mask.png')
        # mask.save('mask.png')
        mask = np.array(mask)
    else:  # black & white mask, white pixels are used as edit area
        mask = image.convert("L")
        # logger.debug('writing the mask to mask.png')
        # mask.save('mask.png')
        mask = 255 - np.array(mask)
    # print('mask', mask.shape) # [1024, 1024]
    # Image.fromarray(mask).save('/home/dell/workspace/js-sd-svc/3_mask.png')
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None][None]
    mask = torch.from_numpy(mask)
    return mask


def concatenate_rowwise(matrix_list, overlap_size):
    """
    将一组矩阵按行方向拼接，重叠部分进行线性插值
    
    参数:
        matrix_list: 包含多个矩阵的列表，所有矩阵必须具有相同的设备和数据类型
        overlap_size: 相邻矩阵之间的重叠列数
        
    返回:
        拼接后的矩阵
    """
    # 获取第一个矩阵的设备类型和数据类型作为基准
    device, dtype = matrix_list[0].device, matrix_list[0].dtype
    
    num_matrices = len(matrix_list)
    rows = matrix_list[0].shape[-2]  # 矩阵行数
    cols = matrix_list[0].shape[-1]  # 矩阵列数
    
    # 如果只有一个矩阵，直接返回
    if num_matrices == 1:
        return matrix_list[0]
    
    # 计算结果矩阵的总宽度
    total_width = num_matrices * (cols - overlap_size // 2)
    
    # 初始化结果矩阵
    result_matrix = torch.zeros(matrix_list[0].shape[:-2] + (rows, total_width)).to(device, dtype)
    
    current_col_position = 0
    
    # 处理第一个矩阵的非重叠部分
    non_overlap_cols = cols - overlap_size
    result_matrix[..., :, :non_overlap_cols] = matrix_list[0][..., :, :non_overlap_cols]
    current_col_position += non_overlap_cols
    
    # 创建重叠区域的权重
    weights_prev = torch.linspace(1, 0, steps=overlap_size).view(1, -1).to(device, dtype)  # 前一个矩阵权重
    weights_next = torch.linspace(0, 1, steps=overlap_size).view(1, -1).to(device, dtype)  # 后一个矩阵权重
    
    for matrix_idx in range(1, num_matrices):
        current_cols = matrix_list[matrix_idx].shape[-1]
        
        # 获取相邻矩阵的重叠区域
        prev_matrix_overlap = matrix_list[matrix_idx - 1][..., :, -overlap_size:]
        current_matrix_overlap = matrix_list[matrix_idx][..., :, :overlap_size]
        
        # 计算重叠区域的插值
        interpolated_overlap = weights_prev * prev_matrix_overlap + weights_next * current_matrix_overlap
        result_matrix[..., :, current_col_position:current_col_position + overlap_size] = interpolated_overlap
        current_col_position += overlap_size
        
        # 处理当前矩阵的非重叠部分
        if matrix_idx < num_matrices - 1:
            # 中间矩阵: 处理剩余的非重叠部分
            remaining_cols = current_cols - overlap_size * 2
            result_matrix[..., :, current_col_position:current_col_position + remaining_cols] = \
                matrix_list[matrix_idx][..., :, overlap_size:]
            current_col_position += remaining_cols
        else:
            # 最后一个矩阵: 处理剩余的所有列
            result_matrix[..., :, current_col_position:] = matrix_list[matrix_idx][..., :, overlap_size:]
    
    return result_matrix

def concatenate_rowwise_og(matrix_list, overlap):
    device, dtype = matrix_list[0].device, matrix_list[0].dtype

    num, rows, cols = len(matrix_list), matrix_list[0].shape[-2], matrix_list[0].shape[-1]
    if num == 1:
        return matrix_list[0]

    result_width = num * (cols - overlap // 2)  # all width

    result_row = torch.zeros(matrix_list[0].shape[:-2] + (rows, result_width)).to(device, dtype)

    current_col = 0
    result_row[..., :, : cols - overlap] = matrix_list[0][..., :, : cols - overlap]
    current_col += cols - overlap

    weights_A = torch.linspace(1, 0, steps=overlap).view(1, -1).to(device, dtype)
    weights_B = torch.linspace(0, 1, steps=overlap).view(1, -1).to(device, dtype)

    for j in range(1, num):
        cols = matrix_list[j].shape[-1]

        overlap_region_A = matrix_list[j - 1][..., :, -overlap:]
        overlap_region_B = matrix_list[j][..., :, :overlap]

        interpolated = weights_A * overlap_region_A + weights_B * overlap_region_B
        result_row[..., :, current_col: current_col + overlap] = interpolated
        current_col += overlap

        if j < num - 1:
            result_row[:, current_col: current_col + (cols - overlap)] = matrix_list[j][
                                                                      ..., :, overlap:
                                                                      ]
            current_col += cols - overlap * 2
        else:
            result_row[..., :, current_col:] = matrix_list[j][..., :, overlap:]

    return result_row


def concatenate_columnwise(matrices_col, overlap):
    device, dtype = (
        matrices_col[0].device,
        matrices_col[0].dtype
    )

    k, m, n = (
        len(matrices_col),
        matrices_col[0].shape[-2],
        matrices_col[0].shape[-1],
    )

    if k == 1:
        return matrices_col[0]

    result_height = k * (m - overlap // 2)
    result_col = torch.zeros(matrices_col[0].shape[:-2] + (result_height, n)).to(device, dtype)

    current_row = 0
    result_col[..., : m - overlap, :] = matrices_col[0][..., : m - overlap, :]
    current_row += m - overlap

    weights_A = torch.linspace(1, 0, steps=overlap).view(-1, 1).to(device, dtype)
    weights_B = torch.linspace(0, 1, steps=overlap).view(-1, 1).to(device, dtype)

    for i in range(1, k):
        m = matrices_col[i].shape[-2]

        overlap_region_A = matrices_col[i - 1][..., -overlap:, :]
        overlap_region_B = matrices_col[i][..., :overlap, :]

        interpolated = weights_A * overlap_region_A + weights_B * overlap_region_B
        result_col[..., current_row: current_row + overlap, :] = interpolated
        current_row += overlap

        if i < k - 1:
            result_col[..., current_row: current_row + (m - overlap), :] = (
                matrices_col[i][..., overlap:, :]
            )
            current_row += m - overlap * 2
        else:
            result_col[..., current_row:, :] = matrices_col[i][..., overlap:, :]

    return result_col


# ---  model utils ------

def crop_image(image, width, height, position="middle"):
    w, h = image.size
    if position == "start":
        box = (0, 0, width, height)
    elif position == "end":
        box = (w - width, h - height, w, h)
    else:
        box = ((w - width) // 2, (h - height) // 2, (w + width) // 2, (h + height) // 2)
    image = image.crop(box)
    return image


def extend_image(image, width, height, position="both"):
    w, h = image.size
    if position == "start":
        box = (width - w, height - h)
    elif position == "end":
        box = (0, 0)
    else:
        box = ((width - w) // 2, (height - h) // 2)
    extended = Image.new("RGBA", (width, height))
    extended.paste(image, box)
    return extended


def center_image(image, width, height, position=None):
    w, h = image.size
    background = Image.new("RGBA", (width, height))
    background.paste(image, ((width - w) // 2, (height - h) // 2))
    return background


def pad_image_with_offcut(image, width, height):
    w, h = image.size
    target = Image.new(image.mode, (width, height))
    target.paste(image, (0, 0))
    line_v = target.crop((w - 1, 0, w, height))
    for x in range(w, width):
        target.paste(line_v, (x, 0))
    line_h = target.crop((0, h - 1, width, h))
    for y in range(h, height):
        target.paste(line_h, (0, y))
    return target

def resize_numpy_image_long(image, resize_long_edge=768):
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def resize_image(image, width, height, esrgan=None, resizer=RESIZER[0]):
    if image:
        resizer = resizer.split("_")  # {method}_{position}
        resizer = resizer + [""] * (2 - len(resizer))
        method, position = resizer
        w, h = image.size

        if esrgan is not None and (width / w) == (height / h) and width / w >= 2:
            factor = width / w
            image = esrgan(image, factor)
        elif w != width or h != height:
            if method == "crop":
                factor = min(w / width, h / height)
                processor = crop_image
            elif method == "extend":
                factor = max(w / width, h / height)
                processor = extend_image
            elif method == "center":
                factor = 1
                processor = center_image
            image = image.resize(
                (int(w / factor), int(h / factor)), resample=Image.Resampling.BICUBIC
            )
            image = processor(image, width, height, position)
            w, h = image.size
        if w % 16 != 0 or h % 16 != 0:
            image = pad_image_with_offcut(
                image, math.ceil(w / 16) * 16, math.ceil(h / 16) * 16
            )
    return image


def has_transparency(image):
    if image is not None and image.mode == "RGBA":
        extrema = image.getextrema()
        if extrema[3][0] < 255:
            return True
    return False


def create_np_image(image, normalize=True):
    # set invisible pixels to black
    if has_transparency(image):
        background = Image.new("RGB", image.size)
        background.paste(image, mask=image)
        image = background
    else:
        image = image.convert("RGB")
    # logger.debug('writing the image to img.png')
    # image.save('img.png')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    if normalize:
        image = 2.0 * image - 1.0
    return image


def slerp(t, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD=0.9995):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2)

    return v2


def seed_to_int(seed, batch_size=1):
    seed = seed[:batch_size] + [-1] * (batch_size - len(seed))
    for i in range(len(seed)):
        if seed[i] == -1:
            random.seed()
            seed[i] = random.randint(0, 2 ** 32 - 1)
    return seed


def load_parameters(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
    """
    load model parameters
    """
    for prefix, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(recurse=False):
            key = f"{prefix}.{parameter_name}" if prefix else parameter_name
            if key in state_dict:
                tensor = state_dict.pop(key)
                module.register_parameter(
                    parameter_name, torch.nn.Parameter(tensor, requires_grad=False)
                )
    if len(state_dict):
        logger.warn(
            f"{model.__class__.__name__} load parameters remained key is {state_dict.keys()}"
        )


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


def calculate_hash(state_dict):
    keys = list(state_dict.keys())
    keys.sort()

    length = len(keys)
    if length == 0:
        return "0" * 32

    density = min(length, 10)
    step = length / density
    offset = step / density

    sample = bytes(0)
    for i in range(1, density + 1):
        index = int(i * step - i * offset)
        tensor = state_dict[keys[index]]
        sample += tensor.data.cpu().half().numpy().tobytes()

    return hashlib.md5(sample).hexdigest()


def save_lora(
        state_dict, output_dir, family, dtype=torch.float16, name=None, hash_=None
):
    if hash_ is None:
        hash_ = calculate_hash(state_dict)

    if name is None:
        name = hash_

    output_dir = os.path.join(output_dir, name)
    with statistical_runtime(f'Create "{output_dir}"'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    model_filename = (
        "model.fp16.safetensors" if dtype == torch.float16 else "model.safetensors"
    )
    model_pathname = os.path.join(output_dir, model_filename)
    with statistical_runtime(f'Save "{model_pathname}"'):
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.to(dtype)
        save_file(state_dict, model_pathname, {"format": "pt"})

    meta_pathname = os.path.join(output_dir, "_meta.json")
    with statistical_runtime(f'Save "{meta_pathname}"'):
        meta = {
            "hash": hash_,
            "name": name,
            "family": family,
            "type": "lora",
            "main": model_filename,
        }
        with open(meta_pathname, "w") as f:
            f.write(json.dumps(meta, indent=4))
    return meta

if __name__ == "__main__":
    matrices = [
        [torch.ones((4, 5)), torch.ones((4, 5))],
        [torch.ones((4, 5)), torch.ones((4, 5))],
    ]
    overlap = 2

    rowwise_results = [concatenate_rowwise(row, overlap) for row in matrices]

    final_result = concatenate_columnwise(rowwise_results, overlap)

    print(final_result)
    print(final_result.shape)

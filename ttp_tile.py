import torch

import numpy as np
from PIL import Image
from typing import Tuple, List, Union
from torch import Tensor

def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor
def tensor2pil(image):
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))


class BBox:
    def __init__(self, x:int, y:int, w:int, h:int,
        left_lap=None, up_lap=None, right_lap = None, down_lap = None, scale = 1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)
        self.latent_slicer = slice(None), slice(None), slice(y // scale, (y+h) // scale ), slice(x // scale, (x+w) // scale)
        self.left_lap = self.related_box(left_lap)  # get final overlap box~
        self.up_lap = self.related_box(up_lap)
        self.right_lap = self.related_box(right_lap)
        self.down_lap = self.related_box(down_lap)
        self.left_slicer = self.sub_slicer(self.left_lap)
        self.left_slicer_lat = self.sub_slicer(self.left_lap, scale=scale, latent_mode=True)
        self.up_slicer = self.sub_slicer(self.up_lap)
        self.up_slicer_lat = self.sub_slicer(self.up_lap, scale=scale, latent_mode=True)
        self.right_slicer = self.sub_slicer(self.right_lap)
        self.right_slicer_lat = self.sub_slicer(self.right_lap, scale=scale, latent_mode=True)
        self.down_slicer = self.sub_slicer(self.down_lap)
        self.down_slicer_lat = self.sub_slicer(self.down_lap, scale=scale, latent_mode=True)
        self.prompt = ''
        self.image = None
        self.cond = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.text_ids = None

    def related_box(self, box):
        if box is None:
            return box
        x,y = self.x + box[0], self.y + box[1]
        w,h = x + box[2], y + box[3]
        return [x, y, w, h]
    @staticmethod
    def sub_slicer(box, scale = 1, latent_mode = False):
        if box is not None:
            scale = scale if latent_mode else 1
            x, y = box[0] // scale, box[1] // scale
            x_w, y_h = box[2] // scale, box[3] // scale
            return slice(None), slice(None), slice(y, y_h), slice(x, x_w)
            # slice(None), slice(None), slice(y, y+h), slice(x, x+w)
        else:
            return None

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

def split_4bboxes(w:int, h:int, overlap:int=16,device = 'cuda' , factor=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    # TODO ,gussain weight
    cols = 2
    rows = 2
    tile_w = w // 2 + overlap
    tile_h = h // 2 + overlap
    bbox_1 = BBox(
        0,          0,        tile_w, tile_h,  scale=factor,
        left_lap=None, up_lap=None, 
        right_lap=[tile_w - overlap, 0, overlap, tile_h], 
        down_lap=[0, tile_h  - overlap, tile_w, overlap], 
    )
    bbox_2 = BBox(
        w - tile_w, 0, tile_w, tile_h, scale=factor,
        left_lap=[0, 0, overlap, tile_h], 
        up_lap=None, right_lap=None, 
        down_lap=[0, tile_h - overlap, tile_w, overlap]
    )
    # bbox_3 = BBox(0,          h-tile_h, tile_w, tile_h, scale=factor)
    # 左下角bbox
    bbox_3 = BBox(
        0, h - tile_h, tile_w, tile_h, scale=factor,
        left_lap=None, 
        up_lap=[0, 0, tile_w, overlap], 
        right_lap=[tile_w - overlap, 0, overlap, tile_h], 
        down_lap=None
    )
    # bbox_4 = BBox(w - tile_w, h-tile_h, tile_w, tile_h, scale=factor)
    bbox_4 = BBox(
        w - tile_w, h - tile_h, tile_w, tile_h, scale=factor,
        left_lap=[0, 0, overlap, tile_h], 
        up_lap=[0, 0, tile_w, overlap], 
        right_lap=None, down_lap=None
    )
    bbox_list = [bbox_1, bbox_2, bbox_3, bbox_4]
    weight = torch.ones((1, 1, h, w), device=device, dtype=torch.float32)
    latent_weight = torch.ones((1, 1, h//factor, w//factor), device=device, dtype=torch.float32)

    dtype = torch.float32
    
    weights_right = torch.linspace(0.5, 0, steps=overlap, device=device, dtype=dtype).view(1, 1, overlap).expand(1, tile_h, overlap)  # [1, 1, overlap]  水平方向
    weights_left = torch.linspace(0, 0.5, steps=overlap, device=device, dtype=dtype).view(1, 1, overlap).expand(1, tile_h, overlap)  # [1, 1, overlap]  水平方向
    weights_up = torch.linspace(0.5, 0, steps=overlap, device=device, dtype=dtype).view(1, overlap, 1).expand(1, overlap, tile_w)  # [1, overlap, 1]
    weights_down = torch.linspace(0, 0.5, steps=overlap, device=device, dtype=dtype).view(1, overlap, 1).expand(1, overlap, tile_w)

    weights_right_lt = torch.linspace(0.5, 0, steps=overlap//factor, device=device, dtype=dtype).view(1, 1, overlap//factor).expand(1, tile_h//factor, overlap//factor)  # [1, 1, overlap]  水平方向
    weights_left_lt = torch.linspace(0, 0.5, steps=overlap//factor, device=device, dtype=dtype).view(1, 1, overlap//factor).expand(1, tile_h//factor, overlap//factor)  # [1, 1, overlap]  水平方向
    weights_up_lt = torch.linspace(0.5, 0, steps=overlap//factor, device=device, dtype=dtype).view(1, overlap//factor, 1).expand(1, overlap//factor, tile_w//factor)  # [1, overlap, 1]
    weights_down_lt = torch.linspace(0, 0.5, steps=overlap//factor, device=device, dtype=dtype).view(1, overlap//factor, 1).expand(1, overlap//factor, tile_w//factor)

    for bbox in bbox_list:
        # weight[bbox.slicer] += init_weight
        # latent_weight[bbox.latent_slicer] += init_weight

        if bbox.right_lap is not None:  # bbox.right_slicer is not None
            weight[bbox.right_slicer] += weights_right - 1.0
            latent_weight[bbox.right_slicer_lat] += weights_right_lt  - 1.0
        if bbox.left_lap is not None:
            weight[bbox.left_slicer] += weights_left
            latent_weight[bbox.left_slicer_lat] += weights_left_lt - 1.0
        if bbox.up_lap is not None:
            weight[bbox.up_slicer] += weights_up
            latent_weight[bbox.up_slicer_lat] += weights_up_lt - 1.0
        if bbox.down_lap is not None:
            weight[bbox.down_slicer] += weights_down
            latent_weight[bbox.down_slicer_lat] += weights_down_lt - 1.0
    return bbox_list, latent_weight


def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, factor=16, device='cuda', init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    latent_weight = torch.zeros((1, 1, h//factor, w//factor), device=device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h, scale=factor)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight
            latent_weight[bbox.latent_slicer] += init_weight

    return bbox_list, latent_weight

def split_bboxes_bac(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, factor=16, device='cuda', init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    latent_weight = torch.zeros((1, 1, h//factor, w//factor), device=device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h, scale=factor)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight
            latent_weight[bbox.latent_slicer] += init_weight

    return bbox_list, latent_weight



class TTP():
    def image_width_height(self, image, width_factor, height_factor, overlap_rate):
        #  先确定要将图像划分为几个块进行运算，一般都是W划分为几块，宽划分为几块，总tiles数为 width_factor * height_factor,
        #  width_factor, height_factor, overlap_rate = 3，2，0.1  image： 1152*4， 768*4   4608* 3072
        # _, raw_H, raw_W, _ = image.shape 
        raw_H, raw_W = image.size
        w,h = image.size
        if overlap_rate == 0:
            # 水平方向
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / width_factor)
                if tile_width % 8 != 0:
                    tile_width = ((tile_width + 7) // 8) * 8
            # 垂直方向
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / height_factor)
                if tile_height % 8 != 0:
                    tile_height = ((tile_height + 7) // 8) * 8

        else:
            # 水平方向
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / (1 + (width_factor - 1) * (1 - overlap_rate)))  # 4608/ (1+  (3-1) * (1-0.1)  )  4608 / (1+ 2*0.9) = 1645
                if tile_width % 8 != 0:
                    tile_width = (tile_width // 8) * 8                         # tile_width = 1647 // 8 * 8 = 1640
            # 垂直方向
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / (1 + (height_factor - 1) * (1 - overlap_rate)))        #  3072 / (1+ 1*0.9) = 1616
                if tile_height % 8 != 0:
                    tile_height = (tile_height // 8) * 8

        return tile_width, tile_height      # 1640, 1616
    
    def tile_image(self, image, tile_width=1024, tile_height=1024, factor=1):
        # -- tile image in to batches   but eache image may not same size~~~
        # image = tensor2pil(image.squeeze(0))
        img_width, img_height = image.size

        if img_width <= tile_width and img_height <= tile_height:
            # return (pil2tensor(image), [(0, 0, img_width, img_height)], (img_width, img_height), (1, 1))
            return image, [(0, 0, img_width, img_height)], (img_width, img_height), (1, 1)

        def calculate_step(size, tile_size):
            #  tiles num, step 
            if size <= tile_size:
                return 1, 0
            else:
                num_tiles = (size + tile_size - 1) // tile_size
                overlap = (num_tiles * tile_size - size) // (num_tiles - 1)
                step = tile_size - overlap
                return num_tiles, step

        num_cols, step_x = calculate_step(img_width, tile_width)
        num_rows, step_y = calculate_step(img_height, tile_height)

        new_tiles, tiles = [], []
        positions = []
        for y in range(num_rows):
            for x in range(num_cols):
                left = x * step_x
                upper = y * step_y
                right = min(left + tile_width, img_width)
                lower = min(upper + tile_height, img_height)

                if right - left < tile_width:
                    left = max(0, img_width - tile_width)
                if lower - upper < tile_height:
                    upper = max(0, img_height - tile_height)

                tile = image.crop((left, upper, right, lower))
                # tile_tensor = pil2tensor(tile)
                # tiles.append(tile_tensor)
                positions.append((left, upper, right, lower)) # bbox.box

                obj_tile_bbox = BBox(left, upper, right - left, lower - upper, scale=factor)
                obj_tile_bbox.image = tile

                new_tiles.append(obj_tile_bbox)  #  BBox(0,          0,        tile_w, tile_h)
                # print(new_tiles[-1].box, (left, upper, right, lower))

        # tiles = torch.stack(tiles, dim=0).squeeze(1)
        # return (tiles, positions, (img_width, img_height), (num_cols, num_rows))
        return new_tiles, (num_cols, num_rows)
    
    def get_captions_cond_embedding_by_clip(self, tiles, clip):
        # after tile_image  tiles[
        #  Bbox :  area(left, top, right, botthon), tile_width, tile_height
        #    ]
        for tile in tiles:
            # tile.prompt = get_caption(tile.image)
            # tile.conditioning = clip(tile.prompt)
            pass
        # comfyui 支持区域提示词~ conditioning_set_values

useing_in_pixel = '''
    # Normal liner result WAY |
    result_list = [pil2tensor(im_).permute(0, 3, 1,2) for im_ in result_list]
    result_list = [
        result_list[i: i + k_col] for i in range(0, len(result_list), k_col)
    ]
    # 线性拼接~ overlap
    stitched_image_matrix_tensor = [
        concatenate_rowwise(elem, overlap * 2) for elem in result_list
    ]
    stitched_image_matrix_tensor = concatenate_columnwise(stitched_image_matrix_tensor, overlap * 2)
    stitched_image = tensor2pil(
        stitched_image_matrix_tensor.permute(0, 2, 3 ,1)
    )
'''

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

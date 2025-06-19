# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from einops import rearrange
from PIL import Image


def convert_to_numpy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        image = image.copy()
    else:
        raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
    return image

class DepthV2Annotator:
    def __init__(self, cfg, device=None):
        from .dpt import DepthAnythingV2
        pretrained_model = cfg['PRETRAINED_MODEL']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                pretrained_model,
                map_location=self.device
            )
        )
        self.model.eval()

    @torch.inference_mode()
    @torch.autocast('cuda', enabled=False)
    def forward(self, image):
        image = convert_to_numpy(image)
        depth = self.model.infer_image(image)

        depth_pt = depth.copy()
        depth_pt -= np.min(depth_pt)
        depth_pt /= np.max(depth_pt)
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

        depth_image = depth_image[..., np.newaxis]
        depth_image = np.repeat(depth_image, 3, axis=2)
        return depth_image


class DepthV2VideoAnnotator(DepthV2Annotator):
    def forward(self, frames):
        ret_frames = []
        for frame in frames:
            anno_frame = super().forward(np.array(frame))
            ret_frames.append(anno_frame)
        return ret_frames

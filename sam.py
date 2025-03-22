import cv2
import time
import random
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import matplotlib.pyplot as plt
from torchvision.utils import save_image


import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple

def load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name):
    """
    Load precomputed FD map and filtered image based on the image name.
    Args:
        fd_dir (str): Directory where FD maps are stored.
        filtered_image_dir (str): Directory where filtered images are stored.
        image_name (str): Name of the input image file.
    Returns:
        fd_map (torch.Tensor): Loaded FD map as a tensor.
        filtered_image (torch.Tensor): Loaded filtered image as a tensor.
    """
    # Remove extension to get the base name
    base_name = os.path.splitext(image_name)[0]

    # FD map path (assuming saved as .npy)
    fd_map_path = os.path.join(fd_dir, f"{base_name}.npy")

    # Ensure FD map path doesn't include unexpected suffixes like ':Zone.Identifier'
    fd_map_path = fd_map_path.split(":")[0]

    # Debugging log
   # print(f"FD map path: {fd_map_path}")

    # Check if FD map file exists
    if not os.path.exists(fd_map_path):
        raise FileNotFoundError(f"FD map file not found: {fd_map_path}")

    # Load FD map
    try:
        fd_map = np.load(fd_map_path)
        fd_map = torch.tensor(fd_map, dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load FD map from {fd_map_path}: {e}")

    # Filtered image path (assuming same extension as input)
    filtered_image_path = os.path.join(filtered_image_dir, image_name)

    # Debugging log
    # print(f"Filtered image path: {filtered_image_path}")

    # Check if filtered image file exists
    if not os.path.exists(filtered_image_path):
        raise FileNotFoundError(f"Filtered image file not found: {filtered_image_path}")

    # Load filtered image
    try:
        filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR)
        if filtered_image is None:
            raise ValueError("Failed to load filtered image: returned None")
        filtered_image = torch.tensor(filtered_image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
    except Exception as e:
        raise RuntimeError(f"Failed to load filtered image from {filtered_image_path}: {e}")

    return fd_map, filtered_image




def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.original_size = None  # 🔹 초기화 추가
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()
        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask, image_name, fd_dir, filtered_image_dir, original_size=None):
        """
        입력 데이터를 모델에 설정합니다.
        Args:
            input (torch.Tensor): 원본 입력 이미지.
            gt_mask (torch.Tensor): Ground Truth 마스크.
            image_name (str): 이미지 파일 이름.
            fd_dir (str): FD map이 저장된 디렉토리.
            filtered_image_dir (str): 필터링된 이미지가 저장된 디렉토리.
        """
        # 원본 크기 저장
        self.original_size = input.size()[-2:]  # (Height, Width)

        # 모델이 기대하는 크기 (1024, 1024)로 이미지 리사이즈
        input_resized = F.interpolate(input, size=(1024, 1024), mode='bilinear', align_corners=False)
        self.input = input_resized.to(self.device)

        # Ground Truth 마스크 리사이즈
        gt_mask_resized = F.interpolate(gt_mask, size=(1024, 1024), mode='nearest')
        self.gt_mask = gt_mask_resized.to(self.device)

        # FD map과 필터링된 이미지 로드
        fd_map, filtered_image = load_precomputed_fd_and_image(fd_dir, filtered_image_dir, image_name)

        # FD map의 차원을 확인하고, 필요한 경우 4D로 변환
        if fd_map.ndim == 2:  # FD map이 2D인 경우
            fd_map = fd_map.unsqueeze(0).unsqueeze(0)  # (H, W) → (1, 1, H, W)
        elif fd_map.ndim == 3:  # 이미 채널 차원이 있는 경우
            fd_map = fd_map.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
        else:
            raise ValueError(f"Unexpected fd_map dimensions: {fd_map.shape}")

        # FD map 리사이즈 및 GPU로 이동
        self.fd_map = F.interpolate(fd_map, size=(1024, 1024), mode='bilinear', align_corners=False).to(self.device)

        # 필터링된 이미지를 4D로 변환하고 리사이즈
        filtered_image = filtered_image.unsqueeze(0)  # (C, H, W) → (1, C, H, W)
        self.filtered_image = F.interpolate(filtered_image, size=(1024, 1024), mode='bilinear', align_corners=False).to(self.device)

    def forward(self):
        """
        SAM 모델의 Forward 패스에서 FD map과 필터링된 이미지를 포함한 연산 수행.
        """
        # Ensure 4D tensors for self.input, self.filtered_image, and self.fd_map
        if self.input.dim() == 3:
            self.input = self.input.unsqueeze(0)  # 배치 차원 추가
        if self.filtered_image.dim() == 3:
            self.filtered_image = self.filtered_image.unsqueeze(0)  # 배치 차원 추가
        if self.fd_map.dim() == 3:
            self.fd_map = self.fd_map.unsqueeze(0)  # 배치 차원 추가

        # 모든 데이터를 GPU로 이동
        self.input = self.input.to(self.device)
        self.filtered_image = self.filtered_image.to(self.device)
        self.fd_map = self.fd_map.to(self.device)

        # 어댑터 통합 부분
        # FD map과 필터링된 이미지를 ImageEncoderViT의 입력으로 추가 전달
        combined_features = self.image_encoder(self.input, self.fd_map, self.filtered_image)

        # Mask Decoder 및 마스크 생성
        bs = self.input.size(0)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=combined_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=torch.empty(
                (bs, 0, self.prompt_embed_dim), device=self.device
            ),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 마스크를 원본 이미지 해상도로 업스케일
        self.pred_mask = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)


    def infer(self, input, image_name, fd_dir, filtered_image_dir):
        """
        Perform inference using the SAM model.
        """
        # Ensure input is 4D
        if input.dim() == 3:
            input = input.unsqueeze(0)
        input_resized = F.interpolate(input, size=(1024, 1024), mode='bilinear', align_corners=False)
        self.input = input_resized.to(self.device)

        # Load FD map and filtered image
        fd_map_path = os.path.join(fd_dir, image_name.replace('.jpg', '.npy').replace('.jpeg', '.npy'))
        filtered_image_path = os.path.join(filtered_image_dir, image_name)

        fd_map = np.load(fd_map_path)
        filtered_image = cv2.imread(filtered_image_path, cv2.IMREAD_COLOR)

        self.fd_map = torch.tensor(fd_map, dtype=torch.float32).to(self.device)
        filtered_image_resized = cv2.resize(filtered_image, (1024, 1024))
        self.filtered_image = (
            torch.tensor(filtered_image_resized).permute(2, 0, 1).float().to(self.device) / 255.0
        )

        # Ensure filtered_image is 4D
        if self.filtered_image.dim() == 3:
            self.filtered_image = self.filtered_image.unsqueeze(0)

        # Combine features
        combined_features = self.image_encoder(self.input) + self.image_encoder(self.filtered_image)

        # Decoder and mask generation
        bs = self.input.size(0)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=combined_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=torch.empty(
                (bs, 0, self.prompt_embed_dim), device=self.device
            ),
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        return self.postprocess_masks(low_res_masks, self.inp_size, self.original_size)
    


    def get_dense_pe(self):
        """
        Get positional encoding for dense point prompts.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def postprocess_masks(self, masks, input_size, original_size=None):
        """
        Remove padding and upscale masks to the original image size.
        """
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear", align_corners=False
        )
        masks = masks[..., :input_size, :input_size]

        if original_size:
            # Resize to the original size
            masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    
        return masks

    def backward_G(self):
        """
        Backpropagation with segmentation loss.
        """
        # 배치 크기 동기화
        if self.pred_mask.shape[0] != self.gt_mask.shape[0]:
            min_batch_size = min(self.pred_mask.shape[0], self.gt_mask.shape[0])
            self.pred_mask = self.pred_mask[:min_batch_size]
            self.gt_mask = self.gt_mask[:min_batch_size]

        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += self.criterionIOU(self.pred_mask, self.gt_mask)
        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Optimize parameters during training.
        """
        self.forward()
        self.optimizer.zero_grad()  # Reset gradients
        self.backward_G()  # Backpropagation
        self.optimizer.step()  # Update weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
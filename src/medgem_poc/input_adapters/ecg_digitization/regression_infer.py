from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RegressionInferConfig:
    """
    Inference config for prob_map regression.

    model_format:
      - "torchscript" (recommended for product): torch.jit.load(ckpt)
      - "state_dict"  (supported but requires a model_factory in future; currently raises)

    add_coord_channels:
      - If True, builds [gray, X_coord, Y_coord] channels (1PS-style).
      - Your model must be trained to accept 3 channels.

    input_hw:
      - Input resize for the regressor.
      - You should align this with the model training (e.g., 1280x2528 or 256x512).
    """
    ckpt_path: str
    model_format: str = "torchscript"   # torchscript | state_dict
    device: Optional[str] = None        # "cpu"|"cuda"|None
    input_hw: Tuple[int, int] = (256, 512)  # (H,W) default matches many baselines
    add_coord_channels: bool = False
    sigmoid: bool = True


def _read_gray_image(path: str) -> np.ndarray:
    # Read grayscale image via cv2, fallback PIL; returns uint8 HxW
    img = None
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None

    if img is None:
        try:
            from PIL import Image  # type: ignore
            img = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path} err={e}")

    if img.ndim != 2:
        img = img[:, :, 0]
    return img.astype(np.uint8, copy=False)


def _resize_gray(img: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = int(hw[0]), int(hw[1])
    if H <= 0 or W <= 0:
        raise ValueError("input_hw must be positive")

    try:
        import cv2  # type: ignore
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return out.astype(np.uint8, copy=False)
    except Exception:
        from PIL import Image  # type: ignore
        pil = Image.fromarray(img)
        pil = pil.resize((W, H), resample=Image.BILINEAR)
        return np.asarray(pil, dtype=np.uint8)


def _make_coord_channels(hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    H, W = int(hw[0]), int(hw[1])
    x = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)
    return x, y


def _load_torchscript(ckpt_path: str, device: str):
    import torch  # type: ignore
    m = torch.jit.load(ckpt_path, map_location=device)
    m.eval()
    return m


def infer_prob_map_from_path(input_path: str, cfg: RegressionInferConfig) -> np.ndarray:
    """
    Returns prob_map float32 HxW.

    Expected model output:
      - either (1,1,H,W) or (1,H,W) or (H,W)
    """
    ckpt = str(cfg.ckpt_path)
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"ckpt not found: {ckpt}")

    import torch  # type: ignore

    device = cfg.device
    if device not in ("cpu", "cuda"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.model_format.lower() != "torchscript":
        raise RuntimeError(
            "model_format=state_dict is not enabled in MVP. "
            "Use torchscript export for product inference."
        )

    model = _load_torchscript(ckpt, device)

    img_u8 = _read_gray_image(input_path)
    img_u8 = _resize_gray(img_u8, cfg.input_hw)

    gray = (img_u8.astype(np.float32) / 255.0)
    if cfg.add_coord_channels:
        xch, ych = _make_coord_channels(cfg.input_hw)
        inp = np.stack([gray, xch, ych], axis=0)  # (3,H,W)
    else:
        inp = gray[None, :, :]  # (1,H,W)

    x = torch.from_numpy(inp).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,C,H,W)

    with torch.inference_mode():
        y = model(x)

        # Normalize output shape -> (H,W)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if hasattr(y, "detach"):
            y = y.detach()

        if cfg.sigmoid:
            y = torch.sigmoid(y)

        y_cpu = y.float().cpu()

        if y_cpu.ndim == 4:
            pm = y_cpu[0, 0].numpy()
        elif y_cpu.ndim == 3:
            pm = y_cpu[0].numpy()
        elif y_cpu.ndim == 2:
            pm = y_cpu.numpy()
        else:
            raise RuntimeError(f"Unexpected model output shape: {tuple(y_cpu.shape)}")

    pm = np.asarray(pm, dtype=np.float32)
    pm = np.nan_to_num(pm, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure bounds [0,1] loosely
    pm = np.clip(pm, 0.0, 1.0)
    return pm

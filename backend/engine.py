"""
Loads the trained MSFS-Net (UNet) once and exposes `dehaze(PIL.Image) -> uint8 RGB array`.

Pre/post-processing mirrors the notebook's inference cell exactly:
    input  : ToTensor()   -> float32 in [0, 1]
    output : clamp(0, 1)  -> *255, round, uint8
"""
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_utils import limit_size, pad_to_multiple
from model.network import UNet

WEIGHTS_PATH = Path(__file__).parent / "weights" / "SateHaze_MSFSNet_bestPSNR.pth"

# Must match the architecture used in training (see notebook).
MODEL_CFG = dict(
    img_channel=3,
    width=48,
    middle_blk_num=1,
    enc_blk_nums=[3, 2],
    dec_blk_nums=[1, 1],
    patch_size=8,
)

# Training images are 512x512. Larger uploads are downscaled to this longest side
# to keep CPU inference fast. Raise it if you have a GPU.
MAX_SIDE = 1024


class DehazeEngine:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()  # one forward pass at a time
        self.model = self._load(weights_path)

    # ------------------------------------------------------------------ #
    def _load(self, path: Path) -> torch.nn.Module:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = UNet(**MODEL_CFG)

        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            # Older checkpoints may hold non-tensor objects; this file is your own, so it's trusted.
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "params", "model_state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break

        # strip the "module." prefix left by nn.DataParallel
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint does not match architecture. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

        model.to(self.device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[engine] MSFS-Net loaded on {self.device} | {n_params / 1e6:.2f} M params")
        self.n_params = n_params
        return model

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def dehaze(self, image: Image.Image):
        """Returns (hazy_u8, dehazed_u8) as HxWx3 uint8 arrays of identical size."""
        image = limit_size(image.convert("RGB"), MAX_SIDE)
        hazy = np.asarray(image, dtype=np.uint8)

        padded, (h, w) = pad_to_multiple(hazy)
        x = torch.from_numpy(padded.astype(np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with self.lock:
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        out = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)[:h, :w]  # crop padding away
        return hazy, out

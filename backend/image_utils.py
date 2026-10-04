"""
Torch-free helpers: padding, metrics and a haze-density estimate.

PSNR / SSIM are the SAME implementations used in the original repo
(utils/metrics.py -> cv2.PSNR + Gaussian-window SSIM on uint8 images),
so numbers are directly comparable with the paper / notebook results.
"""
import cv2
import numpy as np
from PIL import Image

# The network = patch_embed (x8) + 2 encoder downsamples (x2 each) -> multiple of 32
DIVISOR = 32


# --------------------------------------------------------------------------- #
# Padding
# --------------------------------------------------------------------------- #
def pad_to_multiple(arr: np.ndarray, divisor: int = DIVISOR):
    """Reflect-pad an HxWxC array so H and W are multiples of `divisor`."""
    h, w = arr.shape[:2]
    ph = (divisor - h % divisor) % divisor
    pw = (divisor - w % divisor) % divisor
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return arr, (h, w)


def limit_size(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale (keeping aspect ratio) if the longest side exceeds max_side."""
    w, h = img.size
    longest = max(w, h)
    if max_side and longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


# --------------------------------------------------------------------------- #
# Metrics  (copied from the repo's utils/metrics.py)
# --------------------------------------------------------------------------- #
def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    def _ssim_single_channel(x, y, data_range=255):
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu_x = cv2.filter2D(x, -1, window)[5:-5, 5:-5]
        mu_y = cv2.filter2D(y, -1, window)[5:-5, 5:-5]

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = cv2.filter2D(x * x, -1, window)[5:-5, 5:-5] - mu_x_sq
        sigma_y_sq = cv2.filter2D(y * y, -1, window)[5:-5, 5:-5] - mu_y_sq
        sigma_xy = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        return ssim_map.mean()

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return float(_ssim_single_channel(img1, img2))
    if img1.ndim == 3 and img1.shape[2] == 3:
        return float(np.mean([_ssim_single_channel(img1[..., i], img2[..., i]) for i in range(3)]))
    raise ValueError("Wrong input dimensions for SSIM. Expected HxW or HxWx3.")


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    val = float(cv2.PSNR(img1, img2))
    return min(val, 100.0)  # identical images -> cap (cv2 returns inf / ~361)


def compute_metrics(output_u8: np.ndarray, reference_u8: np.ndarray) -> dict:
    h, w = output_u8.shape[:2]
    if h < 11 or w < 11:
        raise ValueError("Image is too small to compute SSIM (needs at least 11x11 px).")
    return {
        "psnr": round(calculate_psnr(output_u8, reference_u8), 2),
        "ssim": round(calculate_ssim(output_u8, reference_u8), 4),
    }


# --------------------------------------------------------------------------- #
# Haze-density estimate (dark channel prior heuristic, informational only)
# --------------------------------------------------------------------------- #
def estimate_haze(rgb_u8: np.ndarray, patch: int = 15) -> dict:
    img = rgb_u8.astype(np.float32) / 255.0
    dark = cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))

    # atmospheric light: mean colour of the brightest 0.1% dark-channel pixels
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = float(max(img.reshape(-1, 3)[idx].mean(), 1e-3))

    density = float(np.clip(dark.mean() / A, 0.0, 1.0))
    if density < 0.30:
        label = "thin"
    elif density < 0.55:
        label = "moderate"
    else:
        label = "thick"
    return {"density": round(density, 3), "label": label}


# --------------------------------------------------------------------------- #
# Error / difference map  (same logic as the Streamlit error-map tool)
#   error = mean over RGB of |restored - reference|, shown with the "jet" colormap, range 0 - 0.3
# --------------------------------------------------------------------------- #
ERR_VMAX = 0.3


def error_map(output_u8: np.ndarray, reference_u8: np.ndarray) -> np.ndarray:
    a = output_u8.astype(np.float32) / 255.0
    b = reference_u8.astype(np.float32) / 255.0
    return np.abs(a - b).mean(axis=2)  # HxW in [0, 1]


def error_map_raw(err: np.ndarray) -> np.ndarray:
    """Colour-mapped error map as an HxWx3 uint8 image (no colorbar)."""
    from matplotlib import colormaps

    rgba = colormaps["jet"](np.clip(err / ERR_VMAX, 0.0, 1.0))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def error_map_plot_png(err: np.ndarray) -> bytes:
    """PNG of the error map WITH a colorbar (thread-safe: no pyplot)."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(err, cmap="jet", vmin=0, vmax=ERR_VMAX)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

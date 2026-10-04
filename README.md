# MSFS-Net · Aerial Image Dehazing Web App

### Multi Scale Deep Learning based Frequency Selective Dehazing Network

A local web app that removes haze from aerial / satellite images with a trained **MSFS-Net**.
Upload a hazy image and get the dehazed result side by side with the input, an **error map**,
and **PSNR / SSIM** metrics. Built with a **FastAPI** backend, a plain HTML/CSS/JS frontend and **PyTorch**.

<!-- Add your own screenshot: save it as docs/preview.png, then uncomment the next line -->
<!-- ![App preview](docs/preview.png) -->

## Features

- Drag-and-drop upload with preview
- Haze type chips: Auto-detect, Thin, Moderate, Thick
- Hazy input · dehazed output · error map shown side by side, with download buttons
- **PSNR**, **SSIM**, mean absolute error, inference time and an estimated haze density underneath
- Optional ground-truth upload for true full-reference metrics and a true error map
- Runs on CPU or automatically on a CUDA GPU
- Interactive API docs at `/docs`

## Project structure

```
msfsnet-dehazing/
├── backend/
│   ├── app.py                 # FastAPI server: API + serves the frontend
│   ├── engine.py              # loads the checkpoint once, runs inference
│   ├── image_utils.py         # padding, PSNR / SSIM, error map, haze estimate
│   ├── model/network.py       # MSFS-Net (UNet) definition
│   └── weights/SateHaze_MSFSNet_bestPSNR.pth
├── frontend/                  # index.html · style.css · script.js
├── requirements.txt
├── run.bat / run.sh
└── .vscode/launch.json        # press F5 in VS Code
```

## Quick start

Requires **Python 3.9+**.

```bash
git clone <your-repo-url>
cd msfsnet-dehazing

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 1) PyTorch - choose your build at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 2) everything else
pip install -r requirements.txt

cd backend
python app.py
```

Open **http://localhost:8000**. A green pill at the top reading *"Model ready"* means the weights loaded.

**VS Code:** open the folder and press **F5** (uses `.vscode/launch.json`), or double-click `run.bat` / run `./run.sh`.

## How it works

| Stage | Detail |
|---|---|
| Pre-processing | RGB → float32 in `[0,1]`; reflect-padded to a multiple of 32 (network stride = 8 × 2 × 2) |
| Model | `UNet(width=48, enc_blk_nums=[3,2], dec_blk_nums=[1,1], middle_blk_num=1, patch_size=8)` |
| Post-processing | `clamp(0,1)` → uint8, padding cropped off |
| PSNR / SSIM | `cv2.PSNR` and a Gaussian-window SSIM on uint8 RGB (same as the training repo's `utils/metrics.py`) |
| Error map | `mean_over_RGB(\|output − reference\|)`, drawn with the `jet` colormap on a fixed **0 – 0.3** scale |

### Error map

The error map is the per-pixel absolute difference between the dehazed output and the reference,
averaged over the colour channels. Blue = small error, red = large error (values ≥ 0.3 saturate).
Two downloads are provided: a **plot with colorbar** and the **raw colour-mapped map** (no colorbar,
same size as the output image).

## API

### `POST /api/dehaze`  (multipart/form-data)

| Field | Required | Description |
|---|---|---|
| `image` | yes | Hazy image (PNG, JPG, BMP, TIFF; max 25 MB) |
| `haze_type` | no | `auto` (default), `thin`, `moderate`, `thick` |
| `reference` | no | Ground-truth clear image of the same scene |

Returns JSON: `hazy`, `dehazed`, `error_map`, `error_map_plot` (base64 PNG data URLs), `metrics`
(`psnr`, `ssim`, `mae`, `mode`), `haze_estimate`, `width`, `height`, `inference_seconds`, `device`.

```bash
curl -X POST http://localhost:8000/api/dehaze \
  -F "image=@hazy.png" -F "reference=@clear.png" -F "haze_type=auto"
```

### `GET /api/health`
Model / device status.

## Good to know

- **PSNR, SSIM and the error map are full-reference measures and need a ground truth.**
  Without one, the app compares the output with the *hazy input* and labels this on screen
  ("Change Map · vs Hazy Input"); that shows how much the image changed, not how good the result is.
- **The haze-type selector does not switch models.** A single checkpoint was trained on thin,
  moderate and thick haze, so the same weights run in every mode. The "estimated haze density"
  is an uncalibrated dark-channel-prior heuristic and is informational only.
- **Image size.** The model was trained on 512 × 512 images. Uploads with a longest side above
  1024 px are downscaled for speed (`MAX_SIDE` in `backend/engine.py`). The reference is resized to
  the output size if they differ.
- `timm` is not required for inference.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Backend offline" pill | Start the server (`cd backend && python app.py`) |
| `Checkpoint does not match architecture` | The `.pth` must come from the config listed above |
| Port 8000 already in use | Change the port at the bottom of `backend/app.py` |
| Slow on CPU | Lower `MAX_SIDE` in `engine.py` or install a CUDA build of PyTorch |

## Acknowledgements

Model definition and training code: [arkapravo-dey/Lightweight_Aerial_Dehazing](https://github.com/arkapravo-dey/Lightweight_Aerial_Dehazing).
Trained on the SateHaze1K dataset. Please check the original repository's license and add your own
`LICENSE` file before publishing.

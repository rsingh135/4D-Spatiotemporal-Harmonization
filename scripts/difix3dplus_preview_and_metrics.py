"""Single-frame preview + numeric metrics for the Difix3D+ distillation experiment.

Picks one mid-video frame, crops the FG region, and renders a 1x4 strip plus a
zoomed strip. Also reports per-method PSNR/SSIM against the render-time Difix
output (treated as a reference 'clean' image) over a sampled set of frames.
"""
import os

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

OUT_DIR = "/home/ubuntu/new_sa4d/sa4d/output/composite_torchocolateBigger"
VIDS = {
    "raw":     os.path.join(OUT_DIR, "composite_cookie_chocBigger.mp4"),
    "rtDifix": os.path.join(OUT_DIR, "difix3d/difix_cleaned.mp4"),
    "d3+_fg":  os.path.join(OUT_DIR, "composite_cookie_chocBigger_difix3dplus_fg.mp4"),
    "d3+_all": os.path.join(OUT_DIR, "composite_cookie_chocBigger_difix3dplus_all.mp4"),
}
LABELS = {
    "raw":     "raw composite",
    "rtDifix": "render-time Difix only",
    "d3+_fg":  "Difix3D+ distilled (FG SH)",
    "d3+_all": "Difix3D+ distilled (all SH)",
}
NUM_FRAMES = 539
PREVIEW_T = 270
SAMPLE_TS = list(range(20, NUM_FRAMES, 30))
CROP = (170, 360, 410, 720)  # (left, top, right, bottom) for chocolate ROI
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def at(path, idx):
    return np.asarray(iio.imread(path, index=idx))


def label(im, text, font_path=FONT, size=20):
    pil = Image.fromarray(im)
    draw = ImageDraw.Draw(pil)
    f = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0] + 14
    h = bbox[3] - bbox[1] + 12
    draw.rectangle([4, 4, 4 + w, 4 + h], fill=(0, 0, 0, 200))
    draw.text((11, 9), text, fill=(255, 255, 255), font=f)
    return np.asarray(pil)


def main():
    rendered = {k: at(p, PREVIEW_T) for k, p in VIDS.items()}
    full = np.concatenate([label(rendered[k], LABELS[k]) for k in ("raw", "rtDifix", "d3+_fg", "d3+_all")], axis=1)
    Image.fromarray(full).save(os.path.join(OUT_DIR, "difix3dplus_preview_full.png"), optimize=True)

    L, T, R, B = CROP
    crops = [rendered[k][T:B, L:R] for k in ("raw", "rtDifix", "d3+_fg", "d3+_all")]
    crop_strip = np.concatenate([label(c, LABELS[k]) for c, k in zip(crops, ("raw", "rtDifix", "d3+_fg", "d3+_all"))], axis=1)
    Image.fromarray(crop_strip).save(os.path.join(OUT_DIR, "difix3dplus_preview_zoom.png"), optimize=True)
    print(f"Saved difix3dplus_preview_full.png ({full.shape}) and difix3dplus_preview_zoom.png ({crop_strip.shape})")

    print("\nMetrics against render-time-Difix reference (mean over 18 sampled frames):")
    print(f"{'method':24s} | {'PSNR (dB)':>10s} | {'SSIM':>6s}")
    print("-" * 50)
    refs = {t: at(VIDS["rtDifix"], t) for t in SAMPLE_TS}
    for k in ("raw", "rtDifix", "d3+_fg", "d3+_all"):
        psnrs, ssims = [], []
        for t in SAMPLE_TS:
            r = refs[t]
            cur = at(VIDS[k], t)
            if cur.shape != r.shape:
                continue
            psnrs.append(_psnr(r, cur, data_range=255))
            ssims.append(_ssim(r, cur, channel_axis=2, data_range=255))
        m_p = float(np.mean(psnrs))
        m_s = float(np.mean(ssims))
        print(f"{LABELS[k]:24s} | {m_p:>10.3f} | {m_s:>6.4f}")


if __name__ == "__main__":
    main()

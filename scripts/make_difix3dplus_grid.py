"""Build a 4 x 6 grid (4 methods x 6 timestamps) PNG comparing
raw / render-time-only / Difix3D+ FG-only / Difix3D+ all on the
composite_cookie_chocBigger scene."""
import os
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "/home/ubuntu/new_sa4d/sa4d/output/composite_torchocolateBigger"
VIDS = [
    ("raw composite", os.path.join(OUT_DIR, "composite_cookie_chocBigger.mp4")),
    ("render-time Difix only", os.path.join(OUT_DIR, "difix3d/difix_cleaned.mp4")),
    ("Difix3D+ conservative (300 iter, all SH)",
     os.path.join(OUT_DIR, "composite_cookie_chocBigger_difix3dplus_conservative.mp4")),
    ("Difix3D+ aggressive FG-only (1k iter)",
     os.path.join(OUT_DIR, "composite_cookie_chocBigger_difix3dplus_fg.mp4")),
    ("Difix3D+ aggressive all SH (1.5k iter)",
     os.path.join(OUT_DIR, "composite_cookie_chocBigger_difix3dplus_all.mp4")),
]
NUM_FRAMES = 539
TS = [int(round(NUM_FRAMES * f)) for f in (0.05, 0.30, 0.50, 0.70, 0.92)]
TILE_W, TILE_H = 360, 636
LABEL_PAD = 36
ROW_LABEL_W = 240

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def grab_frames(path, idxs):
    arr = []
    for i in idxs:
        img = iio.imread(path, index=i)
        arr.append(img)
    return arr


def main():
    grid = []
    for label, vp in VIDS:
        frames = grab_frames(vp, TS)
        grid.append((label, frames))

    n_cols = len(TS)
    n_rows = len(VIDS)
    total_w = ROW_LABEL_W + n_cols * TILE_W
    total_h = LABEL_PAD + n_rows * TILE_H
    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_col = ImageFont.truetype(FONT, 16)
    font_row = ImageFont.truetype(FONT, 14)

    for c, t in enumerate(TS):
        x = ROW_LABEL_W + c * TILE_W + 8
        draw.text((x, 8), f"t={t}/{NUM_FRAMES}", fill=(20, 20, 20), font=font_col)

    for r, (label, frames) in enumerate(grid):
        y = LABEL_PAD + r * TILE_H
        for line_i, line in enumerate(label.split(" (")):
            ly = y + 18 + line_i * 22
            txt = line if line_i == 0 else "(" + line
            draw.text((10, ly), txt, fill=(20, 20, 20), font=font_row)
        for c, fr in enumerate(frames):
            tile = Image.fromarray(fr).resize((TILE_W, TILE_H), Image.LANCZOS)
            canvas.paste(tile, (ROW_LABEL_W + c * TILE_W, LABEL_PAD + r * TILE_H))

    out = os.path.join(OUT_DIR, "difix3dplus_grid.png")
    canvas.save(out, optimize=True)
    print(f"Saved {out}  ({canvas.size})")


if __name__ == "__main__":
    main()

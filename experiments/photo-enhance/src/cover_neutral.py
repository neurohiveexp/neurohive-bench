"""Cover without text labels, shared by the RU and EN articles.

Polylang copies the featured image between translations, so one cover has to work in both
languages: same frame as figs/cover-{ru,en}.jpg (red macaw, left noisy, right SCUNet), no ДО/ПОСЛЕ or BEFORE/AFTER.
"""
import os
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
k, box = "kodim23", (376, 30, 760, 462)  # 384x432 region -> two 800x900 panels, as in figs.py
a = Image.open(f"{ROOT}/inputs/noise_{k}.png").convert("RGB").crop(box).resize((800, 900), Image.LANCZOS)
b = Image.open(f"{ROOT}/out/scunet/noise_{k}.png").convert("RGB").crop(box).resize((800, 900), Image.LANCZOS)
cov = Image.new("RGB", (1600, 900), (255, 255, 255))
cov.paste(a, (0, 0)); cov.paste(b, (800, 0))
ImageDraw.Draw(cov).rectangle([797, 0, 802, 900], fill=(255, 255, 255))
out = f"{ROOT}/figs/cover.jpg"
cov.save(out, "JPEG", quality=88, optimize=True, progressive=True)
print(out, cov.size, os.path.getsize(out) // 1024, "KB")

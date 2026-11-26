"""Neurohive photo-enhancement test: build degraded inputs from Kodak GT.

Scenarios
  small  - photo shrunk 4x and re-saved as JPEG (messenger / old phone)
  face   - portrait shrunk 4x, light (blur 1.0, JPEG 60) and heavy (blur 2.0, JPEG 40)
  text   - synthetic notice with 6 text lines, blur 0.6, shrunk 4x, JPEG 75
  noise  - full-size photo with Gaussian noise sigma 25

Everything is deterministic (fixed seeds), so the inputs can be rebuilt.
"""
import io, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = f"{ROOT}/gt"
OUT = f"{ROOT}/inputs"
os.makedirs(OUT, exist_ok=True)


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q, subsampling=2)  # 4:2:0 like phones and messengers
    buf.seek(0)
    return Image.open(buf).convert("RGB"), buf.getvalue()


def shrink(img, f=4, blur=0.0):
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    w, h = img.size
    return img.resize((w // f, h // f), Image.BICUBIC)  # PIL bicubic is antialiased


manifest = {"small": [], "face": [], "text": [], "noise": []}

# 1) small photo
for k in ["kodim01", "kodim08", "kodim13", "kodim23"]:
    gt = Image.open(f"{GT}/{k}.png").convert("RGB")
    lr, raw = jpeg(shrink(gt, 4), 75)
    open(f"{OUT}/small_{k}.jpg", "wb").write(raw)
    manifest["small"].append({"id": f"small_{k}", "gt": f"{k}.png", "input": f"small_{k}.jpg", "lr_size": lr.size})

# 2) faces, two strengths
for k in ["kodim04", "kodim15"]:
    gt = Image.open(f"{GT}/{k}.png").convert("RGB")
    for lvl, blur, q in [("light", 1.0, 60), ("heavy", 2.0, 40)]:
        lr, raw = jpeg(shrink(gt, 4, blur), q)
        open(f"{OUT}/face_{lvl}_{k}.jpg", "wb").write(raw)
        manifest["face"].append({"id": f"face_{lvl}_{k}", "gt": f"{k}.png", "input": f"face_{lvl}_{k}.jpg", "level": lvl})

# 3) synthetic text
LINES = [
    ("ОБЪЯВЛЕНИЕ № 2417", 44, True),
    ("Отключение воды 14.03.2026", 34, False),
    ("с 09:00 до 18:30, дом 18, корп. 3", 28, False),
    ("Справки: кабинет 205, этаж 2", 24, False),
    ("Free Wi-Fi: CAFE_2417 / pass 7781", 20, False),
    ("Цена 12 990 руб., скидка 15%", 18, False),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
rng = np.random.default_rng(2417)
W, H = 768, 512
paper = np.full((H, W, 3), (242, 239, 230), np.float32)
paper += rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)  # paper grain
canvas = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(canvas)
y = 34
boxes = []
for text, size, bold in LINES:
    f = ImageFont.truetype(FONTB if bold else FONT, size)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    x = 40
    d.text((x, y), text, font=f, fill=(28, 30, 38))
    bx = (x + l - 6, y + t - 6, x + r + 6, y + b + 6)
    boxes.append({"text": text, "size": size, "lr_px": round(size / 4, 1), "box": bx})
    y += int(size * 1.75)
assert y < H, y
canvas.save(f"{GT}/text_notice.png")
lr, raw = jpeg(shrink(canvas, 4, 0.6), 75)
open(f"{OUT}/text_notice.jpg", "wb").write(raw)
manifest["text"].append({"id": "text_notice", "gt": "text_notice.png", "input": "text_notice.jpg", "lines": boxes})

# 4) noise
for k in ["kodim03", "kodim23"]:
    gt = np.asarray(Image.open(f"{GT}/{k}.png").convert("RGB"), np.float32)
    rng = np.random.default_rng(int(k[-2:]))
    noisy = np.clip(gt + rng.normal(0, 25, gt.shape), 0, 255).round().astype(np.uint8)
    Image.fromarray(noisy).save(f"{OUT}/noise_{k}.png")
    manifest["noise"].append({"id": f"noise_{k}", "gt": f"{k}.png", "input": f"noise_{k}.png", "sigma": 25})

json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: len(v) for k, v in manifest.items()}))
for b in boxes:
    print(b["size"], b["lr_px"], b["box"], b["text"])

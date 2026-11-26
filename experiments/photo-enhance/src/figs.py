"""Article figures (RU and EN labels) + cover. Output: figs/*.jpg"""
import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT, INP, OUT, FIG = f"{ROOT}/gt", f"{ROOT}/inputs", f"{ROOT}/out", f"{ROOT}/figs"
os.makedirs(FIG, exist_ok=True)
M = json.load(open(f"{OUT}/metrics.json"))
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
INK, SUB, BG = (24, 26, 33), (95, 99, 110), (255, 255, 255)

L = {
    "ru": {"in": "Вход", "in_s": "что прислали", "bic": "Обычное увеличение", "esr": "Real-ESRGAN",
           "gen": "Real-ESRGAN General", "swin": "SwinIR", "gfp": "GFPGAN", "gt": "Оригинал",
           "noisy": "Шумное фото", "nlm": "Классический фильтр", "scu": "SCUNet",
           "id": "сходство {v}", "db": "{v} дБ", "small_px": "{w}×{h} px",
           "before": "ДО", "after": "ПОСЛЕ", "ref": "эталон"},
    "en": {"in": "Input", "in_s": "what you have", "bic": "Plain upscale", "esr": "Real-ESRGAN",
           "gen": "Real-ESRGAN General", "swin": "SwinIR", "gfp": "GFPGAN", "gt": "Original",
           "noisy": "Noisy photo", "nlm": "Classic filter", "scu": "SCUNet",
           "id": "identity {v}", "db": "{v} dB", "small_px": "{w}×{h} px",
           "before": "BEFORE", "after": "AFTER", "ref": "reference"},
}


def num(v, lang, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace(".", ",") if lang == "ru" else s


def load(p):
    return Image.open(p).convert("RGB")


def up_nearest(p, size):
    return load(p).resize(size, Image.NEAREST)


def strip(panels, cw, ch, gap=10, top=30, bottom=26, pad=0):
    """panels: [(title, sub, PIL image already cropped)] -> one row figure"""
    n = len(panels)
    W = n * cw + (n - 1) * gap + 2 * pad
    H = top + ch + bottom
    fig = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(fig)
    ft, fs = ImageFont.truetype(FB, 17), ImageFont.truetype(FR, 15)
    for i, (title, sub, im) in enumerate(panels):
        x = pad + i * (cw + gap)
        fig.paste(im.resize((cw, ch), Image.LANCZOS), (x, top))
        d.text((x + cw / 2, top - 8), title, fill=INK, font=ft, anchor="ms")
        if sub:
            d.text((x + cw / 2, top + ch + 19), sub, fill=SUB, font=fs, anchor="ms")
    return fig


def crop_all(paths, box, nearest_first=True, full=(768, 512)):
    ims = []
    for i, p in enumerate(paths):
        im = up_nearest(p, full) if (i == 0 and nearest_first) else load(p)
        ims.append(im.crop(box))
    return ims


def save(fig, name):
    fig.save(f"{FIG}/{name}.jpg", "JPEG", quality=88, optimize=True, progressive=True)
    print(name, fig.size, os.path.getsize(f"{FIG}/{name}.jpg") // 1024, "KB")


for lang in ["ru", "en"]:
    t = L[lang]
    # 1) small photo x4: kodim08 houses
    k = "kodim08"; box = (440, 150, 640, 350)
    paths = [f"{INP}/small_{k}.jpg", f"{OUT}/bicubic/small_{k}.png", f"{OUT}/realesrgan_x4plus/small_{k}.png",
             f"{OUT}/swinir_gan/small_{k}.png", f"{GT}/{k}.png"]
    ims = crop_all(paths, box)
    subs = [t["small_px"].format(w=192, h=128), "", "", "", "768×512 px"]
    titles = [t["in"], t["bic"], t["esr"], t["swin"], t["gt"]]
    save(strip(list(zip(titles, subs, ims)), 232, 232), f"fig1-small-{lang}")

    # 2) faces, heavy damage, kodim15 + kodim04
    rows = []
    for k, box, full in [("kodim15", (395, 20, 705, 400), (768, 512)), ("kodim04", (130, 200, 400, 530), (512, 768))]:
        it = f"face_heavy_{k}"
        r = M["face"][it]
        paths = [f"{INP}/{it}.jpg", f"{OUT}/bicubic/{it}.png", f"{OUT}/realesrgan_x4plus/{it}.png",
                 f"{OUT}/gfpgan/{it}.png", f"{GT}/{k}.png"]
        ims = crop_all(paths, box, full=full)
        subs = ["", t["id"].format(v=num(r["bicubic"]["identity"], lang)),
                t["id"].format(v=num(r["realesrgan_x4plus"]["identity"], lang)),
                t["id"].format(v=num(r["gfpgan"]["identity"], lang)), t["ref"]]
        titles = [t["in"], t["bic"], t["esr"], t["gfp"], t["gt"]]
        cw = 232; ch = int(cw * (box[3] - box[1]) / (box[2] - box[0]))
        rows.append(strip(list(zip(titles, subs, ims)), cw, ch))
    W = max(r.width for r in rows); H = sum(r.height for r in rows) + 12
    fig = Image.new("RGB", (W, H), BG); y = 0
    for r in rows:
        fig.paste(r, (0, y)); y += r.height + 12
    save(fig, f"fig2-faces-{lang}")

    # 3) text: 2 columns x 3 rows
    box = (30, 105, 590, 330)
    it = "text_notice"
    paths = [f"{INP}/{it}.jpg", f"{OUT}/bicubic/{it}.png", f"{OUT}/realesrgan_x4plus/{it}.png",
             f"{OUT}/realesr_general_v3/{it}.png", f"{OUT}/swinir_gan/{it}.png", f"{GT}/text_notice.png"]
    ims = crop_all(paths, box)
    titles = [t["in"], t["bic"], t["esr"], t["gen"], t["swin"], t["gt"]]
    cw, ch, gap = 590, int(590 * 225 / 560), 20
    fig = Image.new("RGB", (2 * cw + gap, 3 * (ch + 40)), BG)
    d = ImageDraw.Draw(fig); ft = ImageFont.truetype(FB, 17)
    for i, (ti, im) in enumerate(zip(titles, ims)):
        x = (i % 2) * (cw + gap); y = (i // 2) * (ch + 40)
        d.text((x + 2, y + 26), ti, fill=INK, font=ft, anchor="ls")
        fig.paste(im.resize((cw, ch), Image.LANCZOS), (x, y + 34))
        d.rectangle([x, y + 34, x + cw - 1, y + 34 + ch - 1], outline=(220, 222, 228))
    save(fig, f"fig3-text-{lang}")

    # 4) noise: kodim23 macaw head
    k = "kodim23"; box = (120, 60, 360, 300)
    paths = [f"{INP}/noise_{k}.png", f"{OUT}/nlmeans/noise_{k}.png", f"{OUT}/scunet/noise_{k}.png", f"{GT}/{k}.png"]
    ims = [load(p).crop(box) for p in paths]
    r = M["noise"][f"noise_{k}"]
    subs = [t["db"].format(v=num(r["input"]["psnr"], lang, 1)), t["db"].format(v=num(r["nlmeans"]["psnr"], lang, 1)),
            t["db"].format(v=num(r["scunet"]["psnr"], lang, 1)), ""]
    titles = [t["noisy"], t["nlm"], t["scu"], t["gt"]]
    save(strip(list(zip(titles, subs, ims)), 292, 292, gap=10), f"fig4-noise-{lang}")

    # cover 1600x900: the SAME red macaw twice, left noisy / right SCUNet
    # (kodim23 has two parrots side by side; splitting the whole frame showed a different bird on each side)
    k = "kodim23"
    box = (376, 30, 760, 462)  # 384x432 region -> two 800x900 panels
    a = load(f"{INP}/noise_{k}.png").crop(box).resize((800, 900), Image.LANCZOS)
    b = load(f"{OUT}/scunet/noise_{k}.png").crop(box).resize((800, 900), Image.LANCZOS)
    cov = Image.new("RGB", (1600, 900), BG); cov.paste(a, (0, 0)); cov.paste(b, (800, 0))
    d = ImageDraw.Draw(cov)
    d.rectangle([797, 0, 802, 900], fill=(255, 255, 255))
    fb = ImageFont.truetype(FB, 44)
    for x, s in [(60, t["before"]), (1540, t["after"])]:
        anchor = "ls" if x < 800 else "rs"
        tb = d.textbbox((x, 860), s, font=fb, anchor=anchor)
        d.rectangle([tb[0] - 14, tb[1] - 10, tb[2] + 14, tb[3] + 10], fill=(17, 19, 26))
        d.text((x, 860), s, font=fb, fill=(255, 255, 255), anchor=anchor)
    save(cov, f"cover-{lang}")

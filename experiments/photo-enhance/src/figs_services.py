"""Figures for the consumer-service test: one example per service card + text across services.
Output: figs/svc-*.jpg, figs/fig5-text-services-*.jpg"""
import json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT, INP, OUT, FIG = f"{ROOT}/gt", f"{ROOT}/inputs", f"{ROOT}/out", f"{ROOT}/figs"
os.makedirs(FIG, exist_ok=True)
MS = json.load(open(f"{OUT}/metrics_services.json"))
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
INK, SUB, BG = (24, 26, 33), (95, 99, 110), (255, 255, 255)
FACE_BOX, SIGN_BOX = (330, 20, 730, 370), (448, 232, 768, 512)
TEXT_BOX = (30, 256, 400, 326)


def load(p):
    return Image.open(p).convert("RGB")


def num(v, lang, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace(".", ",") if lang == "ru" else s


def strip(panels, cw, ch, gap=10, top=30, bottom=26):
    n = len(panels)
    fig = Image.new("RGB", (n * cw + (n - 1) * gap, top + ch + bottom), BG)
    d = ImageDraw.Draw(fig)
    ft, fs = ImageFont.truetype(FB, 17), ImageFont.truetype(FR, 15)
    for i, (title, sub, im) in enumerate(panels):
        x = i * (cw + gap)
        fig.paste(im.resize((cw, ch), Image.LANCZOS), (x, top))
        d.text((x + cw / 2, top - 8), title, fill=INK, font=ft, anchor="ms")
        if sub:
            d.text((x + cw / 2, top + ch + 19), sub, fill=SUB, font=fs, anchor="ms")
    return fig


def vstrip(rows, scale, label_w=200, gap=12):
    """rows: [(title, sub, PIL crop)] -> rows stacked, label column on the left"""
    w, h = rows[0][2].size
    W, H = label_w + int(w * scale), int(h * scale)
    fig = Image.new("RGB", (W, len(rows) * (H + gap) - gap), BG)
    d = ImageDraw.Draw(fig)
    ft, fs = ImageFont.truetype(FB, 17), ImageFont.truetype(FR, 14)
    for i, (title, sub, im) in enumerate(rows):
        y = i * (H + gap)
        fig.paste(im.resize((int(w * scale), H), Image.LANCZOS), (label_w, y))
        d.text((label_w - 14, y + H / 2 - (9 if sub else 0)), title, fill=INK, font=ft, anchor="rm")
        if sub:
            d.text((label_w - 14, y + H / 2 + 12), sub, fill=SUB, font=fs, anchor="rm")
    return fig


def three(svc, iid, gt, box, name, sub, lang):
    inp = load(f"{INP}/{iid}.jpg").resize((768, 512), Image.NEAREST).crop(box)
    res = load(f"{OUT}/{svc}/{iid}.png").crop(box)
    ref = load(f"{GT}/{gt}").crop(box)
    t = {"ru": ("Вход", "что прислали", "Оригинал", "эталон"), "en": ("Input", "what you have", "Original", "reference")}[lang]
    w, h = box[2] - box[0], box[3] - box[1]
    cw = 360
    return strip([(t[0], t[1], inp), (name, sub, res), (t[2], t[3], ref)], cw, round(cw * h / w))


F = MS["face"]
for lang in ["ru", "en"]:
    idt = {"ru": "сходство с оригиналом {v}", "en": "identity {v}"}[lang]
    wm = {"ru": ", водяной знак", "en": ", watermarked"}[lang]
    figs = {
        "upscayl": three("svc_upscayl", "small_kodim08", "kodim08.png", SIGN_BOX, "Upscayl",
                         {"ru": "мягко, без выдумок", "en": "soft, nothing invented"}[lang], lang),
        "gemini": three("svc_gemini", "small_kodim08", "kodim08.png", SIGN_BOX, "Gemini",
                        {"ru": "чётко, но вывеска другая", "en": "sharp, but a different sign"}[lang], lang),
        "krea": three("svc_krea", "face_heavy_kodim15", "kodim15.png", FACE_BOX, "Krea",
                      idt.format(v=num(F["svc_krea"]["identity"], lang)), lang),
        "remini": three("svc_remini", "face_heavy_kodim15", "kodim15.png", FACE_BOX, "Remini",
                        idt.format(v=num(F["svc_remini"]["identity"], lang)) + wm, lang),
        "chatgpt": three("svc_chatgpt", "face_heavy_kodim15", "kodim15.png", FACE_BOX, "ChatGPT",
                         idt.format(v=num(F["svc_chatgpt"]["identity"], lang)), lang),
    }
    for k, im in figs.items():
        im.save(f"{FIG}/svc-{k}-{lang}.jpg", quality=90)

    # text across services: Wi-Fi password and price lines
    T = {"ru": {"gt": ("Оригинал", "эталон"), "in": ("Вход", "что прислали")},
         "en": {"gt": ("Original", "reference"), "in": ("Input", "what you have")}}[lang]
    rows = [(T["gt"][0], T["gt"][1], load(f"{GT}/text_notice.png").crop(TEXT_BOX)),
            (T["in"][0], T["in"][1], load(f"{INP}/text_notice.jpg").resize((768, 512), Image.NEAREST).crop(TEXT_BOX))]
    for svc, name in [("svc_upscayl", "Upscayl"), ("svc_krea", "Krea"), ("svc_remini", "Remini"),
                      ("svc_chatgpt", "ChatGPT"), ("svc_gemini", "Gemini")]:
        rows.append((name, None, load(f"{OUT}/{svc}/text_notice.png").crop(TEXT_BOX)))
    vstrip(rows, 2.0).save(f"{FIG}/fig5-text-services-{lang}.jpg", quality=90)
print(sorted(f for f in os.listdir(FIG) if f.startswith(("svc-", "fig5"))))

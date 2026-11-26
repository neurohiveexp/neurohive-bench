"""Run every method on every degraded input. Outputs: out/<method>/<id>.png, out/timing.json"""
import json, os, sys, time
import numpy as np
import cv2
import torch
from PIL import Image
import spandrel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = f"{ROOT}/weights"
INP = f"{ROOT}/inputs"
OUT = f"{ROOT}/out"
torch.set_num_threads(2)
torch.manual_seed(0)

man = json.load(open(f"{INP}/manifest.json"))
only = set(sys.argv[1:])  # optional: method names to (re)run

_models = {}


def load(name, file):
    if name not in _models:
        _models[name] = spandrel.ModelLoader().load_from_file(f"{W}/{file}").eval()
    return _models[name]


def to_t(img):  # PIL RGB -> 1x3xHxW [0,1]
    a = np.asarray(img, np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def from_t(t):
    a = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((a * 255.0).round().astype(np.uint8))


def sr(name, file):
    def f(img):
        return from_t(load(name, file)(to_t(img)))
    return f


def bicubic(img):
    w, h = img.size
    return img.resize((w * 4, h * 4), Image.BICUBIC)


def nlmeans(img):
    a = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    d = cv2.fastNlMeansDenoisingColored(a, None, 15, 15, 7, 21)
    return Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))


def scunet(img):
    return from_t(load("scunet", "scunet_color_real_psnr.pth")(to_t(img)))


_helper = None


def gfpgan(img):
    """Official GFPGAN recipe: faces by GFPGAN v1.4 (512 px aligned), background by Real-ESRGAN x4plus."""
    global _helper
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    if _helper is None:
        _helper = FaceRestoreHelper(4, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
                                    save_ext="png", use_parse=True, device=torch.device("cpu"), model_rootpath=W)
    g = load("gfpgan", "GFPGANv1.4.pth")
    h = _helper
    h.clean_all()
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h.read_image(bgr)
    n = h.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    h.align_warp_face()
    for face in h.cropped_faces:
        x = torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        x = (x - 0.5) / 0.5
        torch.manual_seed(0)
        with torch.inference_mode():
            y = g.model(x, return_rgb=False)[0]
        y = ((y.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) / 2 * 255.0).round().numpy().astype(np.uint8)
        h.add_restored_face(cv2.cvtColor(y, cv2.COLOR_RGB2BGR))
    bg = cv2.cvtColor(np.asarray(sr("x4plus", "RealESRGAN_x4plus.pth")(img)), cv2.COLOR_RGB2BGR)
    h.get_inverse_affine(None)
    out = h.paste_faces_to_input_image(upsample_img=bg)
    gfpgan.faces = n
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


METHODS = {
    "bicubic": bicubic,
    "realesrgan_x4plus": sr("x4plus", "RealESRGAN_x4plus.pth"),
    "realesr_general_v3": sr("gen3", "realesr-general-x4v3.pth"),
    "swinir_gan": sr("swinir", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"),
    "gfpgan": gfpgan,
    "nlmeans": nlmeans,
    "scunet": scunet,
}
PLAN = {
    "small": ["bicubic", "realesrgan_x4plus", "realesr_general_v3", "swinir_gan"],
    "face": ["bicubic", "realesrgan_x4plus", "swinir_gan", "gfpgan"],
    "text": ["bicubic", "realesrgan_x4plus", "realesr_general_v3", "swinir_gan"],
    "noise": ["nlmeans", "scunet"],
}

tpath = f"{OUT}/timing.json"
timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
for scen, items in man.items():
    for it in items:
        img = Image.open(f"{INP}/{it['input']}").convert("RGB")
        for m in PLAN[scen]:
            if only and m not in only:
                continue
            os.makedirs(f"{OUT}/{m}", exist_ok=True)
            t0 = time.time()
            res = METHODS[m](img)
            dt = time.time() - t0
            res.save(f"{OUT}/{m}/{it['id']}.png")
            timing[f"{m}/{it['id']}"] = round(dt, 2)
            extra = f" faces={gfpgan.faces}" if m == "gfpgan" else ""
            print(f"{m:20s} {it['id']:24s} {img.size} -> {res.size} {dt:6.1f}s{extra}", flush=True)
            json.dump(timing, open(tpath, "w"), indent=1)

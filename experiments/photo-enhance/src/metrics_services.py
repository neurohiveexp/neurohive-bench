"""Same metrics as metrics.py, for results captured from consumer services (out/svc_*/).
Writes out/metrics_services.json.

Service outputs larger than 4x were downscaled to 768x512 (in the browser, canvas with
imageSmoothingQuality=high; Alice AI and GigaChat by align_services.py), so every result is
compared with the original at the original size. Generative services may also shift or crop
the frame by a few pixels, so before comparing, each result is brought to the original's frame
with a similarity transform found from SIFT matches (align_services.similarity); PSNR and SSIM
are taken only where the warped result covers the frame."""
import json, os
import numpy as np
import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT, INP, OUT = f"{ROOT}/gt", f"{ROOT}/inputs", f"{ROOT}/out"
man = json.load(open(f"{INP}/manifest.json"))
SERVICES = ["svc_upscayl", "svc_krea", "svc_letsenhance", "svc_remini", "svc_chatgpt", "svc_gemini", "svc_alice", "svc_gigachat"]
REF = ["bicubic", "realesrgan_x4plus", "swinir_gan", "gfpgan"]


def arr(p):
    return np.asarray(Image.open(p).convert("RGB"))


def ps(gt, x, b=4):
    g, y = (gt[b:-b, b:-b], x[b:-b, b:-b]) if b else (gt, x)
    return round(float(psnr(g, y, data_range=255)), 2), round(float(ssim(g, y, channel_axis=2, data_range=255)), 4)


sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
from align_services import similarity


def aligned(m, p, gt):
    """result warped to the original's frame and the mask of covered pixels; references are left as they are"""
    x = arr(p)
    if not m.startswith("svc_"):
        return x, np.ones(x.shape[:2], bool)
    M, inl, _ = similarity(x, gt)
    if M is None or inl < 15:
        return x, np.ones(x.shape[:2], bool)
    sc, ang = float(np.hypot(M[0, 0], M[1, 0])), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if abs(sc - 1) < 0.002 and abs(ang) < 0.1 and abs(M[0, 2]) < 1 and abs(M[1, 2]) < 1:
        return x, np.ones(x.shape[:2], bool)  # already in the original's frame: no resampling
    w = cv2.warpAffine(x, M, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    v = cv2.warpAffine(np.ones(x.shape[:2], np.uint8), M, (x.shape[1], x.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return w, cv2.erode(v, np.ones((9, 9), np.uint8)) > 0


def ps_masked(gt, x, mask, b=4):
    m = mask.copy(); m[:b, :] = m[-b:, :] = False; m[:, :b] = m[:, -b:] = False
    mse = float(((gt.astype(np.float64) - x.astype(np.float64)) ** 2)[m].mean())
    _, smap = ssim(gt, x, channel_axis=2, data_range=255, full=True)
    return round(10 * np.log10(255 ** 2 / mse), 2), round(float(smap.mean(axis=2)[m].mean()), 4)


def item(scen, iid):
    return next(i for i in man[scen] if i["id"] == iid)


res = {"small": {}, "face": {}, "text": {}}

# --- small photo: PSNR / SSIM against the original
it = item("small", "small_kodim08")
gt = arr(f"{GT}/{it['gt']}")
for m in REF + SERVICES:
    p = f"{OUT}/{m}/{it['id']}.png"
    if os.path.exists(p):
        x, mask = aligned(m, p, gt)
        res["small"][m] = dict(zip(["psnr", "ssim"], ps_masked(gt, x, mask)))
        res["small"][m]["psnr_unaligned"], res["small"][m]["ssim_unaligned"] = ps(gt, arr(p))

# --- face: ArcFace identity. Landmarks from the original (as in metrics.py) and,
# because generative services can move the face a little, also from the result itself.
import onnxruntime as ort
from facexlib.detection import init_detection_model
from skimage.transform import SimilarityTransform

det = init_detection_model("retinaface_resnet50", half=False, device="cpu", model_rootpath=f"{ROOT}/weights")
sess = ort.InferenceSession(f"{ROOT}/dl/insightface/buffalo_l/w600k_r50.onnx", providers=["CPUExecutionProvider"])
DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def landmarks(rgb):
    with torch.no_grad():
        b = det.detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.9)
    b = sorted(b, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))[0]
    return np.array(b[5:15], np.float32).reshape(5, 2), [int(v) for v in b[:4]]


def embed(rgb, lm):
    t = SimilarityTransform()
    t.estimate(lm, DST)
    face = cv2.warpAffine(rgb, t.params[0:2, :], (112, 112), borderValue=0.0)
    x = ((face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    e = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    return e / np.linalg.norm(e)


it = item("face", "face_heavy_kodim15")
g = arr(f"{GT}/{it['gt']}")
lm, box = landmarks(g)
ge = embed(g, lm)
x0, y0, x1, y1 = box
pad = int(0.25 * (x1 - x0))
crop = (max(0, y0 - pad), min(g.shape[0], y1 + pad), max(0, x0 - pad), min(g.shape[1], x1 + pad))
for m in REF + SERVICES:
    p = f"{OUT}/{m}/{it['id']}.png"
    if not os.path.exists(p):
        continue
    x, _ = aligned(m, p, g)
    own_lm, _ = landmarks(x)
    fp, fs = ps(g[crop[0]:crop[1], crop[2]:crop[3]], x[crop[0]:crop[1], crop[2]:crop[3]], b=0)
    res["face"][m] = {"identity": round(float(embed(x, lm) @ ge), 3),
                      "identity_own_landmarks": round(float(embed(x, own_lm) @ ge), 3),
                      "landmark_shift_px": round(float(np.abs(own_lm - lm).mean()), 1),
                      "face_psnr": fp, "face_ssim": fs}

# --- text: OCR line by line, boxes known from rendering (same as metrics.py)
import easyocr

reader = easyocr.Reader(["ru", "en"], gpu=False, model_storage_directory=f"{ROOT}/dl/easyocr_models",
                        download_enabled=False, verbose=False)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(s.replace("№", "N").split()).lower()


it = man["text"][0]
for m in ["gt"] + REF + SERVICES:
    p = f"{GT}/{it['gt']}" if m == "gt" else f"{OUT}/{m}/{it['id']}.png"
    if not os.path.exists(p):
        continue
    img = cv2.cvtColor(arr(p) if m == "gt" else aligned(m, p, arr(f"{GT}/{it['gt']}"))[0], cv2.COLOR_RGB2GRAY)
    lines, errs, tot = [], 0, 0
    for ln in it["lines"]:
        x0, y0, x1, y1 = ln["box"]
        txt = " ".join(reader.recognize(img[max(0, y0):y1, max(0, x0):x1], detail=0))
        e = lev(norm(txt), norm(ln["text"]))
        errs += e; tot += len(norm(ln["text"]))
        lines.append({"px": ln["lr_px"], "read": txt, "cer": round(e / len(norm(ln["text"])), 3)})
    res["text"][m] = {"cer": round(errs / tot, 3), "lines": lines}

json.dump(res, open(f"{OUT}/metrics_services.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "text"}, ensure_ascii=False, indent=1))
for m, r in res["text"].items():
    print(m, r["cer"], [l["cer"] for l in r["lines"]], "|", r["lines"][4]["read"], "|", r["lines"][5]["read"])

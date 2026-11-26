"""Metrics: PSNR/SSIM vs original, face identity (ArcFace cosine), OCR character error rate.
Writes out/metrics.json."""
import json, os
import numpy as np
import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT, INP, OUT = f"{ROOT}/gt", f"{ROOT}/inputs", f"{ROOT}/out"
man = json.load(open(f"{INP}/manifest.json"))
PLAN = {
    "small": ["bicubic", "realesrgan_x4plus", "realesr_general_v3", "swinir_gan"],
    "face": ["bicubic", "realesrgan_x4plus", "swinir_gan", "gfpgan"],
    "text": ["bicubic", "realesrgan_x4plus", "realesr_general_v3", "swinir_gan"],
    "noise": ["input", "nlmeans", "scunet"],
}


def arr(p):
    return np.asarray(Image.open(p).convert("RGB"))


def out_path(m, it):
    return f"{INP}/{it['input']}" if m == "input" else f"{OUT}/{m}/{it['id']}.png"


def ps(gt, x, b=4):
    g, y = (gt[b:-b, b:-b], x[b:-b, b:-b]) if b else (gt, x)
    return round(float(psnr(g, y, data_range=255)), 2), round(float(ssim(g, y, channel_axis=2, data_range=255)), 4)


res = {"small": {}, "face": {}, "text": {}, "noise": {}}

# --- PSNR / SSIM
for scen in ["small", "noise", "text"]:
    for it in man[scen]:
        gt = arr(f"{GT}/{it['gt']}")
        for m in PLAN[scen]:
            p = out_path(m, it)
            if os.path.exists(p):
                res[scen].setdefault(it["id"], {})[m] = dict(zip(["psnr", "ssim"], ps(gt, arr(p))))

# --- faces: identity with ArcFace (InsightFace buffalo_l / w600k_r50), landmarks from the ORIGINAL
import onnxruntime as ort
from facexlib.detection import init_detection_model
from skimage.transform import SimilarityTransform

det = init_detection_model("retinaface_resnet50", half=False, device="cpu", model_rootpath=f"{ROOT}/weights")
sess = ort.InferenceSession(f"{ROOT}/dl/insightface/buffalo_l/w600k_r50.onnx", providers=["CPUExecutionProvider"])
DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def landmarks(rgb):
    with torch.no_grad():
        b = det.detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.97)
    b = sorted(b, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))[0]
    return np.array(b[5:15], np.float32).reshape(5, 2), [int(v) for v in b[:4]]


def embed(rgb, lm):
    t = SimilarityTransform.from_estimate(lm, DST) if hasattr(SimilarityTransform, "from_estimate") else None
    if t is None:
        t = SimilarityTransform(); t.estimate(lm, DST)
    face = cv2.warpAffine(rgb, t.params[0:2, :], (112, 112), borderValue=0.0)
    x = (face.astype(np.float32) - 127.5) / 127.5
    x = x.transpose(2, 0, 1)[None]
    e = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    return e / np.linalg.norm(e)


gt_emb = {}
for it in man["face"]:
    g = arr(f"{GT}/{it['gt']}")
    lm, box = landmarks(g)
    if it["gt"] not in gt_emb:
        gt_emb[it["gt"]] = embed(g, lm)
    x0, y0, x1, y1 = box
    pad = int(0.25 * (x1 - x0))
    crop = (max(0, y0 - pad), min(g.shape[0], y1 + pad), max(0, x0 - pad), min(g.shape[1], x1 + pad))
    r = {"_box": box}
    for m in PLAN["face"]:
        p = out_path(m, it)
        if not os.path.exists(p):
            continue
        x = arr(p)
        e = embed(x, lm)
        fp, fs = ps(g[crop[0]:crop[1], crop[2]:crop[3]], x[crop[0]:crop[1], crop[2]:crop[3]], b=0)
        r[m] = {"identity": round(float(e @ gt_emb[it["gt"]]), 3), "face_psnr": fp, "face_ssim": fs}
    res["face"][it["id"]] = r
ks = list(gt_emb)
res["face"]["_different_people"] = round(float(gt_emb[ks[0]] @ gt_emb[ks[1]]), 3)

# --- OCR on the text notice, line by line, boxes known from rendering
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
ocr = {}
for m in ["gt"] + PLAN["text"]:
    p = f"{GT}/{it['gt']}" if m == "gt" else out_path(m, it)
    if not os.path.exists(p):
        continue
    img = cv2.cvtColor(arr(p), cv2.COLOR_RGB2GRAY)
    lines, errs, tot = [], 0, 0
    for ln in it["lines"]:
        x0, y0, x1, y1 = ln["box"]
        crop = img[max(0, y0):y1, max(0, x0):x1]
        out = reader.recognize(crop, detail=0)
        txt = " ".join(out)
        e = lev(norm(txt), norm(ln["text"]))
        errs += e; tot += len(norm(ln["text"]))
        lines.append({"px": ln["lr_px"], "read": txt, "cer": round(e / len(norm(ln["text"])), 3)})
    ocr[m] = {"cer": round(errs / tot, 3), "lines": lines}
res["text"]["_ocr"] = ocr

json.dump(res, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:6000])

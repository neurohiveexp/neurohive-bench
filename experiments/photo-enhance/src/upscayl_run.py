"""Run Upscayl's default model (Upscayl Standard 4x, NCNN files from the Upscayl repo) on CPU,
the same way upscayl-bin / realesrgan-ncnn does: RGB 0..1 in, x4 out, clamp, round.
Output: out/svc_upscayl/<id>.png"""
import os, sys, time, json
import numpy as np
import ncnn
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{ROOT}/dl/upscayl_models/upscayl-standard-4x"
OUT = f"{ROOT}/out/svc_upscayl"
os.makedirs(OUT, exist_ok=True)

net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 2
net.load_param(M + ".param")
net.load_model(M + ".bin")

PAD = 10  # realesrgan-ncnn prepadding: reflect-pad the input, crop the output
items = sys.argv[1:] or ["text_notice", "face_heavy_kodim15", "small_kodim08"]
times = {}
for it in items:
    img = np.asarray(Image.open(f"{ROOT}/inputs/{it}.jpg").convert("RGB"))
    h, w = img.shape[:2]
    padded = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]
    t0 = time.time()
    mat = ncnn.Mat.from_pixels(np.ascontiguousarray(padded), ncnn.Mat.PixelType.PIXEL_RGB, pw, ph)
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)
    ex = net.create_extractor()
    ex.input("data", mat)
    ret, out = ex.extract("output")
    o = np.array(out)  # 3 x 4ph x 4pw, 0..1
    o = o[:, PAD * 4:PAD * 4 + h * 4, PAD * 4:PAD * 4 + w * 4]
    o = (np.clip(o, 0, 1) * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
    times[it] = round(time.time() - t0, 2)
    Image.fromarray(o).save(f"{OUT}/{it}.png")
    print(it, img.shape, "->", o.shape, times[it], "s")
json.dump(times, open(f"{OUT}/timing.json", "w"))

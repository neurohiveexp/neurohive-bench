# Photo enhancement test: what AI restores and what it invents

Open AI photo enhancers on four everyday cases — a small photo, a face, small text and noise. Used in the Neurohive articles «Нейросети для улучшения качества фото» (RU) and «Best AI Photo Enhancers» (EN), November 2026. Everything runs on CPU, no API calls.

## Data
- 7 images from the Kodak Lossless True Color Image Suite (kodim01, 03, 04, 08, 13, 15, 23), fetched by `get_data.sh` from the [GitHub mirror](https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite); not stored here.
- One synthetic notice rendered with DejaVu Sans: `gt/text_notice.png` (made by `src/prep.py`).

## Degradations (`src/prep.py`, fixed seeds) → `inputs/`
| Scenario | Images | Degradation |
| --- | --- | --- |
| small | kodim01, 08, 13, 23 | bicubic 4x downscale, JPEG q75 (4:2:0) |
| face | kodim04, 15 × light / heavy | Gaussian blur 1.0 / 2.0 px, 4x downscale, JPEG q60 / q40 |
| text | text_notice | blur 0.6 px, 4x downscale, JPEG q75; capital letters 8.0–3.3 px tall in the small image |
| noise | kodim03, 23 | Gaussian noise σ = 25 at full size |

## Methods (`src/run.py`) → `out/<method>/`
| Folder | Method |
| --- | --- |
| `bicubic` | Pillow bicubic 4x |
| `realesrgan_x4plus` | Real-ESRGAN x4plus |
| `realesr_general_v3` | realesr-general-x4v3 (compact model from the Real-ESRGAN project) |
| `swinir_gan` | SwinIR-L Real-SR GAN (003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN) |
| `gfpgan` | GFPGAN v1.4 faces (facexlib RetinaFace, paste-back) + Real-ESRGAN x4plus background |
| `scunet` | SCUNet, color real PSNR |
| `nlmeans` | OpenCV fastNlMeansDenoisingColored (h = 15) |

Weights from the official GitHub releases, loaded with spandrel 0.4.2 on PyTorch 2.15. Timing on 2 CPU cores: `out/timing.json`.

## Metrics (`src/metrics.py`) → `out/metrics.json`
- PSNR and SSIM (scikit-image), 4 px border cropped.
- Face identity: cosine similarity of ArcFace embeddings (InsightFace buffalo_l, `w600k_r50.onnx`), face landmarks taken from the original. Two different people from the set: 0.064.
- Text: EasyOCR 1.7.2 (ru + en) on known line boxes, character error rate by Levenshtein distance.

## Results
**Text**, share of misread characters by capital height in the small image:

| Capital height | Bicubic | Real-ESRGAN | General v3 | SwinIR |
| --- | --- | --- | --- | --- |
| 8.0 px | 6% | 6% | 6% | 6% |
| 6.2 px | 4% | 12% | 8% | 0% |
| 5.1 px | 42% | 46% | 30% | 30% |
| 4.4 px | 100% | 75% | 61% | 64% |
| 3.6 px | 97% | 61% | 76% | 73% |
| 3.3 px | 96% | 75% | 75% | 86% |
| all lines | 62% | 49% | 46% | 47% |

OCR on the original: 2.4% (it reads «№» as «Ng», hence 6% in the top line). All three GAN models render the password 7781 as 7701; Real-ESRGAN and SwinIR turn «15%» into «10%» (see `figs/fig3-text-*.jpg`).

**Faces**, ArcFace similarity to the original (kodim04 / kodim15):

| Damage | Bicubic | Real-ESRGAN | SwinIR | GFPGAN |
| --- | --- | --- | --- | --- |
| light | 0.88 / 0.90 | 0.83 / 0.87 | 0.85 / 0.89 | 0.83 / 0.84 |
| heavy | 0.84 / 0.86 | 0.76 / 0.81 | 0.78 / 0.84 | 0.78 / 0.76 |

**Small photo**, mean PSNR / SSIM: bicubic 22.17 / 0.537; Real-ESRGAN 21.82 / 0.547; General v3 21.57 / 0.563; SwinIR 21.72 / 0.552.

**Noise**, PSNR / SSIM: noisy 20.4 / 0.22; NL-means 29.9–30.4 / 0.77–0.83; SCUNet 34.9–35.0 / 0.91–0.92.

Article figures (RU and EN labels, 1200 px; covers 1600×900): `figs/`.

## Reproduce
```bash
pip install -r requirements.txt
bash get_data.sh
python src/prep.py && python src/run.py && python src/metrics.py && python src/figs.py
```
`prep.py` and `figs.py` use DejaVu fonts from `/usr/share/fonts/truetype/dejavu/` (package `fonts-dejavu-core`).

## Limits
- Degradations are synthetic; real photos mix motion blur, perspective and glare.
- Two portraits: an illustration, not statistics. ArcFace is robust to blur, which keeps blurry bicubic results similar to the original.
- OCR reads worse than people do; the text threshold holds for one font on a flat sheet.
- Paid services were not run (zero budget).

License: MIT for code, CC BY 4.0 for figures and results. The InsightFace model is licensed for non-commercial research and is used only as a measuring tool; it is not redistributed.

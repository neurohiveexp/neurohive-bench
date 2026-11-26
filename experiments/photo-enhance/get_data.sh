#!/usr/bin/env bash
# Downloads what is not stored in the repository: 7 Kodak originals, model weights,
# the ArcFace model (InsightFace buffalo_l) and EasyOCR models. Everything comes from GitHub.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p gt weights dl/insightface/buffalo_l dl/easyocr_models

K=https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992
for n in 01 03 04 08 13 15 23; do
  curl -fsSL -o "gt/kodim$n.png" "$K/$n.png"
done

(cd weights && for u in \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
  https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
  https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
  https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth; do
  curl -fsSLO "$u"
done)

curl -fsSL -o dl/insightface/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -oq dl/insightface/buffalo_l.zip -d dl/insightface/buffalo_l

for z in pre-v1.1.6/craft_mlt_25k.zip v1.6.1/cyrillic_g2.zip v1.3/english_g2.zip; do
  f="dl/easyocr_models/$(basename "$z")"
  curl -fsSL -o "$f" "https://github.com/JaidedAI/EasyOCR/releases/download/$z"
  unzip -oq "$f" -d dl/easyocr_models
done
echo "done"

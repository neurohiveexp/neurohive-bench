"""Bring service results to the original's frame (768x512) for comparison.

Some services return a picture of another size and framing (Alice AI 1248x848,
GigaChat 2048x1376, both slightly cropped or zoomed). A plain resize would
shift every pixel, so we find the similarity transform (scale, rotation, shift)
from SIFT matches with the original and warp the result into its frame.

Usage: python src/align_services.py svc_alice svc_gigachat   -> out/<svc>/<id>.png from out/<svc>/orig/<id>.*
       python src/align_services.py --check svc_gemini ...   -> only report the transform for results already at 768x512
"""
import glob, json, os, sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT, OUT = f"{ROOT}/gt", f"{ROOT}/out"
GTS = {"face_heavy_kodim15": "kodim15.png", "text_notice": "text_notice.png", "small_kodim08": "kodim08.png"}
W, H = 768, 512


def read(p):
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def similarity(res, gt):
    """res already resized to width 768; returns 2x3 matrix res->gt, inliers, matches"""
    sift = cv2.SIFT_create(5000)
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    r = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    kg, dg = sift.detectAndCompute(g, None)
    kr, dr = sift.detectAndCompute(r, None)
    if dg is None or dr is None:
        return None, 0, 0
    good = [a for a, b in cv2.BFMatcher().knnMatch(dr, dg, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return None, 0, len(good)
    src = np.float32([kr[a.queryIdx].pt for a in good])
    dst = np.float32([kg[a.trainIdx].pt for a in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    return M, int(inl.sum()) if inl is not None else 0, len(good)


def describe(M):
    s = float(np.hypot(M[0, 0], M[1, 0]))
    a = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return {"scale": round(s, 4), "angle_deg": round(a, 2), "dx": round(float(M[0, 2]), 1), "dy": round(float(M[1, 2]), 1)}


def main():
    args = sys.argv[1:]
    check = "--check" in args
    svcs = [a for a in args if not a.startswith("--")]
    report = {}
    for svc in svcs:
        for iid, gname in GTS.items():
            gt = read(f"{GT}/{gname}")
            if check:
                p = f"{OUT}/{svc}/{iid}.png"
            else:
                c = glob.glob(f"{OUT}/{svc}/orig/{iid}.*")
                p = c[0] if c else None
            if not p or not os.path.exists(p):
                continue
            res = read(p)
            h = round(res.shape[0] * W / res.shape[1])
            res_w = cv2.resize(res, (W, h), interpolation=cv2.INTER_AREA)
            M, inl, n = similarity(res_w, gt)
            rec = {"file": os.path.relpath(p, ROOT), "size": [res.shape[1], res.shape[0]], "matches": n, "inliers": inl}
            if M is None or inl < 15:
                rec["aligned"] = False  # fall back to a plain resize
                out = cv2.resize(res, (W, H), interpolation=cv2.INTER_AREA)
            else:
                rec["aligned"] = True
                rec.update(describe(M))
                out = cv2.warpAffine(res_w, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                valid = cv2.warpAffine(np.ones(res_w.shape[:2], np.uint8), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                rec["covered"] = round(float(valid.mean()), 3)
            if not check:
                cv2.imwrite(f"{OUT}/{svc}/{iid}.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            report[f"{svc}/{iid}"] = rec
    print(json.dumps(report, ensure_ascii=False, indent=0))
    if not check:
        prev = {}
        f = f"{OUT}/alignment.json"
        if os.path.exists(f):
            prev = json.load(open(f))
        prev.update(report)
        json.dump(prev, open(f, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

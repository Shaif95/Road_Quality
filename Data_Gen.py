#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from pathlib import Path
from datetime import timedelta
import cv2
from tqdm import tqdm
import csv
import re

# ---------- CONFIG ----------
src = Path(r"E:\Traffic_Vidoes")            # folder with videos
base = Path(r"E:\traffic data")             # root output
frames_root = base / "frames"
loc_root = base / "location"

fmt = "jpg"                                 # 'jpg' or 'png' for TOP frames
jpg_quality = 95
# split ratios (tune if your overlay height/left map width differ)
bottom_ratio = 0.32                         # ~bottom 32% is overlay
left_map_ratio = 0.28                       # ~left 28% is Google map thumbnail
# --------------------------------

# Ensure outputs exist
frames_root.mkdir(parents=True, exist_ok=True)
loc_root.mkdir(parents=True, exist_ok=True)

# Enumerate videos (non-recursive; use rglob("*.mp4") if you want recursion)
vids = sorted(src.glob("*.mp4"))

if len(vids) == 0:
    print(f"[warn] no .mp4 found in {src}")

# Optional (Windows): set the Tesseract path if pytesseract can’t find it
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_latlon(img_bgr):
    """
    OCR latitude & longitude from the bottom overlay crop of a frame.
    Returns (lat_str, lon_str) or (None, None).
    """
    try:
        import pytesseract
    except Exception:
        return None, None

    # 1) remove left mini-map to focus on text
    h, w = img_bgr.shape[:2]
    x_cut = int(w * left_map_ratio)
    roi = img_bgr[:, x_cut:, :]

    # 2) upscale + grayscale + CLAHE + tophat + Otsu (invert)
    roi = cv2.resize(roi, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 3) OCR
    cfg = "--oem 3 --psm 6 -l eng"
    txt = pytesseract.image_to_string(bw, config=cfg)

    # 4) Parse robustly
    pats = [
        r"Lat\s*([0-9\.\-]+)[^\d\-]+Long\s*([0-9\.\-]+)",
        r"Lat\s*([0-9\.\-]+)\s*[°]?\s*Long\s*([0-9\.\-]+)",
        r"([0-9]+\.[0-9]+)\s*[°]?\s*[, ]+\s*(-?[0-9]+\.[0-9]+)"
    ]
    for p in pats:
        m = re.search(p, txt, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
    return None, None

def save_top_frame(img_bgr, out_path, fmt="jpg", jpg_quality=95):
    if fmt.lower() == "jpg":
        cv2.imwrite(str(out_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    elif fmt.lower() == "png":
        cv2.imwrite(str(out_path), img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        raise ValueError("fmt must be 'jpg' or 'png'.")

# Outer progress: videos
pbar_vids = tqdm(vids, desc="All videos", unit="video")
for v in pbar_vids:
    name = v.stem

    # Per-video output
    out_frames = frames_root / name
    out_loc = loc_root / name
    out_frames.mkdir(parents=True, exist_ok=True)
    out_loc.mkdir(parents=True, exist_ok=True)

    # CSV path
    csv_path = out_loc / f"{name}.csv"
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(fcsv)
    wr.writerow(["video_name", "frame_number", "lat", "lon"])

    cap = cv2.VideoCapture(str(v))
    if not cap.isOpened():
        fcsv.close()
        print(f"[error] cannot open video: {v}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inner progress: frames for this video
    desc = f"{name} ({wv}x{hv} @ {fps:.2f} fps)"
    pbar_frames = tqdm(total=n if n > 0 else None, desc=desc, unit="frame", leave=False)

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Split top vs bottom
        h, w = frame.shape[:2]
        cut_y = int(h * (1.0 - bottom_ratio))
        top = frame[:cut_y, :, :]
        bottom = frame[cut_y:, :, :]

        # OCR first
        lat, lon = ocr_latlon(bottom)

        if lat and lon:
            # Save top frame only when coordinates exist
            fn = out_frames / f"frame_{idx:06d}.{fmt}"
            save_top_frame(top, fn, fmt=fmt, jpg_quality=jpg_quality)
            saved = saved + 1

            # Write CSV row
            wr.writerow([name, idx, lat, lon])
        else:
            # If for any reason a file with this index already exists, remove it
            fn = out_frames / f"frame_{idx:06d}.{fmt}"
            if fn.exists():
                try:
                    fn.unlink()
                except Exception:
                    pass  # ignore cleanup errors

        idx = idx + 1
        pbar_frames.update(1)

    pbar_frames.close()
    cap.release()
    fcsv.close()

    # Optional summary per video
    dur_sec = saved / fps if fps and fps > 0 else 0
    print(f"[done] {name}: kept {saved} frames -> {out_frames} | CSV -> {csv_path} | duration ~ {timedelta(seconds=dur_sec)}")

pbar_vids.close()
print(f"[all done] Outputs under: {base}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





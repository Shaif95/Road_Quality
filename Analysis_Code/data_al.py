import os
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

# ---------- CONFIG ----------
videos_dir = r"E:\Traffic_Vidoes"
frames_dir = r"E:\traffic data\frames"
location_dir = r"E:\traffic data\location"

# ---------- STEP 1: BASIC INFO ----------
video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
frame_folders = sorted([f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))])
location_folders = sorted([f for f in os.listdir(location_dir) if os.path.isdir(os.path.join(location_dir, f))])

print("\n========== DATASET V1 STRUCTURE ==========")
print(f"📹 Total video files (.mp4): {len(video_files)}")
print(f"🖼️  Total frame folders:      {len(frame_folders)}")
print(f"📍 Total location folders:   {len(location_folders)}\n")

# ---------- STEP 2: SCAN AND ANALYZE ----------
summary = []
all_lat, all_lon = [], []
total_valid_frames = 0
missing_latlon_files = 0

for folder in tqdm(location_folders, desc="Analyzing CSVs"):
    csv_files = glob.glob(os.path.join(location_dir, folder, "*.csv"))
    folder_frame_count = 0
    lat_count, lon_count = 0, 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=None, engine="python")
            df.columns = [c.strip().lower() for c in df.columns]

            if not {'lat', 'lon'}.issubset(df.columns):
                missing_latlon_files += 1
                continue

            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]

            folder_frame_count += len(df)
            all_lat.extend(df['lat'].tolist())
            all_lon.extend(df['lon'].tolist())

        except Exception as e:
            print(f"[WARN] Could not read {csv_file}: {e}")

    summary.append({
        "Folder": folder,
        "ValidFrames": folder_frame_count
    })
    total_valid_frames += folder_frame_count

# ---------- STEP 3: COMPUTE STATISTICS ----------
video_count = len(video_files)
avg_frames_per_video = total_valid_frames / video_count if video_count > 0 else 0

print("\n========== DATASET SUMMARY ==========")
print(f"📹 Total Videos:           {video_count}")
print(f"🖼️  Total Valid Frames:     {total_valid_frames}")
print(f"⚙️  Avg Frames per Video:   {avg_frames_per_video:.2f}")
print(f"⚠️  CSVs Missing lat/lon:   {missing_latlon_files}")

if all_lat and all_lon:
    print(f"🌍 Latitude range:          {np.min(all_lat):.6f} → {np.max(all_lat):.6f}")
    print(f"🧭 Longitude range:         {np.min(all_lon):.6f} → {np.max(all_lon):.6f}")

# ---------- STEP 4: SHOW PER-FOLDER DETAILS ----------
print("\n========== FRAMES PER VIDEO ==========")
for s in summary:
    print(f"{s['Folder']:<45} | Frames: {s['ValidFrames']}")

print("\n✅ Analysis complete — all summary printed above.")

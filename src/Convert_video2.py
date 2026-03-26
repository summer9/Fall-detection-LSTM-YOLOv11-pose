# ================================================================
# This file is to convert from videos to CSV file with 17 body keypoints. I convert for each subject to control the accuracy
# Instead of 29fps, I use 10fps
# GMDCSA24 - Subject 1: Extract keypoints + correct timed labels
# → Processes ALL videos in ADL and Fall folders
# Output: GMDCSA24_yolo11pose_S1.csv
# ================================================================

from ultralytics import YOLO
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import glob

# -------------------------- CONFIG --------------------------
BASE_PATH = "E:\PROJECT\HAMK\YEAR 3\THESIS\PRACTICAL PART\DATA\GMDCSA24\Subject 4"
ADL_FOLDER = os.path.join(BASE_PATH, "ADL")
FALL_FOLDER = os.path.join(BASE_PATH, "Fall")
ADL_CSV = os.path.join(BASE_PATH, "ADL.csv")
FALL_CSV = os.path.join(BASE_PATH, "Fall.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "GMDCSA24_yolo11pose_S4_10fps_a.csv")
SUBJECT = "S4"

model = YOLO("yolo11s-pose.pt")
FPS = 10

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
# -------------------------- Parse ALL annotations from CSV --------------------------
def parse_all_annotations(csv_path, folder_name):
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    file_col = next((c for c in df.columns if 'file' in c.lower() and 'name' in c.lower()), df.columns[0])
    class_col = next((c for c in df.columns if c.lower() == 'classes'), None)
    if class_col is None:
        class_col = df.columns[-1]  # fallback

    segments_dict = {}
    pattern = r"([A-Za-z]+)\s*(?:\([^)]*\))?\s*\[([\d.]+)\s*to\s*([\d.]+)\]"

    print(f"Parsing {len(df)} rows from {os.path.basename(csv_path)}...")
    for _, row in df.iterrows():
        filename = str(row[file_col]).strip()
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        text = str(row[class_col])
        matches = re.findall(pattern, text)
        segments = []
        for label_raw, start, end in matches:
            label = label_raw.strip()
            if label in ["Fall", "Falling", "fall", "FALL"]:
                label = "Falling"
            elif label in ["Sit", "Sitting"]:
                label = "Sitting"
            elif label in ["Sleep", "Sleeping", "Lay", "Lying"]:
                label = "Sleeping"
            # add more mappings if needed
            try:
                segments.append((label, float(start), float(end)))
            except ValueError:
                continue

        if segments:
            segments_dict[filename] = segments
            print(f"  → {folder_name}/{filename} → {segments}")
        else:
            print(f"  Warning: No valid segments found for {filename}")

    print(f"  Found annotations for {len(segments_dict)} videos in {folder_name}")
    return segments_dict


# -------------------------- Load ALL annotations --------------------------
print("Loading ALL annotations from CSV files...")
adl_segments = parse_all_annotations(ADL_CSV, "ADL")
fall_segments = parse_all_annotations(FALL_CSV, "Fall")
all_segments = {**adl_segments, **fall_segments}

# -------------------------- Find actual video files --------------------------
adl_videos = {os.path.basename(p): p for p in glob.glob(os.path.join(ADL_FOLDER, "*.mp4"))}
fall_videos = {os.path.basename(p): p for p in glob.glob(os.path.join(FALL_FOLDER, "*.mp4"))}

print(f"Found {len(adl_videos)} video files in ADL folder")
print(f"Found {len(fall_videos)} video files in Fall folder")

# -------------------------- Match videos that have annotations --------------------------
valid_adl_videos = {name: path for name, path in adl_videos.items() if name in adl_segments}
valid_fall_videos = {name: path for name, path in fall_videos.items() if name in fall_segments}

print(f"Videos with annotations → ADL: {len(valid_adl_videos)}, Fall: {len(valid_fall_videos)}")


# -------------------------- Label function --------------------------
def get_label(segments, sec):
    for label, s, e in segments:
        if s <= sec <= e:
            return label
    return "Other"  # background / no annotation


# -------------------------- Main Processing --------------------------
all_rows = []
# loop over all videos
for folder_dict, folder_name, segments_dict in [
    (valid_adl_videos, "ADL", adl_segments),
    (valid_fall_videos, "Fall", fall_segments)
]:
    print(f"\nProcessing {folder_name} folder ({len(folder_dict)} videos)...")

    for video_file, video_path in tqdm(folder_dict.items(), desc=folder_name):
        if video_file not in segments_dict:
            print(f"  Skipping {video_file} - no annotation")
            continue

        segments = segments_dict[video_file]
        # open each video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: Cannot open video {video_path}")
            continue
        # videos become frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        # Frame index → time
            second = round(frame_idx / FPS, 4)
            label = get_label(segments, second)
        # Keypoints extraction
           
            results = model(frame, conf=0.3, verbose=False)[0]

            kpts = np.zeros(51)  

            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                # Since only one person → safely take index 0
                conf_array = results.keypoints.conf[0].cpu().numpy()  # shape (17,)
                # Calculate average confidence across all 17 keypoints
                avg_conf = np.mean(conf_array)
                # Optional: also check nose confidence specifically (index 0 = nose)
                nose_conf = conf_array[0]
                # Print for debugging 
                #print(f"Frame {frame_idx:4d} | avg_conf: {avg_conf:.3f} | nose_conf: {nose_conf:.3f}")

                
                if avg_conf >= 0.25: # can change this threshold
                    xy = results.keypoints.xy[0].cpu().numpy()  # (17, 2)
                   
                    i = 0
                    for j in range(17):
                        kpts[i]   = xy[j, 0]      # x
                        kpts[i+1] = xy[j, 1]      # y
                        kpts[i+2] = conf_array[j] # c
                        i += 3
                        kpts = np.round(kpts, 4)
            # Build rows
            row =  [
                video_file,
                f"{SUBJECT}/{folder_name}/{video_file}",
                frame_idx,
                second,
                label
            ]+ list(kpts)
            all_rows.append(row)
            frame_idx += 1

        cap.release()

# -------------------------- Save CSV --------------------------
kp_cols = [f"{k}_{c}" for k in KEYPOINT_NAMES for c in ['x', 'y', 'c']]
columns = ['video_name', 'video_path', 'frame', 'second', 'class'] + kp_cols

df = pd.DataFrame(all_rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("Full Subject  dataset extracted")
print(f"   Output → {OUTPUT_CSV}")
print(f"   Total frames → {len(df):,}")
print(f"   Unique classes → {sorted(df['class'].unique())}")
print(f"   Total videos processed → {df['video_path'].nunique()}")
print("=" * 80)
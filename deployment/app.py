# app.py
import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
from tqdm import tqdm

# ────────────────────────────────────────────────
# 1. Load models (do this once at startup)
# ────────────────────────────────────────────────

pose_model = YOLO("yolo11s-pose.pt")        

class FallingBinaryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])

device = "cuda" if torch.cuda.is_available() else "cpu"

lstm_model = FallingBinaryLSTM().to(device)
lstm_model.load_state_dict(torch.load("BEST_MODEL_Fall_Nonfall_LSTM_3layers_f1_30fps_optimized.pth", map_location=device))
lstm_model.eval()

# Hyperparams — must match training!
WINDOW_SIZE = 90
STRIDE      = 30
FPS_TARGET  = 30               # keypoints extraction rate
CONF_THRESHOLD = 0.25          # min avg keypoint confidence

KEYPOINT_NAMES = [...]         # copy the list of 17 names

# ────────────────────────────────────────────────
# 2. Core inference function
# ────────────────────────────────────────────────

def detect_fall_in_video(video_path: str):
    if not video_path:
        return None, "No video uploaded", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video", None

    orig_fps   = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # We will subsample to ~FPS_TARGET
    frame_interval = max(1, round(orig_fps / FPS_TARGET))

    keypoints_list = []
    timestamps = []
    frame_indices = []

    frame_idx = 0

    with tqdm(total=frame_count, desc="Extracting keypoints") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                results = pose_model(frame, conf=0.25, verbose=False)[0]

                if results.keypoints is not None and len(results.keypoints) > 0:
                    confs = results.keypoints.conf[0].cpu().numpy()  # (17,)
                    xy = results.keypoints.xy[0].cpu().numpy()       # (17,2)

                    kpts = np.zeros(51, dtype=np.float32)

                    i = 0
                    for j in range(17):
                        kpts[i]   = xy[j, 0]   # raw x (NO normalization)
                        kpts[i+1] = xy[j, 1]   # raw y (NO normalization)
                        kpts[i+2] = confs[j]   # confidence
                        i += 3

                    keypoints_list.append(kpts)
                    timestamps.append(frame_idx / orig_fps)
                    frame_indices.append(frame_idx)

                # ❗ IMPORTANT: skip frame if no detection (no zero padding)

            frame_idx += 1
            pbar.update(1)

    cap.release()

    if len(keypoints_list) < WINDOW_SIZE:
        if len(keypoints_list) == 0:
            return None, "No valid pose detected", None

        last = keypoints_list[-1]
        while len(keypoints_list) < WINDOW_SIZE:
            keypoints_list.append(last)
            timestamps.append(timestamps[-1])

    # convert to array

    keypoints_array = np.stack(keypoints_list)          # (T, 51)
    num_keypoints = len(keypoints_list)

    # ── DEBUG (VERY IMPORTANT – keep this for now) ──────────────────────

    print("DEBUG INPUT STATS:")
    print("Shape:", keypoints_array.shape)
    print("Mean:", np.mean(keypoints_array))
    print("Std:", np.std(keypoints_array))
    print("Min:", np.min(keypoints_array))
    print("Max:", np.max(keypoints_array))



    

    # ── Sliding window inference ──────────────────────────────────────
    probs_fall = []
    window_centers = []

    for start in range(0, len(keypoints_array) - WINDOW_SIZE + 1, STRIDE):
        window = keypoints_array[start : start + WINDOW_SIZE]
        tensor = torch.from_numpy(window).unsqueeze(0).to(device)   # (1, W, 51)

        with torch.no_grad():
            logits = lstm_model(tensor)                     # (1, 2)
            prob_fall = torch.softmax(logits, dim=1)[0,1].item()

        probs_fall.append(prob_fall)
        # center of window (in extracted-frame index)
        center = start + WINDOW_SIZE // 2
        window_centers.append(timestamps[center])

   

    # ── Decision logic (simple version) ───────────────────────────────
    THRESHOLD = 0.6

    max_prob = max(probs_fall) if probs_fall else 0.0

    high_windows = [p > THRESHOLD for p in probs_fall]

    consecutive = 0
    max_consecutive = 0

    for flag in high_windows:
        if flag:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    num_high = sum(high_windows)
    avg_prob = float(np.mean(probs_fall)) if probs_fall else 0.0


    is_fall = (
        max_prob > 0.65
        or max_consecutive >= 2
        or (avg_prob > 0.5 and num_high >= 2)
    )



    num_windows = len(probs_fall)

    if len(timestamps) > 1:
        video_duration = timestamps[-1] - timestamps[0]
        effective_fps = len(timestamps) / video_duration if video_duration > 0 else 0
    else:
        video_duration = 0.0
        effective_fps = 0.0

    summary_lines = []
    summary_lines.append("Fall Status")
    summary_lines.append("────────────")
    if is_fall:
        summary_lines.append("**FALL DETECTED**")
    else:
        summary_lines.append("No fall detected")

    summary_lines.append("")
    summary_lines.append("Confidence")
    summary_lines.append("──────────")
    summary_lines.append(f"Number of windows > {THRESHOLD}: {num_high}")
    summary_lines.append(f"Thresholds: max_prob > 0.65 OR consecutive >= 2 OR avg_prob > 0.5")

    summary_lines.append("")
    summary_lines.append("Video information")
    summary_lines.append("─────────────────")
    summary_lines.append(f"Duration: {video_duration:.2f} seconds")
    summary_lines.append(f"The number of frames: {num_keypoints} (~{effective_fps:.1f} fps)")
    summary_lines.append(f"Number of sliding window: {num_windows}")

    if num_keypoints < 90 or num_windows < 1:
        summary_lines.append("")
        summary_lines.append("⚠️ Warning: Video is very short")

    summary = "\n".join(summary_lines)   # ← now use this
    
    

    # Optional: prepare timeline plot (prob vs time)
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    fig, ax = plt.subplots(figsize=(10, 3))
    if probs_fall and window_centers:
        ax.plot(window_centers, probs_fall, 'r-', label='Fall prob')
        ax.axhline(THRESHOLD, color='gray', ls='--')
        ax.set_xlim(min(window_centers)-0.5, max(window_centers)+0.5)
    else:
        ax.text(0.5, 0.5, "No data to plot\n(try longer video)", 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fall probability")
    ax.set_ylim(0,1)
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    timeline_img = Image.open(buf)
    plt.close(fig)

    # Optional: annotated video (slow!)
    # You can re-open the video, overlay text "FALL!" when window covering current frame has high prob
    # → see extended version below if needed

    return timeline_img, summary

# ────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────

with gr.Blocks(title="Fall Detection – YOLO11 Pose + LSTM") as demo:
    gr.Markdown("""
    # Fall Detection Demo
    Upload a short video clip (preferably < 30 seconds).
    The system extracts 17 keypoints @ ~30 fps using YOLO11s-pose,  
    then classifies sliding windows with a 3-layer LSTM.
    """)

    with gr.Row():
        video_input = gr.Video(
            label="Upload video",
            sources=["upload"],
            format="mp4",
            height=360
        )

    btn = gr.Button("Analyze Video", variant="primary")

    with gr.Row():
        timeline_plot = gr.Image(label="Fall Probability Timeline", type="pil")
        text_output   = gr.Textbox(label="Result Summary", lines=6)

    # video_output  = gr.Video(label="Annotated Video")   

    btn.click(
        fn=detect_fall_in_video,
        inputs=video_input,
        outputs=[timeline_plot, text_output]   # , video_output
    )

demo.launch(share=True)   # or just demo.launch() on HF Spaces
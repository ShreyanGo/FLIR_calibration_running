import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import os
import json

# Boson+ Thermal Gradient Analysis
# Mode 1: relative heating (background subtraction)
# Mode 2: substrate snapshot (absolute or relative temperature)

sys.path.append(r"C:\Users\LAb B 208\SDK\SDK\SDK")
from SDK_USER_PERMISSIONS import *

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
COM_PORT    = "COM3"
NUM_FRAMES  = 8  # frames averaged per capture
BASE_PATH   = r"path\to\SDK_USER_PERMISSIONS\parent"  # update to your SDK folder
OUTPUT_PATH = r"path\to\output\folder"               # update to your preferred output folder
DARK_PATH   = BASE_PATH + r"\dark.tiff"
BG_PATH     = BASE_PATH + r"\background_raw.tiff"
CAL_PATH    = BASE_PATH + r"\calibration.json"

# optical params from ray diagram — 72.71mm relay lens, 1.0431x magnification
PIXEL_PITCH_UM = 12.0
MAGNIFICATION  = 1.0431
UM_PER_PIXEL   = PIXEL_PITCH_UM / MAGNIFICATION  # ~11.5 um/pixel on substrate
FOV_WIDTH_MM   = (640 * UM_PER_PIXEL) / 1000.0   # ~7.4 mm
FOV_HEIGHT_MM  = (512 * UM_PER_PIXEL) / 1000.0   # ~5.9 mm

os.makedirs(OUTPUT_PATH, exist_ok=True)

def save_path(filename):  # all outputs go to FLIR images folder
    return os.path.join(OUTPUT_PATH, filename)

# -------------------------------------------------------
# Load dark frame
# -------------------------------------------------------
if not os.path.exists(DARK_PATH):
    print("ERROR: dark.tiff not found. Run boson_dark.py first.")
    sys.exit()
dark = tifffile.imread(DARK_PATH).astype(np.float32)
print("Loaded dark frame. Mean: " + str(round(dark.mean(), 1)))

# -------------------------------------------------------
# Load calibration
# -------------------------------------------------------
calibrated = False
degrees_per_count = 1.0
intercept = 0.0

if os.path.exists(CAL_PATH):
    with open(CAL_PATH, "r") as f:
        cal = json.load(f)
    degrees_per_count = cal["degrees_per_count"]
    intercept = cal["intercept"]
    calibrated = True
    print("Loaded calibration:")
    print("  R²:                " + str(round(cal["r_squared"], 4)))
    print("  Degrees per count: " + str(round(degrees_per_count, 6)))
    print("  Intercept:         " + str(round(intercept, 3)) + " C")
else:
    print("No calibration.json found — results in raw counts.")

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def capture_frame(myCam):
    result = myCam.captureSingleFrameWithSrc(FLR_CAPTURE_SRC_E.FLR_CAPTURE_SRC_NUC)
    if result != FLR_RESULT.R_SUCCESS:
        raise RuntimeError("Capture failed: " + str(result))
    result, total_bytes, rows, cols = myCam.memGetCaptureSize()
    CHUNK = 240
    raw_data = bytearray()
    offset = 0
    while offset < total_bytes:
        to_read = min(CHUNK, total_bytes - offset)
        result, chunk = myCam.memReadCapture(0, offset, to_read)
        if result != FLR_RESULT.R_SUCCESS:
            raise RuntimeError("Read failed at offset " + str(offset))
        raw_data.extend(chunk)
        offset += to_read
    return np.frombuffer(raw_data, dtype=np.uint16).reshape((rows, cols)).astype(np.float32)

def capture_avg(myCam, n):
    stack = []
    for i in range(n):
        f = capture_frame(myCam)
        stack.append(f)
        print("  Frame " + str(i+1) + "/" + str(n))
    return np.mean(stack, axis=0)

def remove_bad_pixels(frame):
    median = cv2.medianBlur(frame.astype(np.float32), 3)
    diff = np.abs(frame - median)
    mad = np.median(diff)
    bad = diff > 5.0 * mad
    cleaned = frame.copy()
    cleaned[bad] = median[bad]
    return cleaned

def compute_gradient(frame):
    cleaned = remove_bad_pixels(frame)
    gx = cv2.Sobel(cleaned, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cleaned, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2) / (UM_PER_PIXEL / 1000.0)

def to_absolute_temp(raw_counts):
    """Absolute temperature in degrees C — uses dark-subtracted counts."""
    dark_sub = raw_counts - dark
    return intercept + dark_sub * degrees_per_count

def to_relative_temp(raw_counts):
    """ΔT from scene mean in millidegrees C."""
    abs_C = to_absolute_temp(raw_counts)
    return (abs_C - abs_C.mean()) * 1000.0

def to_temp_diff(count_diff):
    """For Mode 1 difference images — count difference to mC."""
    return count_diff * degrees_per_count * 1000.0

def make_extent():
    w = FOV_WIDTH_MM / 2
    h = FOV_HEIGHT_MM / 2
    return [-w, w, h, -h]

def annotate_center(ax, frame_temp, extent, temp_unit):
    """Mark center pixel with crosshair and temperature label."""
    rows, cols = frame_temp.shape
    cy, cx = rows // 2, cols // 2

    # Center pixel temperature
    center_temp = frame_temp[cy, cx]

    # Convert pixel coords to mm coords using extent
    w = FOV_WIDTH_MM / 2
    h = FOV_HEIGHT_MM / 2
    cx_mm = -w + (cx / cols) * FOV_WIDTH_MM
    cy_mm =  h - (cy / rows) * FOV_HEIGHT_MM

    # 5x5 pixel box size in mm
    box_half_mm = (5 * UM_PER_PIXEL) / 1000.0 / 2

    # Draw crosshair
    ax.plot(cx_mm, cy_mm, '+', color='cyan', markersize=16, markeredgewidth=1.5)

    # Draw box outline around center pixel
    rect = plt.Rectangle(
        (cx_mm - box_half_mm, cy_mm - box_half_mm),
        box_half_mm * 2, box_half_mm * 2,
        fill=False, edgecolor='cyan', linewidth=1.5)
    ax.add_patch(rect)

    # Temperature label
    label = str(round(center_temp, 2)) + " " + temp_unit
    ax.text(cx_mm + box_half_mm + 0.05, cy_mm,
            label, color='cyan', fontsize=9,
            va='center', ha='left',
            bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))

    return center_temp

def show_results(frame_temp, gradient, title, temp_unit, grad_unit):
    threshold = np.percentile(gradient, 90)
    high_grad = gradient.copy()
    high_grad[high_grad < threshold] = 0
    extent = make_extent()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title + (" [calibrated]" if calibrated else " [uncalibrated]"), fontsize=13)

    im0 = axes[0].imshow(frame_temp, cmap='inferno', extent=extent, aspect='auto')
    axes[0].set_title("Thermal Frame")
    axes[0].set_xlabel("mm")
    axes[0].set_ylabel("mm")
    plt.colorbar(im0, ax=axes[0], label=temp_unit)
    center_temp = annotate_center(axes[0], frame_temp, extent, temp_unit)

    im1 = axes[1].imshow(gradient, cmap='hot', extent=extent, aspect='auto')
    axes[1].set_title("Thermal Gradient Magnitude")
    axes[1].set_xlabel("mm")
    axes[1].set_ylabel("mm")
    plt.colorbar(im1, ax=axes[1], label=grad_unit)
    annotate_center(axes[1], frame_temp, extent, temp_unit)

    im2 = axes[2].imshow(high_grad, cmap='plasma', extent=extent, aspect='auto')
    axes[2].set_title("Top 10% Gradient Regions")
    axes[2].set_xlabel("mm")
    axes[2].set_ylabel("mm")
    plt.colorbar(im2, ax=axes[2], label=grad_unit)
    annotate_center(axes[2], frame_temp, extent, temp_unit)

    plt.tight_layout()
    fname = save_path(title.replace(" ", "_") + ".png")
    plt.savefig(fname, dpi=150)
    print("Saved: " + fname)
    print("Center pixel temperature: " + str(round(center_temp, 3)) + " " + temp_unit)
    plt.show()

def save_overlay(frame_temp, gradient, filename, grad_unit):
    extent = make_extent()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(frame_temp, cmap='inferno', extent=extent, aspect='auto')
    ax.imshow(gradient, cmap='hot', alpha=0.5, extent=extent, aspect='auto')
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title("Gradient Overlay (" + grad_unit + ")")
    plt.savefig(save_path(filename), dpi=150)
    print("Saved: " + save_path(filename))
    plt.show()

# -------------------------------------------------------
# Mode selection
# -------------------------------------------------------
print("\n==============================================")
print("  BOSON+ THERMAL GRADIENT ANALYSIS")
print("==============================================")
print("Optical config:")
print("  Scale: " + str(round(UM_PER_PIXEL, 2)) + " um/pixel")
print("  FOV:   " + str(round(FOV_WIDTH_MM, 2)) + " x " + str(round(FOV_HEIGHT_MM, 2)) + " mm")
print()
print("Mode 1 — Relative heating")
print("  Captures cold background then heated frame.")
print("  Shows temperature CHANGE only.")
print()
print("Mode 2 — Substrate snapshot")
print("  Single capture.")
print("  Choose absolute temp (C) or relative ΔT from mean (mC).")
print()

while True:
    mode = input("Select mode (1 or 2): ").strip()
    if mode in ("1", "2"):
        break
    print("Please enter 1 or 2.")

# -------------------------------------------------------
# MODE 1 — Relative heating
# -------------------------------------------------------
if mode == "1":
    print("\n--- STEP 1: Cold background ---")
    input("Press Enter when scene is cold and ready...")

    myCam = CamAPI.pyClient(manualport=COM_PORT)
    print("Capturing background (" + str(NUM_FRAMES) + " frames)...")
    bg_raw = capture_avg(myCam, NUM_FRAMES)
    myCam.Close()
    tifffile.imwrite(BG_PATH, bg_raw.astype(np.float32))

    bg_temp = to_relative_temp(bg_raw)
    bg_gradient = compute_gradient(bg_temp)
    print("Background std: " + str(round(bg_temp.std(), 2)) + " mC")
    show_results(bg_temp, bg_gradient, "Background_No_Heating", "mC (ΔT from mean)", "mC/mm")

    print("\n--- STEP 2: Heated frame ---")
    input("Press Enter when heating is applied and ready...")

    myCam = CamAPI.pyClient(manualport=COM_PORT)
    print("Capturing heated frame (" + str(NUM_FRAMES) + " frames)...")
    heated_raw = capture_avg(myCam, NUM_FRAMES)
    myCam.Close()
    tifffile.imwrite(save_path("heated_raw.tiff"), heated_raw.astype(np.float32))

    print("\n--- STEP 3: Relative heating analysis ---")
    diff_counts = heated_raw - bg_raw
    if calibrated:
        diff_temp = to_temp_diff(diff_counts)
        t_unit = "mC"
        g_unit = "mC/mm"
    else:
        diff_temp = diff_counts
        t_unit = "counts"
        g_unit = "counts/mm"

    diff_gradient = compute_gradient(diff_temp)

    print("Heating signal:")
    print("  Min:  " + str(round(diff_temp.min(), 2)) + " " + t_unit)
    print("  Max:  " + str(round(diff_temp.max(), 2)) + " " + t_unit)
    print("  Mean: " + str(round(diff_temp.mean(), 2)) + " " + t_unit)
    print("  Std:  " + str(round(diff_temp.std(), 2)) + " " + t_unit)
    print("Gradient max: " + str(round(diff_gradient.max(), 2)) + " " + g_unit)

    tifffile.imwrite(save_path("heating_signal.tiff"), diff_temp.astype(np.float32))
    show_results(diff_temp, diff_gradient, "Relative_Heating_Signal", t_unit, g_unit)

    heated_display = to_relative_temp(heated_raw)
    save_overlay(heated_display, diff_gradient, "gradient_overlay.png", g_unit)

# -------------------------------------------------------
# MODE 2 — Substrate snapshot
# -------------------------------------------------------
else:
    if calibrated:
        print()
        print("Temperature display options:")
        print("  A — Absolute temperature (degrees C)")
        print("  B — Relative ΔT from scene mean (mC)")
        print()
        while True:
            temp_mode = input("Choose A or B: ").strip().upper()
            if temp_mode in ("A", "B"):
                break
            print("Please enter A or B.")
    else:
        temp_mode = "B"

    print()
    input("Press Enter when ready to capture...")

    myCam = CamAPI.pyClient(manualport=COM_PORT)
    print("Capturing (" + str(NUM_FRAMES) + " frames)...")
    frame_raw = capture_avg(myCam, NUM_FRAMES)
    myCam.Close()
    tifffile.imwrite(save_path("substrate_raw.tiff"), frame_raw.astype(np.float32))

    if temp_mode == "A":
        frame_temp = to_absolute_temp(frame_raw)
        t_unit = "degrees C"
        g_unit = "C/mm"
        title = "Absolute_Temperature"
    else:
        frame_temp = to_relative_temp(frame_raw)
        t_unit = "mC (ΔT from mean)"
        g_unit = "mC/mm"
        title = "Absolute_Substrate"

    frame_gradient = compute_gradient(frame_temp)

    print("\nStats:")
    print("  Min:  " + str(round(frame_temp.min(), 3)) + " " + t_unit)
    print("  Max:  " + str(round(frame_temp.max(), 3)) + " " + t_unit)
    print("  Mean: " + str(round(frame_temp.mean(), 3)) + " " + t_unit)
    print("  Std:  " + str(round(frame_temp.std(), 3)) + " " + t_unit)
    print("Gradient max: " + str(round(frame_gradient.max(), 3)) + " " + g_unit)

    tifffile.imwrite(save_path("substrate_processed.tiff"), frame_temp.astype(np.float32))
    show_results(frame_temp, frame_gradient, title, t_unit, g_unit)
    save_overlay(frame_temp, frame_gradient, "gradient_overlay_" + temp_mode + ".png", g_unit)

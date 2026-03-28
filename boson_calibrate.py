import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import json
import os

sys.path.append(r"C:\Users\LAb B 208\SDK\SDK\SDK")
from SDK_USER_PERMISSIONS import *

COM_PORT = "COM3"
NUM_FRAMES = 16       # more frames = cleaner calibration
DARK_PATH = "dark.tiff"
CAL_PATH = "calibration.json"

# -------------------------------------------------------
def capture_frame(myCam):
    result = myCam.captureSingleFrameWithSrc(FLR_CAPTURE_SRC_E.FLR_CAPTURE_SRC_NUC)
    if result != FLR_RESULT.R_SUCCESS:
        raise RuntimeError("Capture failed: " + str(result))
    result, total_bytes, rows, cols = myCam.memGetCaptureSize()
    if result != FLR_RESULT.R_SUCCESS:
        raise RuntimeError("Failed to get capture size")
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
    frame = np.frombuffer(raw_data, dtype=np.uint16).reshape((rows, cols))
    return frame.astype(np.float32)

def capture_avg(myCam, n):
    stack = []
    for i in range(n):
        f = capture_frame(myCam)
        stack.append(f)
        print("  Frame " + str(i+1) + "/" + str(n) +
              "  mean=" + str(round(f.mean(), 1)))
    return np.mean(stack, axis=0)

# -------------------------------------------------------
# Load dark frame
if not os.path.exists(DARK_PATH):
    print("ERROR: dark.tiff not found. Run boson_dark.py first.")
    sys.exit()
dark = tifffile.imread(DARK_PATH).astype(np.float32)
print("Loaded dark frame. Mean: " + str(round(dark.mean(), 1)))

# -------------------------------------------------------
print("\n==============================================")
print("  BOSON+ RELATIVE CALIBRATION")
print("==============================================")
print("You will need:")
print("  - A flat uniform surface (cardboard or foam board)")
print("  - Your heat gun")
print("  - The heat gun temperature reading (in Celsius)")
print()
print("IMPORTANT TIPS for best calibration:")
print("  - Fill as much of the frame as possible with the surface")
print("  - Keep camera distance fixed throughout")
print("  - Let heat gun stabilize on surface before capturing hot frame")
print("  - Use center ROI for mean calculation to avoid edge effects")
print()

# -------------------------------------------------------
# Step 1: Cold frame
print("----------------------------------------------")
print("STEP 1: Cold frame")
print("Point camera at the unheated flat surface.")
cold_temp = float(input("Enter current surface temperature in C (or room temp): "))
input("Press Enter when ready to capture cold frame...")

myCam = CamAPI.pyClient(manualport=COM_PORT)
print("Capturing cold frame (" + str(NUM_FRAMES) + " frames)...")
cold_raw = capture_avg(myCam, NUM_FRAMES)
myCam.Close()

cold = cold_raw - dark
tifffile.imwrite("cal_cold.tiff", cold.astype(np.float32))

# Use center ROI (middle 50% of frame) for stable mean
r0, r1 = cold.shape[0]//4, 3*cold.shape[0]//4
c0, c1 = cold.shape[1]//4, 3*cold.shape[1]//4
cold_mean = cold[r0:r1, c0:c1].mean()
cold_std = cold[r0:r1, c0:c1].std()
print("Cold ROI mean: " + str(round(cold_mean, 2)) + " counts")
print("Cold ROI std:  " + str(round(cold_std, 2)) + " counts")

# -------------------------------------------------------
# Step 2: Hot frame
print()
print("----------------------------------------------")
print("STEP 2: Hot frame")
print("Heat the surface with your heat gun.")
print("Keep the heat gun moving to get a uniform heated surface.")
hot_temp = float(input("Enter the heat gun set temperature in C: "))
print("Hold the heat gun steady on the surface and let it stabilize.")
input("Press Enter when ready to capture hot frame...")

myCam = CamAPI.pyClient(manualport=COM_PORT)
print("Capturing hot frame (" + str(NUM_FRAMES) + " frames)...")
hot_raw = capture_avg(myCam, NUM_FRAMES)
myCam.Close()

hot = hot_raw - dark
tifffile.imwrite("cal_hot.tiff", hot.astype(np.float32))

hot_mean = hot[r0:r1, c0:c1].mean()
hot_std = hot[r0:r1, c0:c1].std()
print("Hot ROI mean: " + str(round(hot_mean, 2)) + " counts")
print("Hot ROI std:  " + str(round(hot_std, 2)) + " counts")

# -------------------------------------------------------
# Step 3: Compute calibration factor
delta_T = hot_temp - cold_temp
delta_counts = hot_mean - cold_mean

if delta_counts <= 0:
    print("\nWARNING: Hot frame is not warmer than cold frame.")
    print("Check your setup and re-run.")
    sys.exit()

counts_per_degree = delta_counts / delta_T
degrees_per_count = delta_T / delta_counts

print()
print("==============================================")
print("  CALIBRATION RESULTS")
print("==============================================")
print("Cold temperature:   " + str(cold_temp) + " C")
print("Hot temperature:    " + str(hot_temp) + " C")
print("Delta T:            " + str(round(delta_T, 2)) + " C")
print("Delta counts:       " + str(round(delta_counts, 2)))
print("Counts per degree:  " + str(round(counts_per_degree, 4)))
print("Degrees per count:  " + str(round(degrees_per_count, 6)))
print()

# Estimate minimum detectable temperature (noise floor)
noise_floor_C = cold_std * degrees_per_count
print("Noise floor (1-sigma): " + str(round(noise_floor_C * 1000, 2)) + " mK")
print("  (Camera spec is ~10mK NETD -- this is your empirical estimate)")

# -------------------------------------------------------
# Step 4: Save calibration
cal = {
    "counts_per_degree": counts_per_degree,
    "degrees_per_count": degrees_per_count,
    "cold_temp_C": cold_temp,
    "hot_temp_C": hot_temp,
    "delta_T_C": delta_T,
    "delta_counts": delta_counts,
    "cold_mean_counts": cold_mean,
    "hot_mean_counts": hot_mean,
    "noise_floor_mK": round(noise_floor_C * 1000, 2)
}

with open(CAL_PATH, "w") as f:
    json.dump(cal, f, indent=2)
print("\nCalibration saved to calibration.json")

# -------------------------------------------------------
# Step 5: Visual verification
diff = hot - cold
diff_K = diff * degrees_per_count

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Calibration Verification", fontsize=13)

im0 = axes[0].imshow(cold, cmap='inferno')
axes[0].set_title("Cold Frame (dark subtracted)")
plt.colorbar(im0, ax=axes[0], label="Counts")

im1 = axes[1].imshow(hot, cmap='inferno')
axes[1].set_title("Hot Frame (dark subtracted)")
plt.colorbar(im1, ax=axes[1], label="Counts")

im2 = axes[2].imshow(diff_K, cmap='RdBu_r')
axes[2].set_title("Hot - Cold (degrees C)")
axes[2].add_patch(plt.Rectangle((c0, r0), c1-c0, r1-r0,
    fill=False, edgecolor='white', linewidth=2, linestyle='--'))
axes[2].text(c0+5, r0+20, "Calibration ROI", color='white', fontsize=9)
plt.colorbar(im2, ax=axes[2], label="Delta T (C)")

plt.tight_layout()
plt.savefig("calibration_verification.png", dpi=150)
plt.show()
print("Saved: calibration_verification.png")

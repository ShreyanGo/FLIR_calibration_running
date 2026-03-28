# Boson+ Thermal Gradient Analysis
**Shreyan Goswami**

Python pipeline for relative thermal gradient measurement using a Teledyne FLIR Boson+ 640 thermal camera. Designed for imaging silica substrates in vacuum through a ZnSe viewport, with a 72.71mm relay lens giving ~11.5 µm/pixel spatial resolution over a ~7.4 × 5.9 mm FOV.

---

## Hardware
- **Camera:** Teledyne FLIR Boson+ 640 (640×512, 12µm pixel pitch)
- **Connection:** USB via VPC adapter, COM3 serial for SDK control
- **Lens:** 36mm fixed Boson lens + 72.71mm relay lens (1.0431× magnification)
- **Viewport:** ZnSe window, vacuum-compatible
- **Reference thermometer:** BSIDE HX1 (±2°C, emissivity 0.95)

---

## Files

| File | Purpose |
|------|---------|
| `boson_dark.py` | Captures dark reference frame with lens cap on |
| `boson_calibrate.py` | Multi-point calibration using fridge-to-room temperature block |
| `boson_gradient_v2.py` | Main measurement script — thermal map and gradient analysis |

---

## Setup

### SDK Files (required)
Download the Boson SDK from the official FLIR documentation portal:
**http://flir.com/bosondocuments**

Extract the SDK and place the `SDK_USER_PERMISSIONS` folder somewhere accessible on your machine. Then update `BASE_PATH` in each script to point to the parent folder of `SDK_USER_PERMISSIONS`.

### Dependencies
Install Python dependencies:
```
pip install opencv-python numpy matplotlib tifffile scipy pyserial
```

### SDK Folder Structure
Update the `BASE_PATH` variable at the top of each script to point to the folder containing `SDK_USER_PERMISSIONS`:
```python
BASE_PATH = r"path\to\your\SDK_USER_PERMISSIONS\parent"
```
Also update `OUTPUT_PATH` to your preferred output directory for saved images and data.

All scripts must be run from the `BASE_PATH` folder:
```
cd "path\to\SDK_USER_PERMISSIONS\parent"
python "path\to\scripts\<script>.py"
```

---

## Workflow

### Step 1 — Dark frame (run once per camera reconfiguration)
```
python boson_dark.py
```
- Put lens cap on when prompted
- Captures 16 averaged frames → saves `dark.tiff`
- Corrects for fixed pattern noise and bad pixels

### Step 2 — Calibration (run once per setup)
```
python boson_calibrate.py
```
- Takes block out of fridge, lets it warm naturally to room temperature
- Captures every 60 seconds, enter BSIDE reading after each capture
- Fits linear regression through all (counts, temperature) pairs
- Saves `calibration.json` with slope, intercept, R², noise floor
- Prints run command for `boson_gradient_v2.py` at the end — run it manually when ready
- **Target:** R² > 0.97, temperature range > 15°C

### Step 3 — Measurement (run manually after calibration is complete)
```
cd "path\to\SDK_USER_PERMISSIONS\parent"
python "path\to\scripts\boson_gradient_v2.py"
```
**Mode 1 — Relative heating:**
- Captures cold background then heated substrate
- Subtracts to isolate temperature change only
- Best for: locating where heating occurs and mapping gradients

**Mode 2 — Substrate snapshot:**
- Single capture, dark subtracted
- Option A: absolute temperature in °C per pixel
- Option B: ΔT from scene mean in mC
- Best for: full thermal map at any moment

---

## Output Files
All saved to `Documents\FLIR images\`

| File | Contents |
|------|---------|
| `Absolute_Temperature.png` | Three-panel plot: thermal frame, gradient magnitude, top 10% gradients (Mode 2A) |
| `Absolute_Substrate.png` | Same as above in relative mC units (Mode 2B) |
| `Relative_Heating_Signal.png` | Difference image plots (Mode 1) |
| `gradient_overlay_A.png` | Gradient overlaid on absolute temperature map |
| `gradient_overlay_B.png` | Gradient overlaid on relative temperature map |
| `heating_signal.tiff` | Raw difference data in mC (Mode 1) |
| `substrate_raw.tiff` | Raw uint16 counts from sensor |
| `substrate_processed.tiff` | Processed temperature data |

Calibration and reference files saved to `SDK\SDK\SDK\`:

| File | Contents |
|------|---------|
| `dark.tiff` | Dark reference frame (float32) |
| `calibration.json` | Calibration parameters and fit data |
| `background_raw.tiff` | Cold background frame raw counts (Mode 1) |

---

## Calibration Notes
- Linear fit: `T(°C) = intercept + counts × degrees_per_count`
- Absolute temperature uses dark-subtracted counts as input
- Gradient units: mC/mm (calibrated) or counts/mm (uncalibrated)
- Spatial scale: 11.5 µm/pixel → gradients in temperature per mm on substrate
- Bad pixels removed via median filter before gradient computation

---

## Key Parameters (boson_gradient_v2.py)
```python
NUM_FRAMES     = 8       # frames averaged per capture (increase for cleaner image)
MAGNIFICATION  = 1.0431  # from ray diagram (72.71mm relay lens)
UM_PER_PIXEL   = 11.5    # microns per pixel on substrate
```

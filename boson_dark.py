import sys
import numpy as np
import tifffile

sys.path.append(r"C:\Users\LAb B 208\SDK\SDK\SDK")
from SDK_USER_PERMISSIONS import *

COM_PORT = "COM3"
NUM_FRAMES = 16  # number of frames to average for dark reference

def capture_frame(myCam):
    result = myCam.captureSingleFrameWithSrc(FLR_CAPTURE_SRC_E.FLR_CAPTURE_SRC_NUC)
    if result != FLR_RESULT.R_SUCCESS:
        raise RuntimeError("Capture failed: " + str(result))

    result, total_bytes, rows, cols = myCam.memGetCaptureSize()
    if result != FLR_RESULT.R_SUCCESS:
        raise RuntimeError("Failed to get capture size: " + str(result))

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


print("==============================================")
print("  DARK FRAME CAPTURE")
print("==============================================")
print("Make sure the lens cover is ON before continuing.")
input("Press Enter when lens cover is on and you are ready...")

print("\nConnecting to Boson on " + COM_PORT + "...")
myCam = CamAPI.pyClient(manualport=COM_PORT)
print("Connected.")

print("\nCapturing " + str(NUM_FRAMES) + " frames for dark reference...")
dark_stack = []
for i in range(NUM_FRAMES):
    frame = capture_frame(myCam)
    dark_stack.append(frame)
    print("  Frame " + str(i + 1) + "/" + str(NUM_FRAMES) +
          "  min=" + str(int(frame.min())) +
          "  max=" + str(int(frame.max())))

myCam.Close()

# Average all frames to reduce noise
dark_avg = np.mean(dark_stack, axis=0)

print("\nDark frame stats:")
print("  Min:  " + str(dark_avg.min()))
print("  Max:  " + str(dark_avg.max()))
print("  Mean: " + str(dark_avg.mean()))
print("  Std:  " + str(dark_avg.std()))

# Save as 32-bit float TIFF to preserve the averaged values
tifffile.imwrite("dark.tiff", dark_avg.astype(np.float32))
print("\nSaved dark reference as dark.tiff")
print("You can now remove the lens cover.")

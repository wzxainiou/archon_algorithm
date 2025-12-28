import cv2
import os

os.chdir(r"c:\Users\王照旭\Desktop\CODE\archon_algorithm\data\test_videos")

cap = cv2.VideoCapture("2022-05-08-11-23-59.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Save full frame
    cv2.imwrite("full_frame.png", frame)
    print(f"Saved full_frame.png: {os.path.exists('full_frame.png')}")

    # Also save the three panels separately for analysis
    panel_width = w // 3  # 593

    rgb = frame[:, 0:panel_width]
    thermal = frame[:, panel_width:panel_width*2]
    gray = frame[:, panel_width*2:]

    cv2.imwrite("panel_rgb.png", rgb)
    cv2.imwrite("panel_thermal.png", thermal)
    cv2.imwrite("panel_gray.png", gray)

    print("Saved panel images")

    # List all files
    for f in os.listdir("."):
        if f.endswith(".png"):
            print(f"  {f}: {os.path.getsize(f)} bytes")
else:
    print("Failed to read frame")

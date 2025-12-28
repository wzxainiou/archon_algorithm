import cv2
import os

os.chdir(r"c:\Users\王照旭\Desktop\CODE\archon_algorithm\data\test_videos")

cap = cv2.VideoCapture("2022-05-08-11-23-59.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 从截图观察，三个区域大致等宽
# 总宽度 1780，每部分约 593 像素
# 但实际可能有细微差别，让我们用更精确的边界

# 根据视觉观察调整边界
rgb_start = 0
rgb_end = 640       # RGB 可见光区域 (基于饱和度分析调整)

thermal_start = 640
thermal_end = 1140  # 热成像区域 (基于饱和度分析: 高饱和度在 650-1100)

# 右边的黑白区域我们不需要

max_frames = int(fps * 45)  # 45 seconds

print(f"Source: {w}x{h} @ {fps} FPS")
print(f"RGB region: {rgb_start}-{rgb_end} ({rgb_end - rgb_start} px)")
print(f"Thermal region: {thermal_start}-{thermal_end} ({thermal_end - thermal_start} px)")
print(f"Target: {max_frames} frames (45s)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rgb_writer = cv2.VideoWriter("rgb_test.mp4", fourcc, fps, (rgb_end - rgb_start, h))
thermal_writer = cv2.VideoWriter("thermal_test.mp4", fourcc, fps, (thermal_end - thermal_start, h))

count = 0
while count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = frame[:, rgb_start:rgb_end]
    thermal = frame[:, thermal_start:thermal_end]

    rgb_writer.write(rgb)
    thermal_writer.write(thermal)
    count += 1

    if count % 500 == 0:
        print(f"  Progress: {count}/{max_frames}")

cap.release()
rgb_writer.release()
thermal_writer.release()

print(f"\nDone! Processed {count} frames ({count/fps:.1f}s)")
print(f"RGB: {os.path.getsize('rgb_test.mp4')/1024/1024:.2f} MB")
print(f"Thermal: {os.path.getsize('thermal_test.mp4')/1024/1024:.2f} MB")

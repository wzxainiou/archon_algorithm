"""
Debug script to test bbox handling
"""
import numpy as np

# Simulate YOLO output
boxes = np.array([[120.7, 200.3, 450.8, 520.6]])

# Simulate to_dict_list
bbox_as_list = boxes[0].tolist()
print(f"bbox type: {type(bbox_as_list)}")
print(f"bbox value: {bbox_as_list}")
print(f"bbox elements: {[type(x) for x in bbox_as_list]}")

# Test unpacking in two ways
print("\n=== Method 1: Direct unpack (OLD WAY - WRONG) ===")
x1, y1, x2, y2 = map(int, bbox_as_list)
print(f"After map(int): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"Lost precision: {bbox_as_list[0] - x1}")

print("\n=== Method 2: Keep float, convert at draw (NEW WAY - CORRECT) ===")
x1, y1, x2, y2 = bbox_as_list
print(f"Keep float: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"Convert at draw: int(x1)={int(x1)}, int(y1)={int(y1)}")

# Test centroid
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
print(f"\n=== Centroid (float) ===")
print(f"cx={cx}, cy={cy}")
print(f"int(cx)={int(cx)}, int(cy)={int(cy)}")

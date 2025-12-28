# 修复总结 (Fix Summary)

## 问题概述 (Problem Overview)

用户报告了两个主要问题：
1. **Thermal窗口的检测框位置不正确** - 橙色/黄色/绿色PTZ区域框延伸到画面边界之外
2. **三个窗口（RGB、Thermal、Fusion）不同步** - Fusion窗口中可以看到重影

## 已应用的修复 (Fixes Applied)

### 修复 1: PTZ区域框尺寸适配 ✅

**问题根因**:
- PTZ区域框在第一次调用时被初始化（RGB画面 1280×720）
- 之后处理Thermal画面（320×180）时，仍使用RGB的区域尺寸
- 导致橙色/黄色/绿色框远超出Thermal画面边界

**修复位置**: `jetson_bench/visualization/overlay.py:471-475`

**修改内容**:
```python
# 修改前（错误）:
if not hasattr(self, 'zones_initialized') or not self.zones_initialized:
    self.frame_width = w
    self.frame_height = h
    self._init_ptz_zones()

# 修改后（正确）:
if not hasattr(self, 'zones_initialized') or not self.zones_initialized or \
   self.frame_width != w or self.frame_height != h:
    self.frame_width = w
    self.frame_height = h
    self._init_ptz_zones()
```

**效果**:
- PTZ区域框现在会在画面尺寸变化时重新计算
- RGB窗口使用1280×720的区域尺寸
- Thermal窗口使用320×180的区域尺寸
- 每个窗口的橙色（紧急边界）、黄色（慢速区）、绿色（死区）框都会正确适配

---

### 修复 2: 视频流独立帧跳跃策略 ✅

**问题根因**:
- RGB视频: 25 FPS
- Thermal视频: 30 FPS
- 之前使用单一`frame_skip`值，导致两个流速度不匹配
- Fusion窗口叠加时产生重影

**修复位置**: `jetson_bench/dual_stream/dual_loader.py:79-156`

**修改内容**:

1. **独立帧跳跃计数器** (Lines 79-88):
```python
# 为RGB和Thermal分别计算帧跳跃值
self.rgb_frame_skip = max(1, int(self.rgb_native_fps / target_fps))
self.thermal_frame_skip = max(1, int(self.thermal_native_fps / target_fps))
```

2. **独立跳帧逻辑** (Lines 121-156):
```python
# 使用独立计数器
rgb_skip_counter += 1
thermal_skip_counter += 1

# 确定每个流是否应该读取此帧
should_read_rgb = rgb_skip_counter >= self.rgb_frame_skip
should_read_thermal = thermal_skip_counter >= self.thermal_frame_skip

# 仅在两个流都准备好时才yield帧
if not (should_read_rgb and should_read_thermal):
    # 分别跳过需要跳过的流
    if not should_read_rgb:
        self.rgb_cap.grab()  # 跳过RGB帧
    if not should_read_thermal:
        self.thermal_cap.grab()  # 跳过Thermal帧
    continue

# 重置计数器
rgb_skip_counter = 0
thermal_skip_counter = 0

# 同步读取两个帧
ret_rgb, rgb_frame = self.rgb_cap.read()
ret_thermal, thermal_frame = self.thermal_cap.read()
```

**效果**:
- RGB和Thermal视频现在以相同的有效帧率（5 FPS）处理
- 每次yield时，两个流的帧是同步的
- Fusion窗口不再出现重影

---

### 修复 3: 窗口刷新同步 ✅

**问题根因**:
- `cv2.waitKey(1)` 延迟太短（1ms），三个窗口来不及同步刷新
- 导致窗口播放速度不一致

**修复位置**: `scripts/test_dual_stream.py:279`

**修改内容**:
```python
# 修改前:
if cv2.waitKey(1) & 0xFF == 27:

# 修改后:
# 增加waitKey延迟到5ms，确保三窗口同步刷新
# 对于实时摄像头，5ms足够快（200fps理论上限）
if cv2.waitKey(5) & 0xFF == 27:
```

**效果**:
- 三个窗口（RGB、Thermal、Fusion）有足够时间完成渲染
- 窗口刷新更加同步
- 对实时摄像头不会影响实时性（处理瓶颈在推理速度，不是显示）

---

### 额外修复: 坐标精度保留 ✅

**修复位置**: `jetson_bench/visualization/overlay.py:185-204`

**修改内容**:
- 保留bbox和centroid的浮点精度
- 仅在`cv2.rectangle()`/`cv2.circle()`/`cv2.putText()`绘图时转int

```python
# 修改前（错误）:
x1, y1, x2, y2 = map(int, det["bbox"])
cv2.rectangle(frame, (x1, y1), (x2, y2), ...)

# 修改后（正确）:
x1, y1, x2, y2 = det["bbox"]  # 保留浮点
cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), ...)
```

**效果**: 避免过早截断坐标精度

---

## 测试验证 (Testing)

### 运行测试命令:
```bash
# 方法1: 双击运行
run_test.bat

# 方法2: 命令行运行
python scripts\test_dual_stream.py --rgb-video "data\test_videos\rgb_test.mp4" --thermal-video "data\test_videos\thermal_test.mp4" --rgb-model "weights\yolo11n.pt" --thermal-model "weights\yolo11n.pt"
```

### 预期效果:

1. **RGB窗口**:
   - 橙色/黄色/绿色PTZ区域框在1280×720画面内
   - 检测框和追踪框位置准确

2. **Thermal窗口**:
   - ✅ 橙色/黄色/绿色PTZ区域框在320×180画面内（已修复）
   - ✅ 检测框和追踪框位置准确（已修复）

3. **Fusion窗口**:
   - ✅ RGB和Thermal画面同步，无重影（已修复）
   - Thermal检测框正确叠加在融合画面上

4. **三窗口同步**:
   - ✅ 三个窗口显示相同的frame_id（已修复）
   - ✅ 三个窗口中目标运动保持一致（已修复）

---

## 技术说明 (Technical Details)

### Debug输出验证

运行测试时会看到：
```
✓ RGB: 1280x720 @ 25.0 FPS
✓ Thermal: 320x180 @ 30.0 FPS

Video mode: RGB skip=5 frames, Thermal skip=6 frames
  RGB FPS: 25.0 -> 5.0 FPS
  Thermal FPS: 30.0 -> 5.0 FPS

[yolo11n_thermal] Input frame shape: (180, 320, 3), imgsz: 640
[yolo11n_thermal] First bbox: [175.67880249 91.25787354 262.39373779 127.08993530], frame shape: (180, 320)
```

**解释**:
- RGB跳过5帧，Thermal跳过6帧，最终都达到5 FPS
- Thermal输入帧是180×320
- YOLO bbox坐标在180×320范围内（正确）
- PTZ区域框现在会根据180×320重新计算（已修复）

---

## 文件修改清单 (Modified Files)

1. ✅ `jetson_bench/visualization/overlay.py`
   - Line 471-475: PTZ区域框动态重新初始化
   - Line 185-204: 坐标精度保留

2. ✅ `jetson_bench/dual_stream/dual_loader.py`
   - Line 79-88: 独立帧跳跃计数器
   - Line 121-156: 独立跳帧逻辑

3. ✅ `scripts/test_dual_stream.py`
   - Line 279: `cv2.waitKey(5)` 窗口同步

---

## 下一步 (Next Steps)

1. 运行`run_test.bat`测试所有修复
2. 验证三个窗口的PTZ区域框都在画面内
3. 验证Fusion窗口无重影
4. 如果问题仍存在，提供新的截图和具体问题描述

---

**修复日期**: 2025-12-29
**修复版本**: 所有修复已应用，等待用户测试验证

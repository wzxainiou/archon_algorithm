# 🚀 快速开始指南

## ✅ 已准备好的内容

你的环境已经完全配置好，可以立即运行！

### 📦 已下载文件
- ✅ `yolo11n.pt` - YOLO11 nano 模型（5.4MB）
- ✅ `test_data/images/` - 测试图片（2张）
  - bus.jpg - 公交车图片
  - zidane.jpg - 足球运动员图片

## 🎯 运行方式

### 方式 1：使用快速脚本（推荐）

**基础测试**：
```cmd
run_test.bat
```

**详细日志模式**：
```cmd
run_test_verbose.bat
```

### 方式 2：命令行运行

**基础命令**：
```bash
python -m jetson_bench.cli --source image_dir=test_data/images --model0 yolo11n.pt
```

**完整参数**：
```bash
python -m jetson_bench.cli \
    --source image_dir=test_data/images \
    --model0 yolo11n.pt \
    --max_frames 10 \
    --gpu_mem_limit_gb 8.0 \
    --verbose
```

## 📊 查看结果

运行完成后，在 `outputs/YYYY-MM-DD_HH-MM-SS/` 目录下查看：

1. **report.md** - 📄 人类可读的 Markdown 报告
2. **report.json** - 📦 机器可读的 JSON 数据
3. **metrics.jsonl** - 📈 时间序列指标数据
4. **run.log** - 📝 运行日志

## 🎓 常用命令

### 使用视频文件
```bash
python -m jetson_bench.cli --source video=path/to/video.mp4 --model0 yolo11n.pt --max_frames 100
```

### 使用摄像头
```bash
python -m jetson_bench.cli --source camera=0 --model0 yolo11n.pt --max_frames 50
```

### 调整图片尺寸（优化速度）
```bash
python -m jetson_bench.cli --source image_dir=test_data/images --model0 yolo11n.pt --imgsz 320
```

### 测试 GPU 内存限制
```bash
# 尝试设置更高限制（会自动限制到8GB）
python -m jetson_bench.cli --source image_dir=test_data/images --model0 yolo11n.pt --gpu_mem_limit_gb 12
```

## 🔧 下载更多模型

### YOLO11 系列
```python
from ultralytics import YOLO

# 不同大小的模型
YOLO('yolo11n.pt')  # Nano - 最快
YOLO('yolo11s.pt')  # Small - 平衡
YOLO('yolo11m.pt')  # Medium - 更准确
YOLO('yolo11l.pt')  # Large - 最准确
```

### YOLOv8 系列
```python
YOLO('yolov8n.pt')  # Nano
YOLO('yolov8s.pt')  # Small
```

## 📚 完整文档

- **README.md** - 完整项目说明
- **CLAUDE.md** - 开发者指南
- **GPU_MEMORY_IMPLEMENTATION.md** - GPU 内存限制实现详情

## ⚠️ 注意事项

1. **Windows 系统**：本机是 Windows，不是 Jetson
   - tegrastats 不可用（正常现象）
   - GPU 内存限制类型为 "soft"（依赖 PyTorch 配合）

2. **GPU 支持**：
   - 检测到 NVIDIA RTX 4080（16GB VRAM）
   - 项目会将其限制为 8GB 使用

3. **模型格式**：
   - .pt = PyTorch 格式（通用）
   - .onnx = ONNX 格式（更快）
   - .engine = TensorRT 格式（最快，仅限 Jetson）

## 🎉 测试成功！

你的首次运行结果：
- ✅ 处理了 2 张图片
- ✅ FPS: ~14
- ✅ 检测到 4 个物体（平均）
- ✅ GPU 内存使用正常
- ✅ 生成了完整报告

现在就可以用自己的数据测试了！

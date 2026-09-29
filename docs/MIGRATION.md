# KAVACH Migration Guide: PyTorch to Native C++20 / TensorRT

## 1. Building the Native Extension
```bash
# Windows
python setup.py build_ext --inplace

# Linux
./build_linux.sh
```

## 2. Exporting Model Engine
```bash
python -m cv.export_engine --weights yolo11s.pt --fp16 --output yolo11s.engine
```

## 3. Switching Backend
In your `.env` or `config.py`:
```ini
DETECTOR_BACKEND=tensorrt_native
ENGINE_PATH=yolo11s.engine
```

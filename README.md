## Getting started
### Pip
```bash
pip install -r requirements.txt
cd YOLOX
pip install -v -e .
```
### Nvidia Driver
Make sure to use CUDA Toolkit version 11.2 as it is the proper version for the Torch version used in this repository: https://developer.nvidia.com/cuda-11.2.0-download-archive

### torch2trt
Clone this repository and install: https://github.com/NVIDIA-AI-IOT/torch2trt 

## Download a pretrained model
Download pretrained yolox_s.pth file: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth

Copy and paste yolox_s.pth from your downloads folder into the 'YOLOX' folder of this repository.

## Convert model to TensorRT
```bash
python tools/trt.py -n yolox-s -c yolox_s.pth
```

## Runing with YOLOX-s
```bash
python detector.py
```

## References
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)


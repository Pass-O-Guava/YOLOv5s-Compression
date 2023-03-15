# YOLOV5s-Compression

## Requirements

```shell
# conda
conda create -n yolov5 python=3.9
conda activate yolov5

# pip
pip install -r requirments-01.txt
```

## Download data and weight
```shell
# download dataset
cd ../{project}
mkdir -p datasets/coco128

coco128
链接: https://pan.baidu.com/s/1ya6SAFGp6du5RahaU1BlkA?pwd=jufh
提取码: jufh 

datasets/
├── coco128
│   ├── images
│   │   └── train2017
│   ├── labels
│   │   └── train2017
│   ├── LICENSE
│   └── README.txt

# download pre-trained weights
cd {project}
wget -O yolov5s-v5.0.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
```

## 1. Base Train

```shell
# YOLOv5s
python train.py --data data/coco128.yaml --imgsz 640 --weights yolov5s.pt --cfg models/yolov5s.yaml --epochs 100 --device 0,1 --sync-bn

# [Optional] YOLOv5l-PP-LCNet
python train.py --data data/coco128.yaml --imgsz 640 --weights yolov5lPP-LC.pt --cfg models/lightModels/yolov5lPP-LC.yaml --epochs 60 --device 0,1 --sync-bn
```

## 2. Sparse train

I. Slim (BN-L1)
```shell
# sparse train
python train.py --data data/coco128.yaml --imgsz 640 --weights runs/Base-coco128-mAP05_0.2293.pt --cfg models/prunModels/yolov5s-pruning.yaml --epochs 100 --device 0,1 --sparse

# prune
python pruneSlim.py --data data/coco128.yaml --weights runs/sparse-coco128-mAP05-035504.pt --cfg models/prunModels/yolov5s-pruning.yaml --path yolov5s-pruned.yaml --global_percent 0.5 --device 0,1

# finetune
python train.py --data data/coco128.yaml --imgsz 640 --weights runs/sparse-coco128-mAP05-035504-Slimpruned.pt --cfg yolov5s-pruned.yaml --epochs 100 --device 0,1
```

II. EagleEye (un-test)
```shell
# search best sub-net
python pruneEagleEye.py --data data/coco128.yaml --weights runs/Base-coco128-mAP05_0.2293.pt --cfg models/prunModels/yolov5s-pruning.yaml  --path yolov5s-pruned-eagleeye.yaml --max_iter 100 --remain_ratio 0.5 --delta 0.02

# finetune
python train.py --data data/coco128.yaml --imgsz 640 --weights runs/base-coco128-mAP05-02293-EagleEyepruned.pt --cfg yolov5s-pruned-eagleeye.yaml --epochs 100 --device 0,1
```


## 5. Quantization PTQ

5.1 export onnx
```shell
python models/export.py --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --img 640 --batch 1 --device 0,1 
```
5.2 Build a int8 engine using TensorRT's native PTQ
```shell
rm trt/yolov5s_calibration.cache

python trt/onnx_to_trt.py --model runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.onnx --dtype int8 --batch-size 4 --num-calib-batch 16 --calib-img-dir ../datasets/coco128/images/train2017
```
5.3 Evaluate the accurary of TensorRT inference result.
```shell
# datasets eval (coco128 json label Unfinished)
??? python trt/eval_yolo_trt.py --model ./weights/xxx.trt -l

# image test
python trt/demo.py --model runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5-int8-4-16-minmax.trt

python trt/demo.py --model runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned-int8-4-16-minmax.trt
```

## 6. Quantization QAT (un-done)

```shell
# QAT-finetuning(un-done)
python yolo_quant_flow.py --data data/coco128v2.yaml --cfg yolov5s-pruned-slim.yaml --ckpt-path runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --hyp data/hyp.qat.yaml --skip-layers

# Build TensorRT engine
python trt/onnx_to_trt.py --model ./weights/yolov5s-qat.onnx --dtype int8 --qat

# Evaluate the accuray of TensorRT engine
python trt/eval_yolo_trt.py --model ./weights/yolov5s-qat.trt -l
```

## 7. Export to deploy
```shell
# SlimPrune
python deploy/export_onnx_trt.py --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --device 0,1 --half --simplify

# EagleEye
python deploy/export_onnx_trt.py --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt --device 0,1 --half --simplify
```

## Test
1. test model
```shell
Base-coco128-mAP05_0.2293.pt                        14.8M
# SlimPrune
Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt      4.6M
Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.engine  6.7M
# EagleEye
Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt      6.0M
Finetune_coco128-mAP05_0.0860-EagleEyepruned.engine  8.5M
```
2. inference test
```shell
# Test: original.pt
python detect.py  --weights runs/Base-coco128-mAP05_0.2293.pt
```
> Speed: 0.3ms pre-process, 6.0ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)

```shell
# Test: pruned.pt
python detect.py  --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt
python detect.py  --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt
```
> Speed: 0.3ms pre-process, 6.3ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)  
> Speed: 0.3ms pre-process, 6.5ms inference, 0.4ms NMS per image at shape (1, 3, 640, 640)

```shell
# Test: pruned+FP16.engine
python deploy/detect_trt.py --weights runs/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.engine --device 0,1 --half
python deploy/detect_trt.py --weights runs/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.engine --device 0,1 --half
```
> Speed: 0.3ms pre-process, 1.3ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)  
> Speed: 0.3ms pre-process, 1.5ms inference, 0.7ms NMS per image at shape (1, 3, 640, 640)

## Result

|  Model   | File Size | inference per img |
|  ----  | ---- | ----  |
| Base.pt           | 14.8M | 6.0ms |
| SlimPrune.pt      |  **4.6M** | 6.3ms |
| SlimPrune.engine  |  6.7M | **1.3ms** |
| EagleEye.pt       |  6.0M | 6.5ms |
| EagleEye.engine   |  8.5M | 1.5ms |

## Plan
- [X] Base-train
- [X] Prune（SlimPrune、EagleEye）
- [X] Finetune
- [X] FP16 Quantization
- [X] INT8 PTQ
- [ ] INT8 QAT
- [ ] Change Backbone(mobilev2/...)
- [ ] Distillation


## Acknowledge
> https://github.com/Gumpest/YOLOv5-Multibackbone-Compression  
> https://github.com/maggiez0138/yolov5_quant_sample  
> https://github.com/Syencil/mobile-yolov5-pruning-distillation

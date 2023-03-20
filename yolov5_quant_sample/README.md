QAT Finetuning
yolo_quant_flow.py is the main script for QAT experiment. 

The QDQ insert, calibration, QAT-finetuning and evalution will be performed.

# QAT

QAT-Finetuning takes long time, you can skip this step and download the post-QAT model directly.


    # 1. QAT-finetuning
    python yolo_quant_flow.py --data data/coco128.yaml --cfg models/yolov5s.yaml --ckpt-path weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.pt --hyp data//hyp.qat.yaml --skip-layers
    python yolo_quant_flow.py --data data/coco128.yaml --cfg models/yolov5s.yaml --ckpt-path weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.pt --hyp data//hyp.qat.yaml --skip-layers

> ONNX export success, saved as weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5.onnx
> ONNX export success, saved as weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned.onnx
    

    # 2. Build TensorRT engine
    python trt/onnx_to_trt.py --model weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.onnx --dtype int8 --qat
    python trt/onnx_to_trt.py --model weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.onnx --dtype int8 --qat
    
> [03/20/2023-17:46:39] [TRT] [W] - 4 weights are affected by this issue: Detected subnormal FP16 values.
Serialized the TensorRT engine to file: weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt

> [03/20/2023-18:24:05] [TRT] [W] - 4 weights are affected by this issue: Detected subnormal FP16 values.
Serialized the TensorRT engine to file: weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.trt


    # 3. Evaluate the accuray of TensorRT engine
    python trt/trt_test.py --model  weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt
    python trt/trt_test.py --model  weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.trt

result:

    (yolov5) ubuntu@wilson:~/wy/model_compression/01_yolov5_quant_sample (copy)$ python trt/trt_test.py --model  weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt
    TRT model path:  weights/SlimPrune/Finetune-coco128-mAP05_0.0810-Slimpruned_0.5_skip4.trt
    [03/20/2023-18:27:28] [TRT] [I] Loaded engine size: 9 MiB
    [03/20/2023-18:27:28] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +7, now: CPU 0, GPU 7 (MiB)
    [03/20/2023-18:27:28] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +17, now: CPU 0, GPU 24 (MiB)
    [03/20/2023-18:27:28] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    ==> delay:5.856752395629883ms
    ==> output: weights/SlimPrune/result_000000000192.jpg
    ==> delay:3.1545162200927734ms
    ==> output: weights/SlimPrune/result_000000000154.jpg
    ==> delay:3.1213760375976562ms
    ==> output: weights/SlimPrune/result_000000000081.jpg
    ==> delay:3.749370574951172ms
    ==> output: weights/SlimPrune/result_000000000143.jpg


    (yolov5) ubuntu@wilson:~/wy/model_compression/01_yolov5_quant_sample (copy)$ python trt/trt_test.py --model  weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.trt
    TRT model path:  weights/EagleEye/Finetune_coco128-mAP05_0.0860-EagleEyepruned_skip4.trt
    [03/20/2023-18:26:48] [TRT] [I] Loaded engine size: 9 MiB
    [03/20/2023-18:26:48] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +7, now: CPU 0, GPU 7 (MiB)
    [03/20/2023-18:26:48] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +17, now: CPU 0, GPU 24 (MiB)
    [03/20/2023-18:26:48] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    ==> delay:6.123781204223633ms
    ==> output: weights/EagleEye/result_000000000192.jpg
    ==> delay:3.5889148712158203ms
    ==> output: weights/EagleEye/result_000000000154.jpg
    ==> delay:3.034830093383789ms
    ==> output: weights/EagleEye/result_000000000081.jpg
    ==> delay:3.7088394165039062ms
    ==> output: weights/EagleEye/result_000000000143.jpg
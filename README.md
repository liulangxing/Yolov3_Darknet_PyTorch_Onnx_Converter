# Yolov3_Darknet_PyTorch_Onnx_Converter
This Repository allows to convert *.weights file of darknet format to *.pt (pytorch format) and *.onnx (ONNX format).
Based on ultralytics repository (archive branch).This module converts *.weights files to *.pt and to *.onnx
fork from https://github.com/matankley/Yolov3_Darknet_PyTorch_Onnx_Converter

此存储库允许将darknet格式的*.weights文件转换为*.pt（pytorch格式）和*.onnx（onnx格式）。
基于ultralytics存储库（归档分支）。此模块将*.weights文件转换为*.pt和*.onnx
    
记得在cfg文件中删除如下字段，因为解析器不支持它们，并且它们不会影响推理：
    1. jitter
    2. nms_threshold (不需要删除，cfg文件中没找到这个字段)
    3. threshold (不需要删除，cfg文件中没找到这个字段)
本代码的cfg文件已经屏蔽此字段

下载代码及权重文件：
    git clone https://github.com/liulangxing/Yolov3_Darknet_PyTorch_Onnx_Converter
    cd Yolov3_Darknet_PyTorch_Onnx_Converter
    wget https://pjreddie.com/media/files/yolov3.weights
    wget https://pjreddie.com/media/files/yolov3-tiny.weights 

安装环境:
    conda create -n yolov3 python==3.7.4
    conda activate yolov3
    pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

转换命令为:    
    python converter.py yolov3-onnx.cfg yolov3.weights 608 608
    python converter.py yolov3-tiny-onnx.cfg yolov3-tiny.weights 416 416
    
预测命令为：
    python yolov3_onnx_inference.py
    python yolov3_tiny_onnx_inference.py

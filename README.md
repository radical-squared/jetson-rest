# jetson-rest
Exposes Jetson's TRT models via a rest API.
Based on Dr Robin Cole's Tensorflow-lite-rest-server:
https://github.com/robmarkcole/tensorflow-lite-rest-server

## Object and face detection
Uses ssd-mobilenet-v2 for object detection and facenet for face detection available with "/v1/vision/detection" and "/v1/vision/face" endpoints. 

## run with: 
uvicorn trt:app --reload --port 5000 --host 0.0.0.0

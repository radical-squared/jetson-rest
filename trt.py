#!/usr/bin/python3
"""
Expose Jetson's TRT models via a rest API.
Based on Dr Robin Cole's Tensorflow-lite-rest-server:
https://github.com/robmarkcole/tensorflow-lite-rest-server
"""
import io
import sys

import numpy as np
import cv2
import jetson.inference
import jetson.utils
from fastapi import FastAPI, File, HTTPException, UploadFile

from datetime import datetime


# from helpers import classify_image, read_labels, set_input_tensor

app = FastAPI()

# Settings
MIN_CONFIDENCE = 0.3  # The absolute lowest confidence for a detection.
# URL
FACE_DETECTION_URL = "/v1/vision/face"
OBJ_DETECTION_URL = "/v1/vision/detection"


odNet = jetson.inference.detectNet("ssd-mobilenet-v2")   #, sys.argv, opt.threshold)
fdNet = jetson.inference.detectNet("facenet")



@app.get("/")
async def info():
    return """tflite-server docs at ip:port/docs"""


@app.post(FACE_DETECTION_URL)
async def predict_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        # image_bytes = Image.open(io.BytesIO(contents))


        i = np.frombuffer(contents, dtype=np.uint8)
        im = cv2.imdecode(i, cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(datetime.now().strftime('%m%d_%H%M%S%f')+'.jpg',im)

        tsr_imga = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
        cudaImage = jetson.utils.cudaFromNumpy (tsr_imga)

        detections = fdNet.Detect(cudaImage, im.shape[0], im.shape[1]) 
        # jetson.utils.saveImage(datetime.now().strftime('%m%d_%H%M%S%f')+'.jpg', cudaImage)
        
        
        data = {"success": False}


        if detections:
            
            preds = []
            for detection in detections:
                preds.append(
                    {
                        "confidence": float(detection.Confidence),
                        "label": fdNet.GetClassDesc(detection.ClassID), 
                        "y_min": int(detection.Top),    
                        "x_min": int(detection.Left),   
                        "y_max": int(detection.Bottom), 
                        "x_max": int(detection.Right), 
                    }
                )
            data["predictions"] = preds
            data["success"] = True

        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(OBJ_DETECTION_URL)
async def predict_object(image: UploadFile = File(...)):
    try:
        contents = await image.read()

        i = np.frombuffer(contents, dtype=np.uint8)
        im = cv2.imdecode(i, cv2.IMREAD_UNCHANGED)
        # cv2.imwrite(datetime.now().strftime('%m%d_%H%M%S%f')+'.jpg',im)


        tsr_imga = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
        cudaImage = jetson.utils.cudaFromNumpy (tsr_imga)

        detections = odNet.Detect(cudaImage, im.shape[0], im.shape[1]) 
        # jetson.utils.saveImage(datetime.now().strftime('%m%d_%H%M%S%f')+'.jpg', cudaImage)
    

        data = {"success": False}
        if detections:
            
            preds = []
            for detection in detections:
                preds.append(
                    {
                        "confidence": float(detection.Confidence),
                        "label": odNet.GetClassDesc(detection.ClassID), 
                        "y_min": int(detection.Top),    
                        "x_min": int(detection.Left),   
                        "y_max": int(detection.Bottom), 
                        "x_max": int(detection.Right),  
                    }
                )
            data["predictions"] = preds
            data["success"] = True

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))



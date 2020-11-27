# jetson-rest
Exposes Jetson's TRT models via a rest API.
Based on Dr Robin Cole's Tensorflow-lite-rest-server:
https://github.com/robmarkcole/tensorflow-lite-rest-server

# run with: 
uvicorn trt:app --reload --port 5000 --host 0.0.0.0

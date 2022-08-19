# YoloLite

iOS app for object detection with Yolov7 in Tensorflow Lite. The app files are on the YoloLite folder. The other two folders were for model parsing and optimization. 

In the python folder you can find the implementation of the model in PyTorch and Tensorflow, together with the files for converting the model from PyTorch to ONNX to Tensorflow to TensorflowLite. If you run `python3 PyTorchYolo.py` or `python3 TensorflowYolo.py` you should see an image with the bounding boxes painted on it. For the environment just install dependencies through the `requirements.txt` file.

In the cplusplus folder there is a Makefile for working with OpenCV in C++. I just used this language to see how fast the images can be processed. If you are on MacOS installing OpenCV can be done through HomeBrew by `brew install opencv`.

Right now the bottleneck of the application is moving the channels axis from the last to the first. The algorithm is $O(n)$ passing only one time for each pixel, but in Swift that is terribly slow. For the record, that operation takes normally 0.5sec in Swift and 0.008sec in C++, and something in the middel in python.

In the future I plan on using the Metal library to exploit the GPU capabilities of the iPhone.

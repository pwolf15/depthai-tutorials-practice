#!/usr/bin/env python3
import cv2
import depthai as dai
from scipy.ndimage.filters import median_filter
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# provided by: https://www.idtools.com.au/unsharp-masking-python-opencv/
def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = median_filter(image, sigma)
    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf,cv2.CV_64F)
    # Calculate the sharpened image
    sharp = image-strength*lap
    # Saturate the pixels in either direction
    sharp[sharp>255] = 255
    sharp[sharp<0] = 0
    
    return sharp

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the rgb frames
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    sigma = 5
    while True:
        inRgb = qRgb.get() # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        img = inRgb.getCvFrame()

        sharp1 = np.zeros_like(img)
        for i in range(3):
            sharp1[:,:,i] = unsharp(img[:,:,i], 5, 0.8)

        cv2.imshow('bgr', img)
        cv2.imshow('gray_sharp', sharp1)

        key = cv2.waitKey(1)
        if  key == ord('q'):
            break
        elif key == ord('a'):
            sigma += 2
            blur = cv2.GaussianBlur(img,(sigma, sigma),0)
            print(f'sigma: {sigma}')
        elif key == ord('s'):
            sigma -= 2
            if sigma <= 1:
                sigma = 1
            blur = cv2.GaussianBlur(img,(sigma, sigma),0)
            print(f'sigma: {sigma}')
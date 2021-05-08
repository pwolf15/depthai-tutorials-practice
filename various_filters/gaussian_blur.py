#!/usr/bin/env python3
import cv2
import depthai as dai

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
        blur = cv2.GaussianBlur(img,(sigma, sigma),0)
        cv2.imshow('bgr', blur)
        cv2.imshow('bgr2', img)

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
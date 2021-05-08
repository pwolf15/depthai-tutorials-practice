import cv2
import depthai as dai
import numpy as np

extended_disparity = False
subpixel = False
lr_check = False
pipeline = dai.Pipeline()
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputDepth(False)

median = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
depth.setMedianFilter(median)

depth.setLeftRightCheck(lr_check)

max_disparity = 95

if extended_disparity: max_disparity *= 2
depth.setExtendedDisparity(extended_disparity)

if subpixel: max_disparity *= 32
depth.setSubpixel(subpixel)

multiplier = 255 / max_disparity

left.out.link(depth.left)
right.out.link(depth.right)

xout = pipeline.createXLinkOut()
xout.setStreamName('disparity')
depth.disparity.link(xout.input)

with dai.Device(pipeline) as device:

    device.startPipeline()
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    while True:
        inDepth = q.get()
        frame = inDepth.getFrame()
        frame = (frame * multiplier).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        cv2.imshow("disparity", frame)

        if cv2.waitKey(1) == ord('q'):
            break
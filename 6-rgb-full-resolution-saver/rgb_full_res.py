import time
from pathlib import Path

import cv2
import depthai as dai

# Start defining a pipeline
pipeline = dai.Pipeline()

# define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)

# Create RGB output
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Create encoder to produce JPEG images
videoEnc = pipeline.createVideoEncoder()
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
camRgb.video.link(videoEnc.input)

# Create JPEG output
xoutJpeg = pipeline.createXLinkOut()
xoutJpeg.setStreamName("jpeg")
videoEnc.bitstream.link(xoutJpeg.input)

# Connect and start the pipeline
with dai.Device(pipeline) as device:

    qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    Path('06_data').mkdir(parents=True, exist_ok=True)

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            cv2.imshow("rgb", inRgb.getCvFrame())

        for encFrame in qJpeg.tryGetAll():
            with open(f"06_data/{int(time.time()*10000)}.jpeg", "wb") as f:
                f.write(bytearray(encFrame.getData()))

        if cv2.waitKey(1) == ord('q'):
            break 
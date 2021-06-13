import numpy as np
import cv2
import depthai

# normalize bounding box coordinates (0..1) to actual frame size
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

# pipeline
pipeline = depthai.Pipeline()

# camera node
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# mobilenet-ssd network
#   single-shot detection (SSD) network to perform object detection
#   Caffe framework
#   blob 1x3x300x300 model input

detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath("/home/pwolf/dev/ml/depthai-tutorials-practice/1-hello-world/mobilenet-ssd.blob")

# connect color camera output to nn input
cam_rgb.preview.link(detection_nn.input)

# XLink handles communication between device and host
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# initialize and start depthAI device
device = depthai.Device(pipeline)
device.startPipeline()

# grab camera frames and neural network inferences
q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")

frame = None
bboxes = []

while True:
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1,2,0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16())
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]]  # remove up to delimiter (fixed size array)
        bboxes = bboxes.reshape((bboxes.size // 7, 7)) # group bboxes by size 7
        bboxes = bboxes[bboxes[:,2] > 0.8][:, 3:7] # filter out less confident results
    if frame is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.imshow("preview", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# coding=utf-8
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from utils import getDeviceInfo, non_max_suppression, FPSHandler, process_mask, toTensorResult

ROOT = Path(__file__).parent

# 配置参数
nnWidth, nnHeight = 320, 320  # 模型输入尺寸保持原样
MAX_DEPTH = 9000  # 最大深度显示范围（毫米）
MIN_DEPTH = 200  # 最小深度显示范围

labelMap = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def create_pipeline():
    blob = ROOT.joinpath("yolov8n-seg.blob")
    model = dai.OpenVINO.Blob(blob)
    dim = next(iter(model.networkInputs.values())).dims
    nnWidth, nnHeight = dim[:2]

    pipeline = dai.Pipeline()

    # 定义节点
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    detectionNN = pipeline.create(dai.node.NeuralNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutDepth.setStreamName("depth")

    # RGB相机配置
    camRgb.setPreviewSize(nnWidth, nnHeight)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(30)

    # 双目相机配置
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # 立体深度配置
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(nnWidth, nnHeight)

    # 深度后处理
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    config = stereo.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.thresholdFilter.minRange = MIN_DEPTH
    config.postProcessing.thresholdFilter.maxRange = MAX_DEPTH
    stereo.initialConfig.set(config)

    # 神经网络配置
    detectionNN.setBlob(model)
    detectionNN.setNumInferenceThreads(2)

    # 连接节点
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    stereo.depth.link(xoutDepth.input)
    camRgb.preview.link(detectionNN.input)
    detectionNN.passthrough.link(xoutRgb.input)
    detectionNN.out.link(xoutNN.input)

    return pipeline

def run():
    with dai.Device(create_pipeline(), getDeviceInfo()) as device:
        # 创建队列
        qRgb = device.getOutputQueue("rgb", 4, False)
        qDet = device.getOutputQueue("detections", 4, False)
        qDepth = device.getOutputQueue("depth", 4, False)

        # 初始化参数
        last_detections = []
        last_protos = []
        fpsHandler = FPSHandler()
        BLUE_COLOR = (255, 191, 0)  # BGR格式的蓝色

        # 显示设置
        cv2.namedWindow("Dual View")

        # 视频录制（调整为较低分辨率640x240）
        videoWriter = cv2.VideoWriter(
            ROOT.joinpath(f"dual_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4").as_posix(),
            cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 240)
        )

        while True:
            # 同步获取关键数据流
            inRgb = qRgb.get()
            inDepth = qDepth.get()
            inDet = qDet.tryGet()

            # 处理RGB帧
            frame = inRgb.getCvFrame()
            rgbDisplay = frame.copy()
            fpsHandler.tick("color")

            # 处理检测结果
            if inDet is not None:
                tensor = toTensorResult(inDet)
                last_detections = non_max_suppression(tensor["output0"], 0.5, nc=80)[0]
                last_protos = tensor.get("output1", [np.zeros((32, 160, 160))])[0]
                fpsHandler.tick("det")

            # 处理深度数据
            depthFrame = inDepth.getFrame()
            validDepth = np.where((depthFrame >= MIN_DEPTH) & (depthFrame <= MAX_DEPTH), depthFrame, 0)
            depthNormalized = cv2.normalize(validDepth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depthColored = cv2.applyColorMap(255 - depthNormalized, cv2.COLORMAP_JET)
            depthDisplay = cv2.resize(depthColored, (nnWidth, nnHeight))
            fpsHandler.tick("depth")

            # 处理显示叠加
            h, w = frame.shape[:2]
            depthOverlay = cv2.resize(depthDisplay, (w, h))

            if len(last_detections) > 0:
                masks = process_mask(last_protos, last_detections[:, 6:], last_detections[:, :4], (h, w), upsample=True)

                for mask, det in zip(masks, last_detections):
                    clsId = int(det[5])
                    conf = det[4]
                    bbox = list(map(int, det[:4]))

                    # 计算中心点深度
                    cx = np.clip((bbox[0] + bbox[2]) // 2, 0, w - 1)
                    cy = np.clip((bbox[1] + bbox[3]) // 2, 0, h - 1)
                    depthVal = depthFrame[cy, cx] if (MIN_DEPTH <= depthFrame[cy, cx] <= MAX_DEPTH) else 0

                    # 在所有视图应用蓝色掩码
                    alpha = 0.5
                    for display in [rgbDisplay, depthOverlay]:
                        display[mask > 0.5] = display[mask > 0.5] * (1 - alpha) + np.array(BLUE_COLOR) * alpha

                    # 绘制蓝色边界框和文字
                    for display in [rgbDisplay, depthOverlay]:
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BLUE_COLOR, 2)
                        cv2.putText(display, f"{labelMap[clsId]} {conf:.2f}",
                                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE_COLOR, 2)
                        cv2.putText(display, f"{depthVal}mm",
                                    (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 调整显示尺寸到更小分辨率
            displayRGB = cv2.resize(rgbDisplay, (320, 240))
            displayDepth = cv2.resize(depthOverlay, (320, 240))

            # 组合显示视图
            dualView = cv2.hconcat([displayRGB, displayDepth])

            # 添加视图标题
            titles = ["RGB View", "Depth View"]
            for i, title in enumerate(titles):
                cv2.putText(dualView, title, (10 + 320 * i, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示和录制
            cv2.imshow("Dual View", dualView)
            videoWriter.write(dualView)

            # 退出控制
            if cv2.waitKey(1) == ord('q'):
                videoWriter.release()
                break


if __name__ == "__main__":
    run()
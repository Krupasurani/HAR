from flask import Flask, render_template, Response
from flask import request
import cv2
import socket
import numpy as np
import pickle
# import request as req
import os
import torch
import math
import PIL
from VideoCamera import VideoCamera

app = Flask(__name__)
# camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture(f'{link}')
camera = VideoCamera(0)


@app.route('/')
def index():
    return render_template('index.html')


# def gen_frames(link):  # generate frame by frame from camera
def gen_frames():


    img_size = [640, 640]


    # print(os.getcwd())
    # os.chdir("YOLOv6")
    # print(os.getcwd())


    from yolov6.utils.events import LOGGER, load_yaml
    from yolov6.layers.common import DetectBackend
    from yolov6.data.data_augment import letterbox
    from yolov6.utils.nms import non_max_suppression
    from yolov6.core.inferer import Inferer

    from typing import List, Optional

    device = 'gpu'
    half = False

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    # print(device)

    def check_img_size(img_size, s=32, floor=0):
        def make_divisible( x, divisor):
            # Upward revision the value x to make it evenly divisible by the divisor.
            return math.ceil(x / divisor) * divisor
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2
    

    def precess_image(path, img_size, stride, half):
        '''Process image before image inference.'''
        try:
            from PIL import Image
            img_src = frame
            assert img_src is not None, f'Invalid image: {path}'
        except Exception as e:
            LOGGER.Warning(e)
        image = letterbox(img_src, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src
    

    model = DetectBackend(f"./best_ckpt_2.pt", device=device)
    stride = model.stride
    class_names = load_yaml("./my_dataset.yaml")['names']

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    # camera = cv2.VideoCapture(f'{link}')

    while True:
        # Capture frame-by-frame
        success, frame = camera.videoRead()  # read the camera frame
        # print(frame)

        # =================================================================================================================
        

        # width  = camera.get(3)  # float `width`
        # height = camera.get(4)
        


        # -------------------------------------------------------------------------------------------------------------------

        hide_labels = False
        hide_conf = False

        img_size = 640

        conf_thres = .30
        iou_thres = .45
        max_det = 1000
        agnostic_nms = False

        img_size = check_img_size(img_size, s=stride)

        img, img_src = precess_image(frame, img_size, stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = model(img)
        classes:Optional[List[int]] = None # the classes to keep
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()
        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))


        # =================================================================================================================

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img_ori)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/cam', methods=['POST'])
@app.route('/cam')
def cam():
    # if request.method == 'POST':
    #     link = request.form['link']
    #     print("The client IP is: {}".format(request.environ['REMOTE_ADDR']))
    #     print("The client port is: {}".format(request.environ['REMOTE_PORT']))
    #     return Response(gen_frames(link), mimetype='multipart/x-mixed-replace; boundary=frame')
    # else :
    #     return render_template('cam.html')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # app.run(host='192.168.0.138', port=5000, debug=True)
    app.run(debug=True)

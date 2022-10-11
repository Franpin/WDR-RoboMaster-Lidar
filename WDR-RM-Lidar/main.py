# main.py
# import the necessary packages
from flask import Flask, render_template, Response
import cv2
import time
import darknet
import os
import argparse
from threading import Thread, enumerate
from queue import Queue
from flask import Flask, render_template, request
import configparser
import json
app = Flask(__name__)
app.config.from_object(configparser)


@app.route('/upload', methods=['get', 'post'])
def upload():
    # print(request.files.get("image"))
    imgfile = request.files.get("image")
    # os.system("python3 darknet")
    str1="static/images/handle/upload.jpg"
    # imgfile.save("static/images/handle/upload.jpg")
    # os.system("python3 darknet_images.py --weights=yolov4_mask_4000.weights --config_file=yolov4_mask.cfg --data=mask.data --input="+str1)
    return "static/images/handle/handled.jpg"  # 返回处理后的图片的url


@app.route('/list', methods=['get'])
def list():
    imgList = [{"id": 1, "filename": "1.jpg", "url": "../../../static/images/list/1.jpg"}, \
               {"id": 2, "filename": "2.jpg", "url": "../../../static/images/list/2.jpg"}, \
               {"id": 3, "filename": "3.jpg", "url": "../../../static/images/list/3.jpg"}, \
               {"id": 4, "filename": "4.jpg", "url": "../../../static/images/list/4.jpg"}, \
               {"id": 5, "filename": "5.jpg", "url": "../../../static/images/list/5.jpg"}, \
               {"id": 6, "filename": "6.jpg", "url": "../../../static/images/list/6.jpg"}, \
               {"id": 7, "filename": "7.jpg", "url": "../../../static/images/list/7.jpg"}, \
               {"id": 8, "filename": "8.jpg", "url": "../../../static/images/list/8.jpg"}, \
               {"id": 9, "filename": "9.jpg", "url": "../../../static/images/list/9.jpg"}, \
               {"id": 10, "filename": "10.jpg", "url": "../../../static/images/list/10.jpg"}]

    return json.dumps(imgList)
@app.route('/pages/detailImageList')
def showImageList():
    return render_template("pages/detailImageList/detailImageList.html")
@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='x-requested-with,content-type'
    return environ
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen():
    image=None
    ds_factor = 0.6
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    # args = parser()
    config_file = 'yolov4_mask.cfg'
    data_file = 'mask.data'
    weights = 'yolov4_mask_4000.weights'
    # check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    cap = cv2.VideoCapture('video.mp4')
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        img_for_detect = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)

        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
        detections_queue.put(detections)
        fps = int(1 / (time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, 0)
        darknet.free_image(darknet_image)

        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Inference', image)
            # if args.out_filename is not None:
            #     video.write(image)
            # if not args.dont_show:
            #     cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break
###################sadas
        # frame = cv2.resize(image, None, fx=ds_factor, fy=ds_factor,
        #                    interpolation=cv2.INTER_AREA)
        frame=image
        if len(detections)!=0:
            if detections[0][0] == 'face':
                i=i+1
                if i%100==0:
                    cv2.imwrite("static/images/list/"+str(i/100)+".jpg",frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    # gen(VideoCamera())
    app.run(host='0.0.0.0',port='5000', debug=True)
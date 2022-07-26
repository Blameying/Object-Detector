#!/usr/bin/python3
import cv2
import numpy as np
import argparse
from predict import Detector, scale_coords
import json
import socket
import time
import multiprocess
from multiprocess import Process, Queue


def get_img_size(file):
    return cv2.imread(file, cv2.IMREAD_GRAYSCALE).shape[:2]


def calibration(img, camera_matrix, dist_coefs):
    if img is None:
        return

    h, w = img.shape[:2]
    new_cameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, new_cameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst


def detect_desktop(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # dst = np.zeros_like(img_gray)
    # cv2.normalize(img_gray, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)

    dst = float(2) * img_gray
    dst[dst > 255] = 255
    dst = np.round(dst)
    dst = dst.astype(np.uint8)
    ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    # print(len(contours), hierarchy, sep='\n')

    result = contours[0]
    result = cv2.approxPolyDP(
        result, 0.01 * cv2.arcLength(contours[0], True), True)
    assert(len(result) >= 4)
    cv2.drawContours(img, [result], 0, 255, 3)
    cv2.imshow("contours", img)
    # cv2.waitKey(0)
    result = result.reshape(-1, 2)
    middle = np.sum(result, axis=0) / len(result)

    four_area_split = [[], [], [], []]
    for item in result:
        if (item[0] <= middle[0] and item[1] <= middle[1]):
            # left_top_group area
            four_area_split[0].append(item)
        elif (item[0] >= middle[0] and item[1] <= middle[1]):
            # right_top_group area
            four_area_split[1].append(item)
        elif (item[0] <= middle[0] and item[1] >= middle[1]):
            # left_bottom_group area
            four_area_split[2].append(item)
        elif (item[0] >= middle[0] and item[1] >= middle[1]):
            # right_bottom_group.area
            four_area_split[3].append(item)

    for group in four_area_split:
        group.sort(key=lambda i: np.linalg.norm(i - middle), reverse=True)

    return np.array([item[0] for item in four_area_split])


send_time = time.time()
def send_data(address, port, data):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest_addr = (address, port)
    json_str = json.dumps(data)
    udp_socket.sendto(bytes(json_str, encoding='utf8'), dest_addr)
    udp_socket.close()


if __name__ == '__main__':
    multiprocess.freeze_support()
    # init model
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='coco_phone_best.onnx', help='onnx path(s)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--line-thickness', default=1,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--ip', default='127.0.0.1',
                        type=str, help="game server's ip address")
    parser.add_argument('--port', default=7180,
                        type=int, help="game server's ip port")
    opt = parser.parse_args()

    # init camera
    fs = cv2.FileStorage("camera_params.yml", cv2.FileStorage_READ)
    rms = fs.getNode("rms").real()
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coefs = fs.getNode("dist_coefs").mat()
    h = int(fs.getNode("h").real())
    w = int(fs.getNode("w").real())
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())


    # load model

    QUEUE_SIZE = 5
    data_pipe = Queue(QUEUE_SIZE)
    sender_pipe = Queue(QUEUE_SIZE)

    def productor(data_pipe, camera_matrix, dist_coefs):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, frame = capture.read()
            img = calibration(frame, camera_matrix, dist_coefs)
            corners = detect_desktop(img)
            dst_pos = np.float32([[0, 0], [img.shape[1], 0],
                        [0, img.shape[0]], [img.shape[1], img.shape[0]]])
            transform_mat = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pos)
            img = cv2.warpPerspective(img, transform_mat, (img.shape[1], img.shape[0]))
            cv2.imshow("final", img)
            data_pipe.put(img)
            cv2.waitKey(10)
    
    def consumer(opt, data_pipe, sender_pipe):
        detector = Detector(opt)
        while True:
            img = data_pipe.get()
            sender_pipe.put(detector.detect(img))
    
    def sender(sender_pipe, opt):
        global send_time
        while True:
            im0, pred_boxes, pred_confes = sender_pipe.get()
            data = {'pos': []}
            if len(pred_boxes) > 0:
                for i, _ in enumerate(pred_boxes):
                    box = pred_boxes[i]
                    left, top, width, height = box[0], box[1], box[2], box[3]
                    x = (left + width / 2) / opt.img_size
                    y = (top + height / 2) / opt.img_size
                    data['pos'].append({'x': x, 'y': y})
                    print(data)
                send_data(opt.ip, opt.port, data)
            send_time = time.time()


    threads = []
    threads.append(Process(target=productor, args=(data_pipe, camera_matrix, dist_coefs,)))
    threads.append(Process(target=consumer, args=(opt, data_pipe, sender_pipe,)))
    threads.append(Process(target=sender, args=(sender_pipe, opt,)))
        
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # for corner in corners:
    #    cv2.circle(img, (corner[0], corner[1]), 10, (0, 0, 255), 2),
    #
    # print(img.shape)
    # cv2.imshow("corners", img)
    # cv2.waitKey(0)
    # dst_pos = np.float32([[0, 0], [img.shape[1], 0],
    #                    [0, img.shape[0]], [img.shape[1], img.shape[0]]])
    # transform_mat = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pos)
    # img = cv2.warpPerspective(img, transform_mat, (img.shape[1], img.shape[0]))
    # cv2.imshow("final", img)
    # cv2.waitKey(0)

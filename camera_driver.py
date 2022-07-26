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
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        img, arucoDict, parameters=arucoParams)
    print(ids)
    result = {}
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            if markerID == 0:
                result[0] = (int(topLeft[0]), int(topLeft[1]))
            elif markerID == 1:
                result[1] = (int(topRight[0]), int(topRight[1]))
            elif markerID == 2:
                result[2] = (int(bottomLeft[0]), int(bottomLeft[1]))
            elif markerID == 3:
                result[3] = (int(bottomRight[0]), int(bottomRight[1]))

    return result


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
        adjusted = False
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ret, img = capture.read()
            # cv2.imshow("final", img)
            # cv2.waitKey(10)
            # img = calibration(frame, camera_matrix, dist_coefs)
            if not adjusted:
                corners = detect_desktop(img)
                if len(corners) == 4:
                    print(corners)
                    dst_pos = np.float32([[0, 0], [img.shape[1], 0],
                                          [0, img.shape[0]], [img.shape[1], img.shape[0]]])
                    print(corners)
                    real_pos = []
                    for i in range(0, len(corners)):
                        real_pos.append([corners[i][0], corners[i][1]])

                    transform_mat = cv2.getPerspectiveTransform(
                        np.float32(real_pos), dst_pos)
                    adjusted = True
                    cv2.imshow("final", img)
                    cv2.waitKey()
            else:
                img = cv2.warpPerspective(
                    img, transform_mat, (img.shape[1], img.shape[0]))
                data_pipe.put(img)

    def consumer(opt, data_pipe, sender_pipe, is_ai=False):
        detector = Detector(opt, is_ai=False)
        while True:
            img = data_pipe.get()
            sender_pipe.put(detector.detect(img))

    def sender(sender_pipe, opt, is_ai=False):
        global send_time
        while True:
            data = {'pos': [{"x": 0, "y": 0}, {"x": 0, "y": 0}]}
            updated = False
            if is_ai:
                im0, pred_boxes, pred_confes = sender_pipe.get()
                if len(pred_boxes) > 0:
                    for i, _ in enumerate(pred_boxes):
                        box = pred_boxes[i]
                        left, top, width, height = box[0], box[1], box[2], box[3]
                        x = (left + width / 2) / opt.img_size
                        y = (top + height / 2) / opt.img_size
                        data['pos'].append({'x': x, 'y': y})
                        updated = True
            else:
                result = sender_pipe.get()
                if 4 in result.keys():
                    data['pos'][0] = {'x': result[4][0], 'y': result[4][1]}
                    updated = True
                if 5 in result.keys():
                    data['pos'][1] = {'x': result[5][0], 'y': result[5][1]}
                    updated = True

            if updated:
                send_data(opt.ip, opt.port, data)
            print("Time: {}".format((time.time() - send_time) * 1000))
            send_time = time.time()

    threads = []
    threads.append(Process(target=productor, args=(
        data_pipe, camera_matrix, dist_coefs,)))
    threads.append(
        Process(target=consumer, args=(opt, data_pipe, sender_pipe,)))
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

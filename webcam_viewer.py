import numpy as np
import cv2
from datetime import datetime
import math

devices = []
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        devices.append(cap)


gray = False
sharpen = False
edges = False
lines = False
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
source = devices[0]


def draw_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,255,255], 2)

def find_angle(lines, aoi=np.pi / 30):
    # aoi: angle of interest. e.g: 30deg ->
    # -15..15 -> 0
    # 30..60 -> 45
    # 75..105 -> 90
    # rest discarded
    pi = np.pi
    bounds = {-pi/2: 90, -pi/4: -45, 0: 0, pi/4: 45, pi/2: 90}
    if lines is None:
        return 0
    total_length = {-45:0, 0: 0, 45: 0, 90: 0}
    for line in lines:
        coords = line[0]
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(abs(dx), dy)
        for b, a in bounds.items():
            if b - aoi / 2 < angle < b + aoi / 2:
                total_length[a] += length
                break
    return total_length



while source.isOpened():
    ret, frame = source.read()
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if sharpen:
        frame = cv2.filter2D(frame, -1, kernel)
    if edges:
        frame = cv2.Canny(frame, 150,200)
    if lines and edges:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        l = cv2.HoughLinesP(frame, 1, np.pi / 180, 180, np.array([]), 10, 3)
        draw_lines(frame, l)
        total_length = find_angle(l)
        print(total_length)

    # frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "camera (c): {}/{} / quit (q) save (s) / edges (e) / gray (g): {} / sharpen (p): {}".format(devices.index(source) + 1, len(devices), gray, sharpen)
    cv2.putText(frame, text, (10, 15), font, 0.5, (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    k = chr(cv2.waitKey(1) & 0xFF)
    if k == 'c':
        i = (devices.index(source) + 1) % len(devices)
        source = devices[i]
    elif k == 'q':
        for s in devices:
            s.release()
        cv2.destroyAllWindows()
    elif k == 's':
        filename = './caps/{}.jpg'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
        print(filename)
        cv2.imwrite(filename, frame)

    elif k == 'g':
        gray = not gray
    elif k == 'p':
        sharpen = not sharpen
    elif k == 'e':
        edges = not edges
    elif k == 'l':
        lines = not lines

print("No devices open.")

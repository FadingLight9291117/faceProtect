from pathlib import Path

import cv2
import numpy as np


def decode(data_file):
    data_file = Path(data_file)
    with data_file.open('rb+') as f:
        data = f.read()

    N = 5
    faces = []
    i = 0
    while i < len(data):
        dlen = N * 4
        five = np.frombuffer(data[i:i + dlen], np.int32)
        i += dlen

        dlen = five[N - 1]
        face_data = np.frombuffer(data[i:i + dlen], np.uint8)
        i += dlen

        face = {
            'xmin': five[0],
            'ymin': five[1],
            'xmax': five[2],
            'ymax': five[3],
            'data_length': five[4],
            'rgb_data': face_data.reshape(five[3] - five[1], five[2] - five[0], 3)
        }
        faces.append(face)
    return faces


def replace_face(img_with_mask_path, faces):
    img = cv2.imread(img_with_mask_path)
    for face in faces:
        xmin = face['xmin']
        ymin = face['ymin']
        xmax = face['xmax']
        ymax = face['ymax']
        data = face['rgb_data']
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        img[ymin: ymax, xmin: xmax] = data
    cv2.imshow('1', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    data_files = Path('result/data').rglob('*')
    data_files = sorted([str(data_file) for data_file in data_files])
    img_files = Path('result/images_with_mask').rglob('*')
    img_files = sorted([str(img_file) for img_file in img_files])

    for data_file, img_file in zip(data_files, img_files):
        faces = decode(data_file=str(data_file))
        replace_face(str(img_file), faces)

from pathlib import Path

import cv2

from Face import Faces, FaceUtils


def show():
    data_files = Path('result/data').rglob('*')
    data_files = sorted([str(data_file) for data_file in data_files])
    img_files = Path('result/images_with_mask').rglob('*')
    img_files = sorted([str(img_file) for img_file in img_files])

    for data_file, img_file in list(zip(data_files, img_files))[:2]:
        faces = Faces.load(str(data_file))
        FaceUtils.replace_face(str(img_file), faces)


if __name__ == '__main__':
    show()

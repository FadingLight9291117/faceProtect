import functools
import operator
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import hashlib

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Face:
    xmin: np.int32
    ymin: np.int32
    xmax: np.int32
    ymax: np.int32
    data_length: np.int32
    rgb_data: np.ndarray

    @classmethod
    def deserialize(cls, face_bytes):
        N = 5
        dlen = N * 4
        five = np.frombuffer(face_bytes[:dlen], np.int32)
        xmin = five[0]
        ymin = five[1]
        xmax = five[2]
        ymax = five[3]
        data_length = five[4]
        rgb_data = np.frombuffer(face_bytes[dlen: dlen + data_length], np.uint8).reshape(ymax - ymin, xmax - xmin, 3)
        face = cls(xmin, ymin, xmax, ymax, data_length, rgb_data)
        return face

    def serialize(self):
        data = [value.tobytes() for value in self.__dict__.values()]
        face_bytes = functools.reduce(operator.add, data)
        return face_bytes

    @classmethod
    def from_img(cls, img, pose):
        pose = np.array(pose, dtype=np.int32)
        xmin = pose[0]
        ymin = pose[1]
        xmax = pose[2]
        ymax = pose[3]
        face_data = img.crop((xmin, ymin, xmax, ymax))
        rgb_data = np.array(face_data, dtype=np.uint8)
        data_length = np.int32(rgb_data.size)
        return Face(xmin, ymin, xmax, ymax, data_length, rgb_data)

    def __len__(self):
        return 5 * 4 + self.data_length


class Faces:
    hash_len = 64

    def __init__(self, faces):
        self.faces = faces

    @classmethod
    def from_img(cls, img_file, poses):
        img = Image.open(img_file)
        faces = [Face.from_img(img, pose) for pose in poses]
        return cls(faces)

    def dump(self, file_path, encoder=lambda x, key: x, key=None):
        serialized_data = self.serialize()
        encoded_data = encoder(serialized_data, key)
        hash_code = self.hash(encoded_data)
        data = hash_code.encode() + encoded_data

        with Path(file_path).open('wb') as f:
            f.write(data)

    @classmethod
    def load(cls, faces_file, decoder=lambda x, key: x, key=None):
        with Path(faces_file).open('rb+') as f:
            encoded_data = f.read()
        decoded_data = decoder(encoded_data, key)
        faces = cls.deserialize(decoded_data)
        return cls(faces)

    def serialize(self):
        faces_bytes_list = [face.serialize() for face in self.faces]
        faces_bytes = functools.reduce(operator.add, faces_bytes_list) if faces_bytes_list != [] else b''
        return faces_bytes

    @classmethod
    def deserialize(cls, faces_bytes):
        ptr = cls.hash_len
        faces = []
        while ptr < len(faces_bytes):
            face = Face.deserialize(faces_bytes[ptr:])
            ptr += len(face)
            faces.append(face)
        return faces

    @classmethod
    def hash(cls, data):
        m = hashlib.sha256()
        m.update(data)
        return m.hexdigest()

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        return self.faces[idx]


class FaceUtils:
    @staticmethod
    def replace_face(img_with_mask_path, faces):
        img = Image.open(img_with_mask_path)
        img_data = np.array(img)
        for face in faces:
            face_data = face.rgb_data
            img_data[face.ymin: face.ymax, face.xmin: face.xmax] = face_data
        img = Image.fromarray(img_data)

        img.show()


if __name__ == '__main__':
    faces = Faces.from_img('result/images_with_mask/12024129.jpg', [[0, 0, 1000, 1000], [100, 100, 200, 200]])
    faces.dump('test.dat')
    faces = Faces.load('test.dat')
    img = Image.fromarray(faces[1].rgb_data)
    img.show()

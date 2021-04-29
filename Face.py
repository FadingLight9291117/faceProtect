from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Face:
    xmin: np.int32
    ymin: np.int32
    xmax: np.int32
    ymax: np.int32
    data_length: np.int32
    rgb_data: np.ndarray

    @classmethod
    def of(cls, faces_bytes):
        pass

    def to(self):
        pass


class Faces:

    def __init__(self, img_id, faces=None):
        self.faces = faces if faces is not None else []
        self.img_id = img_id

    def add(self, face: Face):
        self.faces.append(face)

    def serialize(self):
        serialized_data = None
        """
        TODO: 将人脸数据序列化
        """

        return serialized_data

    @staticmethod
    def deserialize(serialized_data):
        faces = []
        """
        TODO: 将序列化的数据转化为对象
        """
        return faces

    def dump(self, path, encoder=lambda x: x):
        serialized_data = self.serialize()
        encoded_data = self.encode(serialized_data)

        with Path(path).open('wb+') as f:
            f.write(encoded_data)

    @staticmethod
    def load(faces_path, img_id, decoder=lambda x: x):
        encoded_data = None
        with Path(faces_path).open('rb+') as f:
            encoded_data = f.read()
        decoded_data = Faces.decode(encoded_data, decoder)

        faces = Faces.deserialize(decoded_data)
        return Faces(img_id, faces)

    def encode(self, serialized_data, encoder):
        return encoder(serialized_data)

    @staticmethod
    def decode(encoded_data, decoder):
        return decoder(encoded_data)

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        return self.faces[idx]

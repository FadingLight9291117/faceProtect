import sys

from typing import List

sys.path.append('../')

import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from utils.renderer import Renderer
from img2pose.img2pose import img2poseModel
from img2pose.model_loader import load_model
from pathlib import Path
import cv2
from functools import partial, reduce
import operator

np.set_printoptions(suppress=True)


def render_plot(img, poses, bboxes, save_path=None):
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy",
        triangles_path="pose_references/triangles.npy"
    )

    (w, h) = img.size
    origin_img = img.copy()
    origin_img = np.asarray(origin_img)

    # 渲染3D人脸
    trans_vertices = renderer.transform_vertices(img, poses)
    img = renderer.render(img, trans_vertices, alpha=1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 扩大人脸框
    expend = partial(expend_bbox, img_size=(w, h))
    bboxes = [expend(bbox) for bbox in bboxes]

    faces = []
    for bbox in bboxes:
        bbox = bbox.astype(np.int32)

        # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        data = origin_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = {
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3],
            'data_length': np.int32(len(data.flatten()[0].tobytes()) * data.size),
            'rgb_data': data
        }
        faces.append(face)

    cv2.imwrite(save_path, img)
    data_path = Path('result/data') / Path(save_path).name
    data_path = data_path.with_suffix('.dat')

    # 验证字节数是否正确
    # data_length = 5 * 4  * len(faces)
    # for face in faces:
    #     data_length += face['data_length']
    # print(data_length)

    save_face(faces, save_file=str(data_path))


def save_face(faces: List[dict], save_file='test.dat'):
    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True)

    face_bytes = bytes()
    for face in faces:
        data = [value.tobytes() for value in face.values()]
        face_bytes += reduce(operator.add, data)

    with save_file.open('wb') as f:
        f.write(face_bytes)


def expend_bbox(bbox, img_size, expend_factor=0.6):
    bbox = bbox.copy()
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    img_w, img_h = img_size

    expend_w = bbox_w * expend_factor / 2
    expend_h = bbox_h * expend_factor / 2

    bbox[0] -= expend_w
    bbox[1] -= expend_h
    bbox[2] += expend_w
    bbox[3] += expend_h

    bbox[0] = bbox[0] if bbox[0] > 0 else 0
    bbox[1] = bbox[1] if bbox[1] > 0 else 0
    bbox[2] = bbox[2] if bbox[2] < img_w else img_w
    bbox[3] = bbox[3] if bbox[3] < img_h else img_h
    return bbox


# change to a folder with images, or another list containing image paths
IMAGES_PATH = '/home/clz/dataset/face/face'
SAVE_PATH = 'result/images_with_mask'

threshold = 0.9


def get_model():
    # load model begin.
    # model params
    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 600
    THREED_POINTS = './pose_references/reference_3d_68_points_trans.npy'
    POSE_MEAN = "models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "models/img2pose_v1.pth"

    threed_points = np.load(THREED_POINTS)
    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)

    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE,
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()
    # load model end.
    return img2pose_model


def save_result(img, poses, bboxes, save_name):
    save_path = Path(SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    save_file = save_path / save_name
    save_file = save_file.with_suffix('.jpg')
    render_plot(img, poses, bboxes, save_path=str(save_file))


transform = transforms.Compose([transforms.ToTensor()])


def predict_one(img_path, model):
    img = Image.open(img_path).convert("RGB")

    res = model.predict([transform(img)])[0]
    all_bboxes = res["boxes"].cpu().numpy().astype('float')

    poses = []
    bboxes = []
    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()
            poses.append(pose_pred)
            bboxes.append(bbox)
    return img, poses, bboxes


def predict():
    img2pose_model = get_model()

    imgs_path = Path(IMAGES_PATH)
    img_paths = [
        str(imgs_path / img_path)
        for img_path
        in imgs_path.iterdir()
    ]

    for img_path in tqdm(img_paths[:10]):
        img, poses, bboxes = predict_one(img_path, img2pose_model)

        image_name = Path(img_path).name
        save_result(img, poses, bboxes, image_name)


if __name__ == '__main__':
    predict()

import sys

sys.path.append('../')

import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from img2pose.img2pose import img2poseModel
from img2pose.model_loader import load_model
from pathlib import Path

np.set_printoptions(suppress=True)

# change to a folder with images, or another list containing image paths
IMAGES_PATH = '/home/clz/dataset/face/face'
SAVE_PATH = 'result/images_with_mask'

threshold = 0.8


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


if __name__ == '__main__':
    predict()

import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
from PIL import Image

import torch
import torchvision
from torch import nn
from preprocess import seed_everything, SigLipNetwork, OpenCLIPNetwork, OpenCLIPNetworkConfig

import pickle


try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"





if __name__ == '__main__':
    """very simple code to generate embeddings for all images in a folder. This is used as 5th feature level in the final model."""
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model', type=str, default="clip")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    model = args.model
    print("[ INFO ] Using model:", model)   

    img_folder = os.path.join(dataset_path, 'images')
    if not os.path.isdir(img_folder):
        img_folder = os.path.join(dataset_path, 'color')
    data_list = [f for f in sorted(os.listdir(img_folder))]
    data_list.sort()

    if model == "siglip":
        model = SigLipNetwork(device="cuda")
    elif model == "clip":
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        model.eval()
    else:
        assert False, "Unknown model"

    img_list = []
    for data_path in tqdm(data_list, desc="Loading images"):
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if model == "clip":
            image = cv2.resize(image, (224, 224))
        img_list.append(image)
       
    imgs = np.stack(img_list, axis=0)
    imgs = torch.from_numpy(imgs.astype("float16")).permute(0,3,1,2) / 255.0


    embeddings = []
    bsize=64
    for batch_i in tqdm(range(0, len(img_list), bsize), desc="Generating embeddigs"):
        batch = imgs[batch_i:batch_i+bsize].cuda()
        with torch.no_grad():
            embeddings.append(model.encode_image(batch).detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

    
    if model == "siglip":
        save_folder = os.path.join(dataset_path, 'full_image_embeddings_siglip')
    else:
        save_folder = os.path.join('/mnt/usb_ssd/opencity-data/', 'openscene-base/full_image_embeddings')
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, "embeddings.npy"), embeddings)
    # for i, e in tqdm(enumerate(embeddings), desc="Saving embeddings"):
    #     np.save(os.path.join(save_folder, f"{i}.npy"), e)

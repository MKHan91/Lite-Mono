import sys
import os
sys.path.append(os.getcwd())
import os.path as osp

import h5py
import cv2
import numpy as np
import torch
import skimage.transform
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import kitti_utils
from multiprocessing import Pool, cpu_count



def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def process_filename(args):
    filename, data_path, model_height, model_width, frame_idxs, side_map = args
    line = filename.split()
    directory = line[0]
    folderName = directory.split('/')[-1]
    
    if len(line) == 3:
        frame_index = int(line[1])
        side = line[2]
    else:
        frame_index = 0
        side = None

    results = {}

    # ─── DEPTH ─────────────────────
    velo_filename = osp.join(data_path, directory, f"velodyne_points/data/{int(frame_index):010d}.bin")
    is_depth = osp.isfile(velo_filename)
    if is_depth:
        calib_path = osp.join(data_path, directory.split("/")[0])
        depth_gt = kitti_utils.generate_depth_map(calib_path, velo_filename, side_map[side])
        depth_gt = skimage.transform.resize(depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant')
        depth_gt = depth_gt.astype(np.float32)

        key = f"{folderName}_velodyne_points_{side_map[side]}_{frame_index}"
        results[key] = depth_gt

    # ─── IMAGE ─────────────────────
    for i in frame_idxs:
        rgb_image_name = f"{frame_index+i:010d}.jpg"
        image_path = osp.join(data_path, directory, f"image_0{side_map[side]}/data", rgb_image_name)

        image = Image.open(image_path).convert("RGB")  # PIL 사용
        for num_scale in range(4):
            s = 2 ** num_scale
            resized_fn = transforms.Resize((model_height // s, model_width // s), interpolation=Image.Resampling.LANCZOS)
            resized_image = resized_fn(image)

            key = f"{folderName}_image_0{side_map[side]}_{frame_index}_{i}_{num_scale}"
            results[key] = np.array(resized_image, dtype=np.uint8)

    return results


def save_tensor_dict_to_hdf5(hdf5_path, train_filenames, data_path, model_height, model_width, frame_idxs, side_map):
    args_list = [(filename, data_path, model_height, model_width, frame_idxs, side_map) for filename in train_filenames]

    with h5py.File(hdf5_path, 'w') as f:
        group = f.create_group('dataset')

        with Pool(processes=cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(process_filename, args_list), total=len(train_filenames)):
                for key, value in result.items():
                    group.create_dataset(key, data=value, dtype=value.dtype, compression=None)
                    
                    
if __name__=="__main__":
    data_path = "/home/dev/Lite_Mono/datasets/kitti_data"
    fpath = f"/home/dev/Lite_Mono/splits/eigen_zhou/train_files.txt"
    frame_idxs = [0, -1, 1]
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    train_filenames = readlines(fpath)
    
    save_tensor_dict_to_hdf5(hdf5_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_tmp.hdf5",
                            train_filenames=train_filenames,
                            data_path=data_path,
                            model_height=192,
                            model_width=640,
                            frame_idxs=frame_idxs,
                            side_map=side_map)
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


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

        
def save_tensor_dict_to_hdf5(hdf5_path):
    data_path = "/home/dev/Lite_Mono/datasets/kitti_data"
    fpath = f"/home/dev/Lite_Mono/splits/eigen_zhou/train_files.txt"
    frame_idxs = [0, -1, 1]
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    train_filenames = readlines(fpath)
    model_height, model_width = 192, 640
    
    with h5py.File(hdf5_path, 'w') as f:
        group = f.create_group('dataset')
        for filename in tqdm(train_filenames):
            line = filename.split()
            directory = line[0]
            folderName = directory.split('/')[-1]
            if len(line) == 3:
                frame_index = int(line[1])
                side = line[2]
            else:
                frame_index = 0
                side = None


            velo_filename = osp.join(data_path, directory, f"velodyne_points/data/{int(frame_index):010d}.bin")
            is_depth = osp.isfile(velo_filename)
            if is_depth:
                calib_path = osp.join(data_path, directory.split("/")[0])
                depth_gt = kitti_utils.generate_depth_map(calib_path, velo_filename, side_map[side])
                depth_gt = skimage.transform.resize(depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant')
                depth_gt = depth_gt.astype(np.float32)
                
                group.create_dataset(f"{folderName}_velodyne_points_{side_map[side]}_{frame_index}", 
                                        data=depth_gt,
                                        dtype=np.float32,
                                        compression=None)
            
            for i in frame_idxs:
                rgb_image_name = f"{frame_index+i:010d}.jpg"
                
                image_path = osp.join(data_path, directory, f"image_0{side_map[side]}/data", rgb_image_name)
                image = Image.open(image_path)
                image = image.convert('RGB')


                for num_scale in range(4):
                    s = 2 ** num_scale
                    resized_fn = transforms.Resize((model_height // s, model_width // s), interpolation=Image.Resampling.LANCZOS)
                    resized_image = resized_fn(image)
                    
                    group.create_dataset(f"{folderName}_image_0{side_map[side]}_{frame_index}_{i}_{num_scale}", 
                                         data=np.array(resized_image),
                                         dtype=np.uint8,
                                         compression=None)

    print('done')


def save_lmdb(lmdb_path):
    import lmdb
    import os
    from PIL import Image
    import io

    data_path = "/home/dev/Lite_Mono/datasets/kitti_data"
    fpath = f"/home/dev/Lite_Mono/splits/eigen_zhou/train_files.txt"
    frame_idxs = [0, -1, 1]
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    train_filenames = readlines(fpath)
    model_height, model_width = 192, 640

    env = lmdb.open(lmdb_path, map_size=30 * 1024**3)  # 대략 1TB 공간 설정

    with env.begin(write=True) as txn:
        # for idx, img_name in enumerate(images):
        for filename in tqdm(train_filenames):
            line = filename.split()
            directory = line[0]
            folderName = directory.split('/')[-1]
            if len(line) == 3:
                frame_index = int(line[1])
                side = line[2]
            else:
                frame_index = 0
                side = None
            
            # === Depth 이미지 ===
            velo_filename = osp.join(data_path, directory, f"velodyne_points/data/{int(frame_index):010d}.bin")
            is_depth = osp.isfile(velo_filename)
            if is_depth:
                calib_path = osp.join(data_path, directory.split("/")[0])
                depth_gt = kitti_utils.generate_depth_map(calib_path, velo_filename, side_map[side])
                depth_gt = skimage.transform.resize(depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant')
                depth_gt = depth_gt.astype(np.float32)
                
                buf = io.BytesIO()
                np.save(buf, depth_gt)
                depth_bytes = buf.getvalue()
                
                lmdb_key = f"{folderName}_velodyne_points_{side_map[side]}_{frame_index}".encode("ascii")
                txn.put(lmdb_key, depth_bytes)
                
            # === RGB 이미지 === 
            for i in frame_idxs:
                rgb_image_name = f"{frame_index+i:010d}.jpg"
                
                image_path = osp.join(data_path, directory, f"image_0{side_map[side]}/data", rgb_image_name)
                image = Image.open(image_path)
                image = image.convert('RGB')

                for num_scale in range(4):
                    s = 2 ** num_scale
                    resized_image = image.resize((model_width // s, model_height // s), resample=Image.LANCZOS)
            
                    buf = io.BytesIO()
                    resized_image.save(buf, format='jpeg')
                    img_bytes = buf.getvalue()

                    # === LMDB key 생성 ===
                    lmdb_key = f"{folderName}_image_0{side_map[side]}_{frame_index}_{i}_{num_scale}".encode("ascii")
                    txn.put(lmdb_key, img_bytes)

    
if __name__=="__main__":
    save_lmdb(lmdb_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti.lmdb")
    # save_tensor_dict_to_hdf5(hdf5_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_tmp.hdf5")
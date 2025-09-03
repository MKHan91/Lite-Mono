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
import pickle


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


def save_lmdb(image_path, gt_path):
    import lmdb
    import os
    from PIL import Image
    import io

    data_path = "/home/dev/Lite_Mono/datasets/kitti_data"
    fpath = f"/home/dev/Lite_Mono/splits/eigen_zhou/train_files.txt"
    frame_idxs = [0, -1, 1]
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    
    train_filenames = readlines(fpath)
    train_filenames = train_filenames[:100]
    model_height, model_width = 192, 640

    env_image = lmdb.open(image_path, map_size=20 * 1024**3)  # 대략 1TB 공간 설정
    env_gt = lmdb.open(gt_path, map_size=20 * 1024**3)  # 대략 1TB 공간 설정


    gt_txn = env_gt.begin(write=True)
    image_txn = env_image.begin(write=True) 
    for idx, filename in enumerate(tqdm(train_filenames)):
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
        # is_depth = osp.isfile(velo_filename)

        calib_path = osp.join(data_path, directory.split("/")[0])
        depth_gt = kitti_utils.generate_depth_map(calib_path, velo_filename, side_map[side])
        depth_gt = skimage.transform.resize(depth_gt, (375, 1242), order=0, preserve_range=True, mode='constant')
        depth_gt = depth_gt.astype(np.float32)
        
        buf = io.BytesIO()
        np.save(buf, depth_gt)
        depth_bytes = buf.getvalue()
        
        lmdb_key = f"{folderName}_velodyne_points_{side_map[side]}_{frame_index}".encode("ascii")
        gt_txn.put(lmdb_key, depth_bytes)
        
        # === RGB 이미지 ===
        for i in frame_idxs:
            rgb_image_name = f"{frame_index+i:010d}.jpg"
            
            image_path = osp.join(data_path, directory, f"image_0{side_map[side]}/data", rgb_image_name)
            image = Image.open(image_path)
            image = image.convert('RGB')

            img_bytes_list = []
            img_lmdb_keys = []
            img_frame_idx = f"{folderName}_image_0{side_map[side]}_{frame_index}_{i}".encode('ascii')
            for num_scale in range(4):
                s = 2 ** num_scale
                resized_image = image.resize((model_width // s, model_height // s), resample=Image.LANCZOS)
        
                buf = io.BytesIO()
                resized_image.save(buf, format='jpeg')
                img_bytes = buf.getvalue()

                # === LMDB key 생성 ===
                lmdb_key = f"{folderName}_image_0{side_map[side]}_{frame_index}_{i}_{num_scale}".encode("ascii")
                img_lmdb_keys.append(lmdb_key)
                img_bytes_list.append(img_bytes)
                
            image_txn.put(img_frame_idx, pickle.dumps((img_lmdb_keys, img_bytes_list)))
    
        # === 일정 주기마다 commit ===
        if (idx + 1) % 1000 == 0:
            gt_txn.commit()
            image_txn.commit()
            gt_txn = env_gt.begin(write=True)
            image_txn = env_image.begin(write=True)

    image_txn.put(b'__len__', pickle.dumps(len(train_filenames)))
    # commit 확실히 해주기
    image_txn.commit()
    gt_txn.commit()
    
    env_image.sync()
    env_image.close()
    env_gt.sync()
    env_gt.close()
    
    
if __name__=="__main__":
    save_lmdb(image_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_image.lmdb",
              gt_path='/home/dev/Lite_Mono/datasets/kitti_data/kitti_gt.lmdb')
    # save_tensor_dict_to_hdf5(hdf5_path="/home/dev/Lite_Mono/datasets/kitti_data/kitti_tmp.hdf5")
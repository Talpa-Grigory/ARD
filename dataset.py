import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.distributed as dist

def load_data(
    *,
    all_files,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    inter = True,
    predx0 = False,
    all =False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # print(all_files)
    # all_files = _list_image_files_recursively(data_dir)
    # with open('all_files.pickle', 'wb') as f:
    #     # 데이터 저장
    #     pickle.dump(all_files, f)
    # assert 0


    classes = None
    if class_cond:
        classes = [int(path.split("/")[-2]) for path in all_files]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=dist.get_rank(),
        num_shards=dist.get_world_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        inter = inter,
        predx0 = predx0,
        all = all,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
    while True:
        yield from loader

# def _list_image_files_recursively(data_dir):
#     results = []
#     for entry in sorted(bf.listdir(data_dir)):
#         full_path = bf.join(data_dir, entry)
#         ext = entry.split(".")[-1]
#         if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npz"]:
#             results.append(full_path)
#         elif bf.isdir(full_path):
#             results.extend(_list_image_files_recursively(full_path))
#     return results



class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        inter = True,
        predx0 = False,
        all = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.inter = inter
        self.predx0 = predx0
        self.all = all
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        ## t=0
        path = self.local_images[idx]
        npz = np.load(path)
    
        ## t=25
        path_end = path.split("/")
        path_end[-3] = "25"
        path_end = '/'.join(path_end)
        npz_end = np.load(path_end)
        if self.all:
            if self.predx0 == False:
                path_5 = path.split("/")
                path_5[-3] = "5"
                path_5 = '/'.join(path_5)
                npz_5 = np.load(path_5)

                path_10 = path.split("/")
                path_10[-3] = "10"
                path_10 = '/'.join(path_10)
                npz_10 = np.load(path_10)

                path_15 = path.split("/")
                path_15[-3] = "15"
                path_15 = '/'.join(path_15)
                npz_15 = np.load(path_15)

                path_20 = path.split("/")
                path_20[-3] = "20"
                path_20 = '/'.join(path_20)
                npz_20 = np.load(path_20)


                return npz["zt"], npz_5["zt"], npz_10["zt"], npz_15["zt"], npz_20["zt"], npz_end["zt"], np.array(self.local_classes[idx], dtype=np.int64)
            
            else:
                path_5 = path.split("/")
                path_5[-3] = "5"
                path_5[-4] = "pred_z0"
                path_5 = '/'.join(path_5)
                npz_5 = np.load(path_5)

                path_10 = path.split("/")
                path_10[-3] = "10"
                path_10[-4] = "pred_z0"
                path_10 = '/'.join(path_10)
                npz_10 = np.load(path_10)

                path_15 = path.split("/")
                path_15[-3] = "15"
                path_15[-4] = "pred_z0"
                path_15 = '/'.join(path_15)
                npz_15 = np.load(path_15)

                path_20 = path.split("/")
                path_20[-3] = "20"
                path_20[-4] = "pred_z0"
                path_20 = '/'.join(path_20)
                npz_20 = np.load(path_20)
   
                return npz["zt"], npz_5["pred_z0"], npz_10["pred_z0"], npz_15["pred_z0"], npz_20["pred_z0"], npz_end["zt"], np.array(self.local_classes[idx], dtype=np.int64)


        elif self.inter:
            ## t=10
            path_inter = path.split("/")
            path_inter[-3] = "10"
            if self.predx0:
                path_inter[-4] = "pred_z0"
            path_inter = '/'.join(path_inter)
            npz_inter = np.load(path_inter)
            if self.predx0:
                return npz["zt"], npz_inter["pred_z0"], npz_end["zt"], np.array(self.local_classes[idx], dtype=np.int64)

            return npz["zt"], npz_inter["zt"], npz_end["zt"], np.array(self.local_classes[idx], dtype=np.int64)
        
        
        else:
            return npz["zt"], npz_end["zt"], np.array(self.local_classes[idx], dtype=np.int64)

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

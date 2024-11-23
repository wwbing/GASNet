#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
from io import UnsupportedOperation
import os
import os.path as osp
import random
import json
import time
import hashlib
from pathlib import Path

from multiprocessing.pool import Pool

import cv2
import numpy as np
from tqdm import tqdm
from PIL import ExifTags, Image, ImageOps

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .data_augment import (
    augment_hsv,
    letterbox,
    mixup,
    random_affine,
    mosaic_augmentation,
)

from yolov6.utils.events import LOGGER
import copy
import psutil
from multiprocessing.pool import ThreadPool


# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''
    def __init__(
        self,
        img_dir,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
        specific_shape = False,
        height=1088,
        width=1920,
        cache_ram=False
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        tik = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()    # Train  Val Test Speed
        self.class_names = data_dict["names"]
        self.img_paths, self.labels = self.get_imgs_labels(self.img_dir)        # 除了返回值，self.img_info还保存了 图像路径，shape，labels信息
        self.rect = rect
        self.specific_shape = specific_shape
        self.target_height = height
        self.target_width = width
        self.cache_ram = cache_ram

        # 设置每个批次的训练图像形状
        if self.rect:
            shapes = [self.img_info[p]["shape"] for p in self.img_paths]
            self.shapes = np.array(shapes, dtype=np.float64)

            if dist.is_initialized():
                # in DDP mode, we need to make sure all images within batch_size * gpu_num
                # will resized and padded to same shape.
                sample_batch_size = self.batch_size * dist.get_world_size()
            else:
                sample_batch_size = self.batch_size
            self.batch_indices = np.floor(
                np.arange(len(shapes)) / sample_batch_size
            ).astype(
                np.int_
            )  # batch indices of each image

            self.sort_files_shapes()

        # 对图像数据进行缓存
        if self.cache_ram:
            self.num_imgs = len(self.img_paths)
            self.imgs, self.imgs_hw0, self.imgs_hw = [None] * self.num_imgs, [None] * self.num_imgs, [None] * self.num_imgs
            self.cache_images(num_imgs=self.num_imgs)

        tok = time.time()

        if self.main_process:
            LOGGER.info(f"%.1fs for dataset initialization." % (tok - tik))
    

    # 缓存图像数据以加快训练过程中的数据读取速度
    def cache_images(self, num_imgs=None):
        assert num_imgs is not None, "num_imgs must be specified as the size of the dataset"

        mem = psutil.virtual_memory()
        mem_required = self.cal_cache_occupy(num_imgs)
        gb = 1 << 30

        if mem_required > mem.available:
            self.cache_ram = False
            LOGGER.warning("Not enough RAM to cache images, caching is disabled.")
        
        else:
            LOGGER.warning(
                f"{mem_required / gb:.1f}GB RAM required, "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB RAM available, "
                f"Since the first thing we do is cache, "
                f"there is no guarantee that the remaining memory space is sufficient"
            )

        print(f"self.imgs: {len(self.imgs)}")
        LOGGER.info("You are using cached images in RAM to accelerate training!")
        LOGGER.info(
            "Caching images...\n"
            "This might take some time for your dataset"
        )
        num_threads = min(16, max(1, os.cpu_count() - 1))
        load_imgs = ThreadPool(num_threads).imap(self.load_image, range(num_imgs))  # 使用线程池ThreadPool并行地对图像索引进行迭代，同时将load_image方法应用于每个图像索引
        pbar = tqdm(enumerate(load_imgs), total=num_imgs, disable=self.rank > 0)    # 使用tqdm创建一个可视化的进度条来显示图像加载的进度
        
        # 将加载的图像数据（x）、高度和宽度（h0、w0）、以及图像的形状（shape）分别存储在self.imgs、self.imgs_hw0和self.imgs_hw这三个列表中的对应位置
        for i, (x, (h0, w0), shape) in pbar:
            self.imgs[i], self.imgs_hw0[i], self.imgs_hw[i] = x, (h0, w0), shape

    def __del__(self):
        if self.cache_ram:
            del self.imgs
    
    # 计算需要占用的内存空间
    def cal_cache_occupy(self, num_imgs):
        '''estimate the memory required to cache images in RAM.
        '''
        cache_bytes = 0
        num_imgs = len(self.img_paths)
        num_samples = min(num_imgs, 32)

        for _ in range(num_samples):
            img, _, _ = self.load_image(index=random.randint(0, len(self.img_paths) - 1))
            cache_bytes += img.nbytes

        mem_required = cache_bytes * num_imgs / num_samples
        return mem_required

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)

    # 根据索引获取图像数据样本，并根据是否启用数据增强来进行图像处理、尺寸调整和标签数据转换
    # 返回tensor张量类型的图片、标签。和图像路径、用于计算coco指标的宽高信息
    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        target_shape = (
                (self.target_height, self.target_width) if self.specific_shape else
                self.batch_shapes[self.batch_indices[index]] if self.rect
                else self.img_size
                )

        # 如果启用马赛克增强并且满足随机概率
        if self.augment and random.random() < self.hyp["mosaic"]:
            img, labels = self.get_mosaic(index, target_shape)
            shapes = None

            # 满足混合概率的条件
            if random.random() < self.hyp["mixup"]:
                img_other, labels_other = self.get_mosaic(
                    random.randint(0, len(self.img_paths) - 1), target_shape
                )

                img, labels = mixup(img, labels, img_other, labels_other)

        else:
            # Load image
            if self.hyp and "shrink_size" in self.hyp:
                img, (h0, w0), (h, w) = self.load_image(index, self.hyp["shrink_size"])
            else:
                img, (h0, w0), (h, w) = self.load_image(index)

            # 调整图像尺寸和填充
            img, ratio, pad = letterbox(img, target_shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h * ratio / h0, w * ratio / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()

            # 把yolo格式的坐标转换成需要回归的坐标，左上，右下:以便之后仿射变换
            if labels.size:
                w *= ratio
                h *= ratio
                # new boxes
                boxes = np.copy(labels[:, 1:])
                boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # top left x
                boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # top left y
                boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # bottom right x
                boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # bottom right y
                labels[:, 1:] = boxes

            if self.augment:
                # 对图片进行随机放射变换
                img, labels = random_affine(
                    img,
                    labels,
                    degrees=self.hyp["degrees"],
                    translate=self.hyp["translate"],
                    scale=self.hyp["scale"],
                    shear=self.hyp["shear"],
                    new_shape=target_shape,
                )

        # 又把左上，右下的坐标转换为中心点和宽高，并归一化
        if len(labels):
            h, w = img.shape[:2]

            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, w - 1e-3)  # x1, x2
            labels[:, [2, 4]] = labels[:, [2, 4]].clip(0, h - 1e-3)  # y1, y2

            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = ((labels[:, 1] + labels[:, 3]) / 2) / w  # x center
            boxes[:, 1] = ((labels[:, 2] + labels[:, 4]) / 2) / h  # y center
            boxes[:, 2] = (labels[:, 3] - labels[:, 1]) / w  # width
            boxes[:, 3] = (labels[:, 4] - labels[:, 2]) / h  # height
            labels[:, 1:] = boxes

        if self.augment:
            img, labels = self.general_augment(img, labels)

        labels_out = torch.zeros((len(labels), 6))

        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)  # 将标签数据填充到张量中

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_paths[index], shapes
    
    # 加载图像，返回 图像，图像的原始宽高， 图像resize之后的宽高
    def load_image(self, index, shrink_size=None):
        """Load image.
        This function loads image by cv2, resize original image to target shape(img_size) with keeping ratio.

        Returns:
            Image, original shape of image, resized image shape
        """
        path = self.img_paths[index]
        
        if self.cache_ram and self.imgs[index] is not None:
            im = self.imgs[index]
            # im = copy.deepcopy(im)
            return self.imgs[index], self.imgs_hw0[index], self.imgs_hw[index]
        
        else:
            try:
                im = cv2.imread(path)
                assert im is not None, f"opencv cannot read image correctly or {path} not exists"
            except Exception as e:
                print(e)
                im = cv2.cvtColor(np.asarray(Image.open(path)), cv2.COLOR_RGB2BGR)
                assert im is not None, f"Image Not Found {path}, workdir: {os.getcwd()}"

            h0, w0 = im.shape[:2]  # origin shape
            if self.specific_shape:
                # keep ratio resize
                ratio = min(self.target_width / w0, self.target_height / h0)

            elif shrink_size:
                ratio = (self.img_size - shrink_size) / max(h0, w0)

            else:
                ratio = self.img_size / max(h0, w0)
            if ratio != 1:
                    im = cv2.resize(
                        im,
                        (int(w0 * ratio), int(h0 * ratio)),
                        interpolation=cv2.INTER_AREA
                        if ratio < 1 and not self.augment
                        else cv2.INTER_LINEAR,
                    )
            return im, (h0, w0), im.shape[:2]
    

    # 将输入的样本列表进行解压缩，处理标签数据，然后合并成一个小批量的tensor的张量数据
    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    
    # 取得img_paths, labels ，元组类型
    def get_imgs_labels(self, img_dirs):
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]
        # we store the cache img file in the first directory of img_dirs
        valid_img_record = osp.join(
            osp.dirname(img_dirs[0]), "." + osp.basename(img_dirs[0]) + "_cache.json"
        )
        # 
        NUM_THREADS = min(8, os.cpu_count())
        img_paths = []
        
        # 得到每一张图片的路径
        for img_dir in img_dirs:
            assert osp.exists(img_dir), f"{img_dir} is an invalid directory path!"
            img_paths += glob.glob(osp.join(img_dir, "**/*"), recursive=True)

        # 对图片路径排序
        img_paths = sorted(
            p for p in img_paths if p.split(".")[-1].lower() in IMG_FORMATS and os.path.isfile(p)
        )

        assert img_paths, f"No images found in {img_dir}."
        
        # 得到图片路径组成的列表的哈希值
        img_hash = self.get_hash(img_paths)
        LOGGER.info(f'img record infomation path is:{valid_img_record}')

        if osp.exists(valid_img_record):
            with open(valid_img_record, "r") as f:
                cache_info = json.load(f)
                if "image_hash" in cache_info and cache_info["image_hash"] == img_hash:

                    # 得到图片信息字典，图片路径是键，shape labels的图片信息是值
                    img_info = cache_info["information"]
                else:
                    self.check_images = True
        else:
            self.check_images = True

        # check images
        if self.check_images and self.main_process:
            img_info = {}
            nc, msgs = 0, []  # number corrupt, messages

            LOGGER.info(
                f"{self.task}: Checking formats of images with {NUM_THREADS} process(es): "
            )
            # Pool 创建一个具有NUM_THREADS个进程的进程池
            with Pool(NUM_THREADS) as pool:
                # 使用 tqdm 创建了一个可视化进度条，同时使用 pool.imap 方法并行地调用 TrainValDataset.check_image 函数来检查图像文件。
                # 这里的 check_image 函数似乎是用于检查图像文件是否损坏，和修复
                pbar = tqdm(
                    pool.imap(TrainValDataset.check_image, img_paths),
                    total=len(img_paths),
                )
                for img_path, shape_per_img, nc_per_img, msg in pbar:       # nc_per_img: 打不开的图像，损坏无法修复 msg:损坏但是修复的图像
                    if nc_per_img == 0:  # not corrupted
                        img_info[img_path] = {"shape": shape_per_img}
                    nc += nc_per_img
                    if msg:
                        msgs.append(msg)
                    pbar.desc = f"{nc} image(s) corrupted"
            pbar.close()

            if msgs:
                LOGGER.info("\n".join(msgs))

            cache_info = {"information": img_info, "image_hash": img_hash}
            # save valid image paths.
            with open(valid_img_record, "w") as f:       #把图片信息写入json文件
                json.dump(cache_info, f)

        # check and load anns

        img_paths = list(img_info.keys())  # 取出img_info的键,即图片路径
        label_paths = img2label_paths(img_paths) # 根据图片路径得到标签路径
        assert label_paths, f"No labels found."
        label_hash = self.get_hash(label_paths)

        if "label_hash" not in cache_info or cache_info["label_hash"] != label_hash:
            self.check_labels = True

        if self.check_labels:
            cache_info["label_hash"] = label_hash
            nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number corrupt, messages
            LOGGER.info(
                f"{self.task}: Checking formats of labels with {NUM_THREADS} process(es): "
            )

            with Pool(NUM_THREADS) as pool:
                pbar = pool.imap(
                    TrainValDataset.check_label_files, zip(img_paths, label_paths)
                )
                pbar = tqdm(pbar, total=len(label_paths)) if self.main_process else pbar
                for (
                    img_path,           # img_path
                    labels_per_file,    # label
                    nc_per_file,        # num_corrupted_label
                    nm_per_file,        # num_missing_label
                    nf_per_file,        # num_found_label
                    ne_per_file,        # num_empty_label
                    msg,                # message
                ) in pbar:
                    if nc_per_file == 0:
                        img_info[img_path]["labels"] = labels_per_file
                    else:
                        img_info.pop(img_path)
                    nc += nc_per_file
                    nm += nm_per_file
                    nf += nf_per_file
                    ne += ne_per_file
                    if msg:
                        msgs.append(msg)
                    if self.main_process:
                        pbar.desc = f"{nf} label(s) found, {nm} label(s) missing, {ne} label(s) empty, {nc} invalid label files"
            if self.main_process:
                pbar.close()
                with open(valid_img_record, "w") as f:   #把label信息写入json文件
                    json.dump(cache_info, f)
            if msgs:
                LOGGER.info("\n".join(msgs))
            if nf == 0:
                LOGGER.warning(
                    f"WARNING: No labels found in {osp.dirname(img_paths[0])}. "
                )

        # 如果是验证，需要给出json文件，如果不是，就自动生成json，保存在annotations下的文件夹json文件中
        if self.task.lower() == "val":
            if self.data_dict.get("is_coco", False): # use original json file when evaluating on coco dataset.
                assert osp.exists(self.data_dict["anno_path"]), "Eval on coco dataset must provide valid path of the annotation file in config file: data/coco.yaml"
            else:
                assert (
                    self.class_names
                ), "Class names is required when converting labels to coco format for evaluating."
                save_dir = osp.join(osp.dirname(osp.dirname(img_dirs[0])), "annotations")
                if not osp.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = osp.join(
                    save_dir, "instances_" + osp.basename(img_dirs[0]) + ".json"
                )

                TrainValDataset.generate_coco_format_labels(
                    img_info, self.class_names, save_path
                )

        # img_info中img_paths和labels转换为列表格式 [img_paths, labels],单独的img_paths和labels是元组格式
        img_paths, labels = list(
            zip(
                *[
                    (
                        img_path,
                        np.array(info["labels"], dtype=np.float32)
                        if info["labels"]
                        else np.zeros((0, 5), dtype=np.float32),
                    )
                    for img_path, info in img_info.items()
                ]
            )
        )
    
        self.img_info = img_info
        LOGGER.info(
            f"{self.task}: Final numbers of valid images: {len(img_paths)}/ labels: {len(labels)}. "
        )

        # 返回元组类型的img_paths和labels
        return img_paths, labels

    # 获取图像和标签数据，然后通过引入其他图像数据，进行马赛克增强，返回增强后的图
    def get_mosaic(self, index, shape):
        """Gets images and labels after mosaic augments"""
        indices = [index] + random.choices(
            range(0, len(self.img_paths)), k=3
        )                                                       # 随机从整个数据集中选三张其他的图片
        random.shuffle(indices)

        imgs, hs, ws, labels = [], [], [], []
        for index in indices:
            img, _, (h, w) = self.load_image(index)
            labels_per_img = self.labels[index]
            imgs.append(img)
            hs.append(h)
            ws.append(w)
            labels.append(labels_per_img)
        img, labels = mosaic_augmentation(shape, imgs, hs, ws, labels, self.hyp, self.specific_shape, self.target_height, self.target_width)
        return img, labels

    # 进行颜色空间增强和随机翻转等普通增强
    def general_augment(self, img, labels):
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space 颜色空间增强
        augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        return img, labels

    # 对图片进行排序，按照宽高比进行排序，并设置每个批次的形状
    def sort_files_shapes(self):
        '''Sort by aspect ratio.'''
        batch_num = self.batch_indices[-1] + 1
        s = self.shapes                                             # [height, width]
        ar = s[:, 1] / s[:, 0]                                      # aspect ratio 计算图片的宽高比
        irect = ar.argsort()                                        # 对宽高比进行排序，得到排序后的索引。
        self.img_paths = [self.img_paths[i] for i in irect]         # 根据排序后的索引对图片的路径进行重新排序。
        self.labels = [self.labels[i] for i in irect]               # 根据排序后的索引对图片的标签进行重新排序。
        self.shapes = s[irect]  # wh
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * batch_num

        # 计算每个批次图片宽高比的最大值和最小值
        for i in range(batch_num):
            ari = ar[self.batch_indices == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [1, maxi]
            elif mini > 1:
                shapes[i] = [1 / mini, 1]
        
        # 设置每个批次的形状
        self.batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(
                np.int_
            )
            * self.stride
        )

    # 检查图片是否损坏，修复损坏的图片，并返回im_file，num_corrupted_image，msg
    @staticmethod
    def check_image(im_file):
        '''Verify an image.'''
        nc, msg = 0, ""
        try:
            im = Image.open(im_file)
            im.verify()                                     # PIL verify验证图像完整性
            im = Image.open(im_file)                        # need to reload the image after using verify()
            shape = (im.height, im.width)                   # (height, width)

            try:                                            
                im_exif = im._getexif()                     # 尝试获取图像的 EXIF 信息，包括方向信息（orientation
                if im_exif and ORIENTATION in im_exif:      # 根据方向信息修正图像的尺寸
                    rotation = im_exif[ORIENTATION]
                    if rotation in (6, 8):
                        shape = (shape[1], shape[0])
            except:
                im_exif = None

            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels" # 检查是否都大于9像素
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"

            if im.format.lower() in ("jpg", "jpeg"):
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)                           # 移动文件指针到距离文件末尾倒数第二个字节的位置
                    if f.read() != b"\xff\xd9":             # 读取文件指针位置开始的两个字节，如果不等于 b"\xff\xd9"，则表示发现了损坏的JPEG文件。
                        ImageOps.exif_transpose(Image.open(im_file)).save(      # 修复图片
                            im_file, "JPEG", subsampling=0, quality=100
                        )
                        msg += f"WARNING: {im_file}: corrupt JPEG restored and saved"
            return im_file, shape, nc, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {im_file}: ignoring corrupt image: {e}"
            return im_file, None, nc, msg
    

    # 检查标签文件是否有效，并返回img_path，labels，num_corrupted_label，num_missing_label，num_found_label，num_empty_label，msg
    @staticmethod
    def check_label_files(args):
        img_path, lb_path = args
        nm, nf, ne, nc, msg = 0, 0, 0, 0, ""  # number (missing, found, empty, message
        try:
            if osp.exists(lb_path):
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    labels = [
                        x.split() for x in f.read().strip().splitlines() if len(x) # 将每一行的内容按空格分割后，去除空行，并将结果存储在一个列表中
                    ]
                    labels = np.array(labels, dtype=np.float32)
                if len(labels):
                    assert all(
                        len(l) == 5 for l in labels
                    ), f"{lb_path}: wrong label format."
                    assert (
                        labels >= 0
                    ).all(), f"{lb_path}: Label values error: all values in label file must > 0"
                    assert (
                        labels[:, 1:] <= 1                      # 对标签数据中的除第一列外的所有元素进行逐个判断，检查它们是否小于等于1
                    ).all(), f"{lb_path}: Label values error: all coordinates must be normalized"

                    _, indices = np.unique(labels, axis=0, return_index=True) # 获得重复标签的索引
                    if len(indices) < len(labels):  # duplicate row check
                        labels = labels[indices]  # 去除重复的行
                        msg += f"WARNING: {lb_path}: {len(labels) - len(indices)} duplicate labels removed"
                    labels = labels.tolist()
                else:
                    ne = 1  # label empty
                    labels = []
            else:
                nm = 1  # label missing
                labels = []

            return img_path, labels, nc, nm, nf, ne, msg
        except Exception as e:
            nc = 1
            msg = f"WARNING: {lb_path}: ignoring invalid labels: {e}"
            return img_path, None, nc, nm, nf, ne, msg

    # 生成coco格式的标签文件
    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_h, img_w = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]                   # yolo格式的bbox是x y w h ，其中X,Y是中心点的坐标，归一化--> (0,1)
                                                                # coco格式的bbox是x y w h ，其中X,Y是左上角的坐标，没有归一化
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],  # 左上角坐标和宽高
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    # 获取路径的哈希值
    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()


class LoadData:
    def __init__(self, path, webcam, webcam_addr):
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        if webcam: # if use web camera
            imgp = []
            vidp = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path
            if os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            else:
                raise FileNotFoundError(f'Invalid path {p}')
            imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
            vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
        self.files = imgp + vidp
        self.nf = len(self.files)
        self.type = 'image'
        if len(vidp) > 0:
            self.add_video(vidp[0])  # new video
        else:
            self.cap = None

    # @staticmethod
    def checkext(self, path):
        if self.webcam:
            file_type = 'video'
        else:
            file_type = 'image' if path.split('.')[-1].lower() in IMG_FORMATS else 'video'
        return file_type

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.checkext(path) == 'video':
            self.type = 'video'
            ret_val, img = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.add_video(path)
                ret_val, img = self.cap.read()
        else:
            # Read image
            self.count += 1
            img = cv2.imread(path)  # BGR
        return img, path, self.cap

    def add_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

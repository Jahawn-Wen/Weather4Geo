"""Dataset subset used by the archived 75.85 recipe; historical label behavior retained."""
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import random

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset_selectID(dir, class_to_idx, extensions):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images[class_to_idx[target]].append(item)
    return images


class ImageFolder_iaa_selectID(Data.Dataset):
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, iaa_transform = None, norm='bn'):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs.keys()) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.norm = norm
    def __getitem__(self, index):
            """
            index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
            """
            path, target = random.choice(self.imgs[index])
            img = self.loader(path)
            if self.iaa_trans is not None:
                img = np.array(img)
                img = self.iaa_trans(image = img)
                # img_aug = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.norm == 'ada-ibn' or self.norm == 'spade':
                return img, target, 1
            else:
                return img, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_multi_weather(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader,
                 iaa_transform=None, iaa_weather_list=[], batchsize=8, shuffle=False,
                 norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if select:
            imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        else:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))
        
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batch = batchsize
        self.img_num = 0
        self.shuffle = shuffle
        self.norm = norm
        self.select = select
        # 注意：不在这里创建临时保存路径，caption匹配只需要全局索引

    def __getitem__(self, index):
        """
        index (int): 全局索引，对应于self.imgs中的样本顺序
        返回：
            如果 norm 为 'ada-ibn' 或 'spade'：
                (img, target, weather_index, global_index)
            否则：
                (img, target, global_index)
        """
        if self.select:
            path, target = random.choice(self.imgs[index])
        else:
            path, target = self.imgs[index]
        img = self.loader(path)

        # 天气增强处理：注意这里我们保留原有的计算方法，不做修改
        if self.iaa_weather_list:
            img = np.array(img)
            if self.shuffle:
                weather_idx = random.choice(range(len(self.iaa_weather_list)))
                if weather_idx == 0:
                    img = img
                else:
                    img = self.iaa_weather_list[weather_idx](image=img)
            else:
                weather_idx = self.img_num // self.batch % (len(self.iaa_weather_list) + 1)
                if weather_idx == 0:
                    img = img
                else:
                    img = self.iaa_weather_list[weather_idx - 1](image=img)
                self.img_num += 1
                if self.img_num == len(self):
                    self.img_num = 0
        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # 此处不改变原有的天气索引返回；如果有天气增强，则返回第三个值为天气索引（原代码使用 idx+1，这里用 weather_idx+1）
        if self.norm == 'ada-ibn' or self.norm == 'spade':
            # 如果有天气增强，返回计算得到的天气索引；否则返回 None
            weather_index = (weather_idx + 1) if self.iaa_weather_list else None
            return img, target, weather_index, index  # 新增全局索引 index
        else:
            return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageFolder_iaa_multi_weather_single(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, iaa_transform=None, iaa_weather_list=None, batchsize=8, shuffle=False, norm='bn', select=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if select:
            imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        else:
            imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.iaa_trans = iaa_transform
        self.iaa_weather_list = iaa_weather_list
        self.batch = batchsize
        self.img_num = 0
        self.shuffle = shuffle
        self.norm = norm
        self.select = select

    def __getitem__(self, index):
        """
        index (int): 全局索引，对应于self.imgs中的样本顺序
        返回：
            如果 norm 为 'ada-ibn' 或 'spade'：
                (img, target, weather_index, global_index)
            否则：
                (img, target, global_index)
        """
        if self.select:
            path, target = random.choice(self.imgs[index])
        else:
            path, target = self.imgs[index]

        img = self.loader(path)
        img = np.array(img)

        idx = 0
        if self.iaa_weather_list:
            if self.shuffle:
                # 如果 iaa_weather_list 为单个增强器:
                if callable(self.iaa_weather_list):
                    img = self.iaa_weather_list(image=img)
                else:
                    # 如果 iaa_weather_list 为列表:
                    idx = random.randint(0, len(self.iaa_weather_list)-1)
                    img = self.iaa_weather_list[idx](image=img)
            else:
                idx = self.img_num // self.batch % (len(self.iaa_weather_list)+1)
                if idx != 0:
                    img = self.iaa_weather_list[idx-1](image=img)

                self.img_num += 1
                if self.img_num == len(self):
                    self.img_num = 0

        if self.iaa_trans is not None:
            img = self.iaa_trans(image=img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.norm in ['ada-ibn', 'spade']:
            return img, target, idx+1, index  # 明确返回索引
        else:
            return img, target, index


    def __len__(self):
        return len(self.imgs)


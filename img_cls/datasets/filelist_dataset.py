import os
import json
import numpy as np
from utils import pil_loader
from torch.utils.data import Dataset


def parse_json(fn):
    assert os.path.exists(fn)
    with open(fn) as f:
        data = json.load(f)
    return data


def build_dataset_furn(filelist, prefix):
    img_lst = []
    lb_lst = []
    lst = parse_json(filelist)
    lb_max = [-1 for _ in range(len(lst[0]) - 1)]
    for l in lst:
        img_lst.append(os.path.join(prefix, l[0]))
        lbs = []
        for i, lb in enumerate(l[1:]):
            lb = int(lb)
            lb_max[i] = max(lb_max[i], lb)
            lbs.append(lb)
        lb_lst.append(lbs)
    assert len(img_lst) == len(lb_lst)
    return img_lst, lb_lst, [n+1 for n in lb_max]

def build_dataset_onelist(filelist, prefix):
    img_lst = []
    lb_lst = []
    with open(filelist) as f:
        data = [i.strip().split() for i in f.readlines()]
    img_lst = [os.path.join(prefix, i[0]) for i in data]
    lb_lst = [int(i[1]) for i in data]
    return img_lst, lb_lst

def build_dataset(filelist, prefix):
    if type(filelist) == str:
        return build_dataset_onelist(filelist, prefix)
    assert(type(filelist) == list)
    img_lst = []
    lb_lst = []
    for fl_one in filelist:
        img_lst_one, lb_lst_one = build_dataset_onelist(fl_one, prefix)
        img_lst += img_lst_one
        lb_lst += lb_lst_one
    return img_lst, lb_lst


class FileListDataset(Dataset):
    def __init__(self, filelist, prefix, transform=None):
        self.img_lst, self.lb_lst \
            = build_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        idx %= self.num
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            img = pil_loader(open(fn, 'rb').read())
            return img, lb
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        img, lb = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb

class FileListDatasetName(Dataset):
    def __init__(self, filelist, prefix, transform=None):
        self.img_lst, self.lb_lst \
            = build_dataset(filelist, prefix)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        idx %= self.num
        fn = self.img_lst[idx]
        lb = self.lb_lst[idx]
        try:
            img = pil_loader(open(fn, 'rb').read())
            return img, lb, fn
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        img, lb, fn = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, lb, fn

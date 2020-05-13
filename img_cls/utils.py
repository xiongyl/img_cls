import os
import io
import json
import numpy as np
import multiprocessing as mp

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.distributed as dist


def parse_json(fn):
    assert os.path.exists(fn)
    with open(fn) as f:
        data = json.load(f)
    return data


def dump2json(ofn, data):
    with open(ofn, 'w') as of:
        json.dump(data, of)


def init_processes(addr, port, gpu_num, backend):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(rank, size)
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    torch.cuda.set_device(rank % gpu_num)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend)
    print('initialize {} successfully (rank {})'.format(backend, rank))
    return rank, size


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history)


def accuracy(output, target, topk=(1,), cls2tcls=None):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        if cls2tcls is not None:
            for i in range(len(target)):
                pred[0, i] = cls2tcls[int(pred[0, i])]
                target[i] = cls2tcls[int(target[i])]

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred


def save_ckpt(state, ckpt, epoch, is_best):
    print(ckpt)
    folder = os.path.dirname(ckpt)
    print(folder)
    fn = '{}_epoch_{}.pth.tar'.format(os.path.basename(ckpt), epoch)
    if folder != ''and not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, fn)
    print('saving to {}'.format(path))
    torch.save(state, '{}'.format(path))
    if is_best:
        best_fn = os.path.join(folder, 'model_best.pth.tar')
        if os.path.exists(best_fn):
            os.unlink(best_fn)
        os.symlink(fn, best_fn)


def load_ckpt(path, model, ignores=[], strict=True, optimizer=None, prefix='module.'):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if 'state_dict' not in checkpoint:
            m = {}
            m['state_dict'] = {}
            for k, v in checkpoint.items():
                if not k.startswith(prefix):
                    m['state_dict'][prefix+k] = v
                else:
                    m['state_dict'][k] = v
            checkpoint = m
            strict = False
        if len(ignores) > 0:
            assert optimizer == None
            keys = list(checkpoint['state_dict'].keys())
            for ignore in ignores:
                if ignore in keys:
                    print('ignoring {}'.format(ignore))
                    del checkpoint['state_dict'][ignore]
                else:
                    raise ValueError('cannot find {} in load_path'.format(ignore))
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if not strict:
            pretrained_keys = set(checkpoint['state_dict'].keys())
            model_keys = set([k for k, _ in model.named_parameters()])
            for k in model_keys - pretrained_keys:
                print('warning: {} not loaded'.format(k))
        if optimizer != None:
            assert len(ignores) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['epoch']))
            return checkpoint['epoch'], checkpoint['best_prec1']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def normalize(feat, axis=1):
    if axis == 0:
        return feat / np.linalg.norm(feat, axis=0)
    elif axis == 1:
        return feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
        return img


def save_imgs(imgs, ofolder):
    '''save pil image array to JPEG image file
    '''
    for i, img in enumerate(imgs):
        opath = os.path.join(ofolder, "{}.jpg".format(i))
        if not os.path.exists(os.path.dirname(opath)):
            print(opath)
            os.makedirs(os.path.dirname(opath))
        img.save(opath, "JPEG")
    else:
        raise TypeError('axis value should be 0 or 1(cannot handel axis {})'.format(axis))


def mkdir_if_no_exist(path, subdirs=['']):
    if path == '':
        return
    for sd in subdirs:
        d = os.path.dirname(os.path.join(path, sd))
        if not os.path.exists(d):
            os.makedirs(d)


def parse_json(fn):
    assert os.path.exists(fn)
    with open(fn) as f:
        data = json.load(f)
    return data


def get_cat2cls(fn='/mnt/SSD/ihome/data/annotation/furniture_category.json'):
    data = parse_json(fn)
    tcats = []
    cat2cls = {}
    cls2tcls = {}
    cls = 0
    for top_cls, (k, v) in enumerate(data.items()):
        tcats.append(k)
        for kv in v:
            assert kv not in cat2cls
            cat2cls[kv] = cls
            cls2tcls[cls] = top_cls
            cls += 1
    print('#cls: {}, #top_cls: {}'.format(cls, top_cls + 1))
    return tcats, cat2cls, cls2tcls

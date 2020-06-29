from collections.abc import Iterable
from typing import Sequence

import torch
import numpy as np


class MemoryBase:

    def __init__(self, lenght, device='cuda', train=False, train_func=None, np_store=True, *args, **kwargs):
        # self.len_negative = 5
        self.K = lenght
        self.train = train
        self.train_func = train_func
        self.numpy_store = np_store

        self.templates = None
        self.bb = None
        self.weights = None
        self.imgs = [None] * self.K if self.numpy_store else None

        self.temp = None
        self.temp_bb = None
        self.temp_w = None
        self.temp_img = []

        self.negative = None
        self.negative_bb = None
        self.negative_w = None
        self.device = torch.device(device)

        self.lr = 0.01

    def update(self, features: torch.Tensor, bb: torch.Tensor, weight):
        raise NotImplementedError()

    @property
    def images(self):
        return [i for i in self.imgs if i is not None]

    @property
    def num_templates(self):
        return self.templates.shape[0]

    @property
    def temp_set(self):
        return self.temp, self.temp_bb, self.temp_w

    def send_to_device(self) -> None:
        tensors = [
            'templates',
            'bb',
            'weights',
            'temp',
            'temp_bb',
            'temp_w',
        ]
        for name in tensors:
            t = getattr(self, name)
            if t is not None:
                if t.device != self.device:
                    setattr(self, name, t.to(self.device))

    @property
    def augmented_template(self) -> Sequence[torch.Tensor]:
        if self.temp is not None:
            return (torch.cat((self.templates, self.temp)),
                    torch.cat((self.bb, self.temp_bb)),
                    torch.cat((self.weights, self.temp_w)))
        return self.templates, self.bb, self.weights

    @property
    def augmented_template_n(self) -> Sequence[torch.Tensor]:
        tmlt, bb, w = self.templates, self.bb, self.weights
        if self.temp is not None:
            tmlt, bb, w = (torch.cat((tmlt, self.temp)),
                           torch.cat((bb, self.temp_bb)),
                           torch.cat((w, self.temp_w)))
        if self.negative is not None:
            # tmlt, bb, w = (torch.cat((tmlt, self.negative)),
            #                torch.cat((bb, self.negative_bb)),
            #                torch.cat((w, self.negative_w)))
            tmlt, bb, w = (torch.cat((tmlt, self.negative)),
                           torch.cat((bb, self.negative_bb)),
                           torch.cat((w, self.negative_w)))

        return tmlt, bb, w

    def add_temp(self, tmp: torch.Tensor, bb: torch.Tensor, w=1, img=None) -> None:
        bb = bb.to(self.device)
        if len(bb.shape) == 1:
            bb = bb.unsqueeze(0)
        w = torch.Tensor([w]).to(self.device)
        if len(w) == 1 and len(tmp) != 1:
            w = w.repeat(len(tmp))
        if self.temp is None:
            self.temp = tmp
            self.temp_bb = bb
            self.temp_w = w
        else:
            self.temp = torch.cat((self.temp, tmp))
            self.temp_bb = torch.cat((self.temp_bb, bb))
            self.temp_w = torch.cat((self.temp_w, w))
        self.send_to_device()
        if img is not None:
            self.temp_img.append(img.cpu().numpy())

    def flush_temp(self) -> None:
        self.temp = None
        self.temp_bb = None
        self.temp_w = []

    def push(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, imgs=None) -> None:
        bb = bb.to(self.device)
        if isinstance(w, int):
            w = torch.Tensor([w]).to(self.device)
        if len(w) == 1 and len(feat) != 1:
            w = w.repeat(len(feat))

        if len(bb.shape) == 1:
            bb = bb.unsqueeze(0)
        if self.templates is None:
            self.templates = feat
            self.bb = bb
            self.weights = w
        else:
            self.templates = torch.cat((self.templates, feat))
            self.bb = torch.cat((self.bb, bb))
            self.weights = torch.cat((self.weights, w))

        if imgs is not None:
            if self.numpy_store:
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.cpu().numpy()

                list_n = [a is None for a in self.imgs]
                if True in list_n:
                    self.imgs[list_n.index(True)] = imgs
                else:
                    self.imgs.append(imgs)
            else:
                if self.imgs is None:
                    self.imgs = imgs
                else:
                    self.imgs = torch.cat((self.imgs, imgs))

    def flush_temp(self):
        self.temp, self.temp_bb, self.temp_w = [None] * 3
        self.temp_img = []

    def after_classifier(self, target_filter: torch.Tensor, templates: torch.Tensor) -> bool:
        pass

    def append_negative(self, feat: torch.Tensor, bb: torch.Tensor, w: int = -1):
        bb = bb.to(self.device)
        if len(bb.shape) == 1:
            bb = bb.unsqueeze(0)
        if isinstance(w, int):
            w = torch.Tensor([w]).to(self.device)
        if self.negative is None:
            self.negative = feat
            self.negative_bb = bb
            self.negative_w = w
        else:
            self.negative = torch.cat((self.negative, feat))
            self.negative_bb = torch.cat((self.negative_bb, bb))
            self.negative_w = torch.cat((self.negative_w, w))
        if len(self.negative) > self.len_negative:
            self.negative = self.negative[1:]
            self.negative_bb = self.negative_bb[1:]
            self.negative_w = self.negative_w[1:]

    def recompute(self):
        if self.imgs is None:
            return
        shape = self.templates.shape
        res = self.imgs
        for _filter in self.train_func:
            res = _filter(res)
        self.templates = res.view(shape)

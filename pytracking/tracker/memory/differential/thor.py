import torch
import torch.nn.functional as F

from typing import Sequence

from pytracking.tracker.memory.MemoryBase import MemoryBase
from pytracking.tracker.memory.thor.thor import LT_Module, ST_Module


class Differential:

    def after_classifier(self, diff: torch.Tensor, templates: torch.Tensor) -> bool:
        if self.temp is None:
            return False
        comulative = diff.sum()
        comulative.backward()

        templates.requires_grad = False

        keep = []
        num_temp = len(self.temp)
        # templates = templates[len(self.templates), :]
        for i, grad in enumerate(templates.grad[-num_temp:, :]):
            contrib = grad.norm()
            if contrib < self._ub:
                keep.append(i)
        for i in keep:
            self.update(self.temp[i, :].unsqueeze(0), self.temp_bb[i, :], self.temp_w[i].unsqueeze(0), self.temp_img[i])

        self.flush_temp()
        return len(keep) < num_temp

    @property
    def get_diff_template_n(self) -> Sequence[torch.Tensor]:
        if self.temp is not None:
            self.temp.requires_grad = True
        templates, bb, w, w_neg = super().augmented_template_n

        return templates, bb.detach(), w.detach(), w_neg

    @property
    def get_diff_template(self) -> Sequence[torch.Tensor]:
        if self.temp is not None:
            self.temp.requires_grad = True
        templates, bb, w = super().augmented_template
        return templates, bb.detach(), w.detach()


class LT_ModuleDiff(Differential, LT_Module):

    def __init__(self, k, lb, ub, negative=None, * args, **kwargs):
        super().__init__(k, lb, *args, **kwargs)
        self._lb = lb
        self._ub = ub
        self.len_negative = negative

    def update_d(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, img=None):
        self.add_temp(feat, bb, w, img)


class ST_ModuleDiff(Differential, ST_Module):

    def __init__(self, k, ub, negative=None, * args, **kwargs):
        super().__init__(k, *args, **kwargs)
        self._ub = ub
        self.len_negative = negative

    def update_d(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, img=None):
        self.add_temp(feat, bb, w, img)

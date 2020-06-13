import torch
import numpy as np

import torch.nn.functional as F

from scipy.signal import tukey

from pytracking.tracker.memory.MemoryBase import MemoryBase


class TemplateBase(MemoryBase):

    def __init__(self, *args, **kwargs):
        res = super().__init__(*args, **kwargs)
        self.eye = torch.eye(self.K).cuda()
        return res

    def _pairwise_similarities(self, templates: torch.Tensor, feat: torch.Tensor):
        res = F.conv2d(templates, feat).squeeze()
        return res

    def _pairwise_similarities_norm(self, templates: torch.Tensor, feat: torch.Tensor):
        res = F.conv2d(torch.cat((templates, feat)), feat).squeeze()
        return res[:-1]/res[-1]

    def _compute_gram_matrix(self, templates: torch.Tensor) -> torch.Tensor:
        res = F.conv2d(templates, templates).squeeze()
        norm = res/res.diag().unsqueeze(0)
        tri1 = norm.triu()
        tri2 = norm.T.triu()
        res = torch.stack([tri1, tri2])
        m, _ = res.min(0)
        double = m + m.T
        return double - torch.eye(double.shape[0]).to(double.device)


class LT_Module(TemplateBase):

    def __init__(self, k, lb, train=False, train_func=None, alpha=0.450157, np_store=True, *args, **kwargs):
        super().__init__(k, train=train, train_func=train_func, np_store=np_store, *args, **kwargs)
        self._lb = lb
        self.kernel_sz = 18
        self.alpha = alpha
        self.offset_template = 5
        self.window = torch.Tensor(np.outer(tukey(self.kernel_sz, alpha), tukey(self.kernel_sz, alpha))).cuda()
        self.gram_matrix = None

    def _lower_bound(self, feat: torch.Tensor) -> bool:
        # Lower bound in order to not introduce too far away image
        return (feat < self._lb).any()

    def _throw_away_or_keep(self, feat: torch.Tensor, gram_matrix: torch.Tensor, similarity: torch.Tensor, self_similarity: torch.Tensor) -> bool:

        g_det = gram_matrix.det()
        c = g_det**(-1/gram_matrix.shape[0])
        num_templates = gram_matrix.shape[0]

        dets = torch.zeros(num_templates)
        for i in range(num_templates):
            gram = gram_matrix.clone()
            gram[i, :] = similarity
            gram[:, i] = similarity
            gram[i, i] = self_similarity
            dets[i] = (c*gram).det()
        m_det, pos = dets.abs().max(0)
        if m_det > (c*gram_matrix).det():
            return pos
        else:
            return -1

    def update(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, imgs=None):
        # feat = feat * self.window
        with torch.no_grad():

            # add it to the templates
            if self.templates.shape[0] < self.K:
                self.push(feat, bb, w, imgs)
                if len(self.templates) > self.offset_template+1:
                    self.gram_matrix = self._compute_gram_matrix(self.templates[self.offset_template:])
                return

            similarity_norm = self._pairwise_similarities_norm(self.templates[self.offset_template:], feat)
            new_similarity = self._pairwise_similarities(self.templates[self.offset_template:], feat)

            if self._lower_bound(new_similarity):
                return

            self_similarity = 1
            if self.gram_matrix is None or self.gram_matrix.shape[0] != self.templates.shape[0] and len(self.templates) > self.offset_template:
                self.gram_matrix = self._compute_gram_matrix(self.templates[self.offset_template:])
            pos = self._throw_away_or_keep(feat, self.gram_matrix, similarity_norm, self_similarity)
            # reject it, does not improve
            if pos == -1:
                return
            self.gram_matrix[pos, :] = similarity_norm
            self.gram_matrix[:, pos] = similarity_norm
            self.gram_matrix[pos, pos] = self_similarity
            if imgs is not None:
                self.imgs[pos] = imgs
            pos += self.offset_template
            self.templates[pos] = feat
            self.bb[pos] = bb
            self.weights[pos] = w
        # self.weights -= self.lr

    def push(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, imgs=None) -> None:
        super().push(feat, bb, w, imgs=imgs)
        # self.gram_matrix = self._compute_gram_matrix(self.templates)


class ST_Module(TemplateBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_template = 5

    def update(self, feat: torch.Tensor, bb: torch.Tensor, w: int = 1, imgs=None):
        self.push(feat, bb, w, imgs=imgs)
        if self.num_templates > self.K:
            index = torch.ones(len(self.templates), device=self.templates.device)
            index[self.offset_template] = 0
            index = index.bool()
            self.templates = self.templates[index]
            self.bb = self.bb[index]
            self.weights = self.weights[index]

            if self.numpy_store:
                if True not in [a is None for a in self.imgs]:
                    self.imgs = [self.imgs[0]]+self.imgs[2:]
            elif len(self.imgs) > self.K:
                ind = torch.ones(len(self.imgs)).bool()
                ind[1] = False
                self.imgs = self.imgs[ind]
        # self.weights -= self.lr

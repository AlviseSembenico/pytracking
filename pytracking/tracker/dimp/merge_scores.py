import os
import torch
import torch.nn as nn
from pytracking.utils.loading import load_network
from pytracking.evaluation.environment import env_settings


class MergerScoreLSTMFeat(nn.Module):

    def __init__(self, dim=19, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Settings
        self.hidden_dim_lstm = 19*19
        self.hidden_layers_lstm = 2
        self.dim_lstm = 19*19

        self.sqr_dim = dim
        dim = dim**2
        self.dim = dim

        self.lstm = nn.LSTM(self.dim_lstm, self.hidden_dim_lstm, self.hidden_layers_lstm, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(dim*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, dim+1),
        )
        self.hidden = None

    def forward(self, x, temporal, hidden=None, flatten=True):
        # Encoding
        pred, hidden = self.lstm(temporal, hidden)

        x.clamp_(0, 1)

        if flatten:
            x = x.flatten(1)
        out = self.linear(torch.cat((x, pred.squeeze(1)), axis=1))
        out, sigma = out[:, :-1], nn.functional.relu(out[:, -1]+1)+1
        return out, sigma, pred, hidden


MergeScore = MergerScoreLSTMFeat()


def getMergeScore(net_path=None, train=True, device='cuda'):
    if net_path is None:
        return MergeScore.train(train).to(device)
    path_full = None
    if os.path.isabs(net_path):
        path_full = net_path
    elif isinstance(env_settings().network_path, (list, tuple)):
        # TODO: add this case
        pass
    else:
        path_full = os.path.join(env_settings().network_path, net_path)
    if os.path.isfile(path_full):
        try:
            model = MergeScore
            model.load_state_dict(torch.load(path_full))
            print('Merger model loaded')
            return model.train(train).to(device)
        except:
            pass
    print('Merger model not found, created new')
    return MergeScore.train(train).to(device)


def saveMergeScore(net_path, net):
    path_full = None
    if os.path.isabs(net_path):
        path_full = net_path
    elif isinstance(env_settings().network_path, (list, tuple)):
        # TODO: add this case
        pass
    else:
        path_full = os.path.join(env_settings().network_path, net_path)

    assert path_full is not None, 'Failed to save network'
    torch.save(net, path_full)

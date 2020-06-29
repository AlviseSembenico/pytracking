import os
import sys
import numpy as np
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.experiments.lib import *


OTB100 = '/hdd/datasets/OTB100'
PATH = '/hdd/projects/pytracking2/pytracking/pytracking/tracking_results/dimp/dimp50'

res = []
for s in tqdm(os.listdir(PATH)):
    sub = os.path.join(PATH, s)
    dimp_out = set(os.listdir(sub))
    dimp_out = set(filter(lambda x: x.split('.')[1] == 'txt' and '_time' not in x, dimp_out))
    dimp_acc, metrics_result = read_bb(dimp_out, OTB100, sub)
    dimp_acc = np.array(dimp_acc)
    value = dimp_acc.mean(axis=0).sum()
    try:
        with open(sub + '/detail.txt') as infile:
            res.append((value, infile.readlines()))
    except:
        continue

res.sort(key=lambda x: x[0], reverse=True)
print(res)

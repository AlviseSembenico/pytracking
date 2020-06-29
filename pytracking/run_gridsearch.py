import os
import sys
import glob
import uuid
import argparse
from copy import deepcopy
import numpy as np
import shutil
from multiprocessing import Process

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import get_dataset
from pytracking.parameter.dimp.dimp50 import parameters
from tracktest.lib import *
from pytracking.grid import parallel_experiment


vars = {
    'hard_negative_lb': np.linspace(0.4, 1, 4),
    'ub_LT': np.linspace(5, 100, 5),
    'lb_certainty_update': np.linspace(1, 100, 5)
}
result_dir = '/hdd/projects/pytracking2/pytracking/pytracking/tracking_results/dimp/dimp50/'
OTB_100 = '/hdd/datasets/OTB100'


def run_search(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
               visdom_info=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """
    global vars
    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    override = {}
    for k, v in vars.items():
        override[k] = v[0]

    process_params = []

    def rec(override, i=-1):
        # global process_params
        if i == len(vars.keys()) - 1:
            params = parameters()
            for kk, vv in override.items():
                setattr(params, kk, vv)

            trackers = [Tracker(tracker_name, tracker_param, run_id, parameters=params)]

            path = os.path.join(result_dir, str(uuid.uuid4()))

            process_params.append(Process(target=run_dataset, args=(dataset, trackers, debug, 0, visdom_info, 3, path)))

        else:
            k, v = list(vars.items())[i + 1]
            for value in v:
                c = deepcopy(override)
                c[k] = value
                rec(c, i + 1)
    rec(override)
    parallel_experiment(process_params, n=3)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_search(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
               args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port})


if __name__ == '__main__':
    main()

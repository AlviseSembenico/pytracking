import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from tqdm.notebook import tqdm

OCC = [
    'Basketball',
    'Biker',
    'Bolt',
    'Box',
    'CarScale',
    'ClifBar',
    'David',
    'DragonBaby',
    'Dudek',
    'Football ',
    'Freeman4 ',
    'Girl ',
    'Human3 ',
    'Human4 ',
    'Ironman ',
    'Jump ',
    'Liquor ',
    'Matrix ',
    'Panda ',
    'RedTeam ',
    'Skating1 ',
    'Skating2 ',
    'Soccer ',
    'Tiger ',
    'Walking ',
    'Walking2',
    'Woman',
    'Bird2 ',
    'Coke ',
    'Coupon ',
    'David3 ',
    'Doll ',
    'FaceOcc1 ',
    'FaceOcc2 ',
    'Girl2 ',
    'Human5',
    'Human7 ',
    'Jogging ',
    'KiteSurf ',
    'Lemming ',
    'Rubik',
    'Singer1 ',
    'Subway ',
    'Suv ',
    'Tiger1 ',
    'Trans',
]

Deformation = [
    'Basketball',
    'Bird1',
    'Bird2',
    'BlurBody',
    'Bolt',
    'Bolt2',
    'Couple',
    'Crossing',
    'Crowds',
    'Dancer',
    'Dancer2',
    'David',
    'David3',
    'Diving',
    'Dog',
    'Dudek',
    'FleetFace',
    'Girl2',
    'Gym',
    'Human3',
    'Human4.2',
    'Human5',
    'Human6',
    'Human7',
    'Human8',
    'Human9',
    'Jogging.1',
    'Jogging.2',
    'Jump',
    'Mhyang',
    'Panda',
    'Singer2',
    'Skater',
    'Skater2',
    'Skating1',
    'Skating2.1',
    'Skating2.2',
    'Skiing',
    'Subway',
    'Tiger1',
    'Tiger2',
    'Trans',
    'Walking',
    'Woman'
]

metrics = {
    'OCC': OCC,
    'Deformation': Deformation
}


def bb_intersection_over_union(boxA, boxB):
    # print(boxA, boxB)
    # make sure all the values are intergers
    for i in range(0, 4):
        boxA[i] = int(boxA[i])
        boxB[i] = int(boxB[i]) - 1

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def transform_gtbox(gt_bbox):
    gt_bbox = (gt_bbox[0], gt_bbox[1],
               gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3])
    return list(gt_bbox)


def smooth(v, lenght=50):
    div = int(v.shape[0] / lenght)
    v = v[:div * lenght]
    return np.mean(v.reshape(-1, div), axis=1)


def append_reverse_np(item, times):
    tmp = np.array([item if i % 2 == 0 else list(reversed(item)) for i in range(times)])
    return tmp.reshape(item.shape[0] * times, 4)


def convert_center_to_bbb(bboxes):
    p1x = int(bboxes[0] - 1)
    p1y = int(bboxes[1] - 1)
    p2x = int(bboxes[0] + bboxes[2] - 1)
    p2y = int(bboxes[1] + bboxes[3] - 1)
    # return (p1x, p1y), (p2x, p2y)
    return np.array([p1x, p1y, p2x, p2y])


def read_bb(dimp_out, OTB100, dimp_dir):
    metrics_result = {}
    for k, v in metrics.items():
        metrics_result[k] = []
    dimp_acc = []
    for filename in tqdm(dimp_out):
        sample = re.search('(.*).txt', filename).group(1)

        try:
            try:
                gt_bbox = np.loadtxt(os.path.join(OTB100, sample, 'groundtruth_rect.txt'), delimiter=',')
            except:
                gt_bbox = np.loadtxt(os.path.join(OTB100, sample, 'groundtruth_rect.txt'), delimiter='\t')
        except:
            continue

        dimp_v = np.loadtxt(os.path.join(dimp_dir, filename), delimiter='\t')
        min_dim = dimp_v.shape[0]

        gt_bbox = append_reverse_np(gt_bbox, 40)
        gt_bbox = gt_bbox[:min_dim, :]

        da = []
        for d, gt in zip(dimp_v, gt_bbox):
            d = convert_center_to_bbb(d)
            da.append(bb_intersection_over_union(d, convert_center_to_bbb(gt)))

        da = smooth(np.asarray(da))
        dimp_acc.append(da)

        for k, v in metrics_result.items():
            if sample in metrics[k]:
                v.append(da)
    for k, v in metrics_result.items():
        metrics_result[k] = np.array(v)
    return dimp_acc, metrics_result


def show_acc_plot_compare(name, OTB100, base_dir, compare_dir, append=''):
    try:
        gt_bbox = np.loadtxt(os.path.join(OTB100, name, 'groundtruth_rect.txt'), delimiter=',')
    except:
        gt_bbox = np.loadtxt(os.path.join(OTB100, name, 'groundtruth_rect.txt'), delimiter='\t')

    base = np.loadtxt(os.path.join(base_dir, name + '.txt'), delimiter='\t')
    compare = np.loadtxt(os.path.join(compare_dir, name + append + '.txt'), delimiter='\t')
    min_dim = min(base.shape[0], compare.shape[0])

    gt_bbox = append_reverse_np(gt_bbox, 40)
    gt_bbox = gt_bbox[:min_dim, :]

    base_r, compare_r = [], []
    for b, c, gt in zip(base, compare, gt_bbox):
        # gt = convert_center_to_bbb(gt)
        b = convert_center_to_bbb(b)
        c = convert_center_to_bbb(c)
        base_r.append(bb_intersection_over_union(b, convert_center_to_bbb(gt)))
        compare_r.append(bb_intersection_over_union(c, convert_center_to_bbb(gt)))

    # da = smooth(np.asarray(da))

    plt.title(f'Plot accuracy {name}')
    plt.plot(base_r, label='Current method')
    plt.plot(compare_r, label='Compare method')
    plt.legend()
    plt.show()


def show_acc_plot(name, OTB100, dimp_dir):

    try:
        gt_bbox = np.loadtxt(os.path.join(OTB100, name, 'groundtruth_rect.txt'), delimiter=',')
    except:
        gt_bbox = np.loadtxt(os.path.join(OTB100, name, 'groundtruth_rect.txt'), delimiter='\t')

    dimp_v = np.loadtxt(os.path.join(dimp_dir, name + '.txt'), delimiter='\t')
    min_dim = dimp_v.shape[0]

    # gt_bbox = append_reverse_np(gt_bbox, 40)
    # gt_bbox = gt_bbox[:min_dim,:]

    da = []
    print(len(dimp_v))
    for d, gt in zip(dimp_v, gt_bbox):
        # gt = convert_center_to_bbb(gt)
        d = convert_center_to_bbb(d)
        da.append(bb_intersection_over_union(d, convert_center_to_bbb(gt)))

    da = np.asarray(da)

    plt.title(f'Plot accuracy {name}')
    plt.plot(da)
    plt.show()

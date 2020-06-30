from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 18 * 16
    params.search_area_scale = 5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1 / 3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    # Memory parameters
    params.LT_module = 20
    params.ST_module = 15
    params.ST_module = 15
    params.tukey_alpha = 0.450157
    params.lt_decay = 0.9
    params.st_decay = 0.9
    # TODO: fine tune this
    params.lb_certainty_update = 1
    params.hard_negative_lb = 10
    params.hard_negative_offset = 6
    params.hard_negative_size = 5
    params.hard_negative_offset_h = int(params.hard_negative_offset / 2)
    params.lb = 0.3
    params.ub_LT = 76.25
    params.ub_ST = 76.25
    params.lb_type = 'ensemble'

    params.net = NetWithBackbone(net_path='/hdd/projects/pytracking2/pytracking/checkpoints/ltr/dimp/dimp50/DiMPnet_ep0035.pth.tar',
                                 use_gpu=params.use_gpu)
    # params.net = NetWithBackbone(net_path='dimp50.pth',
    #                              use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    return params

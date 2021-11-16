import argparse
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

COCO_LEN = 24095 # len(train_set) + len(val_set) + len(test_set)
 
clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    255: 40
}


def convert_to_trainID(maskpath, out_mask_dir, dataset_type):
    mask = np.array(Image.open(maskpath))
    mask_copy = mask.copy()
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = osp.join(
        out_mask_dir, dataset_type,
        osp.basename(maskpath).split('.')[0] + '_labelTrainIds.png') 
    Image.fromarray(mask_copy).save(seg_filename, 'PNG')


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert PartImageNet annotations to mmsegmentation format')  # noqa
    parser.add_argument('part_imagenet_path', help='PartImageNet path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    part_imagenet_path = args.part_imagenet_path
    nproc = args.nproc

    out_dir = args.out_dir or part_imagenet_path
    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')

    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'val'))
    mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'test'))

    if out_dir != part_imagenet_path:
        shutil.copytree(osp.join(part_imagenet_path, 'images'), out_img_dir)

    train_list = glob(osp.join(part_imagenet_path, 'annotations', 'train', '*.png'))
    train_list = [file for file in train_list if '_labelTrainIds' not in file]
    val_list = glob(osp.join(part_imagenet_path, 'annotations', 'val', '*.png'))
    val_list = [file for file in val_list if '_labelTrainIds' not in file]
    test_list = glob(osp.join(part_imagenet_path, 'annotations', 'test', '*.png'))
    test_list = [file for file in test_list if '_labelTrainIds' not in file]
    assert (len(train_list) +
            len(val_list) +
            len(test_list)) == COCO_LEN, 'Wrong length of list {} & {} & {}'.format(
                len(train_list), len(val_list), len(test_list))

    if args.nproc > 1:
        print('converting train data ...')
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='train'),
            train_list,
            nproc=nproc)
        print('converting val data ...')
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='val'),
            val_list,
            nproc=nproc)
        print('converting test data ...')
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='test'),
            test_list,
            nproc=nproc)
    else:
        print('converting train data ...')
        mmcv.track_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='train'),
            train_list)
        print('converting val data ...')
        mmcv.track_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='val'),
            val_list)
        print('converting test data ...')
        mmcv.track_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, dataset_type='test'),
            test_list)

    print('Done!')


if __name__ == '__main__':
    main()

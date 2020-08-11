import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import json
from tqdm import tqdm as tqdm
import pandas as pd

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import mayavi.mlab as mlab


def create_dirs_if_not_exists(directories):
    if type(directories)==str:
        directories=[directories]

    for d in directories:
        if not os.path.isdir(d):
            os.makedirs(d)


def convert_to_numpy(x):
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
    return x

def get_all_files_in_tree(folderPath, extensions = ['jpg','png'], getCompletePaths=True):
    img_names = []
    for ext in extensions:
        img_names += glob.glob("{}/**/*.{}".format(folderPath, ext), recursive=True)
    if not getCompletePaths:
        img_names = [os.path.basename(i) for i in img_names]
    return img_names

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        is_dir = os.path.isdir(self.root_path)
        data_file_list = get_all_files_in_tree(
            self.root_path,
            getCompletePaths=True,
            extensions=[self.ext.lstrip(".")]
        ) if is_dir else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list


    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.csv':
            df = pd.read_csv(self.sample_file_list[index])
            points = np.array(df[['X', 'Y', 'Z', 'intensity']], dtype=np.float32)
            points[:, 3] = points[:, 3] / 255.0
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict, self.sample_file_list[index]

def load_predictions(predictions_file):
    ref_boxes = None
    ref_scores = None
    ref_labels = None
    if not predictions_file=="":
        predictions = np.load(predictions_file, allow_pickle=True).item()
        ref_boxes = np.array(predictions['pred_boxes'])
        ref_scores = np.array(predictions['pred_scores'])
        ref_labels = np.array(predictions['pred_labels'])
    return ref_boxes,ref_scores, ref_labels

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--predictions_file', type=str, default='',
                        help='specify the predictions_file')
    parser.add_argument('--point_cloud_file', type=str, default='',
                        help='specify the point_cloud_file')

    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.point_cloud_file), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    ref_boxes, ref_scores, ref_labels = load_predictions(args.predictions_file)

    for idx, (data_dict, filepath) in tqdm(enumerate(demo_dataset)):
        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        V.draw_scenes(
            points=data_dict['points'][:, 1:], ref_boxes=ref_boxes,
            ref_scores=ref_scores, ref_labels=ref_labels, boxes_converted=True
        )
        mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()


import os
from typing import Any, Dict, List, Iterator, Optional, Tuple
import torch
from torchvision.datasets.folder import ImageFolder
import csv


DIR_LABEL_CSV = "list/clist.csv"

class ImageNet(ImageFolder):

    def __init__(self, root: str, train: bool = True, download: Optional[str] = None, **kwargs: Any) -> None:
        root = self.root = os.path.join(os.path.expanduser(root), 'imagenet')
        self.split = 'train' if train else 'val'
        self.split_folder = os.path.join(root, self.split)

        super(ImageNet, self).__init__(self.split_folder, **kwargs)

        self.root = root
        self.class_to_idx = self._get_corr()

    def _get_corr(self):
        csv_path = os.path.join(self.root, DIR_LABEL_CSV)
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            dic = dict()
            for row in reader:
                if row[2] == "label":  # ignore first line of csv
                    continue
                dic[row[0]] = int(row[2])
        return dic


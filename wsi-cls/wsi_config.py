# -*- coding: utf-8 -*-

import os, sys
import numpy as np

folder_ratio_map = {
    '0Neg':   1.0,
    '1Pos':   1.0,
}

bin_class_map_dict = {
    '0Neg':    0,
    '1Pos':    1,
}

folder_map_dict = {}
for idx, (k, v ) in enumerate(folder_ratio_map.items()):
    folder_map_dict[k] = idx

folder_reverse_map = {}
for k, v in folder_map_dict.items():
    folder_reverse_map[v] = k

class_reverse_map = {}
for k, v in bin_class_map_dict.items():
    class_reverse_map[v] = k

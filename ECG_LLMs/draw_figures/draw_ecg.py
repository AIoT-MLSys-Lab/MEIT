import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import numpy as np
from torchvision.transforms import transforms
import wfdb
import torch
import ecg_plot
import yaml
import ecg_plot

path_1 =  '/fs/scratch/PAS2473/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1223/p12233384/s40478882/40478882'
path_2 = '/fs/scratch/PAS2473/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1917/p19172395/s40581273/40581273'
path_3 = '/fs/scratch/PAS2473/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/p1450/p14505704/s45461332/45461332'
ecg = wfdb.rdsamp(path_3)[0] # 12 * 5000

ecg = ecg.T
ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8) # 12 * dim
# add noise here #################

ecg_plot.plot_12(ecg, sample_rate = 500, title = 'ECG 12')
ecg_plot.save_as_png('path_3_ecg','/users/PAS2473/brucewan666/ECG/ECG/draw_figures/')
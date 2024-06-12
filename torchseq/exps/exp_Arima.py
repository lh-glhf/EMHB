import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from ..data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from ..models.Arima import Arima
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual
from ..utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Arima:

    def __init__(self, args):
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        return Arima(self.args)

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        print("Arima is Fitting")
        return

    def test(self):
        self.model.fit()
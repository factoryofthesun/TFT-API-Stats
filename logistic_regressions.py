"""
Logistic prediction models for the following problem statement:
Given player team composition information and (historical) unit performance information, predict the
winner in a 1v1 situation.

Models to consider
    Basic - dummy variable for every single unit for both players 1 and 2
    Basic v2 - dummy variable for every single trait for players 1 and 2
    

Plot learning curves for the most promising looking models
"""

import logging
import time
import datetime
from platform import python_version
import matplotlib
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from config import PATH_GDRIVE_MAIN_DIR

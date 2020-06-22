"""
Logistic prediction models for the following problem statement:
Given player team composition information and (historical) unit performance information, predict the
winner in a 1v1 situation.

Features under consideration:
    Dummies for units and traits
    Interactions b/w trait permutations
    Interactions b/w unit, level, and items
    Win rates (unit, trait, item)
    "Synergistic" win rates

Plot learning curves for the most promising looking models
"""

import logging
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from config import PATH_GDRIVE_MAIN_DIR

# Global variable
GAME_VERSION = 10.2

# Read in data

# Data preprocessing

# Split train and test

# Basic dummy models

# Intermediate model

# Maxed Dimension Model
# Consider: paring down dimensionality by using the synergy aggregation from the Dota2 papers

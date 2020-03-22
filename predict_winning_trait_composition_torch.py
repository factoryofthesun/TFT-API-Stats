"""
This script is to solve the following problem statement: Given 2 trait compositions, predict which will win the
matchup.

To keep things simple at the start, the data for this will be the integer difference between trait tiers (e.g.
the trait "Inferno" has 3 tiers) between the top 2 players of each match (1st and 2nd place). In particular, the
feature vector (input data x) is player 1's trait tiers minus player 2's.

The output data (y) denotes which player won the matchup: 1 means player 1 won, 2 means player 2 won.
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
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from config import PATH_GDRIVE_MAIN_DIR

warnings.filterwarnings('ignore', category=FutureWarning)

TRAITS_LIST = ["Alchemist", "Assassin", "Avatar", "Berserker", "Blademaster", "Crystal", "Desert", "Druid", "Electric",
               "Glacial", "Inferno", "Light", "Celestial", "Mage", "Mountain", "Mystic", "Ocean", "Poison", "Predator",
               "Ranger", "Shadow", "Soulbound", "Metal", "Summoner", "Warden",  "Wind", "Woodland"]

NUM_UNIQUE_TRAITS = len(TRAITS_LIST)

#########################################################################
# Extract data and separate data into groups (training, test, validation)
#########################################################################

# Contains top 2 (1st and 2nd place) trait compositions from each match
top_2_df = pd.read_csv(PATH_GDRIVE_MAIN_DIR + 'trait_compositions_first_and_second_v2_test2.csv')

# For each match ID, take the difference between trait tiers of 1st place minus 2nd place. We will shuffle
# the 1st/2nd order in the next step.
for trait_name in TRAITS_LIST:
    top_2_df.loc[top_2_df["Place"] == 2, trait_name] = -top_2_df[top_2_df["Place"] == 2][trait_name]
grouped = top_2_df.groupby("GameID")
diff_df = grouped.sum()

# Create a randomized array of y-labels (either 1 or 2, indicating the winner), and negate the differences accordingly
# (if winner is player 2, need to negate all the trait differences for that match)
num_matches = diff_df.shape[0]
diff_df["winner"] = np.random.choice([1, 2], num_matches)

player_2_wins = np.where(diff_df["winner"] == 2)[0]
diff_df = diff_df.reset_index(drop=True)
diff_df.loc[player_2_wins, TRAITS_LIST] = -diff_df.loc[player_2_wins, TRAITS_LIST]

# Finally, shuffle the data entries
diff_df = diff_df.sample(frac=1).reset_index(drop=True)  # Shuffle the entries

# Split into training and validation
num_train = 500
num_validation = 250
num_test = diff_df.shape[0] - num_train - num_validation
assert num_test > 0

x_train = diff_df.loc[:num_train, TRAITS_LIST]
y_train = diff_df.loc[:num_train, "winner"]
x_validation = diff_df.loc[:num_validation, TRAITS_LIST]
y_validation = diff_df.loc[:num_validation, "winner"]
x_test = diff_df.loc[:num_test, TRAITS_LIST]
y_test = diff_df.loc[:num_test, "winner"]

################
# Run neural net predictor
# Framework: Pytorch
################

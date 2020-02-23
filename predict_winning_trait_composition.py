"""
This script is to solve the following problem statement: Given 2 trait compositions, predict which will win the
matchup.

To keep things simple at the start, the data for this will be the integer difference between trait tiers (e.g.
the trait "Inferno" has 3 tiers) between the top 2 players of each match (1st and 2nd place). In particular, the
feature vector (input data x) is player 1's trait tiers minus player 2's.

The output data (y) denotes which player won the matchup: 1 means player 1 won, 2 means player 2 won.
"""

import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from config import PATH_GDRIVE_MAIN_DIR

warnings.filterwarnings('ignore', category=FutureWarning)

TRAITS_LIST = ["Alchemist", "Assassin", "Avatar", "Berserker", "Blademaster", "Crystal", "Desert", "Druid", "Electric",
               "Glacial", "Inferno", "Light", "Lunar", "Mage", "Mountain", "Mystic", "Ocean", "Poison", "Predator",
               "Ranger", "Shadow", "Soulbound", "Steel", "Summoner", "Warden",  "Wind", "Woodland"]

#########################################################################
# Extract data and separate data into groups (training, test, validation)
#########################################################################

# Contains top 2 (1st and 2nd place) trait compositions from each match
top_2_df = pd.read_csv(PATH_GDRIVE_MAIN_DIR + 'trait_compositions_first_and_second.csv')

# For each match ID, take the difference between trait tiers of 1st place minus 2nd place. We will shuffle
# the 1st/2nd order in the next step.
# Will use the split-apply-combine method for pandas array. First, we negate the tier values of 2nd place player
for trait_name in TRAITS_LIST:
    top_2_df[top_2_df["Place"] == 2][trait_name] = -top_2_df[top_2_df["Place"] == 2][trait_name]

# Create a randomized array of y-labels (either 1 or 2), and negate the differences accordingly


# Finally, shuffle the data entries
# data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffle the entries

# Split into training and validation
# num_train = 480
# num_validation = 160
# num_test = len(rgba) - num_train - num_validation
# assert num_test > 0
#
# x_train = rgba[:num_train]
# y_train = ys[:num_train]
# x_validation = rgba[num_train:(num_train + num_validation)]
# y_validation = ys[num_train:(num_train + num_validation)]
# x_test = rgba[-num_test:]
# y_test = ys[-num_test:]


################
# Run neural net predictor
################

# # TODO: do we even need to run this pre-processing?
# def preprocess(x, y):
#     x = tf.cast(x, tf.int64)
#     y = tf.cast(y, tf.int64)
#     return x, y
#
#
# def create_dataset(xs, ys, n_classes, batch_size=20):
#     ys = tf.one_hot(ys, depth=n_classes)
#     return tf.data.Dataset.from_tensor_slices((xs, ys)).map(preprocess).shuffle(ys.shape[0]).batch(batch_size)
#
#
# train_dataset = create_dataset(x_train, y_train, n_classes=2)
# val_dataset = create_dataset(x_validation, y_validation, n_classes=2)
#
# model = keras.Sequential([
#     keras.layers.Reshape(target_shape=(NUM_UNIQUE_TRAITS,),
#                          input_shape=(NUM_UNIQUE_TRAITS,)),
#     keras.layers.Dense(units=256, activation='relu'),
#     keras.layers.Dense(units=192, activation='relu'),
#     keras.layers.Dense(units=128, activation='relu'),
#     keras.layers.Dense(units=2, activation='softmax')
# ])
#
# # # Learn more about learning rate (e.g. exponential decay) and try implementing it further. We possibly suspect the
# # # learning rate because the training accuracy converges too fast (maybe learns too quickly?) and plateaus for the
# # # rest of the epochs.
# # def exp_decay(t):
# #     initial_lrate = 0.1
# #     k = 0.1
# #     lrate = initial_lrate * np.exp(-k*t)
# #     return lrate
# # lrate = tf.keras.callbacks.LearningRateScheduler(exp_decay)
#
# model.compile(optimizer='adam',
#               loss=tf.compat.v1.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(
#     train_dataset.repeat(),
#     epochs=30,
#     steps_per_epoch=25,
#     validation_data=val_dataset.repeat(),
#     validation_steps=20,
#     verbose=2,
#     )
#
# test_dataset = create_dataset(x_test, y_test, n_classes=2)
# predictions = model.evaluate(test_dataset)
# print('test loss, test acc:', predictions)

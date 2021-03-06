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
               "Glacial", "Inferno", "Light", "Celestial", "Mage", "Mountain", "Mystic", "Ocean", "Poison", "Predator",
               "Ranger", "Shadow", "Soulbound", "Metal", "Summoner", "Warden", "Wind", "Woodland"]

NUM_UNIQUE_TRAITS = len(TRAITS_LIST)  # 27 unique traits

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

# Use the values after the training set as the validation set
validation_index = num_validation + num_train
x_validation = diff_df.loc[num_train:validation_index, TRAITS_LIST]
y_validation = diff_df.loc[num_train:validation_index, "winner"]

test_index = validation_index + num_test
x_test = diff_df.loc[validation_index:test_index, TRAITS_LIST]
y_test = diff_df.loc[validation_index:test_index, "winner"]


################
# Run neural net predictor
################
# Training data: 119372/2 = 59686 * .75 = 44764
# Input structure: If we apply "one hot encoding" over every differenced trait we get a possible range from -3 to 3 (assuming 3 tiers for each trait)
#   Then the model will consider 7 * num(traits) features???
#   Consider adjusting the units counts in the subsequent hidden layers accordingly (256 neurons might be too many)

# "Smaller batch sizes offer a regularizing effect and lower generalization error, and are easier to fit into GPU memory"
def create_dataset(xs, ys, n_classes, batch_size=256):
    xs = xs.to_numpy()
    ys = ys.to_numpy()
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(ys.shape[0]).batch(batch_size)


train_dataset = create_dataset(x_train, y_train, n_classes=2)
val_dataset = create_dataset(x_validation, y_validation, n_classes=2)

# train_repeat = train_dataset.repeat()
# val_repeat = val_dataset.repeat()
#
# model = keras.Sequential([
#     keras.layers.Reshape(target_shape=(NUM_UNIQUE_TRAITS,),
#                          input_shape=(NUM_UNIQUE_TRAITS,)),
#     keras.layers.Dense(units=256, activation='relu'),
#     keras.layers.Dense(units=192, activation='relu'),
#     keras.layers.Dense(units=128, activation='relu'),
#     keras.layers.Dense(units=1, activation='sigmoid')  # Give probability as opposed to binary prediction
# ])
#
#
# # # # Learn more about learning rate (e.g. exponential decay) and try implementing it further. We possibly suspect the
# # # # learning rate because the training accuracy converges too fast (maybe learns too quickly?) and plateaus for the
# # # # rest of the epochs.
# def exp_decay(t):
#     initial_lrate = 0.1
#     k = 0.1
#     lrate = initial_lrate * np.exp(-k * t)
#     return lrate
#
#
# lrate = tf.keras.callbacks.LearningRateScheduler(exp_decay)
#
# model.compile(optimizer='adam',
#               loss=tf.compat.v1.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# history = model.fit(
#     train_dataset.repeat(),
#     epochs=50,
#     #TODO: CONSIDER SKIPPING THIS SINCE WE WANT TO RUN THROUGH WHOLE TRAINING SET
#     steps_per_epoch=25,
#     validation_data=val_dataset.repeat(),
#     #TODO: CONSIDER SKIPPING THIS SINCE WE WANT TO RUN THROUGH WHOLE VALIDATION SET
#     validation_steps=20,
#     verbose=2,
#     )
#
# test_dataset = create_dataset(x_test, y_test, n_classes=2)
# predictions = model.evaluate(test_dataset)
# print('test loss, test acc:', predictions)

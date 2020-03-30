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
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from config import PATH_GDRIVE_MAIN_DIR

warnings.filterwarnings('ignore', category=FutureWarning)

TRAITS_LIST = ["Alchemist", "Assassin", "Avatar", "Berserker", "Blademaster", "Crystal", "Desert", "Druid", "Electric",
               "Glacial", "Inferno", "Light", "Celestial", "Mage", "Mountain", "Mystic", "Ocean", "Poison", "Predator",
               "Ranger", "Shadow", "Soulbound", "Metal", "Summoner", "Warden",  "Wind", "Woodland"]

NUM_UNIQUE_TRAITS = len(TRAITS_LIST) #27 unique traits

#########################################################################
# DATA PREPROCESSING
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

# Check for NA values and drop
print("Total samples: {}".format(len(diff_df.index)))
diff_df.dropna(inplace = True)
print("Samples after dropping missing: {}".format(len(diff_df.index)))

# Create a randomized array of y-labels (either 0 or 1, indicating the winner; 0 = player 1, 1 = player 2), and negate the differences accordingly
# (if winner is player 2, need to negate all the trait differences for that match)
num_matches = diff_df.shape[0]
diff_df["winner"] = np.random.choice([0, 1], num_matches)

player_2_wins = np.where(diff_df["winner"] == 1)[0]
diff_df = diff_df.reset_index(drop=True)
diff_df.loc[player_2_wins, TRAITS_LIST] = -diff_df.loc[player_2_wins, TRAITS_LIST]

# Finally, shuffle the data entries
diff_df = diff_df.sample(frac=1).reset_index(drop=True)  # Shuffle the entries

#Make all traits variables categorical
for trait_name in TRAITS_LIST:
    shift_amount = abs(diff_df[trait_name].min()) #IMPORTANT: shift all values to be positive so they can be used as indices for the embedding layers
    diff_df[trait_name] = diff_df[trait_name] + shift_amount
    diff_df[trait_name] = diff_df[trait_name].astype('category')


# Split into training and validation (use smaller sample size for testing)
num_train = 500
num_validation = 250
num_test = 250
#num_test = diff_df.shape[0] - num_train - num_validation
assert num_test > 0

#TODO: clean this up by calling sklearn's train_test_split
x_train = diff_df.loc[:num_train, TRAITS_LIST].astype(int).values
y_train = diff_df.loc[:num_train, "winner"].values
x_validation = diff_df.loc[num_train:(num_validation + num_train), TRAITS_LIST].astype(int).values
y_validation = diff_df.loc[num_train:(num_validation + num_train), "winner"].values
x_test = diff_df.loc[(num_validation + num_train):(num_validation + num_train + num_test), TRAITS_LIST].astype(int).values
y_test = diff_df.loc[(num_validation + num_train):(num_validation + num_train + num_test), "winner"].values

#Convert to PyTorch tensors
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.tensor(y_train)
x_val_tensor = torch.from_numpy(x_validation)
y_val_tensor = torch.tensor(y_validation)
x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.tensor(y_test)

########################################
# Set up and train Neural Network Model
# Framework: Pytorch
# Inspiration: https://jovian.ml/aakanksha-ns/shelter-outcome?fbclid=IwAR3fYXHtsM2OnhtPHWO4ZRdlzIef15F-Vwjur2psdM-oyo15xMiFM_Vu41U
########################################

#Check if we have a GPU available
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not use CPU
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

################
# Categorical Embedding values
################
target_dict = {'Player 1 Win': 1, 'Player 2 Win': 2}

#Create embedding variables
'''
We can use categorical embeddings instead of one-hot encodings in order to
capture the relationship between levels within a trait
e.g. Infernal 3 is closer to Infernal 2 than Infernal 2 is to Infernal 0
'''
embedded_cols = {name: len(col.cat.categories) for name, col in diff_df.loc[:num_train, TRAITS_LIST].items()}

#Embedding size formula: (# of categories + 1)//2 according to Jeremy Howard (fast.ai)
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

################
# Create dataloaders
################
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_data = TensorDataset(x_train_tensor, y_train_tensor)
validation_data = TensorDataset(x_val_tensor, y_val_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

train_sampler = RandomSampler(train_data)
validation_sampler = RandomSampler(validation_data)
test_sampler = RandomSampler(test_data)

train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)
val_dataloader = DataLoader(validation_data, sampler = validation_sampler, batch_size = batch_size)
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)

################
# Define helper functions
################

#Accuracy function NEEDS ADJUSTMENT
def flat_accuracy(preds, labels):
    pred_flat = np.round(preds).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

################
# Create model: structure copied directly from https://jovian.ml/aakanksha-ns/shelter-outcome?fbclid=IwAR3fYXHtsM2OnhtPHWO4ZRdlzIef15F-Vwjur2psdM-oyo15xMiFM_Vu41U
################
class TFTBinaryClassifier(nn.Module):
    def __init__(self, embedding_sizes, emb_dropout = 0.6, dropout = 0.3):
        super().__init__()
        #Embedding "size" is a function of number of levels per "category" (categories = n_levels); our function is (n_levels+1)//2
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb = n_emb
        self.lin1 = nn.Linear(self.n_emb, 150) #Hidden layer optimal neuron num b/w input and output dimensions
        self.lin2 = nn.Linear(150, 75)
        self.lin3 = nn.Linear(75, 1)
        self.bn2 = nn.BatchNorm1d(150)
        self.bn3 = nn.BatchNorm1d(75)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.drops = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_cat):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        output = self.sigmoid(x) #Transforms single linear output to "probability" value between 0 and 1 (necessary form to calculate BCE Loss)
        return output

################
# Initialize
################
from transformers import AdamW

model = TFTBinaryClassifier(embedding_sizes)
model.cuda()

def get_optimizer(model, lr = 0.001, wd = 0.0):
    optim = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim

optimizer = get_optimizer(model)

################
# Training
################
#Set seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#Record loss and accuracy
train_loss = []
val_loss = []
train_acc = []
val_acc = []

epochs = 40

for epoch_i in range(epochs):
  print("")
  print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
  print('Training...')
  t0 = time.time()
  total_loss = 0
  total_accuracy = 0
  steps = 0
  model.train() #Put into training mode

  for step, batch in enumerate(train_dataloader):
    #Progress update every 5 batches
    if step % 10 == 0 and not step == 0:
      elapsed = format_time(time.time() - t0)
      print('  Batch {}  of  {}.    Elapsed: {}.'.format(step, len(train_dataloader), elapsed))

    #Unpack batch and copy tensors to GPU
    b_inputs = batch[0].type(torch.LongTensor).to(device)
    b_targets = batch[1].to(device)

    #Make sure to zero out the gradient
    model.zero_grad()

    #Perform forward pass
    output = model(b_inputs) #Outputs a single value for each input

    #Get loss
    loss = F.binary_cross_entropy(output, torch.unsqueeze(b_targets.float(),1))
    total_loss += loss.item()

    #Move logits and labels to CPU - can't perform accuracy calculation on GPU
    output = output.detach().cpu().numpy()
    targets = b_targets.to('cpu').numpy()

    #Calculate batch accuracy
    tmp_accuracy = flat_accuracy(output, targets)
    total_accuracy += tmp_accuracy
    steps += 1

    #Backward pass to calculate gradients
    loss.backward()

    #Clip norms to prevent "exploding gradients"
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step() #Update parameters using optimizer

  #Average loss/accuracy
  avg_loss = total_loss/steps
  avg_acc = total_accuracy/steps

  train_loss.append(avg_loss)
  train_acc.append(avg_acc)
  print("")
  print("  Average training loss: {0:.2f}".format(avg_loss))
  print("  Average training accuracy: {0:.2f}".format(avg_acc))
  print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

  #====================
  #     Validation
  #====================
  print("")
  print("Running validation...")
  t0 = time.time()
  total_loss = 0
  total_accuracy = 0
  steps = 0

  model.eval() #Evaluation mode

  for batch in test_dataloader:
    #Unpack batch and copy tensors to GPU
    b_inputs = batch[0].type(torch.LongTensor).to(device)
    b_targets = batch[1].to(device)

    #Tell model not to store gradients - speeds up testing
    with torch.no_grad():
      output = model(b_inputs)

    #Get loss
    loss = F.binary_cross_entropy(output, torch.unsqueeze(b_targets.float(),1))
    total_loss += loss.item()

    #Move logits and labels to CPU - can't perform accuracy calculation on GPU
    output = output.detach().cpu().numpy()
    targets = b_targets.to('cpu').numpy()

    #Calculate batch accuracy
    tmp_accuracy = flat_accuracy(output, targets)
    total_accuracy += tmp_accuracy
    steps += 1

    steps += 1 #Track batch num
  #Report the final accuracy for this validation run.
  val_loss.append(total_loss/steps)
  val_acc.append(total_accuracy/steps)
  print("  Average validation loss: {0:.2f}".format(total_loss/steps))
  print("  Average validation accuracy: {0:.2f}".format(total_accuracy/steps))
  print("  Validation took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))

################
# Plot training results
################
#Plot results
import matplotlib.pyplot as plt

plot_path = PATH_GDRIVE_MAIN_DIR + "Plots/"

plt.plot(train_loss, 'r--')
plt.plot(val_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.title("Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(plot_path + "pytorch_loss.png")
plt.clf()

plt.plot(train_acc, 'r--')
plt.plot(val_acc, 'b-')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.title("Accuracy by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(plot_path + "pytorch_accuracy.png")
plt.clf()

################
# Test model
################
probs = []
targets = []
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        b_inputs = batch[0].type(torch.LongTensor).to(device)
        b_targets = batch[1].to(device)
        output = model(b_inputs)
        probs.append(output.to('cpu').numpy())
        targets.append(b_targets.to('cpu').numpy())

#Flatten and convert to numpy arrays
probs = np.concatenate(probs).ravel()
targets = np.concatenate(targets).ravel()

#Check accuracy
preds = np.round(probs)
test_accuracy = np.sum(preds == targets)/len(targets)
print("Test accuracy: {}".format(test_accuracy))

#Record predictions for review
test_out = pd.DataFrame({"probs":probs, "preds":preds, "targets":targets})
test_out.to_csv("test_outputs.csv", index = False)

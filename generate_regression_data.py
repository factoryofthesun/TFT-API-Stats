import numpy as np
import pandas as pd
from config import PATH_GDRIVE_MAIN_DIR

"""

This script is meant to generate the variables and data files to feed directly into the eventual logistic regression prediction algorithms.
The total variables being generated are:
    Ordinal levels for units and traits
    Dummies for every pair of traits (differentiating tiers) (N = )
    Dummies for every unique champion/tier/itemset (N = )
    % win rates for units and traits (differentiating tiers)
    % increase in win rate for a champion with given itemset (if sample size too small then use sum over individual items)
        - Proxies for interaction b/w champion and item
    % increase in win rate for a champion with given tier and itemset
        - Proxies for interaction b/w champion, level, and item


"""


"""

The data will be taken from {patch_num}_compositions_data.pkl which is formatted with the following columns:

| Game Version | Game Date | GameID | Region | PUUID | Place | Traits | Units | Damage | TeamSize |

We will only be taking data from a single game version at a time, as the meta changes significantly in between patches.
For initial training and testing purposes we will isolate the sample to patch 10.9
"""

# Read in compositions data
comps_data = pd.read_csv(PATH_GDRIVE_MAIN_DIR+'compositions_data.csv')

# Since the meta shifts significantly per patch, let's restrict our analysis to one historic patch period for now
game_version = "10.2"

# Drop data points with empty units
comps_data = comps_data.loc[pd.notna(comps_data['Units']) & (comps_data['Units'] != ''),]

# Take only specified game version and 1st/2nd place finishes
comps_data['Game Version Short'] = comps_data['Game Version'].str.extract(r'(\d*[.][^.]*)', expand=False) # Everything up until but excluding 2nd period
comps_data = comps_data.loc[(comps_data['Game Version Short'] == game_version) & (comps_data['Place'].isin([1,2]))]

# Take only the columns with GameID, Place, and Traits and place into a summary dataframe
summ_df = comps_data.loc[:,['GameID', 'Game Date','Place','Traits']]

# Remove games in which there is no 2nd place
good_games = summ_df.loc[(summ_df['Place'] == 2), 'GameID'].tolist()
summ_df = summ_df.loc[summ_df.GameID.isin(good_games),]

print("Number of first place: {}, Number of second place: {}".\
        format(sum(summ_df.Place == 1), sum(summ_df.Place == 2))) #Check counts

# Clean up traits column string
summ_df['Traits'] = summ_df['Traits'].str.replace('Set2_', '')
summ_df['Traits'] = summ_df['Traits'].str.replace('[^a-zA-Z\d]+', '')

# Explode traits column
summ_df['Traits'] = summ_df.Traits.apply(lambda x: x.split(',')) # Convert traits column into lists
summ_long = summ_df.explode('Traits')

#Split out traits into trait name and tier
summ_long = summ_long.loc[pd.notna(summ_long['Traits']) & (summ_long['Traits'] != ''),] # Drop empty traits values
summ_long['Tier'] = pd.to_numeric(summ_long['Traits'].str.extract('(\d+)', expand = False)) # Only numeric tier value
summ_long['Traits'] = summ_long['Traits'].str.extract('([a-zA-Z]+)', expand = False) # Only text trait data

summ_long.to_csv(PATH_GDRIVE_MAIN_DIR + f"comps_traits_units_clean_{game_version}.csv", index = False)

#Sort by trait name, then pivot long to wide with trait names as columns and tiers as values
summ_long = summ_long.sort_values(by = ['Traits'])
summ_wide = summ_long.pivot_table(index = ['GameID', 'Game Date', 'Place'], columns = 'Traits', values = 'Tier').reset_index()
summ_wide.index.name = summ_wide.columns.name = None

#Sort by gameid, then place
summ_wide.sort_values(by = ['GameID', 'Place'], inplace = True)

# Assign empty cell values to 0 for data analysis purposes later
summ_wide.fillna(0, inplace = True)

# Output the summary dataframe to a .csv file titled "trait_compositions_first_and_second.csv"

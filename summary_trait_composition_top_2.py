
import numpy as np
import pandas as pd
from config import PATH_GDRIVE_MAIN_DIR

"""

This script is meant to aggregate the synergies of the first and second place players into a .csv file for direct
entry into a neural net in the future. The processed data will take the form:

| GameID | Place | Alchemist | Assassin | Avatar | Berserker | Blademaster | Crystal | Desert | Druid | Electric

| Glacial | Inferno | Light | Lunar | Mage | Mountain | Mystic | Ocean | Poison | Predator | Ranger | Shadow | Soulbound

| Steel | Summoner | Warden |  Wind | Woodland |

"""


"""

The data will be taken from compositions_data.csv which is formatted with the following columns:

| Game Version | Game Date | GameID | Region | PUUID | Place | Traits | Units | Damage | TeamSize |

We will only be taking data from Game Versions following 9.22 (release date Nov 6, 2019) which includes only data from TFT Set 2. Furthermore,
currently we will only be using the GameID, Place, and Traits columns to simplify the inputs for the neural net.

"""

#Read in compositions data
comps_data = pd.read_csv(PATH_GDRIVE_MAIN_DIR+'compositions_data.csv')
set2_date = '11-06-2019'
# Take only the columns with GameID, Place, and Traits and place into a summary dataframe
summ_df = comps_data.loc[:,['GameID', 'Game Date','Place','Traits']]

# Remove data from Game Versions prior to 9.22 and not 1st/2nd place
summ_df['Game Date'] = pd.to_datetime(summ_df['Game Date'])
set2summ = summ_df.loc[(summ_df['Game Date'] >= set2_date) & (summ_df['Place'].isin([1,2])),]

#Convert traits column into lists
set2summ['Traits'] = set2summ.Traits.apply(lambda x: x[1:-1].split(','))

#Collapse list of traits into individual rows
set2long = pd.DataFrame({col: np.repeat(set2summ[col].values, set2summ['Traits'].str.len())
                        for col in set2summ.columns.drop('Traits')}).\
                        assign(**{'Traits': np.concatenate(set2summ['Traits'].values)})[set2summ.columns]

#Split out traits into trait name and tier
set2long['Traits'] = set2long['Traits'].str.replace('Set2', '') #Remove Set2 substring to simplify the regex
set2long = set2long.loc[pd.notna(set2long['Traits']) & (set2long['Traits'] != ''),] #Drop empty traits values
set2long['Tier'] = pd.to_numeric(set2long['Traits'].str.extract('(\d+)', expand = False))
set2long['Traits'] = set2long['Traits'].str.extract('([a-zA-Z]+)', expand = False)

#Sort by trait name, then pivot long to wide with trait names as columns and tiers as values
set2long = set2long.sort_values(by = ['Traits'])
set2wide = set2long.pivot_table(index = ['GameID', 'Game Date', 'Place'], columns = 'Traits', values = 'Tier').reset_index()
set2wide.index.name = set2wide.columns.name = None

#Sort by gameid, then place
set2wide.sort_values(by = ['GameID', 'Place'], inplace = True)

# Assign empty cell values to 0 for data analysis purposes later
set2wide.fillna(0, inplace = True)

# TODO: Prune out rows that only have data for a first place finish (Dennis ran into an issue where 
# one game only had data from a single player in a game)

# Output the summary dataframe to a .csv file titled "trait_compositions_first_and_second.csv"
set2wide.to_csv(PATH_GDRIVE_MAIN_DIR + "trait_compositions_first_and_second.csv", index = False)

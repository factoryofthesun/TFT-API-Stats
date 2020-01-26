
import pandas as pd

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

We will only be taking data from Game Versions following 9.22 which includes only data from TFT Set 2. Furthermore,
currently we will only be using the GameID, Place, and Traits columns to simplify the inputs for the neural net.

"""

# Generate the new dataframe with the columns as described above with the GameID, Place, and 28 different synergies

# Take only the columns with GameID, Place, and Traits and place into a summary dataframe

# Remove data from Game Versions prior to 9.22

# Remove data from placements in the Place column that are not 1st or 2nd place

# Order the summary dataframe first by GameID, then by 1st and 2nd places

# For each row in the summary dataframe, first input the GameID and Place for the 1st place player.

# Then, for each element in the Traits list, (i.e. Alchemist1), parse the string into two parts (i.e. "Alchemist" + "1").

# Take the first part of the string to locate the column in the new dataframe, then input the second part of the string

# Repeat for the 2nd place player

# Repeat for the entirety of the GameIDs in the summary dataframe

# Output the summary dataframe to a .csv file titled "trait_compositions_first_and_second.csv"
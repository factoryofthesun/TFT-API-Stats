import os
import glob
import pandas as pd
import requests
from datetime import datetime
import json
import time
from config import PATH_GDRIVE_MAIN_DIR, PATH_GDRIVE_JSON_DIR

match_df = pd.read_csv(PATH_GDRIVE_MAIN_DIR + 'compositions_data.csv')

#Some basic analysis
#Sort by date and filter by latest game version
analysis_df = match_df.sort_values(by = ['Game Date', 'Region'], ascending = False)
latest_version = analysis_df.loc[0, 'Game Version']
print(latest_version)
analysis_df = analysis_df.loc[analysis_df['Game Version'] == latest_version,]
analysis_df['Traits'] = analysis_df['Traits'].map(lambda x: tuple(x))

#Group by trait compositions and export ranking frequencies
comp_group_df = analysis_df.loc[:,['Traits', 'Place']].groupby(['Traits', 'Place']).size().unstack(fill_value = 0)
comp_group_df.to_csv(PATH_GDRIVE_MAIN_DIR + 'traits_compositions.csv', index_label = 'Comp')

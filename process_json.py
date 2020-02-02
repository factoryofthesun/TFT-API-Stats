import os
import glob
import pandas as pd
import requests
from datetime import datetime
import json
import time
from config import PATH_GDRIVE_MAIN_DIR, PATH_GDRIVE_JSON_DIR

#TODO: CREATE FUNCTIONALITY FOR APPENDING NEW DATA
#Functions
def processMatchJson(match_data, gameid):
    #Every player end-game state is a single row in the return dataframe
    flat_data_list = []
    game_dtime = datetime.fromtimestamp(match_data['info']['game_datetime']/1000.0)
    game_version = match_data['info']['game_version']
    region = gameid.split('_')[0]
    for partic in match_data['info']['participants']:
        #TODO: ASSIGN ITEMS TO CHAMPIONS
        puuid = partic['puuid']
        place = partic['placement']
        #Sort lists for easiser comparison in CSV ouput
        traitlist = sorted([t['name']+str(t['tier_current']) for t in partic['traits'] if t['tier_current'] > 0])
        unitlist = sorted([unit['name'] + str(unit['tier']) for unit in partic['units']])
        dmg = partic['total_damage_to_players']
        if unitlist:
            num_units = len(unitlist)
        else:
            num_units = 0
        partic_data = [game_version, game_dtime, gameid, region, puuid, place, traitlist, unitlist, dmg, num_units]
        flat_data_list.append(partic_data)
    flatdf = pd.DataFrame(flat_data_list, columns = ['Game Version','Game Date','GameID',
                                                    'Region','PUUID', 'Place', 'Traits',
                                                    'Units', 'Damage', 'TeamSize'])
    return flatdf

#Process JSON match data and create groups of compositions and placement frequencies
df_list = []

for f in glob.glob(os.path.join(PATH_GDRIVE_JSON_DIR, '*.json')):
    with open(f, 'r') as file:
        match_data = json.load(file)
        alt_path = PATH_GDRIVE_JSON_DIR[:-1] + '\\'
        gameid = f.replace(alt_path,"")
        gameid = gameid.replace('.json',"")
        processed_data = processMatchJson(match_data, gameid)
        df_list.append(processed_data)

final_df = pd.concat(df_list, ignore_index = True, sort = False)
final_df.to_csv(PATH_GDRIVE_MAIN_DIR + 'compositions_data.csv', index = False)

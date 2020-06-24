"""

Script for aggregating all the individual match data JSONs.
Our output file will be compositions_data.pickle in order to preserve object information (e.g. lists)

"""
import os
import sys
import glob
import pandas as pd
import requests
from datetime import datetime
import json
import time
from config import PATH_GDRIVE_MAIN_DIR, PATH_GDRIVE_JSON_DIR

#TODO: CREATE FUNCTIONALITY FOR APPENDING NEW DATA
#TODO: MAKE THIS MORE EFFICIENT BY APPLYING PANDAS READ_JSON FUNCTION DIRECTLY (MAYBE???)
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
        traitlist = sorted([t['name'].replace("Set3_", "") +str(t['tier_current']) for t in partic['traits'] if t['tier_current'] > 0])

        # Units will be list of dictionaries: [{name, item, tier}, ...]
        unit_names = [unit['name'] if unit['name'] != '' else unit['character_id'].replace("TFT3_", "") for unit in partic['units']]
        if len(unit_names) == 0:
            unit_dict = []
        else:
            unit_tiers = [unit['tier'] for unit in partic['units']]
            unit_items = [unit['items'] for unit in partic['units']]
            assert len(unit_names) == len(unit_tiers) == len(unit_items)
            unit_dict = [{'name': name, 'tier': tier, 'items': items} for name, tier, items in zip(unit_names, unit_tiers, unit_items)]

        dmg = partic['total_damage_to_players']
        num_units = len(unit_dict)
        partic_data = [game_version, game_dtime, gameid, region, puuid, place, traitlist, unit_dict, dmg, num_units]
        flat_data_list.append(partic_data)
    flatdf = pd.DataFrame(flat_data_list, columns = ['Game Version','Game Date','GameID',
                                                    'Region','PUUID', 'Place', 'Traits',
                                                    'Units', 'Damage', 'TeamSize'])
    return flatdf

if __name__ == "__main__":
    # Process JSON match data and create groups of compositions and placement frequencies
    set_num = sys.argv[1]
    df_list = []
    for f in glob.glob(os.path.join(PATH_GDRIVE_JSON_DIR, f'{set_num}/*.json')):
        with open(f, 'r') as file:
            match_data = json.load(file)
            extended_path = PATH_GDRIVE_JSON_DIR + f'{set_num}\\'
            gameid = f.replace(extended_path,"")
            gameid = gameid.replace('.json',"")
            processed_data = processMatchJson(match_data, gameid)
            df_list.append(processed_data)

    final_df = pd.concat(df_list, ignore_index = True, sort = False)

    final_df.to_pickle(f'{PATH_GDRIVE_MAIN_DIR}{set_num}_compositions_data.pkl')

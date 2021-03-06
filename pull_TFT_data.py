'''
This script pulls every match played by every "high tier" player (i.e. Challenger, GM, Master) since the most recent patch,
and saves each match result in a JSON format.
'''

import os
import pandas as pd
import requests
from datetime import datetime
import json
import time
import random
from datetime import datetime
from config import PATH_GDRIVE_MAIN_DIR, PATH_GDRIVE_JSON_DIR

patch_date = datetime(2020, 6, 11) # TODO: FILTER BY GAME VERSION INSTEAD OF DATE (different regions have patches hit at different times)
API_KEY = os.environ.get("API_KEY")
API_KEY_SUFFIX = '?api_key=' + API_KEY

PLATFORM_DICT = {'BR1':'br1.api.riotgames.com',
                'EUN1':'eun1.api.riotgames.com',
                'EUW1':'euw1.api.riotgames.com',
                'JP1':'jp1.api.riotgames.com',
                'KR':'kr.api.riotgames.com',
                'LA1':'la1.api.riotgames.com',
                'LA2':'la2.api.riotgames.com',
                'NA1':'na1.api.riotgames.com',
                'OC1':'oc1.api.riotgames.com',
                'TR1':'tr1.api.riotgames.com',
                'RU':'ru.api.riotgames.com'}

REGIONAL_DICT = {'AMERICAS':'americas.api.riotgames.com',
                'ASIA':'asia.api.riotgames.com',
                'EUROPE':'europe.api.riotgames.com'}

AMERICAS_PLATFORMS = ['NA1', 'BR1', 'LA1', 'LA2', 'OC1']
ASIA_PLATFORMS = ['KR', 'JP1']
EUROPE_PLATFORMS = ['EUN1', 'EUW1', 'TR1', 'RU']

tft_league_prefix = '/tft/league/v1/'
tft_match_prefix = '/tft/match/v1/matches/'
tft_summoner_prefix = '/tft/summoner/v1/summoners/'

tft_league_dict = {'Challenger':tft_league_prefix+'challenger',
                'Grandmaster':tft_league_prefix+'grandmaster',
                'Master':tft_league_prefix+'master',
                'League from ID': tft_league_prefix+'leagues/{}',
                'Summoner League Entries':tft_league_prefix+'entries/by-summoner/{}',
                'League Entries': tft_league_prefix + 'entries/{tier}/{division}',
                }
tft_match_dict = {'Matches from PUUID':tft_match_prefix + 'by-puuid/{}/ids',
                    'Match ID Details':tft_match_prefix + '{}'}
tft_summoner_dict = {'Account ID': tft_summoner_prefix + 'by-account/{}',
                    'Summoner Name':tft_summoner_prefix + 'by-name/{}',
                    'PUUID':tft_summoner_prefix + 'by-puuid/{}',
                    'Summoner ID':tft_summoner_prefix + '{}'}

with open(f"{PATH_GDRIVE_MAIN_DIR}match_ids.txt") as f:
    existing_match_ids = [line.strip() for line in f]

match_id_file = open(f"{PATH_GDRIVE_MAIN_DIR}match_ids.txt", "a+")
#Keep in mind: Rate limits - make sure to count and set sleep pauses if necessary
def processReturnCodes(code):
    if code == 200:
        return True
    if code == 400:
        print('Bad Request')
        return False
    if code == 401:
        print("Unauthorized")
        return False
    if code == 403:
        print("Forbidden")
        return False
    if code == 404:
        print("Not found")
        return False
    if code == 415:
        print("Unsupported Media Type")
        return False
    if code == 429:
        print("Rate Limit Exceeded")
        return False
    if code == 500:
        print("Internal Server Error")
        return False
    if code == 503:
        print("Service Unavailable")
        return False
    else:
        print("Unknown Code {}".format(code))
        return False

#Get all challenger, grandmaster, and master players from all regions PUUIDs
tiers = ['Challenger', 'Grandmaster', 'Master']

fail_count = 0
match_ids = []
# Randomize the loop so we aren't just pulling brazil data everytime
for platform_key, platform_link in random.sample(PLATFORM_DICT.items(), len(PLATFORM_DICT)):
    #Ad Hoc way to skip finished regions while the script is still crashing
    #if platform_key != "NA1":
    #    continue
    if platform_key in AMERICAS_PLATFORMS:
        region = 'AMERICAS'
    elif platform_key in ASIA_PLATFORMS:
        region = 'ASIA'
    elif platform_key in EUROPE_PLATFORMS:
        region = 'EUROPE'
    else:
        raise SystemExit("ERROR: Platform key {} not found in any region.".format(platform_key))
    for tier in tiers:
        print('API tier call: https://' + platform_link + tft_league_dict[tier] + API_KEY_SUFFIX)
        tier_response = requests.get('https://' + platform_link + tft_league_dict[tier] + API_KEY_SUFFIX)
        while tier_response.status_code == 429:
            fail_count += 1
            if fail_count >= 5:
                raise SystemExit("ERROR: API request failed 5 times in a row.")
            wait_time = float(tier_response.headers['Retry-After'])
            print("WARNING: Rate Limit Exceeded...retrying request after {} seconds".format(wait_time))
            time.sleep(wait_time)
            tier_response = requests.get('https://' + platform_link + tft_league_dict[tier] + API_KEY_SUFFIX)
        code_response = processReturnCodes(tier_response.status_code)
        if not code_response:
            fail_count += 1
            if fail_count >= 5:
                raise SystemExit("ERROR: API request failed 5 times in a row.")
            pass
        fail_count = 0
        tier_dict = tier_response.json()
        # Get every PUUID for every summonerID
        for entry in tier_dict['entries']:
            summonerID = entry['summonerId']
            summoner_prefix = tft_summoner_dict['Summoner ID'].format(summonerID)
            print('API Summoner Call: https://' + platform_link + summoner_prefix + API_KEY_SUFFIX)
            summoner_response = requests.get('https://' + platform_link + summoner_prefix + API_KEY_SUFFIX)
            while summoner_response.status_code == 429:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                wait_time = float(summoner_response.headers['Retry-After'])
                print("WARNING: Rate Limit Exceeded...retrying request after {} seconds".format(wait_time))
                time.sleep(wait_time)
                summoner_response = requests.get('https://' + platform_link + summoner_prefix + API_KEY_SUFFIX)
            code_response = processReturnCodes(summoner_response.status_code)
            if not code_response:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                continue
            fail_count = 0
            summoner_dict = summoner_response.json()
            puuid = summoner_dict['puuid']
            # Get every match ID from PUUID - gotta use regional routing values
            region_link = REGIONAL_DICT[region]
            matches_prefix = tft_match_dict['Matches from PUUID'].format(puuid)
            print('API Match List call: https://' + region_link + matches_prefix + API_KEY_SUFFIX)
            # Get last 1000 games
            match_response = requests.get('https://' + region_link + matches_prefix + API_KEY_SUFFIX + "&?count=1000" )
            while match_response.status_code == 429:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                wait_time = float(match_response.headers['Retry-After'])
                print("WARNING: Rate Limit Exceeded...retrying request after {} seconds".format(wait_time))
                time.sleep(wait_time)
                match_response = requests.get('https://' + region_link + matches_prefix + API_KEY_SUFFIX)
            code_response = processReturnCodes(match_response.status_code)
            if not code_response:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                continue
            fail_count = 0
            match_list = match_response.json()
            match_list = [i for i in match_list if i not in existing_match_ids]
            if not match_list:
                continue
            # Run through match ids and get details
            for match_id in match_list:
                match_id_prefix = tft_match_dict['Match ID Details'].format(match_id)
                match_detail_response = requests.get("https://" + region_link + match_id_prefix + API_KEY_SUFFIX)
                while match_detail_response.status_code == 429:
                    fail_count += 1
                    if fail_count >= 5:
                        raise SystemExit("ERROR: API request failed 5 times in a row.")
                    wait_time = float(match_detail_response.headers['Retry-After'])
                    print("WARNING: Rate Limit Exceeded...retrying request after {} seconds".format(wait_time))
                    time.sleep(wait_time)
                    match_detail_response = requests.get("https://" + region_link + match_id_prefix + API_KEY_SUFFIX)
                code_response = processReturnCodes(match_detail_response.status_code)
                if not code_response:
                    fail_count += 1
                    if fail_count >= 5:
                        raise SystemExit("ERROR: API request failed 5 times in a row.")
                    continue
                fail_count = 0
                match_details = match_detail_response.json()

                # Matches are list starting with most recent - if match date is before the cutoff patch date, then can skip the rest
                game_dtime = datetime.fromtimestamp(match_details['info']['game_datetime']/1000.0)
                if game_dtime <= patch_date:
                    print("Reached the last game for Set 3.5 for summoner {}".format(summoner_dict['name']))
                    break
                #Add relevant metadata to JSON: summoner name, id, patch, etc
                match_details["metadata"]['summoner_id'] = summonerID
                match_details["metadata"]['puuid'] = puuid
                match_details["metadata"]['summoner_name']  = summoner_dict['name']
                #Save match data as its own JSON file
                json_file_name = PATH_GDRIVE_JSON_DIR + "Set3.5/" + match_id + '.json'
                with open(json_file_name, 'w') as fp:
                    json.dump(match_details, fp)
                print("Saved {} successfully.".format(json_file_name))
                match_id_file.write(f"{match_id}\n")
            #temp_match_df = pd.DataFrame({'Match IDs':match_ids})
            #temp_match_df.to_csv(PATH_GDRIVE_MAIN_DIR + 'match_ids.csv', index = False, header=False, mode = 'a')
    print("Region {} completed.".format(platform_key))

match_id_file.close()

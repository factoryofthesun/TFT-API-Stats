import os
import pandas as pd
import requests
from datetime import datetime
import json
import time

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
                    'Match ID Details':tft_match_prefix + 'matches/{}'}
tft_summoner_dict = {'Account ID': tft_summoner_prefix + 'by-account/{}',
                    'Summoner Name':tft_summoner_prefix + 'by-name/{}',
                    'PUUID':tft_summoner_prefix + 'by-puuid/{}',
                    'Summoner ID':tft_summoner_prefix + '{}'}
#Initial outline: get all NA challenger players with tft-league api, get all PUUIDs with
#tft-summoner api, get all match data with tft-match api
#TODO: Can you batch request match history data? Specify time period of matches to get?
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
platform_summoner_ids = {}
platform_puuids = {}
platform_matchids = {}

fail_count = 0
for platform_key, platform_link in PLATFORM_DICT.items():
    if platform_key in AMERICAS_PLATFORMS:
        region = 'AMERICAS'
    elif platform_key in ASIA_PLATFORMS:
        region = 'ASIA'
    elif platform_key in EUROPE_PLATFORMS:
        region = 'EUROPE'
    else:
        raise SystemExit("ERROR: Platform key {} not found in any region.".format(platform_key))
    for tier in tiers:
        tier_response = requests.get('https://' + platform_link + tft_league_dict[tier] + API_KEY_SUFFIX)
        while tier_response.status_code == 429:
            fail_count += 1
            if fail_count >= 5:
                raise SystemExit("ERROR: API request failed 5 times in a row.")
            wait_time = float(tier_response.headers['Retry-After'])
            time.sleep(wait_time)
            tier_response = requests.get('https://' + platform_link + tft_league_dict[tier] + API_KEY_SUFFIX)
        code_response = processReturnCodes(tier_response.status_code)
        if not code_response:
            fail_count += 1
            if fail_count >= 5:
                raise SystemExit("ERROR: API request failed 5 times in a row.")
            pass
        fail_count = 0
        tier_dict = json.loads(tier_response)
        tier_dict_entries = tier_dict['entries']
        #Get every PUUID for every summonerID
        for entry in tier_dict['entries']:
            summonerID = entry['summonerId']
            summoner_prefix = tft_summoner_dict['Summoner ID'].format(summonerID)
            summoner_response = requests.get('https://' + platform_link + summoner_prefix + API_KEY_SUFFIX)
            while summoner_response.status_code == 429:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                wait_time = float(summoner_response.headers['Retry-After'])
                time.sleep(wait_time)
                summoner_response = requests.get('https://' + platform_link + summoner_prefix + API_KEY_SUFFIX)
            code_response = processReturnCodes(summoner_response.status_code)
            if not code_response:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                pass
            fail_count = 0
            summoner_dict = json.loads(summoner_response)
            puuid = summoner_response['puuid']
            #Get every match ID from PUUID - gotta use regional routing values
            region_link = REGIONAL_DICT['region']
            matches_prefix = tft_match_dict['Matches from PUUID'].format(puuid)
            match_response = requests.get('https://' + region_link + matches_prefix + API_KEY_SUFFIX)
            while match_response.status_code == 429:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                wait_time = float(match_response.headers['Retry-After'])
                time.sleep(wait_time)
                match_response = requests.get('https://' + region_link + matches_prefix + API_KEY_SUFFIX)
            code_response = processReturnCodes(match_response.status_code)
            if not code_response:
                fail_count += 1
                if fail_count >= 5:
                    raise SystemExit("ERROR: API request failed 5 times in a row.")
                pass
            fail_count = 0
            match_list = json.loads(match_response)
            #Run through match ids and get details
            for match_id in match_list:
                match_id_prefix = tft_match_dict['Match ID Details'].format(match_id)
                match_detail_response = requests.get("https://" + region_link + match_id_prefix + API_KEY_SUFFIX)
                while match_detail_response.status_code == 429:
                    fail_count += 1
                    if fail_count >= 5:
                        raise SystemExit("ERROR: API request failed 5 times in a row.")
                    wait_time = float(match_detail_response.headers['Retry-After'])
                    time.sleep(wait_time)
                    match_detail_response = requests.get("https://" + region_link + match_id_prefix + API_KEY_SUFFIX)
                code_response = processReturnCodes(match_detail_response.status_code)
                if not code_response:
                    fail_count += 1
                    if fail_count >= 5:
                        raise SystemExit("ERROR: API request failed 5 times in a row.")
                    pass
                fail_count = 0
                match_details = json.loads(match_response)
                #TODO: RECORD MATCH DETAILS IN A SYSTEMATIC WAY 





#Expansion: collect all high level player match history data across all regions
#Side visualization project: create infographic of the meta comps by region if significantly different

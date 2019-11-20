import os
import pandas as pd
import requests

API_KEY = os.environ.get("API_KEY")
PLATFORM_BASE = 'na1.api.riotgames.com'
REGIONAL_BASE = 'americas.api.riotgames.com'

#Initial outline: get all NA challenger players with tft-league api, get all PUUIDs with
#tft-summoner api, get all match data with tft-match api
#TODO: Can you batch request match history data? Specify time period of matches to get?
#Keep in mind: Rate limits - make sure to count and set sleep pauses if necessary

#Expansion: collect all high level player match history data across all regions
#Side visualization project: create infographic of the meta comps by region if significantly different

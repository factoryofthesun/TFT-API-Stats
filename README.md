# TFT-API-Stats

To use the script to pull data, we use a 24-hr Riot API key (need to re-generate every 24 hours). Go to
`https://developer.riotgames.com/apis#tft-match-v1` to get it. Then, assign your key to an os environment variable
`API_KEY`, so the script can use the key.

Next, create a file named "config.py" in the repo's root directory, with the following contents, replacing
the items in brackets with your actual directory:
```
PATH_GDRIVE_MAIN_DIR = "{repo root directory}"
PATH_GDRIVE_JSON_DIR = f"{PATH_GDRIVE_MAIN_DIR}/Match JSON"
```

The basic flow of script usage is as follows:
1. Run `pull_TFT_data.py` to query Riot database and save match JSONs into `PATH_GDRIVE_JSON_DIR`
2. Run `process_json.py` to build the comprehensive CSV "compositions_data.csv" and save into `PATH_GDRIVE_MAIN_DIR`.
   This comprehensive CSV contains all relevant data regarding a players end-game composition (e.g. champions, traits)
   and stats (e.g. place).
3. Run `summary_stats` to give a quick summary of the best trait compositions.
4. Run `summary_trait_composition_top_2.py` to generate CSV for only the top 2 players of each match (1st and 2nd place)
5. Run `predict_winning_trait_composition.py` to run a neural net on the aforementioned "top 2" data.
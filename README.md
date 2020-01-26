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
"""
Script to maintain a running list of pulled match ids so that we're not making repeated match API calls.
"""

import pandas as pd
import os
from config import PATH_GDRIVE_JSON_DIR, PATH_GDRIVE_MAIN_DIR

# Recursive walk through JSON storage directory
filenames = []
for dirname, _, files in os.walk(PATH_GDRIVE_JSON_DIR):
    filenames.extend(files)

filenames_clean = [f.replace('.json', "") for f in filenames]

with open(f"{PATH_GDRIVE_MAIN_DIR}match_ids.txt", 'w') as f:
    f.writelines(f"{id}\n" for id in filenames_clean)

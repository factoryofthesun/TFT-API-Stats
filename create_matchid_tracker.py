import pandas as pd
import os
from config import PATH_GDRIVE_JSON_DIR, PATH_GDRIVE_MAIN_DIR

#Look through existing GDRIVE directory and create CSV of existing games so we don't make redundant requests later
filenames = os.listdir(PATH_GDRIVE_JSON_DIR)
filenames_clean = [f.replace('.json', "") for f in filenames]
df = pd.DataFrame(data = {'Match IDs':filenames_clean})
df.to_csv(PATH_GDRIVE_MAIN_DIR + 'match_ids.csv', index = False)

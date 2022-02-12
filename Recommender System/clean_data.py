# Script to clean the data for processing in both recommender systems

import pandas as pd
import numpy as np

recipes = pd.read_csv("../data-recipes/RAW_recipes.csv")
interactions = pd.read_csv("../data-recipes/RAW_interactions.csv")
print(f"rows in recipes: {recipes.shape[0]}, rows in interactions: {interactions.shape[0]}")
# Drop unneccessary columns

# Rename columns
recipes = recipes.rename(columns={"id":"item_id"})
interactions = interactions.rename(columns={"recipe_id":"item_id"})

# Fill in 0s or empty strings
recipes.dropna(inplace=True)
interactions.dropna(inplace=True)

# Reduce the size of the datasets
rows = 3000
recipes = recipes.sample(rows, random_state=42)
interactions = interactions.loc[interactions['item_id'].isin(recipes['item_id'])]

print(f"rows in clean_recipes: {recipes.shape[0]}, rows in clean_interactions: {interactions.shape[0]}")

# Save the datasets for use by the RSes
recipes.to_csv("data/clean_recipes.csv", index=False)
interactions.to_csv("data/clean_interactions.csv",  index=False)
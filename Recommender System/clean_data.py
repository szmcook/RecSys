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
# print(recipes.isna().sum())
recipes.dropna(inplace=True)
interactions.dropna(inplace=True)

# Reduce the size of the datasets
rows = 3000
recipes = recipes.sample(rows, random_state=42)

# Keep only interactions relating to the recipes
interactions = interactions.loc[interactions['item_id'].isin(recipes['item_id'])]

# Keep only interactions from users with at least n interactions
n = 5
users_interactions_count = interactions.groupby(['user_id', 'item_id']).size().groupby('user_id').size()
users_with_enough_interactions = users_interactions_count[users_interactions_count >= n].reset_index()[['user_id']]
interactions = interactions.merge(users_with_enough_interactions, how = 'right', left_on = 'user_id', right_on = 'user_id')

# Keep only recipes which appear in the interactions
recipes = recipes[recipes['item_id'].isin(interactions['item_id'])]

# Order the interactions chronologically for train_test split
interactions = interactions.sort_values(by='date')

print(f"rows in recipes: {recipes.shape[0]}, rows in interactions: {interactions.shape[0]}")

splt = int(interactions.shape[0]*0.8)
interactions_train = interactions[:splt]
interactions_test = interactions[splt:]

assert interactions_train.shape[0] + interactions_test.shape[0] == interactions.shape[0]

# Save the datasets for use by the RSes
recipes.to_csv("data/recipes.csv", index=False)
interactions_train.to_csv("data/interactions_train.csv",  index=False)
interactions_test.to_csv("data/interactions_test.csv",  index=False)
import pandas as pd
recipes = pd.read_csv('data/clean_recipes.csv')
n = recipes.nutrition
n = n.apply(eval)

n_df = pd.DataFrame.from_dict(dict(zip(n.index, n.values)), orient='index', columns=['calories', 'total_fat_PDV', 'sugar_PDV', 'sodium_PDV', 'protein_PDV', 'saturated_fat', 'unknown'])

print(n_df.mean())
print(n_df.std())

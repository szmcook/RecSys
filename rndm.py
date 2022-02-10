import pandas as pd

class RandomRecommender:
    
    type = 'Random recommender'
    
    def __init__(self):
        pass

    def load_data():
        pass

    def prepare_data():
        pass

    def create_model(load=False):
        pass
        
    def recommend_items(self, user_id, items_to_ignore=[], n=10):

        items_df1 = pd.read_csv('data-beer-liquor-wine/wine reviews.csv')
        items_df2 = pd.read_csv('data-beer-liquor-wine/447_1.csv').drop(columns=['primaryCategories', 'quantities'])
        items_df = pd.concat([items_df1, items_df2])
        items_df["descriptions"] = items_df["descriptions"].apply(lambda s: 'Carmex' if 'Carmex' in str(s) else s)
        items_df = items_df[items_df['descriptions'] != 'Carmex']

        recommendations = items_df.sample(n)
        recommendations = list(recommendations['name'])
        return recommendations
import pandas as pd


class ContentRecommender:
    def __init__(self, active_user_id, load=False):
        """Constructor for CollaborativeRecommender"""
        self.type = 'Content Based Filter'
        self.user_id = active_user_id
        self.model = RecipesModel()
        self.model.load_weights("CF")
        
    def recommend_items(self, n=5):
        """Recommend a set of n items to the active user"""

        items = self.recipes['name'].to_list() # All the recipe ids from the dataset as a list
        predictions = []
        for item in items:
            predictions.append( (item, model(self.user_id, item)) )

        p = pd.DataFrame(predictions, columns=["name", "Predicted Rating"])

        p.sort_values(by='Predicted Rating', ascending=False, inplace=True)
        p = p[:n]

        while p.iloc[-1]['Predicted Rating'] < 4:
            p = p[:-1]
        if p.shape[0] < n:
            print(f"We were unable to find {n} items which we believe you'll like, we have shortened the list of recommendations to ensure a high quality of recommendations\n")

        p = p.drop(columns=['item_id', 'nutrition']).rename(columns={"name":"Recipe Name","minutes":"Minutes to prepare"})
        return p

# recommender = ContentRecommender(599450)
# recommendations = recommender.recommend_items(n=5)
# print(recommendations)
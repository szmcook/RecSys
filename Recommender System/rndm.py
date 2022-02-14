import pandas as pd

class RandomRecommender():
    def __init__(self):
        super().__init__()
        self.type = 'Collaborative Filter'
        self.recipes, self.ratings = self.load_data()


    def load_data(self):
        """Load the dataset from files"""
        recipes = pd.read_csv('../data/recipes.csv')
        ratings = pd.read_csv('../data/interactions_train.csv')
        return recipes, ratings


    def prepare_data(self):
        pass


    def create_model(self):
        pass


    def recommend_items(self, user_id, items_to_ignore=[], n=10):
        recommendations = self.data.sample(n)
        recommendations = recommendations['name']
        return recommendations
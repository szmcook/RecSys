import pandas as pd

class CollaborativeRecommender:
    def __init__(self):
        super().__init__()
        self.type = 'Collaborative Filter'
        self.interactions = self.load_data()


    def load_data(self):
        """Load a dataset from files"""
        return pd.read_csv('../data/clean_interactions.csv')


    def prepare_data(self):
        self.R = self.data


    def create_model(self):
        pass


    def recommend_items(self, user_id, items_to_ignore=[], n=10):
        print(type(self.data))
        recommendations = ['cake']*n
        return recommendations

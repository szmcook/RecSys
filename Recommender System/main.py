from CF import CollaborativeRecommender
from CBF_basic import ContentRecommender
from rndm import RandomRecommender
from add_user import add_user

import pandas as pd
interactions = pd.read_csv('data/clean_interactions.csv')

global user_id
user_id = None
# ACTIVE_USER = 599450


def sign_in():
    """Signs the user in"""
    user_id = None
    print("Please enter your username to sign in:")
    while user_id == None:
        u = int(input())
        # Check the user_id isn't in use already
        if u in interactions['user_id'].unique():
            user_id = u
        else:
            print(f"That username is not recognised, if you wish to add a new user please restart the system")

    print(f"signed in as {user_id}")


def recommend():
    """Asks the user to select an RS method and then makes recommendations"""
    res = input('Would you like to use the Deep Content Based filter (enter d) the Context-Aware Collaborative Filter (enter c)? ')
    if res == 'c':
        l = input("Would you like to load the saved model (enter 'l' or train a new one (enter 't')? ")
        load = True if l == 'l' else False
        recommender = CollaborativeRecommender(user_id, load)
    elif res == 'd':
        recommender = ContentRecommender(user_id)
    else:
        print("Please enter either a 'c' or a 'd'\nQuitting system")

    n = int(input('How many items would you like to be recommended? Please enter an integer: '))
    recommendations = recommender.recommend_items(n=n)
    print(f"The recommended items are:\n{recommendations}")


def main():
    if user_id == None:
        activity = input('Sign in (s) or Add user (u)? ')
    else:
        activity = 'r'

    if activity == 's':
        sign_in()
        recommend()
    elif activity == 'u':
        add_user()
    if activity == 'r':
        recommend()


if __name__ == "__main__":
    main()
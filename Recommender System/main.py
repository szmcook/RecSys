from CF import CollaborativeRecommender
from CBF_basic import ContentRecommender
from rndm import RandomRecommender
from add_user import add_user

import pandas as pd
interactions = pd.read_csv('data/interactions_train.csv')


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
    return user_id


def recommend(user_id):
    """Asks the user to select an RS method and then makes recommendations"""
    res = input('Would you like to use the Collaborative filter (enter 1) or the Context-Aware Context Based Filter (enter 2)?\n')
    l = input("\nIf you haven't recently added a new user, it is strongly recommended that you load a saved model to reduce recommendation times.\nWould you like to load the saved model (enter 'l') or train a new one (enter 't')?\n")
    load = True if l == 'l' else False
    if res == '1':
        recommender = CollaborativeRecommender(load)
    elif res == '2':
        recommender = ContentRecommender(load)
    else:
        print("Please enter either a '1' or a '2'\nQuitting system")
        return

    print("We're producing your top 5 recommendations")
    recommendations = recommender.recommend_items(user_id, n=20)
    print(f"The recommended items are:\n{recommendations.head(5)}")

    more = input("\nWe hope you can see something there that you'll like! If you'd like to see 5 more recommendations please enter '5', otherwise enter any other key\n")
    if more == '5':
        print(recommendations.head(10))
        


def main():
    activity = input('Sign in (s) or Add user (u)? ')
    
    if activity == 's':
        user_id = sign_in()
        # user_id = 599450
        recommend(user_id)
    elif activity == 'u':
        add_user()


if __name__ == "__main__":
    main()
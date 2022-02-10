from CF import CollaborativeRecommender
from CBF import ContentRecommender
from rndm import RandomRecommender
from pplr import PopularRecommender
from add_user import add_user


global ACTIVE_USER
ACTIVE_USER = None
ACTIVE_USER = 'szm'


def sign_in():
    """Takes user input, returns a UID"""
    username = input('Please enter your username: ')
    ACTIVE_USER = username
    print(f"signed in as {ACTIVE_USER}")


def recommend():
    """Recommend an RS method"""
    # res = input('Would you like to use the Deep Content Based filter (enter d) the Context-Aware Collaborative Filter (enter c)? ')
    # recommender = ContentRecommender() if res == 'd' else CollaborativeRecommender()
    recommender = PopularRecommender()
    # n = int(input('How many items would you like to be recommended? Please enter an integer: '))
    n = 10
    recommendations = recommender.recommend_items(ACTIVE_USER, n=n)
    print(f"The recommended items are: {recommendations}")


def main():
    if ACTIVE_USER == None:
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
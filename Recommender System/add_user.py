import pandas as pd
from datetime import datetime

recipes = pd.read_csv('data/clean_recipes.csv')
interactions = pd.read_csv('data/clean_interactions.csv')


def add_user():
    """Add a new user to the system"""
    # Take a username
    user_id = None
    print("Please enter an integer to be used as your user_id: ")
    while user_id == None:
        i = int(input())
        # Check the user_id isn't in use already
        if i in interactions['user_id'].unique():
            print(f"{i} is already in use, please enter another integer")
        else:
            user_id = i

    # Take some recommendations for 5 random items
    print("""
In order to build your user profile we need you to rank some random items.
If you have not tried the item please move on to the next one.
This data will be stored in order to provide you with personalised recommendations in the future.
We also store the current date and time.
No other data will be stored, to protect your privacy

If you have not tried the suggested item you are free to estimate a rating based on the name or enter 'n' to see the next item.
"""
    )

    items_rated = 0

    while items_rated < 5:
        # pick an item
        item = recipes.sample(1)
        # get a rating
        rating = input(f"Please provide a rating for {item['name'].iloc[0]}: ")

        if rating == 's':
            print('skipping item')
            continue
        else:
            rating = int(rating)
            date = datetime.now().strftime("%Y-%m-%d")
            # save to RAW_interactions.csv
            s = f"{user_id},{item['item_id'].iloc[0]},{date},{rating},review\n"
            with open('data/clean_interactions.csv', "a") as f:
                f.write(s)

            print("\nThank you for your rating!\n")
            items_rated += 1

    print(f"Thank you for providing these ratings, your data has been stored and you may now log in to the system as {user_id}")
Instructions for using my Recommender System

The Recommender system can be started by running the main.py file ($ python3 main.py)

The user will then be asked whether they wish to sign in or add a new user. In order to make personalised recommendations it is necessary to sign in.

If the user chooses to sign in (this option can be selected by inputting the character 's') they will be asked for their username.
Once an integer username has been entered, the system will begin the recommendation process.
If the user instead chooses to add a new user, the system will ask them to provide a user_id and then rate 5 random items in order to build a user profile, this is all done through the command prompt.

The recommendation process begins by asking the user to choose whether they would like to use the Deep Content Based Filter (selected by entering 'd') or the Collaborative Filter (selected by entering 'c').
If the Collaborative Filter is selected, the user is asked whether they'd like to use the current saved model or train a new model, this option is provided because training a new model can take some time but is necessary to include the new data from a recently added user.
The system then asks the user for an integer number of recommendations to produce.

Once all this has been entered, a list of personalised recommendations for the user is printed to the terminal.
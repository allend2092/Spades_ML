import random

# Define a list of suits and ranks for the cards
suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

# Define a card deck object with a dictionary inside
class CardDeck:
    def __init__(self):
        # Create an empty dictionary to store the cards
        self.cards = {}
        # Loop through the suits and ranks and assign a key and a value to each card
        key = 1
        for suit in suits:
            for rank in ranks:
                # The value of each card is a tuple of rank and suit
                self.cards[key] = (rank, suit)
                # Increment the key by 1
                key += 1

    # Define a method to print the cards in the deck
    def print_cards(self):
        # Loop through the keys and values in the dictionary and print them
        for key, value in self.cards.items():
            print(f"{key}: {value[0]} of {value[1]}")

    def shuffle_cards(self):
        # Convert the dictionary values to a list and shuffle it
        card_values = list(self.cards.values())
        random.shuffle(card_values)

        # Reassign the shuffled values back to the dictionary
        for key in self.cards.keys():
            self.cards[key] = card_values[key - 1]



 # define a scoreboard class that keeps track of the score and other information
class Scoreboard:
    def __init__(self, team_name, opponent_name):
        # data member team_name: variable storing the name of the user's team
        self.team_name = team_name
        # data member opponent_name: variable storing the name of the opponent's team
        self.opponent_name = opponent_name
        # data member team_score: variable storing the score of the user's team
        self.team_score = 0
        # data member opponent_score: variable storing the score of the opponent's team
        self.opponent_score = 0
        # data member round_number: variable storing the number of rounds played
        self.round_number = 0


class game_conditions:
    def __init__(self, human_players, bot_players, winning_score):
        self.number_of_players = human_players
        self.ai_opponents = bot_players
        self.threshold_score = winning_score
        self.blind_nil_allowed = False
        self.current_round = 0
        self.whose_turn = [1,2,3,4]


def welcome():
    print('''
 __    __    ___  _         __   ___   ___ ___    ___      ______   ___        _____ ____    ____  ___      ___  _____ __ 
|  T__T  T  /  _]| T       /  ] /   \ |   T   T  /  _]    |      T /   \      / ___/|    \  /    T|   \    /  _]/ ___/|  T
|  |  |  | /  [_ | |      /  / Y     Y| _   _ | /  [_     |      |Y     Y    (   \_ |  o  )Y  o  ||    \  /  [_(   \_ |  |
|  |  |  |Y    _]| l___  /  /  |  O  ||  \_/  |Y    _]    l_j  l_j|  O  |     \__  T|   _/ |     ||  D  YY    _]\__  T|__j
l  `  '  !|   [_ |     T/   \_ |     ||   |   ||   [_       |  |  |     |     /  \ ||  |   |  _  ||     ||   [_ /  \ | __ 
 \      / |     T|     |\     |l     !|   |   ||     T      |  |  l     !     \    ||  |   |  |  ||     ||     T\    ||  T
  \_/\_/  l_____jl_____j \____j \___/ l___j___jl_____j      l__j   \___/       \___jl__j   l__j__jl_____jl_____j \___jl__j
                                                                                                                          
    ''')


def start_game():
    print('''
  _____ ______   ____  ____  ______  ____  ____    ____       ____   ____  ___ ___    ___  __ 
 / ___/|      T /    T|    \|      Tl    j|    \  /    T     /    T /    T|   T   T  /  _]|  T
(   \_ |      |Y  o  ||  D  )      | |  T |  _  YY   __j    Y   __jY  o  || _   _ | /  [_ |  |
 \__  Tl_j  l_j|     ||    /l_j  l_j |  | |  |  ||  T  |    |  T  ||     ||  \_/  |Y    _]|__j
 /  \ |  |  |  |  _  ||    \  |  |   |  | |  |  ||  l_ |    |  l_ ||  _  ||   |   ||   [_  __ 
 \    |  |  |  |  |  ||  .  Y |  |   j  l |  |  ||     |    |     ||  |  ||   |   ||     T|  T
  \___j  l__j  l__j__jl__j\_j l__j  |____jl__j__jl___,_j    l___,_jl__j__jl___j___jl_____jl__j
                                                                                              
    ''')

def how_many_players():
    # Ask the user for the number of human players
    human_players_input = input("How many human players will be playing? ")

    # Convert the input to an integer
    try:
        human_players = int(human_players_input)

    except ValueError:
        print("Please enter a valid number.")
        # Handle the error or exit the program
        # For example, you might want to ask again or set a default value

    return human_players


def how_many_points():
    # Ask the user for the number of human players
    points = input("What should the point limit be? ")

    # Convert the input to an integer
    try:
        points = int(points)

    except ValueError:
        print("Please enter a valid number.")
        # Handle the error or exit the program
        # For example, you might want to ask again or set a default value

    return points


class Player:
    def __init__(self, name):
        # data member team: variable stating if this bot is a user teammate or opponent
        #self.team = team
        # data member name: variable storing the name of the bot
        self.name = name
        # data member hand: list storing the cards that the bot has in its hand
        self.hand = []
        # data member score: variable storing the number of tricks that the bot has won
        self.score = 0

    def make_bid(self, bids):
        """
        Ask this player to make a bid at the start of a round.

        bids: a dict containing all bids so far, with keys 0-3 (player_id) and values 0-13 or "B" for Blind Nill.
        return value: An integer between 0 (a Nill bid) and 13 minus the teammate's bid (inclusive).
        """
        raise NotImplementedError()



class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

        # Additional initialization for HumanPlayer

class BotPlayer(Player):
    def __init__(self, name, difficulty_level):
        super().__init__(name)
        self.difficulty_level = difficulty_level
        # Additional initialization for BotPlayer

    def make_bid(self):
        # Bot player makes a bid based on some algorithm
        # For simplicity, let's just return a random bid
        import random
        bid = random.randint(1, 13)
        return bid


def create_players(num_human_players, total_players=4):
    players = []
    for i in range(num_human_players):
        players.append(HumanPlayer(f"Human {i+1}"))
    for i in range(total_players - num_human_players):
        players.append(BotPlayer(f"Bot {i+1}", 'hard'))
    return players


def main_game_loop(players):
    game_over = False
    while not game_over:
        for player in players:
            if isinstance(player, HumanPlayer):
                # Human player actions (e.g., make a bid, play a card)
                action = input("What do you want to do? ")
                if action == 'end':
                    game_over = True
                    break
            elif isinstance(player, BotPlayer):
                # Bot player actions (e.g., automated bid, play a card)
                pass
            # Other game logic (e.g., determine winner of a trick)
        # Check for end of game conditions


# Define a main function that instantiates the card deck and runs the print cards function
def main():

    welcome()
    human_players = how_many_players()
    points = how_many_points()
    start_game()

    game_parameters = game_conditions(human_players= human_players, bot_players= 4-human_players, winning_score= points)

    # Create a card deck object
    deck = CardDeck()
    # Print the cards in the deck
    deck.print_cards()
    deck.shuffle_cards()
    deck.print_cards()
    players = create_players(game_parameters.ai_opponents)


    # Main game loop
    main_game_loop(players)


# Call the main function
if __name__ == "__main__":
    main()






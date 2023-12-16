

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


# define a bot that will play spades with or against the user
class GameBot:
    def __init__(self, team, name):
        # data member team: variable stating if this bot is a user teammate or opponent
        self.team = team
        # data member name: variable storing the name of the bot
        self.name = name
        # data member hand: list storing the cards that the bot has in its hand
        self.hand = []
        # data member score: variable storing the number of tricks that the bot has won
        self.score = 0

    def bid(self):
        # function bid: returns the number of tricks that the bot expects to win based on its hand and the game rules
        # TODO: implement some logic to estimate the bid based on the cards in hand
        # For example, you can count how many high cards (A, K, Q, J) or spades you have and adjust your bid accordingly
        # You can also consider other factors like your team's bid, your opponents' bid, etc.
        # For simplicity, let's assume that the bot bids randomly between 1 and 13
        import random
        return random.randint(1, 13)

    def play(self):
        # function play: returns a card from the bot's hand that follows the game rules and tries to win the trick
        # TODO: implement some logic to choose a card based on the game rules and strategy
        # For example, you can check what suit was led, what cards have been played, what cards are left in your hand, etc.
        # You can also consider other factors like your team's score, your opponents' score, etc.
        # For simplicity, let's assume that the bot plays randomly from its valid cards
        import random
        # Get a list of valid cards from the bot's hand based on the game rules
        valid_cards = self.get_valid_cards()
        # Choose a random card from the valid cards
        card = random.choice(valid_cards)
        # Remove the card from the bot's hand
        self.hand.remove(card)
        # Return the card
        return card

    def discard(self):
        # function discard: returns a card from the bot's hand that is not needed or has low value
        # TODO: implement some logic to choose a card based on its value and usefulness
        # For example, you can discard low cards or off-suit cards that are unlikely to win any tricks
        # You can also consider other factors like your team's bid, your opponents' bid, etc.
        # For simplicity, let's assume that the bot discards randomly from its hand
        import random
        # Choose a random card from the bot's hand
        card = random.choice(self.hand)
        # Remove the card from the bot's hand
        self.hand.remove(card)
        # Return the card
        return card

    def get_valid_cards(self):
        # helper function: returns a list of valid cards from the bot's hand based on the game rules
        # TODO: implement some logic to check what suit was led, what cards have been played, etc.
        # For simplicity, let's assume that any card is valid for now
        return self.hand.copy()


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


# Define a main function that instantiates the card deck and runs the print cards function
def main():
    # Create a card deck object
    deck = CardDeck()
    # Print the cards in the deck
    deck.print_cards()

# Call the main function
if __name__ == "__main__":
    main()



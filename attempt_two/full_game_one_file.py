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

    def deal_cards(self, players, cards_per_player):
        for _ in range(cards_per_player):
            for player in players:
                if self.cards:
                    # Get the first card key and its value
                    card_key, card_value = next(iter(self.cards.items()))
                    # Add the card to the player's hand
                    player.hand.append(card_value)
                    # Remove the card from the deck
                    del self.cards[card_key]


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
        self.whose_turn = {}
        self.team1_bid = None
        self.team2_bid = None
        self.game_play_order = []
        self.leading_suit = None
        self.spades_broken = False


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
        self.team = None
        # data member name: variable storing the name of the bot
        self.name = name
        # data member hand: list storing the cards that the bot has in its hand
        self.hand = []
        # data member score: variable storing the number of tricks that the bot has won
        self.score = 0
        self.card_played_last = None
        self.eligible_cards = []
        # Print greeting upon instantiation
        print(f"Hello, I am {self.name}. I am currently not assigned to a team.")

    def make_bid(self, bids):
        """
        Ask this player to make a bid at the start of a round.

        bids: a dict containing all bids so far, with keys 0-3 (player_id) and values 0-13 or "B" for Blind Nill.
        return value: An integer between 0 (a Nill bid) and 13 minus the teammate's bid (inclusive).
        """
        raise NotImplementedError()

    def set_team(self, team_name):
        self.team = team_name

    def play_card(self):
        "logic for how to play a card should be inserted here."
        raise NotImplementedError()

    def announce_myself(self):
        print(f"I am {self.name} and I'm on team {self.team}.")

    def determine_eligible_cards(self, leading_suit, spades_broken):
        self.eligible_cards = []
        #print(f"leading_suit: {leading_suit}")
        #print(f"Spades Broken: {spades_broken}")

        # Check if the player has cards of the leading suit
        has_leading_suit = any(card[1] == leading_suit for card in self.hand)
        #print(f"Has Leading suit: {has_leading_suit}")

        if has_leading_suit:
            # Player must play a card of the leading suit
            self.eligible_cards = [card for card in self.hand if card[1] == leading_suit]
        elif leading_suit == None and not spades_broken:
            # Player can play any card including spades if they don't have the leading suit
            self.eligible_cards = self.hand.copy()

            # If spades haven't been broken and the player has other suits, remove spades from eligible cards
            if not spades_broken and any(card[1] != "Spades" for card in self.hand):
                self.eligible_cards = [card for card in self.eligible_cards if card[1] != "Spades"]
        else:
            # Player can play any card including spades if they don't have the leading suit
            self.eligible_cards = self.hand.copy()


class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def display_cards_in_hand(self):
        if not self.hand:
            print("No cards in hand.")
            return

        # Custom sorting key for cards
        def card_sort_key(card):
            suit_order = {"Hearts": 0, "Diamonds": 1, "Clubs": 2, "Spades": 3}
            rank_order = {"Ace": 14, "King": 13, "Queen": 12, "Jack": 11, "10": 10,
                          "9": 9, "8": 8, "7": 7, "6": 6, "5": 5, "4": 4, "3": 3, "2": 2}
            return (suit_order[card[1]], rank_order[card[0]])

        # Sort the cards
        self.hand.sort(key=card_sort_key)

        # Display cards in a single line
        card_str = ", ".join(f"{card[0]} of {card[1]}" for card in self.hand)
        print(f"{self.name}'s cards: {card_str}")

    def make_bid(self, bids):
        while True:
            print(f"Current bids: {bids}")
            print("You have the following options: \n1. See cards and bid\n2. Bid blind nil\n3. Quit game")
            action = input("What do you want to do? ")

            try:
                action = int(action)
                if action == 1:
                    self.display_cards_in_hand()
                    bid = int(input("What is your bid? "))
                    # Add validation for bid here if necessary
                    return bid
                elif action == 2:
                    return 0  # Blind nil bid
                elif action == 3:
                    # Handle quitting the game
                    # You might need to signal this to the main game loop
                    return None
                else:
                    print("That's not an available option. Please select again.")
            except ValueError:
                print("Please enter a valid number.")

    def play_card(self, leading_suit, spades_broken):
        self.determine_eligible_cards(leading_suit, spades_broken)

        hand_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.hand))
        print(f"Complete hand: {hand_cards_str}")

        # Format and display the eligible cards
        eligible_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.eligible_cards))
        print(f"Eligible cards to play: {eligible_cards_str}")

        while True:
            try:
                chosen_index = int(input("Choose a card to play (by number): ")) - 1
                if 0 <= chosen_index < len(self.eligible_cards):
                    chosen_card = self.eligible_cards.pop(chosen_index)
                    self.card_played_last = chosen_card
                    self.hand.remove(chosen_card)  # Remove the chosen card from the hand
                    return chosen_card
                else:
                    print("Invalid choice. Please choose a valid card number.")
            except ValueError:
                print("Please enter a number.")


class BotPlayer(Player):
    def __init__(self, name, difficulty_level):
        super().__init__(name)
        self.difficulty_level = difficulty_level

    def make_bid(self, bids):
        bid = 0
        suit_counts = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}

        # Count cards in each suit and identify Aces and Kings
        for card in self.hand:
            rank, suit = card
            suit_counts[suit] += 1
            if rank in ["Ace", "King"]:
                bid += 1

        # Check for single cards in non-spade suits
        for suit, count in suit_counts.items():
            if suit != "Spades" and count == 1:
                # Check if there's a non-Ace, non-King spade
                if any(spade[0] not in ["Ace", "King"] for spade in self.hand if spade[1] == "Spades"):
                    bid += 1
                    break  # Only increment once for any such suit

        return bid

    def play_card(self, leading_suit, spades_broken):
        self.determine_eligible_cards(leading_suit, spades_broken)

        hand_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.hand))
        print(f"Complete hand: {hand_cards_str}")

        # Format and display the eligible cards
        eligible_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.eligible_cards))
        print(f"Eligible cards to play: {eligible_cards_str}")

        if self.eligible_cards:
            chosen_card = random.choice(self.eligible_cards)
            self.card_played_last = chosen_card
            self.hand.remove(chosen_card)  # Remove the chosen card from the hand
            return chosen_card
        else:
            # Fallback in case no eligible cards (should not happen in a standard game)
            chosen_card = self.hand.pop()
            self.card_played_last = chosen_card
            return chosen_card


def create_players(num_human_players, total_players=4):
    players = []
    for i in range(num_human_players):
        players.append(HumanPlayer(f"Human {i + 1}"))
        print("appended a human player")
    for i in range(total_players - num_human_players):
        players.append(BotPlayer(f"Bot {i + 1}", 'hard'))
        print("appended a bot player")
    # Randomly select a dealer
    dealer_index = random.randint(0, total_players - 1)
    dealer = players[dealer_index]
    print(f"{dealer.name} is the dealer.")

    return players, dealer


def determine_winning_card_and_team(current_hand):
    # Define the order of ranks
    rank_order = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "Jack": 11, "Queen": 12,
                  "King": 13, "Ace": 14}

    # Determine the suit of the first card played (leading suit)
    leading_suit = current_hand[0][1][1]  # (player, (rank, suit))

    # Check if any Spades are played in the hand
    spades_in_hand = [(player, card) for player, card in current_hand if card[1] == "Spades"]
    spades_in_hand.sort(key=lambda x: rank_order[x[1][0]], reverse=True)

    # If Spades are played, the highest Spade wins
    if spades_in_hand:
        return spades_in_hand[0]

    # If no Spades are played, consider cards of the leading suit
    valid_cards = [(player, card) for player, card in current_hand if card[1] == leading_suit]
    valid_cards.sort(key=lambda x: rank_order[x[1][0]], reverse=True)

    # The first card in sorted valid_cards is the winning card
    return valid_cards[0] if valid_cards else current_hand[0]



def assign_tricks_to_team(current_hand, winning_card, team1_tricks, team2_tricks):
    winning_player = winning_card[0]
    if winning_player.team == "Team 1":
        team1_tricks.extend(current_hand)
    else:
        team2_tricks.extend(current_hand)


def rotate_dealer(players, current_dealer):
    dealer_index = players.index(current_dealer)
    new_dealer_index = (dealer_index + 1) % len(players)
    new_dealer = players[new_dealer_index]
    return new_dealer


def main_game_loop(players, game_parameters):
    game_over = False
    current_bids = {}
    team1_tricks = []
    team2_tricks = []

    while not game_over:
        for player in players:
            bid = player.make_bid(current_bids)
            if bid is not None:
                current_bids[player.name] = bid
            else:
                game_over = True
                break

        # Calculate team bids
        team1_total_bid = sum(current_bids[player.name] for player in players if player.team == "Team 1")
        team2_total_bid = sum(current_bids[player.name] for player in players if player.team == "Team 2")

        # Ensure minimum bid of 4
        game_parameters.team1_bid = max(4, team1_total_bid)
        game_parameters.team2_bid = max(4, team2_total_bid)

        # Display team bids
        print(f"Team 1 Bid: {game_parameters.team1_bid}")
        print(f"Team 2 Bid: {game_parameters.team2_bid}")

        for _ in range(13):  # Play 13 hands
            current_hand = []
            first_card_played = True

            for player in players:
                if first_card_played:
                    # For the first player in the hand, determine eligible cards without a leading suit
                    player.determine_eligible_cards(None, game_parameters.spades_broken)
                else:
                    # For subsequent players, use the current leading suit
                    player.determine_eligible_cards(game_parameters.leading_suit, game_parameters.spades_broken)

                card_played = player.play_card(game_parameters.leading_suit, game_parameters.spades_broken)
                print(card_played)
                print(f"{player.name} of team {player.team} played {card_played}")

                if first_card_played:
                    # Set the leading suit based on the first card played
                    game_parameters.leading_suit = card_played[1]
                    first_card_played = False

                if card_played[1] == 'Spades':
                    game_parameters.spades_broken = True

                current_hand.append((player, card_played))

            # Determine the winning card and player
            winning_card = determine_winning_card_and_team(current_hand)
            print(f"Winning card is {winning_card[1]}. \n\nNext round....\n\n")

            # Assign the tricks to the winning team
            assign_tricks_to_team(current_hand, winning_card, team1_tricks, team2_tricks)

            # Reorder players so that the winning player leads the next hand
            winning_player_index = players.index(winning_card[0])
            players = players[winning_player_index:] + players[:winning_player_index]

            # Reset the leading suit for the next hand
            game_parameters.leading_suit = None

        # Rotate the dealer for the next hand
        dealer = rotate_dealer(ordered_players, dealer)
        ordered_players = arrange_players(ordered_players, dealer)
        game_parameters.spades_broken = False

        # Store the winning hand in the appropriate team's tricks
        # ...

        # Calculate and display scores
        # ...

        # Check for end of game conditions
        # ...

        # Additional code for end of game


def assign_teams(players):
    # Shuffle the players list
    random.shuffle(players)

    # Initialize two teams
    team1 = []
    team2 = []

    # Assign players to teams
    for i, player in enumerate(players):
        if i % 2 == 0:
            team1.append(player)
        else:
            team2.append(player)

    for player in team1:
        player.set_team("Team 1")
        player.announce_myself()

    for player in team2:
        player.set_team("Team 2")
        player.announce_myself()

    return team1, team2


def arrange_players(players, dealer):
    # Find the index of the dealer
    dealer_index = players.index(dealer)

    # Arrange players starting from the left of the dealer
    ordered_players = players[dealer_index + 1:] + players[:dealer_index + 1]
    return ordered_players


# Define a main function that instantiates the card deck and runs the print cards function
def main():
    welcome()
    human_players = 1  # how_many_players()
    points = 200  # how_many_points()
    start_game()

    game_parameters = game_conditions(human_players=human_players, bot_players=4 - human_players, winning_score=points)

    # Create a card deck object
    deck = CardDeck()

    # Print the cards in the deck. This is for diagnostic purposes
    # deck.print_cards()

    # Shuffle the deck prior to starting the game
    deck.shuffle_cards()

    # Print the deck again as a demonstration that the deck shuffle worked. this is a diagnostic
    # deck.print_cards()

    # Generate players based on the number of humans playing the game
    players, dealer = create_players(game_parameters.number_of_players)
    print(f"dealer: {dealer}")

    orderd_players = arrange_players(players, dealer)
    print(f"orderd_players: {orderd_players}")

    # Assign game players and humans to a team based on the number of humans to bots
    assign_teams(orderd_players)

    # give each player their cards for the first hand
    deck.deal_cards(orderd_players, 13)

    # Main game loop. Begin the game loop
    main_game_loop(orderd_players, game_parameters)


# Call the main function
if __name__ == "__main__":
    main()





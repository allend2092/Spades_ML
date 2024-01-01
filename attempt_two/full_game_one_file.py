"""
Author: Daryl

This is a text based game of spades. As of Dec 17th, 2023 this version does work and allow the user to play through multiple hands, make a bid and has functional AI. This is the foundation for a deep learning neural net project.
I built this game so I could train the AI players using deep learning with an Nvidia GPU. I had some assistance writing this code using chatGPT

"""

import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class BidNet(nn.Module):
    def __init__(self):
        super(BidNet, self).__init__()
        self.fc1 = nn.Linear(in_features=51, out_features=128)  # Adjust in_features based on vector size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to get a value between 0 and 1
        return x

class PlayCardNet(nn.Module):
    def __init__(self):
        super(PlayCardNet, self).__init__()
        self.fc1 = nn.Linear(51, 128)  # Assuming 39 features in the input vector
        self.fc2 = nn.Linear(128, 13)  # Output size is 13 for each card in hand

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bid_net = BidNet().to(device)
play_card_net = PlayCardNet().to(device)

# Define suits and ranks for the cards
suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

# CardDeck class represents a deck of playing cards
class CardDeck:
    def __init__(self):
        # Initialize an empty dictionary to store cards
        self.cards = {}
        key = 1
        # Populate the deck with cards
        for suit in suits:
            for rank in ranks:
                self.cards[key] = (rank, suit)
                key += 1

    def print_cards(self):
        # Print all cards in the deck
        for key, value in self.cards.items():
            print(f"{key}: {value[0]} of {value[1]}")

    def shuffle_cards(self):
        # Shuffle the deck of cards
        card_values = list(self.cards.values())
        random.shuffle(card_values)
        for key in self.cards.keys():
            self.cards[key] = card_values[key - 1]

    def deal_cards(self, players, cards_per_player):
        # Deal cards to each player
        for _ in range(cards_per_player):
            for player in players:
                if self.cards:
                    card_key, card_value = next(iter(self.cards.items()))
                    player.hand.append(card_value)
                    del self.cards[card_key]

# Scoreboard class tracks the score of the game
class Scoreboard:
    def __init__(self, team_name, opponent_name):
        self.team_name = team_name
        self.opponent_name = opponent_name
        self.team1_overall_score = 0
        self.team2_overall_score = 0
        self.round_number = 0

    def calculate_score(self, team1_bid, team2_bid, team1_tricks, team2_tricks):
        # Calculate and update scores for both teams
        team1_hand_score = self.calculate_team_score(team1_bid, team1_tricks)
        team2_hand_score = self.calculate_team_score(team2_bid, team2_tricks)
        self.team1_overall_score += team1_hand_score
        self.team2_overall_score += team2_hand_score
        return team1_hand_score, team2_hand_score

    def calculate_team_score(self, bid, tricks):
        # Helper function to calculate score for a team
        score = 0
        if len(tricks) >= bid:
            score += 10 * bid + (len(tricks) - bid)
        else:
            score -= 10 * bid
        return score

# game_conditions class stores the state and rules of the game
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

# Welcome message for the game
def welcome():
    print('''[ASCII Art Welcome Message]''')

# Start game message
def start_game():
    print('''[ASCII Art Start Game Message]''')

# Function to ask the user for the number of human players
def how_many_players():
    human_players_input = input("How many human players will be playing? ")
    try:
        return int(human_players_input)
    except ValueError:
        print("Please enter a valid number.")
        return 0  # Default value or re-prompt

# Function to ask the user for the winning score limit
def how_many_points():
    points_input = input("What should the point limit be? ")
    try:
        return int(points_input)
    except ValueError:
        print("Please enter a valid number.")
        return 100  # Default value or re-prompt

# Player class represents a player in the game
class Player:
    def __init__(self, name):
        self.team = None
        self.name = name
        self.hand = []
        self.score = 0
        self.card_played_last = None
        self.eligible_cards = []
        # print(f"Hello, I am {self.name}. I am currently not assigned to a team.")

    def card_to_number(self, card):
        suit_order = {"Spades": 0, "Hearts": 1, "Diamonds": 2, "Clubs": 3}
        rank_order = {"Ace": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "Jack": 11,
                      "Queen": 12, "King": 13}

        if card is None:
            return 0  # Assuming 0 represents no card

        rank, suit = card
        suit_number = suit_order[suit]
        rank_number = rank_order[rank]

        # Calculate the card number
        card_number = suit_number * 13 + rank_number
        return card_number

    def vectorize_hand(self):
        # Convert each card in hand to its numerical representation
        hand_vector = [self.card_to_number(card) for card in self.hand]

        # Pad the vector with zeros if there are less than 13 cards in hand
        hand_vector.extend([0] * (13 - len(self.hand)))

        return hand_vector

    def vectorize_eligible_cards(self):
        # Convert each eligible card to its numerical representation
        eligible_cards_vector = [self.card_to_number(card) for card in self.eligible_cards]

        # Pad the vector with zeros if there are less than 13 eligible cards
        eligible_cards_vector.extend([0] * (13 - len(self.eligible_cards)))

        return eligible_cards_vector

    def vectorize_player(self):
        vector = []
        # Example: Convert team to a number (0 for Team 1, 1 for Team 2)
        team_number = 0 if self.team == 'Team 1' else 1
        print(team_number)
        vector.append(team_number)

        # Convert hand to numbers (assuming a function card_to_number exists)
        hand_vector = self.vectorize_hand()
        print(hand_vector)
        vector.extend(hand_vector)

        # Add score
        vector.append(self.score)
        print(self.score)

        # Convert last played card to a number
        # last_card_number = self.card_to_number(self.card_played_last) if self.card_played_last else -1
        # vector.append(last_card_number)

        # Convert eligible cards to numbers
        eligible_cards_vector = self.vectorize_eligible_cards()
        vector.extend(eligible_cards_vector)
        print(eligible_cards_vector)

        return vector

    def make_bid(self, bids, vector):
        raise NotImplementedError()

    def set_team(self, team_name):
        self.team = team_name

    def play_card(self):
        raise NotImplementedError()

    def announce_myself(self):
        print(f"I am {self.name} and I'm on team {self.team}.")

    def determine_eligible_cards(self, leading_suit, spades_broken):
        # Determine which cards can be legally played
        self.eligible_cards = []
        has_leading_suit = any(card[1] == leading_suit for card in self.hand)
        if has_leading_suit:
            self.eligible_cards = [card for card in self.hand if card[1] == leading_suit]
        elif leading_suit is None and not spades_broken:
            self.eligible_cards = self.hand.copy()
            if any(card[1] != "Spades" for card in self.hand):
                self.eligible_cards = [card for card in self.eligible_cards if card[1] != "Spades"]
        else:
            self.eligible_cards = self.hand.copy()

# HumanPlayer class represents a human player in the game
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def display_cards_in_hand(self, vector):
        if not self.hand:
            print("No cards in hand.")
            return
        self.hand.sort(key=lambda card: (suits.index(card[1]), ranks.index(card[0])))
        card_str = ", ".join(f"{card[0]} of {card[1]}" for card in self.hand)
        print(f"{self.name}'s cards: {card_str}")

    def make_bid(self, bids, vector):
        # Human player makes a bid
        while True:
            print(f"Current bids: {bids}")
            print("You have the following options: \n1. See cards and bid\n2. Bid blind nil\n3. Quit game")
            action = input("What do you want to do? ")
            try:
                action = int(action)
                if action == 1:
                    self.display_cards_in_hand()
                    bid = int(input("What is your bid? "))
                    return bid
                elif action == 2:
                    return 0  # Blind nil bid
                elif action == 3:
                    return None  # Quit game
                else:
                    print("That's not an available option. Please select again.")
            except ValueError:
                print("Please enter a valid number.")

    def play_card(self, leading_suit, spades_broken, vector):
        # Human player selects a card to play
        self.determine_eligible_cards(leading_suit, spades_broken)
        hand_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.hand))
        print(f"Complete hand: {hand_cards_str}")
        eligible_cards_str = ", ".join(f"{i + 1}. {card[0]} of {card[1]}" for i, card in enumerate(self.eligible_cards))
        print(f"Eligible cards to play: {eligible_cards_str}")
        while True:
            try:
                chosen_index = int(input("Choose a card to play (by number): ")) - 1
                if 0 <= chosen_index < len(self.eligible_cards):
                    chosen_card = self.eligible_cards.pop(chosen_index)
                    self.card_played_last = chosen_card
                    self.hand.remove(chosen_card)
                    return chosen_card
                else:
                    print("Invalid choice. Please choose a valid card number.")
            except ValueError:
                print("Please enter a number.")

# BotPlayer class represents a bot player in the game
class BotPlayer(Player):
    def __init__(self, name, difficulty_level, bid_net, play_card_net):
        super().__init__(name)
        self.difficulty_level = difficulty_level
        self.bid_net = bid_net
        self.play_card_net = play_card_net

    # def make_bid(self, bids, vector):
    #     print(vector)
    #     # Bot player makes a bid based on a simple strategy
    #     bid = 0
    #     suit_counts = {"Hearts": 0, "Diamonds": 0, "Clubs": 0, "Spades": 0}
    #     for card in self.hand:
    #         rank, suit = card
    #         suit_counts[suit] += 1
    #         if rank in ["Ace", "King"]:
    #             bid += 1
    #     for suit, count in suit_counts.items():
    #         if suit != "Spades" and count == 1:
    #             if any(spade[0] not in ["Ace", "King"] for spade in self.hand if spade[1] == "Spades"):
    #                 bid += 1
    #                 break
    #     return bid

    def make_bid(self, bids, vector):
        print(vector)
        # Convert vector to PyTorch tensor and move to GPU
        vector_tensor = torch.tensor(vector, dtype=torch.float).to(device)

        # Get bid from neural network
        with torch.no_grad():
            bid_output = self.bid_net(vector_tensor)

        # Convert NN output to a valid bid (example: scale and round)
        bid = round(bid_output.item() * 13)  # Scale to range 0-13
        return max(1, min(bid, 13))  # Ensure bid is within valid range

    # def play_card(self, leading_suit, spades_broken, vector):
    #     # Bot player selects a card to play
    #     self.determine_eligible_cards(leading_suit, spades_broken)
    #     if self.eligible_cards:
    #         chosen_card = random.choice(self.eligible_cards)
    #         self.card_played_last = chosen_card
    #         self.hand.remove(chosen_card)
    #         return chosen_card
    #     else:
    #         chosen_card = self.hand.pop()
    #         self.card_played_last = chosen_card
    #         return chosen_card
    def play_card(self, leading_suit, spades_broken, vector):
        self.determine_eligible_cards(leading_suit, spades_broken)

        # Convert vector to tensor and pass to the network
        vector_tensor = torch.tensor(vector, dtype=torch.float).unsqueeze(0).to(device)
        card_probabilities = self.play_card_net(vector_tensor).squeeze(0)

        # Choose a card based on the probabilities
        if self.eligible_cards:
            # Map probabilities to eligible cards, ensuring indices are within range
            eligible_probs = []
            for card in self.eligible_cards:
                card_index = self.card_to_number(card) - 1
                if 0 <= card_index < 13:
                    eligible_probs.append(card_probabilities[card_index])
                else:
                    eligible_probs.append(0)  # Assign a low probability for out-of-range cards

            chosen_card_index = eligible_probs.index(max(eligible_probs))
            chosen_card = self.eligible_cards[chosen_card_index]
        else:
            # Fallback if no eligible cards
            chosen_card = self.hand.pop()

        self.card_played_last = chosen_card
        self.hand.remove(chosen_card)
        return chosen_card


# Function to create players for the game
def create_players(num_human_players, total_players=4):
    players = []
    for i in range(num_human_players):
        players.append(HumanPlayer(f"Human {i + 1}"))
        # print("appended a human player")
    for i in range(total_players - num_human_players):
        players.append(BotPlayer(f"Bot {i + 1}", 'hard', bid_net, play_card_net))
        # print("appended a bot player")
    dealer_index = random.randint(0, total_players - 1)
    dealer = players[dealer_index]
    # print(f"{dealer.name} is the dealer.")
    return players, dealer

# Function to determine the winning card and team
def determine_winning_card_and_team(current_hand):
    rank_order = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "Jack": 11, "Queen": 12, "King": 13, "Ace": 14}
    leading_suit = current_hand[0][1][1]
    spades_in_hand = [(player, card) for player, card in current_hand if card[1] == "Spades"]
    spades_in_hand.sort(key=lambda x: rank_order[x[1][0]], reverse=True)
    if spades_in_hand:
        return spades_in_hand[0]
    valid_cards = [(player, card) for player, card in current_hand if card[1] == leading_suit]
    valid_cards.sort(key=lambda x: rank_order[x[1][0]], reverse=True)
    return valid_cards[0] if valid_cards else current_hand[0]

# Function to assign tricks to the winning team
def assign_tricks_to_team(current_hand, winning_card, team1_tricks, team2_tricks):
    winning_player = winning_card[0]
    if winning_player.team == "Team 1":
        team1_tricks.extend(current_hand)
    else:
        team2_tricks.extend(current_hand)

# Function to rotate the dealer for the next round
def rotate_dealer(players, current_dealer):
    dealer_index = players.index(current_dealer)
    new_dealer_index = (dealer_index + 1) % len(players)
    return players[new_dealer_index]

# Function to check if the game has reached the end
def check_end_of_game(scoreboard, winning_score):
    return scoreboard.team1_overall_score >= winning_score or scoreboard.team2_overall_score >= winning_score

def arrange_players(players, dealer):
    # Find the index of the dealer
    dealer_index = players.index(dealer)

    # Arrange players starting from the left of the dealer
    ordered_players = players[dealer_index + 1:] + players[:dealer_index + 1]
    return ordered_players

# Function to assign players to teams
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

# Main game loop function
def main_game_loop(players, game_parameters, dealer, deck):
    game_over = False
    current_bids = {}
    team1_tricks = []
    team2_tricks = []
    scoreboard = Scoreboard("Team 1", "Team 2")
    game_state_and_player_vector = []


    while not game_over:
        for player in players:
            # Vectorize the game state before the bots make a decision
            game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                         team2_tricks, 0, player, players)
            # print(f'This is the vector: {game_state_and_player_vector}')
            bid = player.make_bid(current_bids, game_state_and_player_vector)
            game_state_and_player_vector = []
            if bid is not None:
                current_bids[player.name] = bid
            else:
                game_over = True
                break
        team1_total_bid = sum(current_bids[player.name] for player in players if player.team == "Team 1")
        team2_total_bid = sum(current_bids[player.name] for player in players if player.team == "Team 2")
        game_parameters.team1_bid = max(4, team1_total_bid)
        game_parameters.team2_bid = max(4, team2_total_bid)
        print(f"Team 1 Bid: {game_parameters.team1_bid}")
        print(f"Team 2 Bid: {game_parameters.team2_bid}")
        for i in range(13):
            current_hand = []
            first_card_played = True
            for player in players:
                if first_card_played:
                    player.determine_eligible_cards(None, game_parameters.spades_broken)
                else:
                    player.determine_eligible_cards(game_parameters.leading_suit, game_parameters.spades_broken)
                # Vectorize the game state before the bots make a decision
                game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                                                                    team2_tricks, 0, player, players)

                card_played = player.play_card(game_parameters.leading_suit, game_parameters.spades_broken,
                                               game_state_and_player_vector)
                game_state_and_player_vector = []
                # print(card_played)
                print(f"{player.name} of team {player.team} played {card_played}")
                if first_card_played:
                    game_parameters.leading_suit = card_played[1]
                    first_card_played = False
                if card_played[1] == 'Spades':
                    game_parameters.spades_broken = True
                current_hand.append((player, card_played))
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
        dealer = rotate_dealer(players, dealer)
        players = arrange_players(players, dealer)
        game_parameters.spades_broken = False

        # Calculate and update scores after 13 hands
        team1_score, team2_score = scoreboard.calculate_score(game_parameters.team1_bid, game_parameters.team2_bid,
                                                              team1_tricks, team2_tricks)
        print(f"Round Score - Team 1: {team1_score}, Team 2: {team2_score}")

        # Check if the game has reached the winning score
        if check_end_of_game(scoreboard, game_parameters.threshold_score):
            print("Game Over")
            print(f"Final Score - Team 1: {scoreboard.team1_overall_score}, Team 2: {scoreboard.team2_overall_score}")
            game_over = True
        else:
            # Reset for the next round
            team1_tricks.clear()
            team2_tricks.clear()
            current_bids.clear()

            # Create a card deck object
            deck = CardDeck()

            # Shuffle and deal new cards for the next round
            deck.shuffle_cards()

            # Rotate the dealer for the next round
            dealer = rotate_dealer(players, dealer)
            players = arrange_players(players, dealer)
            # give each player their cards
            deck.deal_cards(players, 13)
            game_parameters.spades_broken = False


def one_hot_encode_round(current_round_number):
    # Create a vector of zeros with length 13
    round_vector = [0] * 13

    # Set the element corresponding to the current round number to 1
    # Subtract 1 from the round number to convert it to a zero-based index
    round_vector[current_round_number - 1] = 1

    return round_vector


def vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                         team2_tricks, current_round_number, player, players):
    # Vectorizing game_over
    game_over_vector = [1 if game_over else 0]
    print(game_over_vector)

    # Vectorizing scoreboard
    scoreboard_vector = [scoreboard.team1_overall_score, scoreboard.team2_overall_score, scoreboard.round_number]
    print(game_over_vector)

    # Vectorizing current bids
    bids_vector = [current_bids.get(p.name, 0) for p in players]  # Assuming 'players' is accessible
    print(game_over_vector)

    # Vectorizing team tricks
    team_tricks_vector = [len(team1_tricks), len(team2_tricks)]
    print(game_over_vector)

    # Vectorizing current hand
    current_round_number_vector = one_hot_encode_round(current_round_number)
    print(game_over_vector)

    # Vectorizing player's perspective
    player_vector = player.vectorize_player()
    print(game_over_vector)

    # Combine all vectors into a single game state vector
    game_state_vector = game_over_vector + scoreboard_vector +\
                        bids_vector + team_tricks_vector + player_vector + current_round_number_vector
    print(game_over_vector)

    return game_state_vector



# Main function to start the game
def main():
    welcome()
    human_players = 0  # how_many_players()
    points = 1000  # how_many_points()
    start_game()

    game_parameters = game_conditions(human_players=human_players, bot_players=4 - human_players,
                                          winning_score=points)

    # Create a card deck object
    deck = CardDeck()

    # Shuffle the deck prior to starting the game
    deck.shuffle_cards()

    # Generate players based on the number of humans playing the game
    players, dealer = create_players(game_parameters.number_of_players)
    # print(f"dealer: {dealer}")

    ordered_players = arrange_players(players, dealer)
    # print(f"ordered_players: {ordered_players}")

    # Assign game players and humans to a team based on the number of humans to bots
    assign_teams(ordered_players)

    # Give each player their cards for the first hand
    deck.deal_cards(ordered_players, 13)

    # Main game loop. Begin the game loop
    main_game_loop(ordered_players, game_parameters, dealer, deck)

# Call the main function
if __name__ == "__main__":
    main()


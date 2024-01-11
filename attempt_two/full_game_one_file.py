"""
Author: Daryl

This is a text based game of spades. As of Dec 17th, 2023 this version does work and allow the user to play through multiple hands, make a bid and has functional AI. This is the foundation for a deep learning neural net project.
I built this game so I could train the AI players using deep learning with an Nvidia GPU. I had some assistance writing this code using chatGPT

"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

class BidNet(nn.Module):
    def __init__(self):
        super(BidNet, self).__init__()
        self.fc1 = nn.Linear(in_features=51, out_features=128)  # Adjust in_features based on vector size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 13)  # Output size is 13 for each possible bid

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No sigmoid here; raw scores for each class
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
        if (len(tricks) / 4) >= bid:
            score += 10 * bid + ((len(tricks) / 4) - bid)
        else:
            score -= 10 * bid
        return score

    def reset(self):
        self.team1_overall_score = 0
        self.team2_overall_score = 0
        self.round_number = 0

# game_conditions class stores the state and rules of the game
class game_conditions:
    # set number of human players, number of bot players, and how many points are needed to win the game
    def __init__(self, human_players: int, bot_players: int, winning_score: int):
        # Set the number of human players
        self.number_of_players: int = human_players
        # Set the number of bot players
        self.ai_opponents: int = bot_players
        # How many points need does a team need to reach to win the game?
        self.threshold_score: int = winning_score
        # Allows for blind nil bids. This feature is not yet implemented
        self.blind_nil_allowed: bool = False
        # Keep track of what round it currently is.
        self.current_round: int = 0
        self.whose_turn = {}
        # Track team 1 bids
        self.team1_bid: int = None
        # Track team 2 bids
        self.team2_bid: int = None
        self.game_play_order = []
        # Helps implement the game rules by tracking what the leading suit is.
        self.leading_suit = None
        # Has any player played a spade in this hand? If no, spades_broken is false
        self.spades_broken: bool = False

    def give_number_of_players(self):
        return self.number_of_players

# Welcome message for the game
def welcome():
    print('''

.------..------..------..------..------..------..------.
|W.--. ||E.--. ||L.--. ||C.--. ||O.--. ||M.--. ||E.--. |
| :/\: || (\/) || :/\: || :/\: || :/\: || (\/) || (\/) |
| :\/: || :\/: || (__) || :\/: || :\/: || :\/: || :\/: |
| '--'W|| '--'E|| '--'L|| '--'C|| '--'O|| '--'M|| '--'E|
`------'`------'`------'`------'`------'`------'`------'

''')

# Start game message
def start_game():
    print('''

   ,-,--.  ,--.--------.   ,---.                  ,--.--------.              _,---.   ,---.             ___      ,----.   .=-.-..=-.-..=-.-. 
 ,-.'-  _\/==/,  -   , -\.--.'  \      .-.,.---. /==/,  -   , -\         _.='.'-,  \.--.'  \     .-._ .'=.'\  ,-.--` , \ /==/_ /==/_ /==/_ / 
/==/_ ,_.'\==\.-.  - ,-./\==\-/\ \    /==/  `   \\==\.-.  - ,-./        /==.'-     /\==\-/\ \   /==/ \|==|  ||==|-  _.-`|==|, |==|, |==|, |  
\==\  \    `--`\==\- \   /==/-|_\ |  |==|-, .=., |`--`\==\- \          /==/ -   .-' /==/-|_\ |  |==|,|  / - ||==|   `.-.|==|  |==|  |==|  |  
 \==\ -\        \==\_ \  \==\,   - \ |==|   '='  /     \==\_ \         |==|_   /_,-.\==\,   - \ |==|  \/  , /==/_ ,    //==/. /==/. /==/. /  
 _\==\ ,\       |==|- |  /==/ -   ,| |==|- ,   .'      |==|- |         |==|  , \_.' )==/ -   ,| |==|- ,   _ |==|    .-' `--`-``--`-``--`-`   
/==/\/ _ |      |==|, | /==/-  /\ - \|==|_  . ,'.      |==|, |         \==\-  ,    (==/-  /\ - \|==| _ /\   |==|_  ,`-._ .=.   .=.   .=.     
\==\ - , /      /==/ -/ \==\ _.\=\.-'/==/  /\ ,  )     /==/ -/          /==/ _  ,  |==\ _.\=\.-'/==/  / / , /==/ ,     /:=; : :=; : :=; :    
 `--`---'       `--`--`  `--`        `--`-`--`--'      `--`--`          `--`------' `--`        `--`./  `--``--`-----``  `=`   `=`   `=`     

''')

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
        self.previous_hand = []
        self.score = 0
        self.card_played_last = None
        self.eligible_cards = []
        self.bot = None
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
        #print(team_number)
        vector.append(team_number)

        # Convert hand to numbers (assuming a function card_to_number exists)
        hand_vector = self.vectorize_hand()
        #print(hand_vector)
        vector.extend(hand_vector)

        # Add score
        vector.append(self.score)
        #print(self.score)

        # Convert last played card to a number
        # last_card_number = self.card_to_number(self.card_played_last) if self.card_played_last else -1
        # vector.append(last_card_number)

        # Convert eligible cards to numbers
        eligible_cards_vector = self.vectorize_eligible_cards()
        vector.extend(eligible_cards_vector)
        #print(eligible_cards_vector)

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
        self.bot = False
        self.won_hands = 0

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
                    self.display_cards_in_hand(vector)
                    bid = int(input("What is your bid? "))
                    return bid
                elif action == 2:
                    return 0  # Blind nil bid
                elif action == 3:
                    sys.exit()  # Quit game
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

    def count_hands_I_won(self, winning_hand):
        if winning_hand[1]:
            self.won_hands += 1

# BotPlayer class represents a bot player in the game
class BotPlayer(Player):
    def __init__(self, name, difficulty_level, bid_net, play_card_net):
        super().__init__(name)
        self.difficulty_level = difficulty_level
        self.bid_net = bid_net
        self.play_card_net = play_card_net
        self.bid_game_vector = [] # instantiated so that I can carry the game vector to different scopes of program
        self.play_card_game_vector = []
        self.bot = True
        self.card_i_played = None
        self.bid_i_made = None
        self.team_bid = None
        self.reward = 0
        self.memory = []
        self.bid_memory = []
        self.optimizer = torch.optim.Adam(self.play_card_net.parameters(), lr=0.001)
        # Check for existing model and load it
        self.load_model_if_exists()
        self.last_played_of_my_hand_card_index = None
        self.bid_reward_value = 0
        self.won_hands = 0

    def count_hands_I_won(self, winning_hand):
        if winning_hand[1]:
            self.won_hands += 1


    def set_team_bid(self, team_bid):
        self.team_bid = team_bid
    def load_model_if_exists(self):
        # Construct the expected filename
        model_filename = f'{self.name}_play_card_net.pth'

        # Check if the file exists in the current directory
        if model_filename in os.listdir():
            print(f"Loading saved model for {self.name}")
            self.play_card_net.load_state_dict(torch.load(model_filename))
        else:
            print(f"No saved model found for {self.name}, starting with a new model.")

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
        self.bid_game_vector = vector
        # Convert vector to PyTorch tensor and move to GPU
        vector_tensor = torch.tensor(vector, dtype=torch.float).to(device)

        # Get bid from neural network
        with torch.no_grad():
            bid_output = self.bid_net(vector_tensor)

        # Get the index of the highest score, which represents the predicted bid
        predicted_bid_index = torch.argmax(bid_output).item()

        # Convert index to bid (index 0 corresponds to bid 1, index 1 to bid 2, etc.)
        predicted_bid = predicted_bid_index + 1

        self.bid_i_made = predicted_bid
        return predicted_bid

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
        self.play_card_game_vector = vector
        # print(self.play_card_game_vector)
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
        self.last_played_of_my_hand_card_index = self.hand.index(chosen_card)
        self.hand.remove(chosen_card)
        return chosen_card

    def play_card_reward_nn(self, current_hand, winning_card, team1_tricks, team2_tricks):
        #print(winning_card)
        reward = 0
        if winning_card[0].name == self.name:
            reward += 50
        else:
            reward += -30
        if winning_card[0].team == self.team:
            reward += 50
        else:
            reward += -30
        self.reward = reward
        #self.memory.append((self.play_card_game_vector, self.card_played_last, self.reward))
        card_index = self.card_to_number(self.card_played_last)  # Convert card to a numerical index
        #print(f'Card index: {card_index}' )
        #print(f'Internal index: {self.last_played_of_my_hand_card_index}')
        self.memory.append((self.play_card_game_vector, self.last_played_of_my_hand_card_index, self.reward))
        return reward

    def train_play_card_network(self):
        # Ensure there's enough data to train
        if len(self.memory) < 1:
            return

        # Convert the memory data into tensors
        #print(self.memory)
        state_tensors = torch.tensor([item[0] for item in self.memory], dtype=torch.float).to(device)
        action_tensors = torch.tensor([item[1] for item in self.memory], dtype=torch.long).to(device)
        reward_tensors = torch.tensor([item[2] for item in self.memory], dtype=torch.float).to(device)

        # Forward pass: compute predicted action probabilities
        predicted_action_probabilities = self.play_card_net(state_tensors)

        # Compute loss using CrossEntropyLoss for multi-class classification
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(predicted_action_probabilities, action_tensors)
        print(f'play card loss: {loss}')

        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after training
        self.memory.clear()

    def bid_reward(self, actual_tricks_I_won, round_score, team_score, team_tricks_won, team_bid):
        # Calculate reward based on the difference between bid and actual tricks won
        bid_error = abs(self.bid_i_made - actual_tricks_I_won)
        reward = -bid_error * 10  # Negative reward for larger errors
        if round_score >0:
            reward += 100
        elif round_score <0:
            reward += -500
        if team_score >0:
            reward += 50
        elif team_score <0:
            reward += -250

        if team_tricks_won > team_bid:
            reward += 75
        elif team_tricks_won < team_bid:
            reward += -75

        self.bid_reward_value = reward
        #print(f'Bid reward output: {self.bid_game_vector} - {self.bid_i_made} - {self.bid_reward_value}')
        self.bid_memory.append((self.bid_game_vector, self.bid_i_made, self.bid_reward_value))
        return reward

    def train_bid_network(self):
        # Ensure there's enough data to train
        if len(self.bid_memory) < 1:
            return

        # Convert the memory data into tensors
        #print(f'Self memory prior to bid training: {self.bid_memory}')
        state_tensors = torch.tensor([item[0] for item in self.bid_memory], dtype=torch.float).to(device)
        bid_tensors = torch.tensor([item[1] - 1 for item in self.bid_memory], dtype=torch.long).to(device)
        #print(bid_tensors)
        reward_tensors = torch.tensor([item[2] for item in self.bid_memory], dtype=torch.float).to(device)

        # Forward pass: compute predicted bid probabilities
        predicted_bid_probabilities = self.bid_net(state_tensors)

        # Compute loss using CrossEntropyLoss for multi-class classification
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(predicted_bid_probabilities, bid_tensors)
        print(f'Bid Loss: {loss}')

        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory after training
        self.memory.clear()

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
# def check_end_of_game(scoreboard, winning_score):
#     return scoreboard.team1_overall_score >= winning_score or scoreboard.team2_overall_score >= winning_score
def check_end_of_game(scoreboard, winning_score, losing_score=-1000):
    # Check if either team has reached or exceeded the winning score
    has_won = scoreboard.team1_overall_score >= winning_score or scoreboard.team2_overall_score >= winning_score

    # Check if either team has fallen below the losing score
    has_lost = scoreboard.team1_overall_score <= losing_score or scoreboard.team2_overall_score <= losing_score

    return has_won or has_lost

# Players is another custom class that contains each player object. I believe dealer indicates which player is dealer.
def arrange_players(players, dealer):
    # Players object contans a list of all players. Dealer is the name of a single player. Given a player name where
    # the player happens to be the dealer, find where in the list of the players object is the one we want as role
    # dealer. Given this players index in the list of players, the dealer index is returned and set as dealer_index
    dealer_index = players.index(dealer)

    # Arrange players starting from the left of the dealer
    rearranged_player_list = players[dealer_index + 1:] + players[:dealer_index + 1]

    # This for loop is not needed at the program outset, but between hands, prior to the start of a new hand, the
    # players get re-arranged based on who the new dealer is. Since its the start of a new round, its also a good time
    # to set the number of won hands to zero. Honestly, this operation could probably be put in a different location
    # of the program to add clarity to when its happening.
    for player in players:
        player.won_hands = 0

    return rearranged_player_list

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
    num_episodes = 999999
    current_bids = {}
    team1_tricks = []
    team2_tricks = []
    scoreboard = Scoreboard("Team 1", "Team 2")

    for episode in range(num_episodes):
        game_over = False
        total_reward = 0

        while not game_over:
            print(f'Current episode: {episode}')
            for outer_player in players:
                # Vectorize the game state before the bots make a decision
                game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                             team2_tricks, 0, outer_player, players)
                bid_game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                                     team2_tricks, 0, outer_player, players)
                #print(game_state_and_player_vector)
                # print(f'This is the vector: {game_state_and_player_vector}')
                bid = outer_player.make_bid(current_bids, bid_game_state_and_player_vector)
                print(f'{outer_player.name} hand: {outer_player.hand}')
                print(f'{outer_player.name} bids {bid}')
                game_state_and_player_vector.clear()
                if bid is not None:
                    current_bids[outer_player.name] = bid
                else:
                    game_over = True
                    break
            team1_total_bid = sum(current_bids[outer_player.name] for outer_player in players if outer_player.team == "Team 1")
            print(f'Team 1 total bid: {team1_total_bid}')
            team2_total_bid = sum(current_bids[outer_player.name] for outer_player in players if outer_player.team == "Team 2")
            print(f'Team 2 total bid: {team2_total_bid}')

            # Cap the team bids at 13
            team1_total_bid = min(team1_total_bid, 13)
            team2_total_bid = min(team2_total_bid, 13)



            game_parameters.team1_bid = max(4, team1_total_bid)
            game_parameters.team2_bid = max(4, team2_total_bid)

            print(f"Team 1 Bid: {game_parameters.team1_bid}")
            print(f"Team 2 Bid: {game_parameters.team2_bid}")

            for i in range(13):
                current_hand = []
                first_card_played = True
                for inner_player in players:

                    if first_card_played:
                        inner_player.determine_eligible_cards(None, game_parameters.spades_broken)
                    else:
                        inner_player.determine_eligible_cards(game_parameters.leading_suit, game_parameters.spades_broken)
                    # Vectorize the game state before the bots make a decision
                    game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                                                                        team2_tricks, 0, inner_player, players)
                    outer_game_state_and_player_vector = vectorize_game_state(game_over, scoreboard, current_bids, team1_tricks,
                                                                        team2_tricks, 0, inner_player, players)

                    card_played = inner_player.play_card(game_parameters.leading_suit, game_parameters.spades_broken,
                                                   game_state_and_player_vector)
                    if outer_player.name == inner_player.name:
                        #print("copy values!")
                        outer_player.play_card_game_vector = outer_game_state_and_player_vector
                        #print(f'printing outer player vector: {outer_player.play_card_game_vector}')

                    #game_state_and_player_vector.clear()
                    # print(card_played)
                    print(f"{inner_player.name} of team {inner_player.team} played {card_played}")
                    if first_card_played:
                        game_parameters.leading_suit = card_played[1]
                        first_card_played = False
                    if card_played[1] == 'Spades':
                        game_parameters.spades_broken = True
                    current_hand.append((inner_player, card_played))
                #print(f'printing outer player vector: {outer_player.play_card_game_vector}')
                winning_card = determine_winning_card_and_team(current_hand)
                print(f"Winning card is {winning_card[1]}. Winning player is {winning_card[0].name}."
                      f" Winning Team {winning_card[0].team}"
                      f" \n\nNext round....\n\n")

                winning_card[0].count_hands_I_won(winning_card)
                print(f'Player {winning_card[0].name} has won {winning_card[0].won_hands}')
                # Assign the tricks to the winning team
                assign_tricks_to_team(current_hand, winning_card, team1_tricks, team2_tricks)

                # Reorder players so that the winning player leads the next hand
                winning_player_index = players.index(winning_card[0])
                players = players[winning_player_index:] + players[:winning_player_index]

                #print(f'training vector: {outer_player.play_card_game_vector}')
                if outer_player.bot == True:
                    outer_player.play_card_reward_nn(current_hand, winning_card, team1_tricks, team2_tricks)
                    outer_player.train_play_card_network()



                # Reset the leading suit for the next hand
                game_parameters.leading_suit = None

            # Rotate the dealer for the next hand
            dealer = rotate_dealer(players, dealer)
            players = arrange_players(players, dealer)
            game_parameters.spades_broken = False

            # Calculate and update scores after 13 hands
            team1_score, team2_score = scoreboard.calculate_score(game_parameters.team1_bid, game_parameters.team2_bid,
                                                                  team1_tricks, team2_tricks)
            print(f"Game Score - Team 1: {scoreboard.team1_overall_score}, Team 2: {scoreboard.team2_overall_score}")
            print(f"Round Score - Team 1: {team1_score}, Team 2: {team2_score}")
            print(f'Round Tricks - Team 1: {(len(team1_tricks) / 4)} , Team 2: {(len(team2_tricks) / 4)}')
            print(f'Round bid - Team 1: {team1_total_bid} , Team 2: {team2_total_bid}')
            print(f'Threshold score: {game_parameters.threshold_score}')


            if outer_player.bot == True:
                #print('tabulating hand:')
                #print(outer_player.team)
                #print(team1_tricks)
                #total_cards = (len(team1_tricks) / 4)
                #print(total_cards)
                #print(team2_tricks)
                #total_cards = (len(team2_tricks) / 4)
                #print(total_cards)
                if outer_player.team == 'Team 1':
                    #print(f"Apply Reward function for {outer_player.name} on team: {outer_player.team}. Tricks: {(len(team1_tricks) / 4)}")
                    outer_player.set_team_bid(team1_total_bid)
                    outer_player.bid_reward(outer_player.won_hands, team1_score, scoreboard.team1_overall_score, (len(team1_tricks) / 4), team1_total_bid)
                elif outer_player.team == 'Team 2':
                    #print(f"Apply Reward function for {outer_player.name} on team: {outer_player.team}. Tricks: {(len(team2_tricks) / 4)}")
                    outer_player.set_team_bid(team2_total_bid)
                    outer_player.bid_reward(outer_player.won_hands, team2_score, scoreboard.team2_overall_score, (len(team2_tricks) / 4), team2_total_bid)
                else:
                    print(f"Team not recognized for {outer_player.name}.")
                outer_player.train_bid_network()

            # Check if the game has reached the winning score
            if check_end_of_game(scoreboard, game_parameters.threshold_score):
                print("Game Over")
                print(f"Final Score - Team 1: {scoreboard.team1_overall_score}, Team 2: {scoreboard.team2_overall_score}")
                game_over = True
                # Reset for the next game. This allows for multiple games so that NN learning can occur
                # Eventually, you should try to fit this into a reset_game()  # Start a new game
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
                scoreboard.reset()
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
    # save trained NNs
    if episode == num_episodes - 1:
        # Save the model's state dictionary
        for player in players:
            torch.save(player.play_card_net.state_dict(), f'{player.name}_play_card_net.pth')


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
    #print(game_over_vector)

    # Vectorizing scoreboard
    scoreboard_vector = [scoreboard.team1_overall_score, scoreboard.team2_overall_score, scoreboard.round_number]
    #print(game_over_vector)

    # Vectorizing current bids
    bids_vector = [current_bids.get(p.name, 0) for p in players]  # Assuming 'players' is accessible
    #print(game_over_vector)

    # Vectorizing team tricks
    team_tricks_vector = [(len(team1_tricks) / 4), (len(team2_tricks)/ 4)]
    #print(game_over_vector)

    # Vectorizing current hand
    current_round_number_vector = one_hot_encode_round(current_round_number)
    #print(game_over_vector)

    # Vectorizing player's perspective
    player_vector = player.vectorize_player()
   # print(game_over_vector)

    # Combine all vectors into a single game state vector
    game_state_vector = game_over_vector + scoreboard_vector +\
                        bids_vector + team_tricks_vector + player_vector + current_round_number_vector
    #print(game_over_vector)

    return game_state_vector



# Main function to start the game
def main():

    # Set intial game conditions
    human_players = 0  # Set players to 0 for bot training
    points = 50  # The team that scores this many points will win. -1,000 points causes team to lose

    # Displays some ascii art displaying welcome for the user
    welcome()

    # More ascii art for the user
    start_game()

    # This is a custom class I made to track game state, such as score, number of players, bids, etc.
    game_parameters = game_conditions(human_players=human_players, bot_players=4 - human_players,
                                          winning_score=points)

    # Create a card deck object
    deck = CardDeck()

    # Shuffle the deck prior to starting the game
    deck.shuffle_cards()

    # Generate players based on the number of humans playing the game
    players, dealer = create_players(game_parameters.give_number_of_players())

    # The purpose of this function is to create a new player list so that the first player is the dealer and the players
    # after the dealer are to the left of the dealer. Obviously, the final player before the dealer will be to the right
    # of the dealer.
    ordered_players = arrange_players(players, dealer)

    # Assign game players and humans to a team based on the number of humans to bots
    assign_teams(ordered_players)

    # Give each player their cards for the first hand
    deck.deal_cards(ordered_players, 13)

    # Main game loop. Begin the game loop
    main_game_loop(ordered_players, game_parameters, dealer, deck)

# main block.
if __name__ == "__main__":
    main()

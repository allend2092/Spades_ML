'''
This version is an improved version of the previous code in version 1. The improvements were made by chatGPT o1.

The purpose of this project is to aid my understanding of machine learning

'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

# Neural network for bidding
class BidNet(nn.Module):
    def __init__(self, input_size):
        super(BidNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 13)  # Output size is 13 for possible bids (1-13)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

# Neural network for playing cards
class PlayCardNet(nn.Module):
    def __init__(self, input_size):
        super(PlayCardNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 13)  # Output size is 13 for cards in hand

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

# Mapping of card to a consistent index
card_to_index = {f"{rank} of {suit}": idx for idx, (suit, rank) in enumerate([(suit, rank) for suit in suits for rank in ranks])}

class CardDeck:
    def __init__(self):
        self.cards = [f"{rank} of {suit}" for suit in suits for rank in ranks]

    def shuffle_cards(self):
        random.shuffle(self.cards)

    def deal_cards(self, players, cards_per_player):
        for _ in range(cards_per_player):
            for player in players:
                if self.cards:
                    player.hand.append(self.cards.pop())

class Scoreboard:
    def __init__(self):
        self.team1_overall_score = 0
        self.team2_overall_score = 0
        self.round_number = 0

    def calculate_score(self, team1_bid, team2_bid, team1_tricks, team2_tricks):
        team1_hand_score = self.calculate_team_score(team1_bid, team1_tricks)
        team2_hand_score = self.calculate_team_score(team2_bid, team2_tricks)
        self.team1_overall_score += team1_hand_score
        self.team2_overall_score += team2_hand_score
        return team1_hand_score, team2_hand_score

    def calculate_team_score(self, bid, tricks):
        score = 0
        if tricks >= bid:
            score += 10 * bid + (tricks - bid)
        else:
            score -= 10 * bid
        return score

    def reset(self):
        self.team1_overall_score = 0
        self.team2_overall_score = 0
        self.round_number = 0

class GameConditions:
    def __init__(self, human_players, bot_players, winning_score, losing_score):
        self.number_of_players = human_players
        self.ai_opponents = bot_players
        self.threshold_score = winning_score
        self.losing_score = losing_score  # Minimum score before game ends
        self.current_round = 0
        self.spades_broken = False

class Player:
    def __init__(self, name):
        self.team = None
        self.name = name
        self.hand = []
        self.score = 0
        self.card_played_last = None
        self.bot = None
        self.won_tricks = 0
        self.bid = 0  # Player's bid

    def set_team(self, team_name):
        self.team = team_name

    def determine_eligible_cards(self, leading_suit, spades_broken):
        self.eligible_cards = []
        if not self.hand:
            return
        has_leading_suit = any(card.split(" of ")[1] == leading_suit for card in self.hand)
        if has_leading_suit and leading_suit:
            self.eligible_cards = [card for card in self.hand if card.split(" of ")[1] == leading_suit]
        elif not leading_suit and not spades_broken:
            self.eligible_cards = [card for card in self.hand if card.split(" of ")[1] != "Spades"]
            if not self.eligible_cards:
                self.eligible_cards = self.hand.copy()
        else:
            self.eligible_cards = self.hand.copy()

class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)
        self.bot = False

    def make_bid(self, bids, game_state_vector):
        while True:
            print(f"Your hand: {', '.join(self.hand)}")
            try:
                bid = int(input("Enter your bid (1-13): "))
                if 1 <= bid <= 13:
                    self.bid = bid
                    return bid
                else:
                    print("Bid must be between 1 and 13.")
            except ValueError:
                print("Please enter a valid number.")

    def play_card(self, leading_suit, spades_broken, game_state_vector):
        self.determine_eligible_cards(leading_suit, spades_broken)
        print(f"Your hand: {', '.join(self.hand)}")
        print(f"Eligible cards: {', '.join(self.eligible_cards)}")
        while True:
            card = input("Enter the card you want to play (e.g., 'Ace of Spades'): ")
            if card in self.eligible_cards:
                self.hand.remove(card)
                return card
            else:
                print("Invalid card or not eligible.")

class BotPlayer(Player):
    def __init__(self, name, difficulty_level):
        super().__init__(name)
        self.bot = True
        self.difficulty_level = difficulty_level
        self.memory = []
        self.bid_memory = []
        self.bid_net = BidNet(input_size=60).to(device)
        self.play_card_net = PlayCardNet(input_size=60).to(device)
        self.play_card_optimizer = torch.optim.Adam(self.play_card_net.parameters(), lr=0.001)
        self.bid_optimizer = torch.optim.Adam(self.bid_net.parameters(), lr=0.001)
        self.load_models()

    def load_models(self):
        bid_model_path = f"{self.name}_bid_net.pth"
        play_model_path = f"{self.name}_play_card_net.pth"

        if os.path.exists(bid_model_path):
            try:
                self.bid_net.load_state_dict(torch.load(bid_model_path))
                print(f"Loaded bid model for {self.name}")
            except Exception as e:
                print(f"Error loading bid model for {self.name}: {e}")
        else:
            print(f"No saved bid model for {self.name}, starting fresh.")

        if os.path.exists(play_model_path):
            try:
                self.play_card_net.load_state_dict(torch.load(play_model_path))
                print(f"Loaded play card model for {self.name}")
            except Exception as e:
                print(f"Error loading play card model for {self.name}: {e}")
        else:
            print(f"No saved play card model for {self.name}, starting fresh.")

    def save_models(self):
        torch.save(self.bid_net.state_dict(), f"{self.name}_bid_net.pth")
        torch.save(self.play_card_net.state_dict(), f"{self.name}_play_card_net.pth")

    def vectorize_game_state(self, game_state_vector):
        return torch.tensor(game_state_vector, dtype=torch.float32).unsqueeze(0).to(device)

    def make_bid(self, bids, game_state_vector):
        self.determine_eligible_cards(None, False)
        vector = self.vectorize_game_state(game_state_vector)
        with torch.no_grad():
            bid_probs = self.bid_net(vector)
        bid = torch.argmax(bid_probs).item() + 1  # Bids are from 1 to 13
        self.last_bid = bid
        self.bid = bid  # Store the bid for reward calculation
        self.bid_memory.append((vector, bid - 1))
        return bid

    def play_card(self, leading_suit, spades_broken, game_state_vector):
        self.determine_eligible_cards(leading_suit, spades_broken)
        vector = self.vectorize_game_state(game_state_vector)
        with torch.no_grad():
            card_probs = self.play_card_net(vector).squeeze(0)
        eligible_indices = [self.hand.index(card) for card in self.eligible_cards]
        probs = torch.zeros(len(self.hand)).to(device)
        for idx in eligible_indices:
            probs[idx] = card_probs[idx]
        chosen_index = torch.argmax(probs).item()
        card = self.hand.pop(chosen_index)
        self.memory.append((vector, chosen_index))
        return card

    def update_rewards(self):
        # Calculate rewards based on performance
        # For bidding: reward if bid is met exactly
        bid_reward = 0
        if self.won_tricks == self.bid:
            bid_reward = 10  # Positive reward for meeting the bid
        else:
            bid_reward = -abs(self.bid - self.won_tricks) * 5  # Penalty for over/underbidding

        # For playing cards: reward for each trick won
        play_reward = self.won_tricks * 2  # Reward per trick won

        total_reward = bid_reward + play_reward

        # Update bid memory
        for i in range(len(self.bid_memory)):
            self.bid_memory[i] = (self.bid_memory[i][0], self.bid_memory[i][1], total_reward)

        # Update play memory
        for i in range(len(self.memory)):
            self.memory[i] = (self.memory[i][0], self.memory[i][1], total_reward)

    def train(self):
        # Train the bid network
        if self.bid_memory:
            states, bids, rewards = zip(*self.bid_memory)
            states = torch.cat(states)
            bids = torch.tensor(bids).to(device)
            rewards = torch.tensor(rewards).to(device)
            logits = self.bid_net(states)
            log_probs = F.log_softmax(logits, dim=1)
            selected_log_probs = log_probs[range(len(bids)), bids]
            loss = -(selected_log_probs * rewards).mean()
            self.bid_optimizer.zero_grad()
            loss.backward()
            self.bid_optimizer.step()
            self.bid_memory = []

        # Train the play card network
        if self.memory:
            states, actions, rewards = zip(*self.memory)
            states = torch.cat(states)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            logits = self.play_card_net(states)
            log_probs = F.log_softmax(logits, dim=1)
            selected_log_probs = log_probs[range(len(actions)), actions]
            loss = -(selected_log_probs * rewards).mean()
            self.play_card_optimizer.zero_grad()
            loss.backward()
            self.play_card_optimizer.step()
            self.memory = []

def create_players(num_human_players):
    players = []
    for i in range(num_human_players):
        players.append(HumanPlayer(f"Human {i+1}"))
    for i in range(4 - num_human_players):
        players.append(BotPlayer(f"Bot {i+1}", 'hard'))
    random.shuffle(players)
    return players

def assign_teams(players):
    team1 = [players[0], players[2]]
    team2 = [players[1], players[3]]
    for player in team1:
        player.set_team("Team 1")
    for player in team2:
        player.set_team("Team 2")
    return team1, team2

def determine_winner(trick, leading_suit):
    def card_rank(card):
        rank = card.split(" of ")[0]
        return ranks.index(rank)

    spades_played = [item for item in trick if "Spades" in item[1]]
    if spades_played:
        winning_card = max(spades_played, key=lambda x: card_rank(x[1]))
        return winning_card
    same_suit_cards = [item for item in trick if leading_suit in item[1]]
    if same_suit_cards:
        winning_card = max(same_suit_cards, key=lambda x: card_rank(x[1]))
        return winning_card
    return trick[0]

def vectorize_game_state(player, game_conditions, scoreboard, bids, team_tricks):
    vector = []

    # Team scores
    vector.append(scoreboard.team1_overall_score)
    vector.append(scoreboard.team2_overall_score)

    # Player's hand
    hand_vector = [0] * 52
    for card in player.hand:
        idx = card_to_index[card]
        hand_vector[idx] = 1
    vector.extend(hand_vector)

    # Bids
    bids_vector = [0] * 4
    for i, p in enumerate(game_conditions.players):
        bids_vector[i] = bids.get(p.name, 0)
    vector.extend(bids_vector)

    # Team tricks
    vector.append(team_tricks[player.team])

    # Current round
    vector.append(game_conditions.current_round)

    return vector

def main():
    num_human_players = 0
    winning_score = 300
    losing_score = -200  # Floor score for early termination
    game_conditions = GameConditions(num_human_players, 4 - num_human_players, winning_score, losing_score)
    players = create_players(num_human_players)
    game_conditions.players = players
    team1, team2 = assign_teams(players)
    scoreboard = Scoreboard()
    game_over = False
    while not game_over:
        deck = CardDeck()
        deck.shuffle_cards()
        for player in players:
            player.hand = []
            player.won_tricks = 0
            player.bid = 0
        deck.deal_cards(players, 13)
        bids = {}
        team_tricks = {"Team 1": 0, "Team 2": 0}
        for player in players:
            game_state_vector = vectorize_game_state(player, game_conditions, scoreboard, bids, team_tricks)
            bid = player.make_bid(bids, game_state_vector)
            bids[player.name] = bid
        team1_bid = sum(bids[player.name] for player in team1)
        team2_bid = sum(bids[player.name] for player in team2)
        for _ in range(13):
            trick = []
            leading_suit = None
            for player in players:
                game_state_vector = vectorize_game_state(player, game_conditions, scoreboard, bids, team_tricks)
                card = player.play_card(leading_suit, game_conditions.spades_broken, game_state_vector)
                if not leading_suit:
                    leading_suit = card.split(" of ")[1]
                    if leading_suit == "Spades":
                        game_conditions.spades_broken = True
                trick.append((player, card))
            winning_play = determine_winner(trick, leading_suit)
            winning_player = winning_play[0]
            winning_player.won_tricks += 1
            team_tricks[winning_player.team] += 1
            # Rotate players so the winner leads next
            winner_index = players.index(winning_player)
            players = players[winner_index:] + players[:winner_index]
        # Calculate scores
        team1_tricks_won = sum(player.won_tricks for player in team1)
        team2_tricks_won = sum(player.won_tricks for player in team2)
        team1_score, team2_score = scoreboard.calculate_score(team1_bid, team2_bid, team1_tricks_won, team2_tricks_won)
        print(f"Team 1 Score: {scoreboard.team1_overall_score}")
        print(f"Team 2 Score: {scoreboard.team2_overall_score}")
        # Update rewards and train bots
        for player in players:
            if player.bot:
                player.update_rewards()
                player.train()
                player.save_models()
        # Check for winning or losing conditions
        if (scoreboard.team1_overall_score >= winning_score or scoreboard.team2_overall_score >= winning_score) or \
           (scoreboard.team1_overall_score <= losing_score or scoreboard.team2_overall_score <= losing_score):
            game_over = True
            print("Game Over")
            if scoreboard.team1_overall_score > scoreboard.team2_overall_score:
                print("Team 1 Wins!")
            else:
                print("Team 2 Wins!")
            # Reset the game for the bots to continue learning
            scoreboard.reset()
            for player in players:
                player.hand = []
                player.won_tricks = 0
                player.bid = 0
            # Optionally, you can set game_over = False to let the bots keep playing indefinitely
            game_over = False  # Uncomment this line if you want continuous training

# Run the main function to start the game
if __name__ == "__main__":
    main()

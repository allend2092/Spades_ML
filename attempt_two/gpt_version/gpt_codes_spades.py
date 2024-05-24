import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Define Neural Networks
class BidNet(nn.Module):
    def __init__(self):
        super(BidNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(51, 128))
        for _ in range(8):  # Add 8 hidden layers
            self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.Linear(128, 13))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  # No activation; raw scores

class PlayCardNet(nn.Module):
    def __init__(self):
        super(PlayCardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(51, 128))
        for _ in range(8):  # Add 8 hidden layers
            self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.Linear(128, 13))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return F.softmax(self.layers[-1](x), dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bid_net = BidNet().to(device)
play_card_net = PlayCardNet().to(device)
optimizer_bid = torch.optim.Adam(bid_net.parameters(), lr=0.001)
optimizer_play = torch.optim.Adam(play_card_net.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
ranks = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

class CardDeck:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cards = {i + 1: (rank, suit) for i, (rank, suit) in enumerate([(rank, suit) for suit in suits for rank in ranks])}

    def shuffle(self):
        items = list(self.cards.items())
        random.shuffle(items)
        self.cards = dict(items)

    def deal(self, players, num_cards=13):
        for player in players:
            player.hand = []
            for _ in range(num_cards):
                if not self.cards:
                    raise KeyError("popitem(): dictionary is empty")
                player.hand.append(self.cards.popitem()[1])

class Scoreboard:
    def __init__(self):
        self.scores = {"Team 1": 0, "Team 2": 0}

    def update(self, bids, tricks):
        for team in self.scores:
            bid = bids[team]
            tricks_won = tricks[team]
            if tricks_won >= bid:
                self.scores[team] += 10 * bid + (tricks_won - bid)
            else:
                self.scores[team] -= 10 * bid

    def reset(self):
        self.scores = {"Team 1": 0, "Team 2": 0}

class Player:
    def __init__(self, name, is_bot=False):
        self.name = name
        self.team = None
        self.hand = []
        self.is_bot = is_bot
        self.net = None
        self.memory = []

    def set_team(self, team_name):
        self.team = team_name

    def vectorize(self):
        return [0] * 51  # Placeholder; actual vectorization logic needed

    def make_bid(self):
        return random.randint(1, 13)

    def play_card(self, leading_suit, spades_broken):
        eligible_cards = [card for card in self.hand if card[1] == leading_suit or (not spades_broken and card[1] == "Spades")]
        card = random.choice(eligible_cards if eligible_cards else self.hand)
        self.hand.remove(card)
        return card

class BotPlayer(Player):
    def __init__(self, name, bid_net, play_card_net):
        super().__init__(name, is_bot=True)
        self.bid_net = bid_net
        self.play_card_net = play_card_net

    def make_bid(self):
        state = torch.tensor(self.vectorize(), dtype=torch.float).to(device)
        with torch.no_grad():
            bid = torch.argmax(self.bid_net(state)).item() + 1
        return bid

    def play_card(self, leading_suit, spades_broken):
        state = torch.tensor(self.vectorize(), dtype=torch.float).unsqueeze(0).to(device)
        with torch.no_grad():
            card_probs = self.play_card_net(state).squeeze(0)
        card = random.choice(self.hand)  # Placeholder; select based on card_probs
        self.hand.remove(card)
        return card

def assign_teams(players):
    random.shuffle(players)
    for i, player in enumerate(players):
        player.set_team("Team 1" if i % 2 == 0 else "Team 2")

def save_models(bid_net, play_card_net):
    torch.save(bid_net.state_dict(), "bid_net.pth")
    torch.save(play_card_net.state_dict(), "play_card_net.pth")
    print("Models saved.")

def load_models(bid_net, play_card_net):
    if os.path.exists("bid_net.pth"):
        bid_net.load_state_dict(torch.load("bid_net.pth"))
        print("BidNet model loaded.")
    if os.path.exists("play_card_net.pth"):
        play_card_net.load_state_dict(torch.load("play_card_net.pth"))
        print("PlayCardNet model loaded.")

def main_game_loop(players, deck, scoreboard):
    num_hands = 13
    winning_score = 300
    num_episodes = 20000
    update_interval = 50  # How often to output progress

    load_models(bid_net, play_card_net)

    for episode in range(num_episodes):
        game_over = False
        deck.reset()
        deck.shuffle()
        deck.deal(players)
        assign_teams(players)

        bids = {"Team 1": 0, "Team 2": 0}
        tricks = {"Team 1": 0, "Team 2": 0}

        state_action_rewards = []

        for player in players:
            state = torch.tensor(player.vectorize(), dtype=torch.float).to(device)
            bid = player.make_bid()
            bids[player.team] += bid
            state_action_rewards.append((state, bid - 1, player.team))  # Store the state, action, and team

        for _ in range(num_hands):
            current_hand = []
            leading_suit = None
            spades_broken = False

            for player in players:
                card = player.play_card(leading_suit, spades_broken)
                if not leading_suit:
                    leading_suit = card[1]
                if card[1] == "Spades":
                    spades_broken = True
                current_hand.append((player, card))

            winning_card = max(current_hand, key=lambda x: (x[1][1] == leading_suit, x[1][0]))
            tricks[winning_card[0].team] += 1
            players = players[players.index(winning_card[0]):] + players[:players.index(winning_card[0])]

        scoreboard.update(bids, tricks)

        # Reward calculation and training
        for state, action, team in state_action_rewards:
            reward = 0
            if tricks[team] >= bids[team]:
                reward += 10 * bids[team] + (tricks[team] - bids[team])
            else:
                reward -= 10 * bids[team]

            if scoreboard.scores[team] > 0:
                reward += 50  # Bonus reward for positive score

            pred = bid_net(state.unsqueeze(0))
            loss = loss_fn(pred, torch.tensor([action], dtype=torch.long).to(device))
            optimizer_bid.zero_grad()
            loss.backward()
            optimizer_bid.step()

        if episode % update_interval == 0:
            print(f"Episode {episode}/{num_episodes}")
            print(f"Scores: {scoreboard.scores}")

        if any(score >= winning_score for score in scoreboard.scores.values()):
            print(f"Game Over at episode {episode}")
            print(f"Final Scores: {scoreboard.scores}")
            scoreboard.reset()

    save_models(bid_net, play_card_net)

def main():
    print("Welcome to Spades!")
    players = [BotPlayer(f"Bot {i}", bid_net, play_card_net) for i in range(4)]
    deck = CardDeck()
    scoreboard = Scoreboard()
    main_game_loop(players, deck, scoreboard)

if __name__ == "__main__":
    main()

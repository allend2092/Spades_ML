"""
Spades RL — Exploration + Reward-weighted updates

This version integrates your existing features (save/load, logging, legal-action masking, score floor)
AND adds the requested improvements so training escapes the "bid 13 forever" trap:

- ε-greedy / temperature **exploration** for bids and plays (configurable)
- **Reward-weighted** cross-entropy (simple policy-gradient-ish update)
- **Heuristic warm-start** for bidding for the first N episodes (configurable)
- Optional **bid clamps** early in training (configurable)
- Keeps: save/load after each episode, logs, score floor, final + checkpoint saves, 74-dim state

Tune knobs in the `Config` block at the top.
"""
from __future__ import annotations
import os
import sys
import csv
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Feature dimension
# -----------------
FEATURE_DIM = 74  # = 1(game_over)+3(sb)+4(bids)+2(tricks)+51(player vec)+13(round one-hot)

# =====================
# Configuration & Utils
# =====================
@dataclass
class Config:
    # Gameplay
    human_play_test: bool = False
    num_humans: int = 1
    target_points: int = 300
    losing_floor_points: int = -1000
    blind_nil_allowed: bool = False

    # Training
    training_mode: bool = True
    episodes: int = 10000
    checkpoint_every_n_episodes: int = 1000
    reward_score_floor: int = -500  # clamp round score contribution to reward

    # Exploration
    bid_eps_start: float = 0.30   # ε for ε-greedy bidding at start
    bid_eps_end: float = 0.05     # ε after anneal
    bid_eps_anneal_episodes: int = 3000
    bid_temp: float = 1.5         # softmax temperature for bids (used if use_temp_for_bids)
    use_temp_for_bids: bool = False

    play_temp: float = 1.20       # softmax temp for card plays
    play_eps: float = 0.10        # small ε to explore a legal random play

    # Heuristic warm-start for bids
    warm_start_episodes: int = 2000
    warm_start_prob: float = 0.5  # probability to use heuristic instead of NN during warm-start

    # Early bid clamp (helps stabilize)
    clamp_bids_for_first_n_episodes: int = 1500
    clamp_bid_min: int = 3
    clamp_bid_max: int = 8

    # Optim / device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_bid: float = 1e-3
    lr_play: float = 1e-3
    seed: int = 42

    # Logging
    log_dir: str = 'logs'
    play_csv: str = 'logs/play_log.csv'
    bids_csv: str = 'logs/bids_log.csv'
    loss_csv: str = 'logs/loss_log.csv'
    console_verbosity: int = logging.INFO

cfg = Config()
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
if cfg.device == 'cuda':
    torch.cuda.manual_seed_all(cfg.seed)

os.makedirs(cfg.log_dir, exist_ok=True)

logger = logging.getLogger('spades_rl')
logger.setLevel(cfg.console_verbosity)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(cfg.console_verbosity)
fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
ch.setFormatter(fmt)
logger.addHandler(ch)

# CSV headers
if not os.path.exists(cfg.play_csv):
    with open(cfg.play_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['episode','hand_index','trick_index','player','team','card','leading_suit','winner_player','winner_team'])
if not os.path.exists(cfg.bids_csv):
    with open(cfg.bids_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['episode','player','team','bid'])
if not os.path.exists(cfg.loss_csv):
    with open(cfg.loss_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['episode','bid_loss','play_loss'])

# ============
# Card helpers
# ============
SUITS = ["Spades", "Hearts", "Diamonds", "Clubs"]
RANKS = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]
RANK_TO_VAL = {"2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10, "Jack":11, "Queen":12, "King":13, "Ace":14}

def card_to_index(card: Tuple[str,str]) -> int:
    rank, suit = card
    return SUITS.index(suit)*13 + RANKS.index(rank)

def index_to_card(idx: int) -> Tuple[str,str]:
    return (RANKS[idx%13], SUITS[idx//13])

# ============
# Networks
# ============
class BidNet(nn.Module):
    def __init__(self, in_features: int = FEATURE_DIM, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 256)
        self.fc3 = nn.Linear(256, 14)  # 0..13
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PlayCardNet(nn.Module):
    def __init__(self, in_features: int = FEATURE_DIM, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 256)
        self.fc3 = nn.Linear(256, 52)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

bid_net = BidNet().to(cfg.device)
play_net = PlayCardNet().to(cfg.device)
opt_bid = torch.optim.Adam(bid_net.parameters(), lr=cfg.lr_bid)
opt_play = torch.optim.Adam(play_net.parameters(), lr=cfg.lr_play)

# Load existing models if present
def try_load_models():
    loaded = False
    if os.path.exists('final_bid.pth'):
        bid_net.load_state_dict(torch.load('final_bid.pth', map_location=cfg.device)); loaded = True
    if os.path.exists('final_play.pth'):
        play_net.load_state_dict(torch.load('final_play.pth', map_location=cfg.device)); loaded = True
    logger.info('Loaded existing weights from final_*.pth' if loaded else 'No existing weights found; starting fresh.')

# =========
# Game types
# =========
class CardDeck:
    def __init__(self):
        self.cards: Dict[int, Tuple[str,str]] = {}
        k = 1
        for s in SUITS:
            for r in RANKS:
                self.cards[k] = (r, s)
                k += 1
    def shuffle(self):
        vals = list(self.cards.values())
        random.shuffle(vals)
        for i, key in enumerate(list(self.cards.keys())):
            self.cards[key] = vals[i]
    def deal(self, players: List['Player'], n: int):
        it = iter(list(self.cards.items()))
        for _ in range(n):
            for p in players:
                _, v = next(it)
                p.hand.append(v)

class Scoreboard:
    def __init__(self):
        self.t1_overall = 0
        self.t2_overall = 0
        self.round_number = 0

# ======
# Agents
# ======
class Player:
    def __init__(self, name: str):
        self.name = name
        self.team: Optional[str] = None
        self.hand: List[Tuple[str,str]] = []
        self.score = 0
        self.card_played_last: Optional[Tuple[str,str]] = None
        self.eligible_cards: List[Tuple[str,str]] = []
        self.bot: Optional[bool] = None
        self.won_hands = 0
    def set_team(self, t: str):
        self.team = t
    def determine_eligible(self, leading_suit: Optional[str], spades_broken: bool):
        self.eligible_cards = []
        has_lead = any(c[1]==leading_suit for c in self.hand)
        if leading_suit is None and not spades_broken:
            if any(c[1] != 'Spades' for c in self.hand):
                self.eligible_cards = [c for c in self.hand if c[1] != 'Spades']
            else:
                self.eligible_cards = self.hand.copy()
        elif has_lead:
            self.eligible_cards = [c for c in self.hand if c[1]==leading_suit]
        else:
            self.eligible_cards = self.hand.copy()
    def vectorize_player(self, players: List['Player']) -> List[float]:
        team_id = 0 if self.team == 'Team 1' else 1
        base = [float(team_id), float(self.score), 0.0, float(self.won_hands)]
        base.extend([0.0]*(51-len(base)))
        return base

class HumanPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name)
        self.bot = False
    def display_hand(self):
        if not self.hand:
            logger.info("No cards in hand.")
            return
        sorted_hand = sorted(self.hand, key=lambda c: (SUITS.index(c[1]), RANKS.index(c[0])))
        logger.info(f"{self.name}'s hand: "+', '.join(f"{r} of {s}" for r,s in sorted_hand))
    def make_bid(self, bids: Dict[str,int], vector: List[float], episode: int) -> int:
        while True:
            logger.info(f"Current bids: {bids}")
            logger.info("Options: 1) See hand & bid  2) Bid NIL  3) Quit")
            try:
                action = int(input("Choose: "))
            except ValueError:
                logger.info("Enter a number, please.")
                continue
            if action == 1:
                self.display_hand()
                try:
                    b = int(input("Your bid (0..13): "))
                    return max(0, min(13, b))
                except ValueError:
                    logger.info("Invalid number.")
            elif action == 2:
                return 0
            elif action == 3:
                sys.exit(0)
            else:
                logger.info("Not an option.")
    def play_card(self, leading_suit: Optional[str], spades_broken: bool, vector: List[float], episode: int) -> Tuple[str,str]:
        self.determine_eligible(leading_suit, spades_broken)
        logger.info("Eligible: "+', '.join(f"{i+1}. {r} of {s}" for i,(r,s) in enumerate(self.eligible_cards)))
        while True:
            try:
                idx = int(input("Play which (number): "))-1
                if 0 <= idx < len(self.eligible_cards):
                    c = self.eligible_cards[idx]
                    self.card_played_last = c
                    self.hand.remove(c)
                    return c
                logger.info("Invalid choice.")
            except ValueError:
                logger.info("Enter a number.")

class BotPlayer(Player):
    def __init__(self, name: str):
        super().__init__(name)
        self.bot = True
        self.bid_memory: List[Tuple[List[float], int, float]] = []
        self.play_memory: List[Tuple[List[float], int, float]] = []
        self.bid_last: Optional[int] = None
        self.last_bid_state: Optional[List[float]] = None
    # --- Heuristic bid ---
    @staticmethod
    def heuristic_bid(hand: List[Tuple[str,str]]) -> int:
        highs = sum(1 for r,s in hand if r in ("Ace","King","Queen","Jack"))
        spades = sum(1 for _,s in hand if s == "Spades")
        est = highs + max(0, spades - 2)//2
        return max(cfg.clamp_bid_min, min(cfg.clamp_bid_max, est))
    # --- Epsilon schedule ---
    @staticmethod
    def eps_for_episode(ep: int) -> float:
        if ep >= cfg.bid_eps_anneal_episodes: return cfg.bid_eps_end
        frac = ep / max(1, cfg.bid_eps_anneal_episodes)
        return cfg.bid_eps_start + (cfg.bid_eps_end - cfg.bid_eps_start)*frac
    # --- Decisions ---
    def make_bid(self, bids: Dict[str,int], vector: List[float], episode: int) -> int:
        assert len(vector) == FEATURE_DIM
        self.last_bid_state = vector[:]
        # Warm-start: sometimes use heuristic early on
        if episode < cfg.warm_start_episodes and random.random() < cfg.warm_start_prob:
            b = self.heuristic_bid(self.hand)
        else:
            x = torch.tensor([vector], dtype=torch.float, device=cfg.device)
            with torch.no_grad():
                logits = bid_net(x).squeeze(0)
                if cfg.use_temp_for_bids:
                    probs = torch.softmax(logits / cfg.bid_temp, dim=0)
                    b = int(torch.multinomial(probs, 1).item())
                else:
                    eps = self.eps_for_episode(episode)
                    if random.random() < eps:
                        b = random.randint(0, 13)
                    else:
                        b = int(torch.argmax(logits).item())
        # Early clamp (optional)
        if episode < cfg.clamp_bids_for_first_n_episodes:
            b = max(cfg.clamp_bid_min, min(cfg.clamp_bid_max, b))
        self.bid_last = b
        return b
    def play_card(self, leading_suit: Optional[str], spades_broken: bool, vector: List[float], episode: int) -> Tuple[str,str]:
        assert len(vector) == FEATURE_DIM
        self.determine_eligible(leading_suit, spades_broken)
        x = torch.tensor([vector], dtype=torch.float, device=cfg.device)
        logits = play_net(x).squeeze(0)
        mask = torch.full((52,), -1e9, device=cfg.device)
        for c in self.eligible_cards:
            mask[card_to_index(c)] = 0.0
        # With small epsilon, randomly pick among legal moves
        if self.eligible_cards and random.random() < cfg.play_eps:
            chosen = random.choice(self.eligible_cards)
            a_idx = card_to_index(chosen)
        else:
            probs = F.softmax((logits + mask) / cfg.play_temp, dim=0)
            a_idx = int(torch.multinomial(probs, 1).item())
            chosen = index_to_card(a_idx)
            if chosen not in self.eligible_cards:
                chosen = random.choice(self.eligible_cards) if self.eligible_cards else self.hand.pop()
                a_idx = card_to_index(chosen)
        self.card_played_last = chosen
        self.hand.remove(chosen)
        self.play_memory.append((vector[:], a_idx, 0.0))
        return chosen
    # --- Rewards ---
    def trick_reward(self, winner_player: 'Player'):
        r = 50.0 if winner_player.name == self.name else -30.0
        r += 50.0 if winner_player.team == self.team else -30.0
        if self.play_memory:
            s, a, _ = self.play_memory[-1]
            self.play_memory[-1] = (s, a, r)
        return r
    def end_round_bid_reward(self, my_tricks: int, round_score: int, team_score: int, team_tricks: int, team_bid: int):
        rscore = max(cfg.reward_score_floor, round_score)
        bid_err = abs((self.bid_last or 0) - my_tricks)
        r = -10.0*bid_err
        r += 100.0 if rscore > 0 else -50.0
        r += 50.0 if team_score > 0 else -25.0
        if team_tricks > team_bid: r += 75.0
        elif team_tricks < team_bid: r -= 75.0
        state = self.last_bid_state if self.last_bid_state is not None else [0.0]*FEATURE_DIM
        self.bid_memory.append((state[:], int(self.bid_last or 0), r))
        return r

# =====================
# Mechanics / Utilities
# =====================

def determine_winning_card(current_hand: List[Tuple[Player, Tuple[str,str]]]) -> Tuple[Player, Tuple[str,str]]:
    leading_suit = current_hand[0][1][1]
    spades = [(p,c) for (p,c) in current_hand if c[1]=="Spades"]
    if spades:
        return max(spades, key=lambda pc: RANK_TO_VAL[pc[1][0]])
    led = [(p,c) for (p,c) in current_hand if c[1]==leading_suit]
    return max(led, key=lambda pc: RANK_TO_VAL[pc[1][0]]) if led else current_hand[0]

def rotate_dealer(players: List[Player], dealer: Player) -> Player:
    i = players.index(dealer)
    return players[(i+1)%len(players)]

def assign_tricks_to_team(current_hand: List, winning: Tuple[Player, Tuple[str,str]], t1: List, t2: List):
    wp = winning[0]
    (t1 if wp.team=="Team 1" else t2).extend(current_hand)

# ==============
# Vectorization
# ==============

def one_hot_round(rn: int) -> List[int]:
    v = [0]*13
    if 1 <= rn <= 13:
        v[rn-1] = 1
    return v

def vectorize_state(game_over: bool, sb, current_bids: Dict[str,int],
                    t1_tricks: List, t2_tricks: List, rn: int, player: Player, players: List[Player]) -> List[float]:
    game_over_v = [1.0 if game_over else 0.0]
    sb_v = [float(sb.t1_overall), float(sb.t2_overall), float(sb.round_number)]
    bids_v = [float(current_bids.get(p.name, 0)) for p in players]
    tricks_v = [float(len(t1_tricks)//4), float(len(t2_tricks)//4)]
    pr_v = player.vectorize_player(players)
    rn_v = one_hot_round(rn)
    vec = game_over_v + sb_v + bids_v + tricks_v + pr_v + rn_v
    assert len(vec) == FEATURE_DIM, f"vectorize_state produced {len(vec)} != {FEATURE_DIM}"
    return vec

# ===================
# Training procedures
# ===================

def _norm_rewards(rs: List[float]) -> torch.Tensor:
    t = torch.tensor(rs, dtype=torch.float, device=cfg.device)
    if t.numel() == 0:
        return t
    t = (t - t.mean()) / (t.std() + 1e-6)
    return torch.clamp(t, -2.0, 2.0)

def train_from_memory(bots: List[BotPlayer], episode: int):
    if not cfg.training_mode:
        return 0.0, 0.0
    play_states, play_actions, play_rewards = [], [], []
    bid_states, bid_targets, bid_rewards = [], [], []
    for b in bots:
        for s,a,r in b.play_memory:
            play_states.append(s); play_actions.append(a); play_rewards.append(r)
        b.play_memory.clear()
    for b in bots:
        for s,a,r in b.bid_memory:
            bid_states.append(s); bid_targets.append(a); bid_rewards.append(r)
        b.bid_memory.clear()
    bid_loss_val = 0.0; play_loss_val = 0.0
    if play_states:
        xs = torch.tensor(play_states, dtype=torch.float, device=cfg.device)
        ya = torch.tensor(play_actions, dtype=torch.long, device=cfg.device)
        rw = _norm_rewards(play_rewards)
        logits = play_net(xs)
        ce = F.cross_entropy(logits, ya, reduction='none')
        loss_play = (ce * (1.0 + rw)).mean()
        opt_play.zero_grad(); loss_play.backward(); opt_play.step()
        play_loss_val = float(loss_play.item())
    if bid_states:
        xs = torch.tensor(bid_states, dtype=torch.float, device=cfg.device)
        yb = torch.tensor(bid_targets, dtype=torch.long, device=cfg.device)
        rw = _norm_rewards(bid_rewards)
        logits = bid_net(xs)
        ce = F.cross_entropy(logits, yb, reduction='none')
        loss_bid = (ce * (1.0 + rw)).mean()
        opt_bid.zero_grad(); loss_bid.backward(); opt_bid.step()
        bid_loss_val = float(loss_bid.item())
    with open(cfg.loss_csv, 'a', newline='') as f:
        csv.writer(f).writerow([episode, bid_loss_val, play_loss_val])
    if (episode % 5) == 0:
        logger.info(f"[ep {episode}] bid_loss={bid_loss_val:.4f} play_loss={play_loss_val:.4f}")
    return bid_loss_val, play_loss_val

# ==========
# Game setup
# ==========

def create_players(num_humans: int, total: int = 4) -> Tuple[List[Player], Player]:
    ps: List[Player] = []
    for i in range(num_humans): ps.append(HumanPlayer(f"Human {i+1}"))
    for i in range(total - num_humans): ps.append(BotPlayer(f"Bot {i+1}"))
    dealer = random.choice(ps)
    return ps, dealer

def arrange_players(players: List[Player], dealer: Player) -> List[Player]:
    di = players.index(dealer)
    ordered = players[di+1:] + players[:di+1]
    for p in ordered: p.won_hands = 0
    return ordered

def assign_teams(players: List[Player]) -> Tuple[List[Player], List[Player]]:
    random.shuffle(players)
    t1, t2 = [], []
    for i,p in enumerate(players):
        (t1 if i%2==0 else t2).append(p)
    for p in t1: p.set_team("Team 1")
    for p in t2: p.set_team("Team 2")
    return t1, t2

# ================
# The main episode
# ================

def play_episode(ep: int, players: List[Player], dealer: Player, params):
    current_bids: Dict[str,int] = {}
    t1_tricks: List = []
    t2_tricks: List = []
    class SB: pass
    sb = SB(); sb.t1_overall = 0; sb.t2_overall = 0; sb.round_number = 0
    game_over = False

    deck = CardDeck(); deck.shuffle(); deck.deal(players, 13)

    while not game_over:
        # Bidding
        for p in players:
            vec = vectorize_state(False, sb, current_bids, t1_tricks, t2_tricks, 0, p, players)
            bid = p.make_bid(current_bids, vec, ep) if isinstance(p, BotPlayer) else p.make_bid(current_bids, vec, ep)
            bid = max(0, min(13, bid))
            current_bids[p.name] = bid
            with open(cfg.bids_csv, 'a', newline='') as f:
                csv.writer(f).writerow([ep, p.name, p.team, bid])
        t1_bid = min(sum(current_bids[p.name] for p in players if p.team=="Team 1"), 13)
        t2_bid = min(sum(current_bids[p.name] for p in players if p.team=="Team 2"), 13)
        params.team1_bid = max(4, t1_bid)
        params.team2_bid = max(4, t2_bid)
        logger.info(f"Bids — Team1:{params.team1_bid} Team2:{params.team2_bid}")

        # 13 tricks
        for trick in range(13):
            current_hand: List[Tuple[Player, Tuple[str,str]]] = []
            first = True
            for p in players:
                p.determine_eligible(None if first else params.leading_suit, params.spades_broken)
                vec = vectorize_state(False, sb, current_bids, t1_tricks, t2_tricks, trick+1, p, players)
                c = p.play_card(params.leading_suit if not first else None, params.spades_broken, vec, ep)
                if first:
                    params.leading_suit = c[1]
                    first = False
                if c[1] == 'Spades': params.spades_broken = True
                current_hand.append((p,c))
            winner = determine_winning_card(current_hand)
            winner[0].won_hands += 1
            assign_tricks_to_team(current_hand, winner, t1_tricks, t2_tricks)
            for p,_ in current_hand:
                if isinstance(p, BotPlayer): p.trick_reward(winner[0])
            wi = players.index(winner[0])
            players = players[wi:] + players[:wi]
            with open(cfg.play_csv, 'a', newline='') as f:
                w = csv.writer(f)
                for pl, cd in current_hand:
                    w.writerow([ep, sb.round_number, trick, pl.name, pl.team, f"{cd[0]} of {cd[1]}", params.leading_suit, winner[0].name, winner[0].team])
            params.leading_suit = None
        # End of hand
        dealer = rotate_dealer(players, dealer)
        players = arrange_players(players, dealer)
        params.spades_broken = False
        # Score hand (same as before)
        def score_team(bid, tricks):
            made = len(tricks)//4
            return 10*bid + (made - bid) if made >= bid else -10*bid
        s1 = score_team(params.team1_bid, t1_tricks)
        s2 = score_team(params.team2_bid, t2_tricks)
        sb.t1_overall += s1; sb.t2_overall += s2
        logger.info(f"Scores — Total: T1 {sb.t1_overall} T2 {sb.t2_overall} | Round: T1 {s1} T2 {s2}")
        logger.info(f"Tricks — T1 {len(t1_tricks)//4} / bid {t1_bid}  |  T2 {len(t2_tricks)//4} / bid {t2_bid}")
        for p in players:
            if isinstance(p, BotPlayer):
                if p.team == 'Team 1': p.end_round_bid_reward(p.won_hands, s1, sb.t1_overall, len(t1_tricks)//4, t1_bid)
                else: p.end_round_bid_reward(p.won_hands, s2, sb.t2_overall, len(t2_tricks)//4, t2_bid)
        bots = [p for p in players if isinstance(p, BotPlayer)]
        train_from_memory(bots, ep)
        # End of game?
        if (sb.t1_overall >= cfg.target_points or sb.t2_overall >= cfg.target_points or
            sb.t1_overall <= cfg.losing_floor_points or sb.t2_overall <= cfg.losing_floor_points):
            logger.info(f"Game Over — Final: Team1 {sb.t1_overall} Team2 {sb.t2_overall}")
            game_over = True
        else:
            t1_tricks.clear(); t2_tricks.clear(); current_bids.clear()
            deck = CardDeck(); deck.shuffle(); deck.deal(players, 13)

# =====
# Main
# =====

def main():
    logger.info("Spades RL — starting up…")
    try_load_models()
    humans = cfg.num_humans if cfg.human_play_test else 0
    class Params: pass
    params = Params(); params.num_humans = humans; params.num_bots = 4-humans; params.threshold_score = cfg.target_points
    params.blind_nil_allowed = cfg.blind_nil_allowed; params.current_round = 0; params.leading_suit=None; params.spades_broken=False
    params.team1_bid=None; params.team2_bid=None

    players, dealer = create_players(humans)
    players = arrange_players(players, dealer)
    assign_teams(players)

    for ep in range(cfg.episodes if cfg.training_mode else 1):
        play_episode(ep, players, dealer, params)
        # Save after each episode
        torch.save(play_net.state_dict(), 'final_play.pth')
        torch.save(bid_net.state_dict(), 'final_bid.pth')
        if cfg.training_mode and (ep+1) % cfg.checkpoint_every_n_episodes == 0:
            torch.save(play_net.state_dict(), f'chk_play_ep{ep+1}.pth')
            torch.save(bid_net.state_dict(), f'chk_bid_ep{ep+1}.pth')
    logger.info("Done.")

if __name__ == '__main__':
    main()

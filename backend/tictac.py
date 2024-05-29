import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from enum import Enum


REWARDS = []


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(18, 128),  # Input layer with 9 features
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, 9),  # Output layer with 9 possible actions
            nn.Softmax(dim=-1),  # Softmax to get probabilities for each action
        )

    def forward(self, x):
        return self.fc(x)


class GameStates(Enum):
    GAMEON = 0
    DRAW = 1
    WON = 2


class Players(Enum):
    ONE = "X"
    TWO = "O"
    EMPTY = " "


# Initialize the game state with empty cells


# Function to place a move in the game state
def place_move(game_state, index, player):
    game_state[index] = player


# Function to check if a player has won
def check_winner(game_state, player):
    # Winning combinations (indices)
    win_combinations = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # Rows
        [0, 3, 4],
        [1, 4, 7],
        [2, 5, 8],  # Columns
        [0, 4, 8],
        [2, 4, 6],  # Diagonals
    ]
    for combo in win_combinations:
        if all(game_state[i] == player for i in combo):
            return True
    return False


def gen_possible_moves(game_state):
    return [i for i, cell in enumerate(game_state) if cell == " "]


# Function to check if the game is a draw
def check_draw(game_state):
    return all(cell != Players.EMPTY.value for cell in game_state)


def print_board(game_state):
    board = [
        game_state[i] if game_state[i] != Players.EMPTY.value else game_state[i]
        for i in range(9)
    ]
    for row in range(3):
        print(" | ".join(board[row * 3 : row * 3 + 3]))
        if row < 2:
            print("---------")


def get_valid_input(board) -> int:
    # print_board(board)
    # Possible moves will be the REAL index
    possible_moves = gen_possible_moves(board)
    shown_moves = [i + 1 for i in possible_moves]
    while True:
        print(f"possible_moves {shown_moves}")
        player_choice = input("Where do you want to place a piece? ")
        if player_choice.isdigit() and int(player_choice) in shown_moves:
            player_choice = int(player_choice) - 1  # type: ignore
            break
        else:
            print("Invalid move! Try again")
    return player_choice  # type: ignore


def game_draw(game_state):
    print("Game got draw. No one wins!")
    print()


def player_turn(game_state, player):
    player_choice = get_valid_input(game_state)
    place_move(game_state, player_choice, player)

    if check_draw(game_state):
        return GameStates.DRAW
    elif check_winner(game_state, player):
        return GameStates.WON
    else:
        return GameStates.GAMEON


def handle_states(board, player, state):
    if state is GameStates.DRAW:
        print("Game over! Game is a draw")
        print()
        return True
    elif state is GameStates.WON:
        print(f"Congratulations Player: {player} you won!")
        print()
        return True
    return False

    pass


def play_one_game():
    player_one, player_two = Players.ONE.value, Players.TWO.value
    board = np.full(9, Players.EMPTY.value)

    # start_players turn
    while True:
        # Move returns correct index in array
        state = player_turn(board, player_one)
        gameover = handle_states(board, player_one, state)
        if gameover:
            break

        state = player_turn(board, player_two)
        gameover = handle_states(board, player_two, state)
        if gameover:
            break

    # change logic


def play_pve(policy_net, human_starts=True):
    player_human = Players.ONE.value if human_starts else Players.TWO.value
    player_ai = Players.TWO.value if human_starts else Players.ONE.value
    board = np.full(9, Players.EMPTY.value)

    current_player = Players.ONE if human_starts else Players.TWO

    while True:
        if current_player.value == player_human:
            state = player_turn(board, player_human)
        else:
            state_tensor = torch.tensor(board == Players.ONE.value, dtype=torch.float32)
            state_tensor = torch.cat(
                (
                    state_tensor,
                    torch.tensor(board == Players.TWO.value, dtype=torch.float32),
                )
            )
            action_probs = policy_net(state_tensor)

            if torch.isnan(action_probs).any():
                print(
                    "NaNs detected in action_probs, replacing with small positive value."
                )
                action_probs = torch.where(
                    torch.isnan(action_probs), torch.tensor(1e-8), action_probs
                )

            action = torch.multinomial(action_probs, 1).item()
            possible_moves = gen_possible_moves(board)
            if action not in possible_moves:
                action = np.random.choice(possible_moves)

            place_move(board, action, player_ai)
            print(f"AI placed {player_ai} at position {action + 1}")
            print_board(board)
            print(len(board))

            state = GameStates.GAMEON
            if check_winner(board, current_player):
                state = GameStates.WON
            elif check_draw(board):
                state = GameStates.DRAW

        gameover = handle_states(board, current_player, state)
        if gameover:
            break

        current_player = Players.TWO if current_player == Players.ONE else Players.ONE


def train_ai_one_game(policy_net1, policy_net2):
    # Initialize the game state
    board = np.full(9, Players.EMPTY.value)
    states, actions, rewards = [], [], []
    current_player = Players.ONE
    policy_nets = {Players.ONE: policy_net1, Players.TWO: policy_net2}

    while True:
        # Create a state tensor representing the current game state
        state_tensor = torch.tensor(
            board == Players.ONE.value, dtype=torch.float32, device="cuda"
        )
        state_tensor = torch.cat(
            (
                state_tensor,
                torch.tensor(
                    board == Players.TWO.value, dtype=torch.float32, device="cuda"
                ),
            )
        )

        # Get action probabilities from the policy network
        action_probs = policy_nets[current_player](state_tensor)

        if torch.isnan(action_probs).any():
            print("NaNs detected in action_probs, replacing with small positive value.")
            action_probs = torch.where(
                torch.isnan(action_probs),
                torch.tensor(1e-8, device="cuda"),
                action_probs,
            )

        print(f"Actions = {action_probs}")
        action = torch.multinomial(action_probs, 1).item()

        # Ensure the chosen action is valid
        possible_moves = gen_possible_moves(board)
        if action not in possible_moves:
            action = np.random.choice(possible_moves)

        # Place the move and record the state and action
        place_move(board, action, current_player.value)
        states.append(state_tensor.cpu().numpy())
        actions.append(action)

        # Check game outcome
        if check_winner(board, current_player.value):
            rewards = [
                1 if current_player == Players.ONE else -2 for _ in range(len(states))
            ]
            return states, actions, rewards

        elif check_draw(board):
            rewards = [-1 for _ in range(len(states))]
            return states, actions, rewards

        # Switch player
        current_player = Players.TWO if current_player == Players.ONE else Players.ONE


def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        discounted_rewards.std() + 1e-3
    )

    return discounted_rewards


def train(env, policy_net1, policy_net2, optimizer1, optimizer2, episodes=1000):
    for episode in range(episodes):
        states, actions, rewards = train_ai_one_game(policy_net1, policy_net2)

        print(rewards)
        if len(rewards) == 0:

            print(f"Episode {episode}: No rewards generated. Skipping update.")
            continue

        rewards = torch.tensor(rewards, dtype=torch.float32)
        print(f"Episode {episode}: Tensor Rewards: {rewards}")
        # Discount rewards

        discounted_rewards = compute_discounted_rewards(rewards)

        for state, action, reward in zip(states, actions, discounted_rewards):
            state = torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0)
            action = torch.tensor([action], device="cuda")
            if not state.is_cuda or not action.is_cuda:
                print("bad things happen")
                print(state.is_cuda)
                print(action.is_cuda)
                exit()

            # Update policy_net1
            optimizer1.zero_grad()
            log_prob1 = torch.log(policy_net1(state).gather(1, action.view(-1, 1)))
            loss1 = -log_prob1 * reward
            loss1.mean().backward()
            optimizer1.step()

            # Update policy_net2
            optimizer2.zero_grad()
            log_prob2 = torch.log(policy_net2(state).gather(1, action.view(-1, 1)))
            loss2 = -log_prob2 * reward
            loss2.mean().backward()
            optimizer2.step()


def try_ai():
    policy_net1 = PolicyNetwork()
    policy_net2 = PolicyNetwork()

    # policy_net1 = policy_net1.todevice("cuda")
    # policy_net2 = policy_net2.todevice("cuda")
    policy_net1.cuda(0)
    policy_net2.cuda(0)

    optimizer1 = optim.Adam(policy_net1.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(policy_net2.parameters(), lr=1e-3)
    # policy_net1.to(device)
    # policy_net2.to(device)

    train(None, policy_net1, policy_net2, optimizer1, optimizer2, episodes=1000)

    play_pve(policy_net1.cpu())


if __name__ == "__main__":
    try_ai()
    # play_one_game()

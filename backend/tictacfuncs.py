from enum import Enum
import numpy as np
import random


class GameStates(Enum):
    GAMEON = 0
    DRAW = 1
    WON = 2


class Players(Enum):
    ONE = 1
    TWO = -1
    EMPTY = 0


# class Players(Enum):
#     ONE = "X"
#     TWO = "O"
#     EMPTY = " "


def place_move(game_state, index, player):
    game_state[index] = player


def place_move_ai_action(board, action: int, player: str):
    valid_actions = gen_possible_moves(board)
    reward = 1 if action in valid_actions else -1

    if action not in valid_actions:
        action = random.choice(valid_actions)

    new_board = np.copy(board)
    new_board[action] = player

    done = False
    if check_winner(new_board, player):
        done = True
        reward = 20
    elif check_draw(new_board):
        done = True
        reward = -10

    return new_board, reward, done, None

    # board, reward, done, info


def get_gamestate(board, player: int):
    if check_winner(board, player):
        return GameStates.WON
    elif check_draw(board):
        return GameStates.DRAW
    else:
        return GameStates.GAMEON


# Function to check if a player has won
def check_winner(game_state, player: int):
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
    return [i for i, cell in enumerate(game_state) if cell == Players.EMPTY.value]


# Function to check if the game is a draw
def check_draw(game_state):
    return all(cell != Players.EMPTY.value for cell in game_state)


def print_board(board):
    if board.size != 9:
        raise ValueError("Board must have exactly 9 elements")

    symbols = {Players.ONE: "X", Players.TWO: "O", Players.EMPTY: " "}

    for i in range(0, 9, 3):
        row = [symbols[Players(board[j])] for j in range(i, i + 3)]
        print(" | ".join(row))
        if i < 6:
            print("-" * 10)


# def print_board(game_state):
#     print(game_state)
#     return
#     board = [
#         game_state[i] if game_state[i] != Players.EMPTY.value else game_state[i]
#         for i in range(9)
#     ]
#     for row in range(3):
#         print(" | ".join(board[row * 3 : row * 3 + 3]))
#         if row < 2:
#             print("---------")


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


def init_board():
    return np.full(9, Players.EMPTY.value, dtype=np.float32)

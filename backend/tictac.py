import numpy as np

from enum import Enum


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
    print_board(board)
    # Possible moves will be the REAL index
    possible_moves = gen_possible_moves(board)
    shown_moves = [i + 1 for i in possible_moves]
    while True:
        print(f"possible_moves {shown_moves}")
        player_choice = input("Where do you want to place a piece? ")
        if player_choice.isdigit() and int(player_choice) in shown_moves:
            player_choice = int(player_choice) - 1
            break
        else:
            print("Invalid move! Try again")
    return player_choice


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


def main_loop():

    while True:
        player_one, player_two = Players.ONE.value, Players.TWO.value
        board = np.full(9, " ")

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


if __name__ == "__main__":
    main_loop()
    pass

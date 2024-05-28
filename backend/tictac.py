def print_board(board=None):
    board = [str(i) for i in range(1, 10)]
    for row in range(3):
        print(" | ".join(board[row * 3 : row * 3 + 3]))
        if row < 2:
            print("---------")


def get_valid_input():
    print_board()
    possible_moves = [i for i in range(1, 10)]
    while True:
        print(f"possible_moves {possible_moves}")
        player_choice = input("Where do you want to place a piece?")
        if player_choice.isdigit() and int(player_choice) in possible_moves:
            break
        else:
            print("Invalid move! Try again")
    return player_choice


def main_loop():
    # start_players turn
    first_player_choice = get_valid_input()
    # change logic


if __name__ == "__main__":
    main_loop()
    pass

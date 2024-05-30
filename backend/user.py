from ai_train import Agent, DeepQNetwork
import tictacfuncs as game
import random


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


def get_valid_input(board) -> int:
    # print_board(board)
    # Possible moves will be the REAL index
    possible_moves = game.gen_possible_moves(board)
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


def handle_gameover(gamestate, player, was_ai):
    if gamestate is game.GameStates.WON:
        print(f"Player {player} won!!")
        exit()

    if gamestate is game.GameStates.DRAW:
        print("It is a draw!!")
        exit()


def player_turn(game_state, player):
    player_choice = get_valid_input(game_state)
    game.place_move(game_state, player_choice, player)


def play_pve(agent: Agent, human_starts=True):
    player_human = game.Players.ONE.value if human_starts else game.Players.TWO.value
    player_ai = game.Players.TWO.value if human_starts else game.Players.ONE.value
    board = game.init_board()

    current_player = game.Players.ONE.value if human_starts else game.Players.TWO.value

    while True:
        if current_player == player_human:
            game.print_board(board)
            player_turn(board, player_human)

        else:
            ai_decicion = agent.choose_action(board)
            valid_moves = game.gen_possible_moves(board)
            if ai_decicion not in valid_moves:
                print("ai did random move")
                ai_decicion = random.choice(valid_moves)
            game.place_move(board, ai_decicion, current_player)

        game_state = game.get_gamestate(board, current_player)
        if game_state is game.GameStates.WON or game_state is game.GameStates.DRAW:
            handle_gameover(game_state, current_player, current_player == player_ai)

        current_player = (
            game.Players.TWO.value
            if current_player == game.Players.ONE.value
            else game.Players.ONE.value
        )


if __name__ == "__main__":
    agent: Agent = Agent.load("agent.pkl")
    play_pve(agent)

    pass

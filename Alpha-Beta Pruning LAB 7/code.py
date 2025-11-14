# auto_tic_tac_toe_alphabeta.py
# Automatic Tic-Tac-Toe demonstrations using Minimax + Alpha-Beta pruning
# No user input required: the script runs games automatically.

import math
import random
import time

EMPTY = " "
X = "X"   # we'll treat X as +1 (maximizer)
O = "O"   # O is -1 (minimizer)

WIN_LINES = [
    (0,1,2), (3,4,5), (6,7,8),   # rows
    (0,3,6), (1,4,7), (2,5,8),   # cols
    (0,4,8), (2,4,6)             # diags
]

def new_board():
    return [EMPTY] * 9

def print_board(board):
    for r in range(3):
        row = board[3*r:3*r+3]
        print(" " + " | ".join(row))
        if r < 2:
            print("---+---+---")
    print()

def available_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]

def is_winner(board, player):
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def is_full(board):
    return all(cell != EMPTY for cell in board)

def evaluate(board):
    """Terminal evaluation: +1 if X wins, -1 if O wins, 0 otherwise."""
    if is_winner(board, X):
        return 1
    if is_winner(board, O):
        return -1
    return 0

def minimax_alpha_beta(board, player, alpha, beta, depth=0):
    """
    General minimax with alpha-beta pruning for Tic-Tac-Toe.
    player: the player to move now (X or O).
    Returns (best_score, best_move_index).
    Depth is used only to prefer faster wins / slower losses.
    """
    score = evaluate(board)
    if score != 0 or is_full(board):
        # If terminal, prefer faster win / slower loss by factoring depth
        if score == 1:
            return 10 - depth, None   # X wins -> positive
        if score == -1:
            return depth - 10, None  # O wins -> negative
        return 0, None               # draw

    if player == X:
        max_eval = -math.inf
        best_move = None
        for move in available_moves(board):
            board[move] = X
            eval_score, _ = minimax_alpha_beta(board, O, alpha, beta, depth+1)
            board[move] = EMPTY

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # beta cut-off
        return max_eval, best_move
    else:  # player == O (minimizer)
        min_eval = math.inf
        best_move = None
        for move in available_moves(board):
            board[move] = O
            eval_score, _ = minimax_alpha_beta(board, X, alpha, beta, depth+1)
            board[move] = EMPTY

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # alpha cut-off
        return min_eval, best_move

def best_move_alphabeta(board, player):
    """Return best move for player using alpha-beta pruning."""
    _, move = minimax_alpha_beta(board, player, -math.inf, math.inf, depth=0)
    return move

def best_move_random(board):
    """Random move for weak opponent."""
    moves = available_moves(board)
    return random.choice(moves) if moves else None

def play_auto_game(starting_player, mode="AIvRandom", verbose=True, delay=0.3):
    """
    Play a single automatic game.
    starting_player: X or O
    mode: "AIvRandom" or "AIvAI"
    verbose: print board and moves if True
    delay: seconds to wait between moves (for readability)
    Returns final board and result string.
    """
    board = new_board()
    current = starting_player

    if verbose:
        print("Starting automatic game. Mode:", mode)
        print("Starting player:", current)
        print_board(board)
        time.sleep(delay)

    while True:
        if mode == "AIvRandom":
            if current == X:
                move = best_move_alphabeta(board, X)
            else:  # O is random
                move = best_move_random(board)
        else:  # "AIvAI"
            move = best_move_alphabeta(board, current)

        if move is None:
            # No moves left (should be handled by terminal checks)
            break

        board[move] = current
        if verbose:
            print(f"{current} -> {move+1}")
            print_board(board)
            time.sleep(delay)

        if is_winner(board, current):
            if current == X:
                result = "X wins"
            else:
                result = "O wins"
            if verbose:
                print("Result:", result)
            return board, result

        if is_full(board):
            if verbose:
                print("Result: Draw")
            return board, "Draw"

        current = O if current == X else X

def demo():
    random.seed(1)  # deterministic randomness for repeatability
    print("\n--- Demo 1: AI (X, alpha-beta) vs Random (O) ---\n")
    board, result = play_auto_game(starting_player=X, mode="AIvRandom", verbose=True, delay=0.15)

    print("\n--- Demo 2: AI (X) vs AI (O) (both alpha-beta) ---\n")
    board2, result2 = play_auto_game(starting_player=X, mode="AIvAI", verbose=True, delay=0.15)

    print("\nSummary:")
    print("Demo 1 result:", result)
    print("Demo 2 result:", result2)

if __name__ == "__main__":
    demo()


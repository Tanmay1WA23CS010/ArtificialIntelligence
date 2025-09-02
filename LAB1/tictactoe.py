import random  # Import random module to allow computer to pick random moves

# Initialize the board as a 3x3 2D list filled with zeros (0 means empty cell)
board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

# List of all winning position combinations based on numbered board positions
win_combinations = [
    [1, 2, 3],  # Top row
    [4, 5, 6],  # Middle row
    [7, 8, 9],  # Bottom row
    [1, 4, 7],  # Left column
    [2, 5, 8],  # Middle column
    [3, 6, 9],  # Right column
    [1, 5, 9],  # Left diagonal
    [3, 5, 7]   # Right diagonal
]

# Function to print the current state of the board in a user-friendly way
def print_board():
    for row in board:
        display = []  # Temporary list to hold string representations of cells
        for cell in row:
            if cell == 1:  # If cell contains 1, print 'X' (user)
                display.append('X')
            elif cell == 2:  # If cell contains 2, print 'O' (computer)
                display.append('O')
            else:  # If cell is 0 (empty), print a blank space
                display.append(' ')
        print(" | ".join(display))  # Join the cells with vertical bars for board format
        print("-" * 9)  # Print a line separator between rows

# Convert a board position number (1-9) into row and column indices for 2D list
def get_position_coords(pos):
    row = (pos - 1) // 3  # Calculate row index (0 to 2)
    col = (pos - 1) % 3   # Calculate column index (0 to 2)
    return row, col

# Check if a given player (1 or 2) has won the game by matching any winning combo
def check_win(player):
    positions = []  # List to store all board positions occupied by the player
    for i in range(3):  # Loop through each row
        for j in range(3):  # Loop through each column
            if board[i][j] == player:  # If player's mark found at this cell
                pos = i * 3 + j + 1  # Convert coordinates to position number (1-9)
                positions.append(pos)  # Add position to player's list
    for combo in win_combinations:  # Check all winning combinations
        if all(pos in positions for pos in combo):  # If player has all positions in a combo
            return True  # Player has won
    return False  # No winning combination found

# Get a list of all empty positions (available moves) on the board
def get_available_moves():
    moves = []  # List to store available position numbers
    for i in range(3):  # Loop through rows
        for j in range(3):  # Loop through columns
            if board[i][j] == 0:  # If cell is empty
                moves.append(i * 3 + j + 1)  # Add its position number to available moves
    return moves  # Return list of free spots

# Handle the user's move by asking for input and placing an 'X' if valid
def user_move():
    while True:  # Keep looping until a valid move is made
        try:
            move = int(input("Enter your move (1-9): "))  # Ask user for move input
            if move < 1 or move > 9:  # Check if input is out of range
                print("Invalid input. Choose a number between 1 and 9.")
                continue  # Ask again
            row, col = get_position_coords(move)  # Convert move to row,col
            if board[row][col] == 0:  # If the spot is empty
                board[row][col] = 1  # Place user's mark (1) on the board
                break  # Exit loop since move is valid
            else:
                print("That spot is already taken.")  # Spot occupied, ask again
        except ValueError:  # Catch non-integer inputs
            print("Please enter a valid number.")  # Prompt again

# Handle the computer's move by randomly choosing one of the available spots
def computer_move():
    available = get_available_moves()  # Get list of free spots
    move = random.choice(available)  # Pick one random move from available ones
    row, col = get_position_coords(move)  # Convert position to row,col
    board[row][col] = 2  # Place computer's mark (2) on the board
    print(f"Computer chose position {move}")  # Show computer's choice

# Main function to play the Tic Tac Toe game
def play_game():
    print("Tic Tac Toe - You (X) vs Computer (O)")  # Game start message
    print_board()  # Display empty board initially
    
    for turn in range(9):  # Maximum of 9 moves possible on the board
        if turn % 2 == 0:  # Even turns: user's move
            user_move()
        else:  # Odd turns: computer's move
            computer_move()
        
        print_board()  # Show updated board after each move
        
        if check_win(1):  # Check if user has won
            print("You win!")
            return  # End the game
        elif check_win(2):  # Check if computer has won
            print("Computer wins!")
            return  # End the game

    print("It's a draw!")  # If loop ends with no winner, it's a draw

# Start the game by calling the main function
play_game()




# i need to make a tic tac toe game using Ai, where there are 2 players, one is the computer and one
#  is the user, i want to use 2d array in order to do this, 
# use 1 for X and 2 for Y, it shouol be a siple code where either
#  the computer wins or the user wins, all win or loose cases must be cinsidered
#the board is numbered :
#  1  2  3
#  4  5  6
#  7  8  9
#and the winning cases are if the player has [1,2,3]. [4,5,6], [7,8,9], [1,4,7], [2,5,8], [3,6,9]
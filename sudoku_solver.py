import math

def find_next_empty(board, empty_val=0):
    # Iterate through the board to find the first empty cell (0 by default)
    num_rows = len(board)
    num_cols = len(board[0])
    
    for row in range(num_rows):
        for col in range(num_cols):
            if board[row][col] == empty_val:
                return (row, col)
                
    return None

def is_valid_number(board, number, position):
    '''
    Check if a number is valid at a specified position on the board.
    
    Args:
        board (list[list[int]]): Current puzzle state
        number (int): Number to test
        position (tuple): (row, col) position to insert number
    Returns:
        bool: True if number is valid, False otherwise
    '''
    num_rows = len(board)
    square_size = int(math.sqrt(num_rows))
    
    row_idx, col_idx = position
    
    # Check row
    if number in board[row_idx]:
        return False
    
    # Check column
    current_column_values = [board[row][col_idx] for row in range(num_rows)]
    if number in current_column_values:
        return False
    
    # Check 3x3 square
    square_x_idx = col_idx // square_size
    square_y_idx = row_idx // square_size
    for row in range(square_y_idx * square_size, (square_y_idx + 1) * square_size):
        for col in range(square_x_idx * square_size, (square_x_idx + 1) * square_size):
            if board[row][col] == number and (row, col) != position:
                return False
    
    return True

def solve(board):
    '''
    Solve the Sudoku puzzle and return the solved board.
    
    Args:
        board (list[list[int]]): 9x9 Sudoku board (0 for empty cells)
    Returns:
        list[list[int]]: Solved 9x9 Sudoku board
    '''
    # Create a deep copy to avoid modifying the input
    board = [row[:] for row in board]
    
    # Base case: no empty cells
    next_empty_pos = find_next_empty(board)
    
    if not next_empty_pos:
        return board
    
    row, col = next_empty_pos
    
    # Try numbers 1-9
    for i in range(1, 10):
        if is_valid_number(board, i, (row, col)):
            board[row][col] = i
            # Recursively solve
            result = solve(board)
            if result:
                return result
            # Backtrack
            board[row][col] = 0
    
    return False

# # Example usage
# if __name__ == "__main__":
#     # Example Sudoku board (0 for empty cells)
#     puzzle = [
#         [5,3,0,0,7,0,0,0,0],
#         [6,0,0,1,9,5,0,0,0],
#         [0,9,8,0,0,0,0,6,0],
#         [8,0,0,0,6,0,0,0,3],
#         [4,0,0,8,0,3,0,0,1],
#         [7,0,0,0,2,0,0,0,6],
#         [0,6,0,0,0,0,2,8,0],
#         [0,0,0,4,1,9,0,0,5],
#         [0,0,0,0,8,0,0,7,9]
#     ]
    
#     solved = solve(puzzle)
#     if solved:
#         for row in solved:
#             print(row)
#     else:
#         print("No solution exists")
# Based on code from https://www.askpython.com/python/examples/sudoku-solver-in-python
M = 9


def puzzle(a):
    for i in range(M):
        for j in range(M):
            print(a[i][j],end = " ")
        print()


def solve(grid, row, col, num):
    # Check for collisions in the column
    for x in range(M):
        if grid[row][x] == num:
            return False
                     
    # Check for collisions in the row
    for x in range(M):
        if grid[x][col] == num:
            return False

    # Check for collisions in the box
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False

    return True
 

def Suduko(grid, row, col):
    # Check if the sudoku is solved by looking at 
    # which row and col is being considered now.
    if (row == M - 1 and col == M):
        return True
    # Handle column overflow by proceeding to the next row.
    if col == M:
        row += 1
        col = 0
    # If there already is a number in the current location
    # move to the next location.
    if grid[row][col] > 0:
        return Suduko(grid, row, col + 1)
    # Try all possible numbers in the current position and then 
    # check if it works by calling the Sudoku function recursively.
    for num in range(1, M + 1, 1): 
        if solve(grid, row, col, num):
            grid[row][col] = num
            if Suduko(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
 

def test_sudoku_solver():
    '''0 means the cells where no value is assigned'''
    grid = [[2, 5, 0, 0, 3, 0, 9, 0, 1],
            [0, 1, 0, 0, 0, 4, 0, 0, 0],
            [4, 0, 7, 0, 0, 0, 2, 0, 8],
            [0, 0, 5, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 8, 1, 0, 0],
            [0, 4, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 3, 6, 0, 0, 7, 2],
            [0, 7, 0, 0, 0, 0, 0, 0, 3],
            [9, 0, 3, 0, 0, 0, 6, 0, 4]]

    if (Suduko(grid, 0, 0)):
        puzzle(grid)
    else:
        print("Solution does not exist:(")

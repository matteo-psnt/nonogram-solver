import numpy as np

"""
0 = _
1 = ■
-1 = ⨯

board = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]]

top_clues = np.array([[1], [3], [1, 2], [1, 2], [2, 1]], dtype=object)
side_clues = np.array([[3], [1], [2], [3], [4]], dtype=object)

top_clues = np.array([[2, 2], [2], [4], [1], [2]], dtype=object)
side_clues = np.array([[2], [3, 1], [3], [1, 1], [1, 1]], dtype=object)
"""


# checks if line is complete and fills in blanks
def complete_line(line, clues):
    if np.sum([line != 0]) == len(line):
        return True

    if np.sum([line == 1]) == np.sum(clues):
        line[line == 0] = -1
        return True

    return False


# Uses varius methods to do an initial run through which can not be repeated
def initial_line_fill(line, clues):
    # Takes a line and uses "simple boxes" to fill any cells
    if len(clues) == 1:
        line[len(line) - clues[0]: clues[0]] = 1

    # calculates if a line is full and then fills it
    if np.sum(clues) + len(clues) - 1 == len(line):
        line[:] = 1
        pos = 0
        for clue in clues[:-1]:
            pos = pos + clue
            line[pos] = -1
            pos += 1


def side_line_fill(line, clues):
    # Side fills
    if complete_line(line, clues):
        return True

    if line[0] == 1:
        line[:clues[0]] = 1

    if line[-1] == 1:
        line[-clues[-1]:] = 1


def is_valid_line(line, clues):

    total_clues = np.sum(clues)
    total_filled = np.sum([line == 1])
    total_empty = np.sum([line == 0])

    if total_filled > total_clues:
        return False

    if total_empty + total_filled < total_clues:
        return False

    if total_empty == 0 and total_filled != total_clues:
        return False


    psb_spots = []
    spot_open = 0
    for pos in line:
        if pos == 0:
            spot_open += 1
        elif pos == 1:
            spot_open += 1
        elif pos == -1:
            psb_spots.append(spot_open)
            spot_open = 0
    if spot_open != 0:
        psb_spots.append(spot_open)

    i = 0
    for spot in range(len(psb_spots)):
        if i >= len(clues):
            return True

        for j in range(len(clues) - i, 0, -1):
            if sum(clues[i: i + j]) + j - 1 <= psb_spots[spot]:
                i += j
                break

    if i >= len(clues):
        return True
    return False


class Nonogram:
    def __init__(self, top_clues, side_clues, board=None):
        self.top_clues = top_clues
        self.side_clues = side_clues
        self.n_rows, self.n_cols = len(top_clues), len(side_clues)
        self.top_wdt = max(len(i) for i in top_clues)
        self.side_wdt = max(len(i) for i in side_clues)
        if board is None:
            self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        else:
            self.board = board
        self.t_board = self.board.transpose()

    # Prints board
    def print_board(self):
        for i in range(self.top_wdt, 0, -1):
            print('  ' * self.side_wdt, end='  ')
            for j in self.top_clues:
                try:
                    print(hex(j[-i]).lstrip('0x'), end=' ')
                except IndexError:
                    print('  ', end='')
            print()
        print('  ' * self.side_wdt + '┌' + '──' * self.n_rows)
        for j in range(self.n_cols):
            for i in range(self.side_wdt, 0, -1):
                try:
                    print(hex(self.side_clues[j][-i]).lstrip('0x'), end=' ')
                except IndexError:
                    print('  ', end='')
            print('│ ', end='')
            for sqr in self.board[j]:
                if sqr == 0:
                    print(' ', end=' ')
                elif sqr == 1:
                    print('■', end=' ')
                elif sqr == -1:
                    print('⨯', end=' ')
            print()

    # Finds the next empty box
    def find_empty(self):
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                if self.board[row][col] == 0:
                    return row, col  # row, col
        return False

    # initial_line_fill on each row and col
    def initial_fill(self):
        for i in range(self.n_rows):
            initial_line_fill(self.t_board[i], self.top_clues[i])

        for i in range(self.n_cols):
            initial_line_fill(self.board[i], self.side_clues[i])

    # Completes line and final_line_fill on each row and col
    def side_fill(self):
        for i in range(self.n_rows):
            side_line_fill(self.t_board[i], self.top_clues[i])

        for i in range(self.n_cols):
            side_line_fill(self.board[i], self.side_clues[i])

        for i in range(self.n_rows):
            complete_line(self.t_board[i], self.top_clues[i])

        for i in range(self.n_cols):
            complete_line(self.board[i], self.side_clues[i])

    def is_valid_line_cached(self, line, clue, cache={}):
        key = (tuple(line), tuple(clue))
        if key in cache:
            return cache[key]

        # perform the validation
        result = is_valid_line(line, clue)

        # cache the result
        cache[key] = result
        return result

    def valid(self, pos):
        col = self.t_board[pos[1]]
        row = self.board[pos[0]]
        col_clue = self.top_clues[pos[1]]
        row_clue = self.side_clues[pos[0]]
        if is_valid_line(col, col_clue) and is_valid_line(row, row_clue):
            return True
        return False

    def rec_solve(self):
        empty = self.find_empty()
        if not empty:
            return True
        row, col = empty

        for typ in [-1, 1]:
            self.board[row][col] = typ

            if not self.valid((row, col)):
                self.board[row][col] = 0

            else:
                if self.rec_solve():
                    return True

                self.board[row][col] = 0

        return False

    def solve_board(self, showPreRec=False):
        self.initial_fill()
        self.side_fill()
        if showPreRec:
            self.print_board()
        self.rec_solve()
        self.print_board()


tc = np.array([[2, 2], [2], [4], [1], [2]], dtype=object)
sc = np.array([[2], [3, 1], [3], [1, 1], [1, 1]], dtype=object)
n1 = Nonogram(tc, sc)

tc = np.array([[1], [3], [1, 2], [1, 2], [2, 1]], dtype=object)
sc = np.array([[3], [1], [2], [3], [4]], dtype=object)
n2 = Nonogram(tc, sc)

tc = np.array([[3], [4], [1, 2], [2, 1], [1, 2], [2, 2, 1], [2, 1, 2], [3, 1, 1, 1], [4, 5], [1, 5]], dtype=object)
sc = np.array([[3], [9], [2, 1, 4], [3, 2], [5, 1], [6], [2], [4], [2, 1], [2]], dtype=object)
n3 = Nonogram(tc, sc)

tc = np.array([[6, 1],[3, 6],[3, 6],[3, 6],[4, 3, 1],[2, 3, 2, 1],[4, 3, 1, 1, 1],[9, 1, 1],[10, 1, 1],[10, 1, 1],[6, 3, 1],[5, 6],[3, 6],[1, 7],[2, 3, 1]], dtype=object)
sc = np.array([[3],[5],[6],[6],[3, 6],[12],[14],[2, 8, 1],[1, 3, 1],[4, 1, 2, 3],[6, 6],[6, 5],[5, 9],[3, 3],[15]], dtype=object)
n4 = Nonogram(tc, sc)

tc = np.array([[2], [3], [5], [7], [2, 9], [4, 4, 4], [1, 7], [3, 3], [1, 4, 4], [4, 1, 1, 2], [3, 3, 1, 2], [1, 7], [5], [3], [2]], dtype=object)
sc = np.array([[7], [2, 1, 3], [6], [2, 2], [3], [5], [4, 1], [3, 1], [4, 3], [4, 3], [5, 4], [6, 7], [4, 2, 2], [4, 1, 3], [3, 1, 2]], dtype=object)
n5 = Nonogram(tc, sc)


# img_c.NonogramImage(pass)

if __name__ == "__main__":
    print("\nNonogram: n1")
    n1.solve_board()

    print("\nNonogram: n2")
    n2.solve_board()

    print("\nNonogram: n3")
    n3.solve_board()

    print("\nNonogram: n4")
    n4.solve_board()

    print("\nNonogram: n5")
    n5.solve_board(True)

def brute_force(line, clues):
    pass


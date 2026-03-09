import numpy as np
from typing import Optional
import random

ROWS = 6
COLS = 7
EMPTY = 0
INAROW = 4  # number of pieces a player needs to get in a row in order to win


class Player:
    """
    Connect 4 player.
    """

    def __init__(self, player_id: int):
        self.name = f"P{player_id}"
        self.symbol = player_id

    def __str__(self):
        return self.name


P1 = Player(1)
P2 = Player(2)


def check_winner(board: np.ndarray, player: Player) -> bool:
    """Return True if `player` has four in a row."""
    b = board

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(b[r][c + i] == player.symbol for i in range(4)):
                return True

    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(b[r + i][c] == player.symbol for i in range(4)):
                return True

    # Diagonal /
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(b[r - i][c + i] == player.symbol for i in range(4)):
                return True

    # Diagonal \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(b[r + i][c + i] == player.symbol for i in range(4)):
                return True

    return False


def get_winning_cells(board: np.ndarray, player: Player) -> list[tuple[int, int]]:
    """Return list of (row, col) cells forming the winning four. Empty list if no win."""
    b = board
    for r in range(ROWS):
        for c in range(COLS - 3):
            cells = [(r, c + i) for i in range(4)]
            if all(b[rc][cc] == player.symbol for rc, cc in cells):
                return cells
    for r in range(ROWS - 3):
        for c in range(COLS):
            cells = [(r + i, c) for i in range(4)]
            if all(b[rc][cc] == player.symbol for rc, cc in cells):
                return cells
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            cells = [(r - i, c + i) for i in range(4)]
            if all(b[rc][cc] == player.symbol for rc, cc in cells):
                return cells
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            cells = [(r + i, c + i) for i in range(4)]
            if all(b[rc][cc] == player.symbol for rc, cc in cells):
                return cells
    return []


# Calculates score if agent drops piece in selected column
def score_move(game, col, mark):
    next_grid = drop_piece(game, col, mark)
    score = get_heuristic(next_grid, mark)
    return score


# Helper function for score_move: gets board at next step if agent drops piece in selected column
def drop_piece(game, col, mark):
    next_grid = game.copy()
    for row in range(game.rows - 1, -1, -1):
        if next_grid.board[row][col] == 0:
            break
    next_grid.board[row][col] = mark  # type: ignore
    return next_grid


# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark):
    num_threes = count_windows(grid, 3, mark)
    num_fours = count_windows(grid, 4, mark)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1)
    score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours
    return score


# Helper function for get_heuristic: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, game):
    return (
        window.count(piece) == num_discs and window.count(0) == game.inarow - num_discs
    )


# Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
def count_windows(game, num_discs, piece):
    num_windows = 0
    # horizontal
    for row in range(game.rows):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(game.board[row, col : col + game.inarow])
            if check_window(window, num_discs, piece, game):
                num_windows += 1
    # vertical
    for row in range(game.rows - (game.inarow - 1)):
        for col in range(game.cols):
            window = list(game.board[row : row + game.inarow, col])
            if check_window(window, num_discs, piece, game):
                num_windows += 1
    # positive diagonal
    for row in range(game.rows - (game.inarow - 1)):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(
                game.board[range(row, row + game.inarow), range(col, col + game.inarow)]
            )
            if check_window(window, num_discs, piece, game):
                num_windows += 1
    # negative diagonal
    for row in range(game.inarow - 1, game.rows):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(
                game.board[
                    range(row, row - game.inarow, -1), range(col, col + game.inarow)
                ]
            )
            if check_window(window, num_discs, piece, game):
                num_windows += 1
    return num_windows


class Connect4:
    """
    Connect 4 game engine.
    Board is a (ROWS x COLS) numpy array.
    Designed to be used as an RL environment — see step(), reset(), render().
    """

    def __init__(self):
        self.board: np.ndarray = np.zeros((ROWS, COLS), dtype=int)
        self.current_player: Player = random.choice([P1, P2])
        self.done: bool = False
        self.winner: Optional[Player] = None
        self.cols: int = COLS
        self.rows: int = ROWS
        self.inarow: int = INAROW

    def copy(self) -> "Connect4":
        new_game = Connect4()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.cols = self.cols
        new_game.rows = self.rows
        new_game.inarow = self.inarow
        return new_game

    # ------------------------------------------------------------------
    # Core RL-style interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the game and return the initial board state."""
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.current_player = random.choice([P1, P2])
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, col: int) -> tuple[np.ndarray, bool, dict]:
        """
        Drop a piece in `col` for the current player.

        Returns:
            observation  : board state (np.ndarray)
            reward       : +1 win, -1 loss, 0 draw/ongoing
            done         : whether the game has ended
            info         : {'winner': int | None, 'valid_moves': list[int]}
        """
        if self.done:
            raise RuntimeError("Game is over. Call reset().")
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is full or out of range.")
        if col not in self.get_valid_moves():
            self.done = True
            self.winner = P2 if self.current_player == P1 else P1
            info = {"winner": self.winner, "valid_moves": self.get_valid_moves()}
            return self.board.copy(), self.done, info

        self._drop_piece(col, self.current_player)

        if check_winner(self.board, self.current_player):
            self.done = True
            self.winner = self.current_player
        elif not self.get_valid_moves():
            # Draw
            self.done = True
        else:
            self.current_player = P2 if self.current_player == P1 else P1

        info = {"winner": self.winner, "valid_moves": self.get_valid_moves()}
        return self.board.copy(), self.done, info

    # ------------------------------------------------------------------
    # Game logic helpers
    # ------------------------------------------------------------------

    def get_valid_moves(self) -> list[int]:
        """Return list of columns that are not full."""
        return [c for c in range(COLS) if self.board[0][c] == EMPTY]

    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < COLS and self.board[0][col] == EMPTY

    def _drop_piece(self, col: int, player: Player) -> int:
        """Drop piece into column; returns the row it landed on."""
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = player.symbol
                return row
        raise ValueError(f"Column {col} is full.")

    # ------------------------------------------------------------------
    # Terminal rendering
    # ------------------------------------------------------------------

    def render(self):
        symbols = {EMPTY: ".", 1: "X", 2: "O"}
        print("\n" + "  ".join(str(c + 1) for c in range(COLS)))
        for row in self.board:
            print("  ".join(symbols[cell] for cell in row))
        print()
        if self.done:
            if self.winner:
                print(
                    f"Player {self.winner} ({'X' if self.winner == P1 else 'O'}) wins!"
                )
            else:
                print("It's a draw!")
        else:
            print(
                f"Player {self.current_player}'s turn ({'X' if self.current_player == P1 else 'O'})"
            )

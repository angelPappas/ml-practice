import numpy as np
from typing import Optional
import random

ROWS = 6
COLS = 7
EMPTY = 0


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
            print("Ay yo")
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
                print(f"Player {self.winner} ({'X' if self.winner == P1 else 'O'}) wins!")
            else:
                print("It's a draw!")
        else:
            print(f"Player {self.current_player}'s turn ({'X' if self.current_player == P1 else 'O'})")

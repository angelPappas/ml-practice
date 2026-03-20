import numpy as np
from typing import Optional
import random

import gymnasium
from gymnasium import spaces

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
    """Return True if `player` has INAROW pieces (4) in a row in any direction."""

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - (INAROW - 1)):
            if all(board[r][c + i] == player.symbol for i in range(INAROW)):
                return True

    # Vertical
    for r in range(ROWS - (INAROW - 1)):
        for c in range(COLS):
            if all(board[r + i][c] == player.symbol for i in range(INAROW)):
                return True

    # Diagonal /
    for r in range(INAROW - 1, ROWS):
        for c in range(COLS - (INAROW - 1)):
            if all(board[r - i][c + i] == player.symbol for i in range(INAROW)):
                return True

    # Diagonal \
    for r in range(ROWS - (INAROW - 1)):
        for c in range(COLS - (INAROW - 1)):
            if all(board[r + i][c + i] == player.symbol for i in range(INAROW)):
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


def score_move(game, col, mark):
    """Calculates score if agent drops piece in selected column."""
    next_grid = drop_piece(game, col, mark)
    score = get_heuristic(next_grid, mark)
    return score


def score_move_minimax(game, col, mark, n_steps):
    """Calculates score if agent drops piece in selected column."""
    next_grid = drop_piece(game, col, mark)
    score = minimax(next_grid, n_steps - 1, False, mark, game)
    return score


def drop_piece(game, col, mark):
    """Helper function for score_move: gets board at next step if agent drops piece in selected column."""
    next_grid = game.copy()
    for row in range(game.rows - 1, -1, -1):
        if next_grid.board[row][col] == 0:
            break
    next_grid.board[row][col] = mark  # type: ignore
    return next_grid


def get_heuristic(grid, mark):
    """Helper function for score_move: calculates value of heuristic for grid."""
    num_threes = count_windows(grid, 3, mark)
    num_fours = count_windows(grid, 4, mark)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1)
    score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours
    return score


def get_heuristic_minimax(grid, mark):
    """Helper function for score_move: calculates value of heuristic for grid."""
    num_threes = count_windows(grid, 3, mark)
    num_fours = count_windows(grid, 4, mark)
    num_threes_opp = count_windows(grid, 3, mark % 2 + 1)
    num_fours_opp = count_windows(grid, 4, mark % 2 + 1)
    score = num_threes - 1e2 * num_threes_opp + 1e6 * num_fours - 1e4 * num_fours_opp
    return score


def check_window(window, num_discs, piece, game):
    """Helper function for get_heuristic: checks if window satisfies heuristic conditions."""
    return (
        window.count(piece) == num_discs and window.count(0) == game.inarow - num_discs
    )


def count_windows(game, num_discs, piece):
    """Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions."""
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


def is_terminal_window(window, game):
    """Helper function for minimax: checks if agent or opponent has four in a row in the window."""
    return (
        window.count(game.current_player.symbol) == game.inarow
        or window.count(game.current_player.symbol % 2 + 1) == game.inarow
    )


def is_terminal_node(game):
    """Helper function for minimax: checks if game has ended."""
    # Check for draw
    if list(game.board[0, :]).count(0) == 0:
        return True
    # Check for win: horizontal, vertical, or diagonal
    # horizontal
    for row in range(game.rows):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(game.board[row, col : col + game.inarow])
            if is_terminal_window(window, game):
                return True
    # vertical
    for row in range(game.rows - (game.inarow - 1)):
        for col in range(game.cols):
            window = list(game.board[row : row + game.inarow, col])
            if is_terminal_window(window, game):
                return True
    # positive diagonal
    for row in range(game.rows - (game.inarow - 1)):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(
                game.board[range(row, row + game.inarow), range(col, col + game.inarow)]
            )
            if is_terminal_window(window, game):
                return True
    # negative diagonal
    for row in range(game.inarow - 1, game.rows):
        for col in range(game.cols - (game.inarow - 1)):
            window = list(
                game.board[
                    range(row, row - game.inarow, -1), range(col, col + game.inarow)
                ]
            )
            if is_terminal_window(window, game):
                return True
    return False


def minimax(node, depth, maximizingPlayer, mark, game):
    """Minimax implementation"."""
    is_terminal = is_terminal_node(game)
    valid_moves = game.get_valid_moves()
    if depth == 0 or is_terminal:
        return get_heuristic_minimax(node, mark)
    if maximizingPlayer:
        value = -np.inf
        for col in valid_moves:
            child = drop_piece(node, col, mark)
            value = max(value, minimax(child, depth - 1, False, mark, game))
        return value
    else:
        value = np.inf
        for col in valid_moves:
            child = drop_piece(node, col, mark % 2 + 1)
            value = min(value, minimax(child, depth - 1, True, mark, game))
        return value


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


class ConnectFourGym(gymnasium.Env):
    def __init__(self, agent2="random"):
        self.game = Connect4()
        self.rows = ROWS
        self.columns = COLS
        self.agent2 = agent2  # "random" or a callable(board, valid_moves) -> col

        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(1, self.rows, self.columns), dtype=np.int_
        )
        self.reward_range = (-10, 1)
        self.spec = None

    def _get_obs(self):
        return self.game.board.reshape(1, self.rows, self.columns).copy()

    def _run_opponent(self):
        """Let agent2 play its turn."""
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return
        if self.agent2 == "random":
            col = random.choice(valid_moves)
        elif callable(self.agent2):
            col = self.agent2(self.game.board, valid_moves)
        else:
            col = random.choice(valid_moves)
        self.game.step(col)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.game.reset()

        if self.game.current_player == P2:
            self._run_opponent()

        return self._get_obs(), {}

    def change_reward(self, done):
        if self.game.winner == P1:  # agent won
            return 1
        elif done:  # opponent won or draw
            return -1
        else:
            return 1 / (self.rows * self.columns)

    def step(self, action):
        action = int(action)
        if not self.game.is_valid_move(action):
            return self._get_obs(), -10, True, False, {}

        _, done, _ = self.game.step(action)

        if not done:
            self._run_opponent()
            done = self.game.done

        reward = self.change_reward(done)
        return self._get_obs(), reward, done, False, {}

import numpy as np
import random
from typing import Callable
from game.game import Connect4, ROWS, COLS, P1, P2, get_winning_cells, score_move

# ── Built-in agents ───────────────────────────────────────────────────


def random_agent(game: Connect4) -> int:
    """Picks a random valid column."""
    return random.choice(list(range(game.cols)))


def agent_leftmost(game: Connect4) -> int:
    """Always picks the leftmost valid column."""
    return game.get_valid_moves()[0]


def agent_middle(game: Connect4) -> int:
    """Picks the middle column."""
    return game.cols // 2


def agent_heuristic(game: Connect4) -> int:
    """Heuristic agent."""
    # Get list of valid moves
    valid_moves = game.get_valid_moves()
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(
        zip(
            valid_moves,
            [score_move(game, col, game.current_player.symbol) for col in valid_moves],
        )
    )
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)


BUILTIN_AGENTS = {
    "random": random_agent,
    "leftmost": agent_leftmost,
    "middle": agent_middle,
    "heuristic": agent_heuristic,
}


class ConnectXEnv:
    """
    Kaggle-style Connect 4 environment.

    Usage:
        env = ConnectXEnv()
        env.run(["random", "random"])
        env.render(mode="ipython")   # animated in Jupyter
        env.render(mode="terminal")  # ASCII in terminal
    """

    def __init__(self):
        self.game = Connect4()
        self.history: list[np.ndarray] = []  # board snapshot after each move

    # ------------------------------------------------------------------
    # Run a full game between two agents
    # ------------------------------------------------------------------

    def run(self, agents: list) -> dict:
        """
        Run a complete game between two agents.

        Args:
            agents: list of 2 items — each is either:
                    - a string: "random" | "leftmost"
                    - a callable: fn(obs, config) -> int

        Returns:
            result dict with winner, num_moves, history
        """
        resolved = [self._resolve_agent(a) for a in agents]

        self.game.reset()
        self.history = [self.game.board.copy()]

        while not self.game.done:
            agent_idx = self.game.current_player.symbol - 1  # P1=0, P2=1
            agent_fn = resolved[agent_idx]

            col = agent_fn(self.game)

            self.game.step(col)
            self.history.append(self.game.board.copy())

        return {
            "winner": self.game.winner,
            "num_moves": len(self.history) - 1,
            "history": self.history,
        }

    def run_n_times(self, agents: list, num_games: int) -> dict:
        """Run multiple games and return list of results."""
        if num_games <= 0:
            raise ValueError("num_games must be >= 1 to use run_n_times()")
        elif num_games == 1:
            return self.run(agents)

        agent_1_wins = 0
        agent_2_wins = 0
        draws = 0

        for _ in range(num_games):
            result = self.run(agents)
            if result["winner"] == P1:
                agent_1_wins += 1
            elif result["winner"] == P2:
                agent_2_wins += 1
            else:
                draws += 1
        return {
            "agent_1_wins": agent_1_wins,
            "agent_2_wins": agent_2_wins,
            "draws": draws,
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "ipython"):
        if mode == "ipython":
            self._render_ipython()
        elif mode == "terminal":
            self._render_terminal()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'ipython' or 'terminal'.")

    def _render_ipython(self):
        """Renders an interactive animation widget in a Jupyter notebook."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.animation import FuncAnimation
            from IPython.display import HTML, display
        except ImportError:
            raise ImportError(
                "Install matplotlib and ipython: pip install matplotlib ipython"
            )

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#0f172a")

        # ── Colours ───────────────────────────────────────────────────
        BOARD_COLOR = "#1d4ed8"
        HOLE_COLOR = "#0f172a"
        P1_COL = "#ef4444"
        P2_COL = "#f59e0b"
        TEXT_COL = "#e2e8f0"

        def draw_frame(step_idx):
            ax.clear()
            ax.set_facecolor("#0f172a")
            ax.set_xlim(-0.5, COLS - 0.5)
            ax.set_ylim(-0.5, ROWS - 0.5)
            ax.set_aspect("equal")
            ax.axis("off")

            board = self.history[step_idx]

            # Board background
            board_rect = patches.FancyBboxPatch(
                (-0.55, -0.55),
                COLS + 0.1,
                ROWS + 0.1,
                boxstyle="round,pad=0.1",
                linewidth=0,
                facecolor=BOARD_COLOR,
            )
            ax.add_patch(board_rect)

            # Determine winning cells for last frame
            winning_cells = set()
            if step_idx == len(self.history) - 1 and self.game.winner:
                wc = get_winning_cells(board, self.game.winner)
                winning_cells = set(wc)

            # Draw circles (row 0 = top, so we flip y)
            for r in range(ROWS):
                for c in range(COLS):
                    val = board[r][c]
                    y = ROWS - 1 - r  # flip so row 0 is top

                    if val == P1.symbol:
                        color = P1_COL
                        zorder = 3
                    elif val == P2.symbol:
                        color = P2_COL
                        zorder = 3
                    else:
                        color = HOLE_COLOR
                        zorder = 2

                    radius = 0.42 if (r, c) in winning_cells else 0.38
                    circle = patches.Circle((c, y), radius, color=color, zorder=zorder)
                    ax.add_patch(circle)

            # Draw winning line
            if winning_cells:
                cells = sorted(winning_cells)
                xs = [c for r, c in cells]
                ys = [ROWS - 1 - r for r, c in cells]
                ax.plot(
                    xs,
                    ys,
                    color="white",
                    linewidth=4,
                    alpha=0.8,
                    solid_capstyle="round",
                    zorder=4,
                )

            # Status text
            total = len(self.history) - 1
            if step_idx == len(self.history) - 1 and self.game.done:
                if self.game.winner:
                    pname = (
                        "Player 1 (Red)"
                        if self.game.winner == P1
                        else "Player 2 (Yellow)"
                    )
                    title = f"{pname} wins!"
                else:
                    title = "Draw!"
            else:
                title = f"Move {step_idx} / {total}"

            ax.set_title(title, color=TEXT_COL, fontsize=14, fontweight="bold", pad=10)

            return []  # Return empty list since we clear and redraw everything

        anim = FuncAnimation(
            fig,
            draw_frame,
            frames=len(self.history),
            interval=500,  # ms per frame
            repeat=True,
        )

        plt.close(fig)
        html = anim.to_jshtml()
        display(HTML(html))

    def _render_terminal(self):
        """Print each board state to the terminal."""
        symbols = {0: ".", 1: "X", 2: "O"}
        for i, board in enumerate(self.history):
            print(f"\n── Move {i} ──")
            print("  ".join(str(c + 1) for c in range(COLS)))
            for row in board:
                print("  ".join(symbols[v] for v in row))
        if self.game.winner:
            print(
                f"\n🏆 Player {self.game.winner} wins in {len(self.history) - 1} moves!"
            )
        else:
            print("\n🤝 Draw!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_agent(self, agent) -> Callable:
        if isinstance(agent, str):
            if agent not in BUILTIN_AGENTS:
                raise ValueError(
                    f"Unknown agent '{agent}'. Available: {list(BUILTIN_AGENTS)}"
                )
            return BUILTIN_AGENTS[agent]
        if callable(agent):
            return agent
        raise TypeError(f"Agent must be a string or callable, got {type(agent)}")

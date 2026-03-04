import sys
import pygame
import numpy as np
from game import Connect4, ROWS, COLS, EMPTY, P1, P2

# ── Layout ────────────────────────────────────────────────────────────
CELL = 100  # pixels per cell
RADIUS = CELL // 2 - 8
WIDTH = COLS * CELL
HEIGHT = (ROWS + 1) * CELL  # +1 row at top for the drop preview
FPS = 60

# ── Colors ────────────────────────────────────────────────────────────
BG = (15, 23, 42)  # dark navy
BOARD_BLUE = (29, 78, 216)
HOLE_EMPTY = (15, 23, 42)
P1_COLOR = (239, 68, 68)  # red
P2_COLOR = (245, 158, 11)  # amber
WIN_GLOW = (255, 255, 100)
TEXT_COLOR = (226, 232, 240)
PREVIEW_ALPHA = 120


def draw_board(screen, game: Connect4, font, hover_col: int | None):
    screen.fill(BG)

    player_color = P1_COLOR if game.current_player == P1 else P2_COLOR

    # ── Drop preview (top row) ────────────────────────────────────────
    if hover_col is not None and not game.done and game.is_valid_move(hover_col):
        preview_surf = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        pygame.draw.circle(
            preview_surf, (*player_color, PREVIEW_ALPHA), (CELL // 2, CELL // 2), RADIUS
        )
        screen.blit(preview_surf, (hover_col * CELL, 0))

    # ── Blue board ────────────────────────────────────────────────────
    pygame.draw.rect(
        screen, BOARD_BLUE, (0, CELL, WIDTH, ROWS * CELL), border_radius=16
    )

    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL + CELL // 2
            cy = (r + 1) * CELL + CELL // 2
            cell_val = game.board[r][c]

            if cell_val == EMPTY:
                color = HOLE_EMPTY
            elif cell_val == P1.symbol:
                color = P1_COLOR
            else:
                color = P2_COLOR

            pygame.draw.circle(screen, color, (cx, cy), RADIUS)

    # ── Status text ───────────────────────────────────────────────────
    if game.done:
        if game.winner:
            name = "Player 1 (Red)" if game.winner == P1 else "Player 2 (Yellow)"
            msg = f"{name} wins! 🎉  Press R to restart."
        else:
            msg = "It's a draw!  Press R to restart."
    else:
        name = "Player 1 (Red)" if game.current_player == P1 else "Player 2 (Yellow)"
        msg = f"{name}'s turn"

    text = font.render(msg, True, TEXT_COLOR)
    rect = text.get_rect(center=(WIDTH // 2, CELL // 2))
    screen.blit(text, rect)

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 26, bold=True)

    game = Connect4()
    game.reset()
    hover_col = None

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                    hover_col = None

            if event.type == pygame.MOUSEMOTION:
                mx, _ = event.pos
                col = mx // CELL
                hover_col = col if 0 <= col < COLS else None

            if event.type == pygame.MOUSEBUTTONDOWN and not game.done:
                mx, my = event.pos
                if my > CELL:  # click on board, not header
                    col = mx // CELL
                    game.step(col)

        draw_board(screen, game, font, hover_col)


if __name__ == "__main__":
    main()

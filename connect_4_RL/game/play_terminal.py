from game import Connect4


def main():
    game = Connect4()
    game.reset()

    print("=== Connect 4 ===")
    print("Players: X = Player 1, O = Player 2")
    print("Enter column number (1-7) to drop a piece.\n")

    while True:
        game.render()
        if game.done:
            break

        valid = game.get_valid_moves()
        prompt = f"Player {game.current_player} — choose column {[c+1 for c in valid]}: "

        col = int(input(prompt)) - 1
        game.step(col)


if __name__ == "__main__":
    main()
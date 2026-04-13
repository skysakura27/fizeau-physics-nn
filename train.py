"""Training entry point."""

from utils.config_loader import Config


def main() -> None:
    config = Config()
    print(f"Loaded configs: {config}")
    # TODO: implement training loop (dataset, optimizer, loss, checkpoints)


if __name__ == "__main__":
    main()

"""Inference and benchmarking entry point."""

from utils.config_loader import Config


def main() -> None:
    config = Config()
    print(f"Loaded configs: {config}")
    # TODO: implement inference and benchmarking routines


if __name__ == "__main__":
    main()

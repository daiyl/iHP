import argparse

from Patient.data_generation.health_simulation import DATASET_CONFIGS, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final latent-state sequence CSV datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    args = parser.parse_args()

    for name in args.datasets:
        save_dataset(DATASET_CONFIGS[name])


if __name__ == "__main__":
    main()

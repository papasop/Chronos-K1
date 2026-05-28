from src.lejepa_baseline import BaselineConfig, describe_baseline


def main() -> None:
    print({"status": "baseline_placeholder", "config": describe_baseline(BaselineConfig())})


if __name__ == "__main__":
    main()

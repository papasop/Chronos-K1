def planning_quality(successes: int, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    return successes / trials

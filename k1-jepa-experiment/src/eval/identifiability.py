def identifiability_score(matches: int, total: int) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    return matches / total

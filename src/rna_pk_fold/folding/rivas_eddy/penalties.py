def short_hole_penalty(costs, k: int, l: int) -> float:
    h = l - k - 1
    return costs.short_hole_caps.get(h, 0.0)
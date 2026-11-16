"""Check the within-region connected gradient formula with exact arithmetic."""
from fractions import Fraction as F
from collections import Counter

def sign(x):
    return int(x > 0) - int(x < 0)

def reference(z):
    first = max(0, 2 + z)
    second = max(0, 3 - first)
    slope = -int(2 + z > 0 and 3 - first > 0)
    return second - F(1, 2), slope

def radii(units):
    result = []
    for direction in (-1, 1):
        distances = [-u / (direction * v) for u, v in units if u * direction * v < 0]
        result.append(min(distances, default=None))
    return result

def main():
    cache = F(0)
    base, slope = reference(cache)
    left, right = radii([(F(2), F(1)), (F(1), F(-1))])
    assert (left, right) == (2, 1)
    counts = Counter()
    for n in range(-48, 49):
        delta = F(n, 16)
        if not -left < delta < right:
            counts["abstain"] += 1
            continue
        fresh, fresh_slope = reference(cache + delta)
        cached_sign = sign(base - (cache + delta))
        direct = (1 - fresh_slope) * sign(fresh - (cache + delta)) - (1 - slope) * cached_sign
        predicted = (1 - slope) * (sign(base - cache + (slope - 1) * delta) - cached_sign)
        assert predicted == direct
        counts["different" if direct else "equal"] += 1
    print(dict(sorted(counts.items())))
    for delta, expected in [(F(1,8),0), (F(1,4),-2), (F(3,8),-4), (F(3,4),0)]:
        fresh, derivative = reference(delta)
        difference = (1 - derivative) * sign(fresh - delta) - (1 - slope) * sign(base - delta)
        assert difference == expected
        print(delta, difference)

if __name__ == "__main__":
    main()

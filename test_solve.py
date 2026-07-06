"""Quick check that the calculator/solve path computes correctly and skips non-math."""
import answerer

TESTS = [
    ("coin corrupted", "An unfair coin has a 23 chance of coming up heads, a 16 chance "
     "of landing on its side, and a 61 chance of coming up tails. He wins if heads, "
     "loses if tails, reflips if side. Probability he wins?", "MATH"),
    ("coin clean", "An unfair coin has a 2/3 chance of heads, 1/6 side, 1/6 tails. Win "
     "on heads, lose on tails, reflip on side. P(win)?", "MATH"),
    ("kinematics", "A ball is dropped from 80 meters. Using g=10 m/s^2, how many seconds "
     "until it hits the ground?", "PHYSICS"),
    ("combinatorics", "How many ways can you arrange the letters in the word BANANA?", "MATH"),
    ("recall (skip)", "What organelle is the powerhouse of the cell?", "BIOLOGY"),
    ("partial (skip)", "A projectile is launched at", "PHYSICS"),
]

if __name__ == "__main__":
    for name, q, cat in TESTS:
        print(f"{name:18}: {answerer.solve(q, cat)!r}")
